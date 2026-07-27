import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomYOLOLoss(nn.Module):
    def __init__(self, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma


    def bbox_iou_loss(self, pred_boxes, target_boxes, eps=1e-7):
        """
        计算 CIoU 损失（比普通 IoU 效果好很多）
        pred_boxes / target_boxes: [N, 4] (cx, cy, w, h)
        return: ciou_loss
        """
        # 解析坐标
        pred_cx, pred_cy, pred_w, pred_h = pred_boxes.unbind(1)
        tgt_cx, tgt_cy, tgt_w, tgt_h = target_boxes.unbind(1)

        # 转换为 x1, y1, x2, y2
        pred_x1 = pred_cx - pred_w / 2
        pred_y1 = pred_cy - pred_h / 2
        pred_x2 = pred_cx + pred_w / 2
        pred_y2 = pred_cy + pred_h / 2

        tgt_x1 = tgt_cx - tgt_w / 2
        tgt_y1 = tgt_cy - tgt_h / 2
        tgt_x2 = tgt_cx + tgt_w / 2
        tgt_y2 = tgt_cy + tgt_h / 2

        # 计算交集
        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)

        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter = inter_w * inter_h

        # 计算并集
        pred_area = pred_w * pred_h
        tgt_area = tgt_w * tgt_h
        union = pred_area + tgt_area - inter + eps
        iou = inter / union

        # ---------------------- DIoU：中心点距离 ----------------------
        cw = torch.max(pred_x2, tgt_x2) - torch.min(pred_x1, tgt_x1)  # 最小外接矩形宽
        ch = torch.max(pred_y2, tgt_y2) - torch.min(pred_y1, tgt_y1)  # 最小外接矩形高
        c2 = cw ** 2 + ch ** 2 + eps                                  # 对角线平方
        rho2 = (pred_cx - tgt_cx) ** 2 + (pred_cy - tgt_cy) ** 2      # 中心点距离平方
        diou = iou - rho2 / c2

        # ---------------------- CIoU：宽高比一致性 ----------------------
        v = (4 / (torch.pi ** 2)) * torch.pow(
            torch.atan(tgt_w / (tgt_h + eps)) -
            torch.atan(pred_w / (pred_h + eps)), 2
        )

        alpha = v / (1 - iou + v + eps)
        ciou = diou - alpha * v

        # CIoU 损失 = 1 - CIoU
        ciou_loss = 1.0 - ciou
        return ciou_loss, iou

    def varifocal_loss(self, pred, score, target):
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        weight = target * score + \
            (1 - target) * ((1 - self.focal_alpha) *
                            pred_sigmoid.detach() ** self.focal_gamma)

        loss = F.binary_cross_entropy_with_logits(
            pred, target, weight=weight, reduction="none")

        return loss.mean() if loss.numel() > 0 else 0.0

    def forward(self, pred_box, gt_box, feat_w, feat_h, pred_cls, cls_btm, gt_box_cls_indices):
        device = pred_box.device
        N = feat_w * feat_h
        B = pred_box.shape[0]

        total_ciou = 0
        total_cls = 0
        num_valid = 0
        # 生成网格坐标
        ys, xs = torch.meshgrid(
            torch.arange(feat_h, device=device),
            torch.arange(feat_w, device=device),
            indexing='ij'
        )
        xs = xs.reshape(-1).float()  # [N]
        ys = ys.reshape(-1).float()
        xs_norm = xs / feat_w
        ys_norm = ys / feat_h
        cls_btm_flat = cls_btm.flatten(2)
        
        for b in range(B):
            pb = pred_box[b].permute(1, 0)          # [N,4]
            pc = pred_cls[b].squeeze(0)          # [N]

            dx = torch.sigmoid(pb[:, 0])  # 中心点x偏移
            dy = torch.sigmoid(pb[:, 1])  # 中心点y偏移
            dw = pb[:, 2]  # 宽度比例
            dh = pb[:, 3]  # 高度比例

            # 网格坐标
            cx = xs_norm + dx  # 最终归一化 cx
            cy = ys_norm + dy  # 最终归一化 cy

            w = torch.clamp(torch.exp(dw) * 0.2, 1e-5, 1.0)  # 最终归一化 w
            h = torch.clamp(torch.exp(dh) * 0.2, 1e-5, 1.0)  # 最终归一化 h

            pred_decoded = torch.stack([cx, cy, w, h], dim=1)

            gb = torch.tensor(gt_box[b]).to(
                device)                          # [M,4]
            if gb.numel() == 0:
                continue
            M = gb.shape[0]

            gt_cids = torch.tensor(gt_box_cls_indices[b]).to(device)
            # ===================== 逐张图计算损失 =====================
            pos_mask = torch.zeros(N, dtype=torch.bool, device=device)
            target_box = torch.zeros(N, 4, device=device)
            cls_target = torch.zeros(N, device=device)
            cls_score = torch.zeros(N, device=device)
            best_iou = torch.zeros(N, device=device)
            target_cls_idx = torch.zeros(N, dtype=torch.long, device=device)

            for m in range(M):
                cx, cy, w, h = gb[m]
                cid = gt_cids[m]
                gx = cx * feat_w
                gy = cy * feat_h

                gw = w * feat_w
                gh = h * feat_h
                radius_w = gw / 2+0.5
                radius_w = max(2.0, radius_w)
                radius_h = gh / 2+0.5
                radius_h = max(2.0, radius_h)
                # 先筛选中心候选区
                candidate_mask = (torch.abs(xs - gx) <
                                  radius_w) & (torch.abs(ys - gy) < radius_h)

                if not candidate_mask.any():
                    continue
                cand_idx = torch.where(candidate_mask)[0]

                # 计算分数与IOU
                cand_box = pred_decoded[cand_idx]
                _, iou_cand = self.bbox_iou_loss(
                    cand_box, gb[m].unsqueeze(0).expand(len(cand_idx), 4))

                # 只保留 iou_cand > best_iou 的
                keep_compete = iou_cand > best_iou[cand_idx]
                cand_idx = cand_idx[keep_compete]
                iou_cand = iou_cand[keep_compete]

                if len(cand_idx) == 0:
                    continue

                # 算align_score
                img_cls_all = cls_btm_flat[b]
                cand_cls_logits = img_cls_all[cid, cand_idx]
                cls_sim_weight = torch.sigmoid(cand_cls_logits)
                align_score = iou_cand * torch.sqrt(cls_sim_weight + 1e-7)

                # TopK
                gt_area = w * h
                topk = int(3 + gt_area * 500)
                max_k = min(12, max(4, int(N / 64)))
                topk = min(topk, max_k, len(align_score))
                k = min(topk, len(align_score))
                topk_val, topk_idx = torch.topk(align_score, k)

                current_idx = cand_idx[topk_idx]
                tmp_iou = iou_cand[topk_idx]


                fusion_label = torch.sqrt(tmp_iou + 1e-7)

                if len(current_idx) > 0:
                    best_iou[current_idx] = tmp_iou
                    pos_mask[current_idx] = True
                    target_box[current_idx] = gb[m]
                    target_cls_idx[current_idx] = cid
                    cls_target[current_idx] = fusion_label.detach()

            # ===================== 计算损失 =====================
            num_pos = pos_mask.sum().item()
            if num_pos > 0:
                ciou, iou = self.bbox_iou_loss(
                    pred_decoded[pos_mask], target_box[pos_mask])
                total_ciou += ciou.mean()

                cls_score[pos_mask] = torch.clamp(iou.detach(), 0.0, 1.0)

                cls = self.varifocal_loss(pc, cls_score, cls_target)
                total_cls += cls

                mul_cls_pred = cls_btm_flat[b][:, pos_mask].transpose(0, 1)
                gt_label = target_cls_idx[pos_mask]

                sample_weight = cls_target[pos_mask].detach()
                # 可选增加上下限约束，防止极端值
                sample_weight = torch.clamp(sample_weight, min=0.2, max=1.0)
                log_probs = F.log_softmax(mul_cls_pred, dim=-1)
                nll_per_sample = F.nll_loss(log_probs, gt_label, reduction="none")
                weighted_nll = nll_per_sample * sample_weight
                mul_cls_loss = weighted_nll.mean()
                total_cls += mul_cls_loss

            else:
                pass

            num_valid += 1

        # 批次平均
        if num_valid == 0:
            return pred_box.sum() * 0.0

        total_loss = (total_ciou*5.0 + total_cls) / num_valid
        return total_loss
