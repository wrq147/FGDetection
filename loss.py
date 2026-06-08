import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomYOLOLoss(nn.Module):
    def __init__(self, alpha=0.25, beta=6.0, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.ce_loss = nn.CrossEntropyLoss()

        gauss_kernel = torch.tensor([
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0]
        ], dtype=torch.float32) / 16.0
        self.register_buffer('gauss_kernel', gauss_kernel.view(1, 1, 3, 3))

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
        # 生成网格坐标 (只生成一次)
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
            dw = torch.sigmoid(pb[:, 2])  # 宽度比例
            dh = torch.sigmoid(pb[:, 3])  # 高度比例


            # 网格坐标
            cx = (xs + dx * 3 - 1.5) / feat_w  # 最终归一化 cx
            cy = (ys + dy * 3 - 1.5) / feat_h  # 最终归一化 cy         
            w = dw
            h = dh

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
            best_mask = torch.zeros(N, dtype=torch.bool, device=device)

            target_cls_idx = torch.zeros(N, dtype=torch.long, device=device)

            for m in range(M):
                cx, cy, w, h = gb[m]
                cid = gt_cids[m]
                gx = cx * feat_w
                gy = cy * feat_h

                gw = w * feat_w
                gh = h * feat_h
                radius_w = gw / 2
                radius_w = max(2.5, radius_w)
                radius_h = gh / 2
                radius_h = max(2.5, radius_h)
                # 先筛选中心候选区
                candidate_mask = (torch.abs(xs - gx) <
                                  radius_w) & (torch.abs(ys - gy) < radius_h)
                if not candidate_mask.any():
                    continue
                cand_idx = torch.where(candidate_mask)[0]
                
                # 计算分数与IOU
                cand_score = pc[cand_idx].sigmoid()
                cand_box = pred_decoded[cand_idx]
                _, iou_cand = self.bbox_iou_loss(
                    cand_box, gb[m].unsqueeze(0).expand(len(cand_idx), 4))
                
                # 只保留 iou_cand > best_iou 的
                keep_compete = iou_cand > best_iou[cand_idx]
                cand_idx = cand_idx[keep_compete]
                iou_cand = iou_cand[keep_compete]
                cand_score = cand_score[keep_compete]

                if len(cand_idx) == 0:
                    continue

                # 算align_score
                align_score = cand_score.pow(self.alpha) * iou_cand.pow(self.beta)

                # TopK
                gt_area = w * h
                topk = int(3 + gt_area * 300)
                topk = max(3, min(topk, 30))
                k = min(topk, len(align_score))
                topk_val, topk_idx = torch.topk(align_score, k)

                current_idx = cand_idx[topk_idx]
                tmp_iou = iou_cand[topk_idx]

                if len(current_idx) > 0:
                    best_iou[current_idx] = tmp_iou
                    pos_mask[current_idx] = True
                    target_box[current_idx] = gb[m]
                    target_cls_idx[current_idx] = cid

                    best_idx_in_current = torch.argmax(tmp_iou)
                    best_single_pos = current_idx[best_idx_in_current]
                    best_mask[best_single_pos] = True

            # ===================== 计算损失 =====================
            num_pos = pos_mask.sum().item()
            if num_pos > 0:
                ciou, iou = self.bbox_iou_loss(
                    pred_decoded[pos_mask], target_box[pos_mask])
                total_ciou += ciou.mean()

                cls_score[pos_mask] = torch.clamp(iou.detach(), 0.0, 1.0)
                
                cls_target[pos_mask] = 0.8
                cls_target[best_mask] = 1.0


                cls = self.varifocal_loss(pc, cls_score, cls_target)
                total_cls += cls

                mul_cls_pred = cls_btm_flat[b][:, pos_mask].transpose(0, 1)
                gt_label = target_cls_idx[pos_mask]
                mul_cls_loss = self.ce_loss(mul_cls_pred, gt_label)
                total_cls += mul_cls_loss

            else:
                pass

            num_valid += 1

        # 批次平均
        if num_valid == 0:
            return pred_box.sum() * 0.0

        total_loss = (total_ciou * 5 + total_cls) / num_valid
        return total_loss
