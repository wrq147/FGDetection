import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from torchvision.ops import nms
from transformers import AutoImageProcessor, AutoTokenizer, AutoModelForCausalLM
# 同目录下你的模型与损失、数据集
from dataset import YOLODataset

import matplotlib
matplotlib.use('Agg')


class AdaptedDetectHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones(1) * 2.6592)
        self.logit_bias = nn.Parameter(torch.zeros(1))

        self.fc = nn.Sequential(
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Linear(768, 768),
        )

        self.box_head = nn.Sequential(
            nn.Conv2d(768, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.GELU(),
            nn.Conv2d(384, 384, kernel_size=3,
                      padding=1, groups=384),
            nn.BatchNorm2d(384),
            nn.GELU(),
            nn.Conv2d(384, 4, kernel_size=1)
        )

        self.cls_head = nn.Sequential(
            nn.Conv2d(768, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.GELU(),
            nn.Conv2d(192, 192, kernel_size=3, padding=1, groups=192),
            nn.BatchNorm2d(192),
            nn.GELU(),
            nn.Conv2d(192, 1, kernel_size=1)
        )

    def forward(self, last_hidden, text_feat):
        B = last_hidden.shape[0]
        featsize = int(last_hidden.shape[1] ** 0.5)  # 28
        N = text_feat.size(1)
        fc_text_feat = F.normalize(text_feat, dim=-1)
        fc_img_feat = self.fc(last_hidden)
        fc_img_feat = F.normalize(fc_img_feat, dim=-1)

        # cls_feat: [B,784,768]
        # text_feat: [B,30,768]
        # out:       [B,784,30]
        cls_sim = torch.matmul(fc_img_feat, fc_text_feat.transpose(-1, -2))
        logit_scale = self.logit_scale.exp()
        cls_sim = cls_sim * logit_scale + self.logit_bias
        cls_btm = cls_sim.view(
            B, featsize, featsize, -1).permute(0, 3, 1, 2).contiguous()  # [B,n,28,28]

        max_sim, _ = cls_sim.max(dim=-1)
        confidence = torch.sigmoid(max_sim)
        sim_mean = cls_sim.mean(dim=-1)
        sim_max_smoothed = sim_mean
        sim_max_min = sim_max_smoothed.amin(dim=1, keepdim=True)
        sim_max_max = sim_max_smoothed.amax(dim=1, keepdim=True)
        sim_mean = (sim_max_smoothed - sim_max_min) / \
            (sim_max_max - sim_max_min + 1e-8)
        real_mm = confidence * sim_mean
        mask = real_mm.unsqueeze(-1)          # [B,784,1]
        cls_feat = fc_img_feat * mask  # [B, 784, 768]
        cls_feat = cls_feat.permute(
            0, 2, 1).reshape(B, -1, featsize, featsize)


        box = self.box_head(cls_feat)  # [B,4,28,28]
        box = box.flatten(2)           # [B,4,784]

        cls_map = self.cls_head(cls_feat)
        cls_map = cls_map.flatten(2)

        return box, cls_map, cls_btm

# ===================== 全局配置(多尺度修改) =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
# 多尺度列表
img_sizes = [512, 768]
# 中心点置信度阈值（按需调）
CENTER_THRESH = 0.5

script_dir = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(script_dir, "fgmodel")

# ===================== 加载模型 =====================
fgmodel = AutoModelForCausalLM.from_pretrained(
    model_file, trust_remote_code=True
).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_file)
image_processor = AutoImageProcessor.from_pretrained(model_file)

dethead = AdaptedDetectHead().to(device).eval()
dethead.load_state_dict(torch.load(
    "dethead_yolo_best.pth", map_location=device))


# ===================== 预处理函数（和训练完全一致） =====================
def resize_and_pad(image, target_size, fill_color=(114, 114, 114)):
    w, h = image.size
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h), Image.BILINEAR)
    padded_img = Image.new("RGB", (target_size, target_size), fill_color)
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    padded_img.paste(image, (paste_x, paste_y))
    return padded_img, paste_x, paste_y, scale


def single_scale_predict(raw_img, target_size, class_names):
    """
    单尺度推理，返回该尺度下所有过滤后的预测框列表
    """
    raw_w, raw_h = raw_img.size
    featuresize = target_size // 16
    maxnumpatches = featuresize * featuresize

    pad_img, pad_x, pad_y, scale = resize_and_pad(raw_img, target_size)
    # 图像前向
    image_input = image_processor(
        images=pad_img,
        max_num_patches=maxnumpatches,
        return_tensors="pt"
    ).to(device)

    # 全部类别文本特征
    text_tokens = tokenizer(
        class_names,
        padding="max_length",
        max_length=196,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        last_hidden = fgmodel.get_vision_feature(**image_input)
        text_feat = fgmodel.get_text_features(**text_tokens, walk_type="long")
        text_feat = text_feat.unsqueeze(0)
        # 检测头前向
        pred_box, pred_cls, pred_sim = dethead(last_hidden, text_feat)
        # 解析输出
        pred_box = pred_box.squeeze(0).permute(1, 0)
        pred_sim = pred_sim.flatten(2).permute(0, 2, 1).squeeze(0)

        pred_cls_score = pred_cls.flatten()
        pred_cls_idx = pred_sim.argmax(dim=-1)
        pred_score = torch.sigmoid(pred_cls_score)  # 类别置信度

    # 遍历所有特征点，仅用 中心阈值过滤
    scale_results = []
    total_grid = featuresize * featuresize
    for grid_idx in range(total_grid):
        c_score = pred_score[grid_idx].item()
        if c_score < CENTER_THRESH:
            continue

        box_tensor = pred_box[grid_idx]
        dx = torch.sigmoid(box_tensor[0])
        dy = torch.sigmoid(box_tensor[1])
        bw = box_tensor[2]
        bh = box_tensor[3]
        w = torch.clamp(torch.exp(bw) * 0.2, 0.0, 1.0)
        h = torch.clamp(torch.exp(bh) * 0.2, 0.0, 1.0)

        gy = grid_idx // featuresize
        gx = grid_idx % featuresize

        cx = gx/featuresize + dx
        cy = gy/featuresize + dy

        # 映射到 padded 图尺度
        cx_pad = cx * target_size
        cy_pad = cy * target_size
        bw_pad = w * target_size
        bh_pad = h * target_size

        # 去除padding偏移
        cx_raw = cx_pad - pad_x
        cy_raw = cy_pad - pad_y

        # 还原原始原图尺度
        cx_orig = cx_raw / scale
        cy_orig = cy_raw / scale
        w_orig = bw_pad / scale
        h_orig = bh_pad / scale

        # 转 x1y1x2y2
        x1 = (cx_orig - w_orig / 2).cpu().item()
        y1 = (cy_orig - h_orig / 2).cpu().item()
        x2 = (cx_orig + w_orig / 2).cpu().item()
        y2 = (cy_orig + h_orig / 2).cpu().item()

        x1 = max(0.0, min(x1, raw_w))
        y1 = max(0.0, min(y1, raw_h))
        x2 = max(0.0, min(x2, raw_w))
        y2 = max(0.0, min(y2, raw_h))
        # 类别
        cls_idx = pred_cls_idx[grid_idx]
        scale_results.append({
            "box": [x1, y1, x2, y2],
            "score": c_score,
            "cls_idx": cls_idx.item()
        })
    return scale_results


# ===================== 多尺度融合推理+可视化 核心函数 =====================
def inference_image_multi_scale(img_path, class_names, save_name="res.jpg"):
    raw_img = Image.open(img_path).convert("RGB")
    all_results = []
    # 遍历全部尺度推理，收集所有预测框
    for sz in img_sizes:
        print(f"正在推理尺度 {sz} ...")
        single_res = single_scale_predict(raw_img, sz, class_names)
        all_results.extend(single_res)
        print(f"尺度 {sz} 检测到 {len(single_res)} 个候选框")

    if len(all_results) == 0:
        print("所有尺度均无目标")
        return

    # 汇总全部尺度结果统一执行NMS
    boxes = torch.tensor([r["box"] for r in all_results], dtype=torch.float32).to(device)
    scores = torch.tensor([r["score"] for r in all_results], dtype=torch.float32).to(device)
    keep_idx = nms(boxes[:, :4], scores, iou_threshold=0.2)
    keep_idx = keep_idx.cpu().numpy()
    results_nms = [all_results[i] for i in keep_idx]

    # 绘图保存
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(raw_img)
    ax.axis("off")
    for res in results_nms:
        x1, y1, x2, y2 = res["box"]
        cls_name = class_names[res["cls_idx"]]
        score = res["score"]
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor="#ff3333", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1-6, f"{cls_name} {score:.2f}",
            color="white", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="red", alpha=0.6)
        )
    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"多尺度融合推理完成，总候选框:{len(all_results)}, NMS后保留 {len(results_nms)} 个目标，结果保存至: {save_name}")


if __name__ == "__main__":
    # 替换为你的测试图片路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_img = os.path.join(script_dir, "car.png")
    # 调用多尺度推理函数
    inference_image_multi_scale(test_img, class_names=["汽车"], save_name="detect_multi_scale_nms.jpg")
