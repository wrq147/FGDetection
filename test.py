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
    def __init__(self, hidden_dim=384):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = nn.Parameter(torch.ones(1) * 0.5)

        self.box_head = nn.Sequential(
            nn.Conv2d(768, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                      padding=1, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 4, kernel_size=1)
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

        gauss_kernel = torch.tensor([
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0]
        ]) / 16.0
        self.register_buffer('gauss_kernel', gauss_kernel.view(1, 1, 3, 3))

    def forward(self, last_hidden, dense_feat, text_feat):
        B = last_hidden.shape[0]
        featsize = int(last_hidden.shape[1] ** 0.5)  # 28
        N = text_feat.size(1)

        text_feat = F.normalize(text_feat, dim=-1)
        img_dense_feat = F.normalize(dense_feat, dim=-1)

        # [B, 784, 768] -> [B, 768, 28, 28]
        img_feat = last_hidden.view(
            B, featsize, featsize, -1).permute(0, 3, 1, 2).contiguous()

        # -------------------------- 框回归 --------------------------
        box = self.box_head(img_feat)  # [B,4,28,28]
        box = box.flatten(2)           # [B,4,784]

        # -------------------------- 核心：einsum 批量相似度 --------------------------

        # cls_feat: [B,784,768]
        # text_feat: [B,30,768]
        # out:       [B,784,30]
        cls_sim = torch.matmul(img_dense_feat, text_feat.transpose(-1, -2))
        cls_sim = cls_sim / self.scale
        cls_btm = cls_sim.view(
            B, featsize, featsize, -1).permute(0, 3, 1, 2).contiguous()  # [B,n,28,28]
        sim_max, _ = cls_sim.max(dim=-1)  # [B,784]

        sim_map = sim_max.view(B, 1, featsize, featsize)
        sim_smooth = F.conv2d(sim_map, self.gauss_kernel.to(sim_map.device), padding=1)
        sim_max_smoothed = sim_smooth.flatten(1)

        sim_max_min = sim_max_smoothed.amin(dim=1, keepdim=True)
        sim_max_max = sim_max_smoothed.amax(dim=1, keepdim=True)
        sim_max = (sim_max_smoothed - sim_max_min) / \
            (sim_max_max - sim_max_min + 1e-8)

        mask = sim_max.unsqueeze(-1)          # [B,784,1]

        final_feat = img_dense_feat * mask  # [B, 784, 768]
        final_feat = final_feat.permute(
            0, 2, 1).reshape(B, -1, featsize, featsize)

        cls_map = self.cls_head(final_feat)
        cls_map = cls_map.flatten(2)

        return box, cls_map, cls_btm


# ===================== 全局配置 =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
imgsize = 512
featuresize = imgsize // 16
maxnumpatches = featuresize * featuresize
# 中心点置信度阈值（按需调）
CENTER_THRESH = 0.8

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
def resize_and_pad(image, target_size=448, fill_color=(114, 114, 114)):
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

# ===================== 推理+可视化 核心函数【无NMS】 =====================


def inference_image(img_path, class_names, save_name="res.jpg"):
    raw_img = Image.open(img_path).convert("RGB")
    pad_img, pad_x, pad_y, scale = resize_and_pad(raw_img, imgsize)
    raw_w, raw_h = raw_img.size
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
        max_length=64,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        dense_feature, last_hidden = fgmodel.get_image_dense_feature(
            **image_input)

        text_feat = fgmodel.get_text_features(**text_tokens, walk_type="box")
        text_feat = text_feat.unsqueeze(0)
        # 检测头前向
        pred_box, pred_cls, pred_sim = dethead(
            last_hidden, dense_feature, text_feat)
        # 解析输出
        pred_box = pred_box.squeeze(0).permute(1, 0)
        pred_sim = pred_sim.flatten(2).permute(0, 2, 1).squeeze(0)

        pred_cls_score = pred_cls.flatten()
        pred_cls_idx = pred_sim.argmax(dim=-1)
        pred_score = torch.sigmoid(pred_cls_score)  # 类别置信度

    # 遍历所有特征点，仅用 中心阈值过滤
    results = []
    total_grid = featuresize * featuresize
    for grid_idx in range(total_grid):
        c_score = pred_score[grid_idx].item()

        if c_score < CENTER_THRESH:
            continue

        box_tensor = pred_box[grid_idx]
        dx = torch.sigmoid(box_tensor[0])
        dy = torch.sigmoid(box_tensor[1])
        bw = torch.sigmoid(box_tensor[2])
        bh = torch.sigmoid(box_tensor[3])

        gy = grid_idx // featuresize
        gx = grid_idx % featuresize

        cx = (gx + dx * 5 - 2.5)/featuresize
        cy = (gy + dy * 5 - 2.5)/featuresize

        # 映射到 padded 图尺度
        cx_pad = cx * imgsize
        cy_pad = cy * imgsize
        bw_pad = bw * imgsize
        bh_pad = bh * imgsize

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
        results.append({
            "box": [x1, y1, x2, y2],
            "score": c_score,
            "cls_idx": cls_idx.item()
        })

    # ===================== 绘图保存 =====================
    if len(results) == 0:
        print("无目标")
        return

    # 转成 NMS 需要的张量
    boxes = torch.tensor([r["box"] for r in results],
                         dtype=torch.float32).to(device)
    scores = torch.tensor([r["score"] for r in results],
                          dtype=torch.float32).to(device)

    # 执行 NMS
    keep_idx = nms(boxes[:, :4], scores, iou_threshold=0.3)
    keep_idx = keep_idx.cpu().numpy()

    # 只保留 NMS 后的结果
    results_nms = [results[i] for i in keep_idx]

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
    print(f"推理完成，共检测 {len(results_nms)} 个目标，结果保存至: {save_name}")


if __name__ == "__main__":
    # 替换为你的测试图片路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_img = os.path.join(script_dir, "cat.jpg")
    inference_image(test_img, class_names=[
                    "黑猫"], save_name="detect_result_nms.jpg")
