import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import os
from loss import CustomYOLOLoss
from dataset import YOLODataset
from torch.utils.data import DataLoader
from tqdm import tqdm  # 新增 tqdm


def load_partial_state_dict(model, pthfile, map_location='cpu'):
    # 1. 加载预训练权重
    pretrained_dict = torch.load(pthfile, map_location=map_location)

    # 2. 获取当前模型的参数字典（键为参数名，值为参数张量）
    model_dict = model.state_dict()

    # 3. 筛选预训练权重：只保留模型中存在的参数
    # 遍历预训练权重的键，若该键在模型参数中存在，则保留
    filtered_dict = {k: v for k, v in pretrained_dict.items()
                     if k in model_dict}

    # （可选）打印信息：哪些参数被加载，哪些被忽略
    print(f"加载的参数数量：{len(filtered_dict)}")
    ignored_keys = [k for k in pretrained_dict if k not in model_dict]
    if ignored_keys:
        print(f"忽略的参数（模型中不存在）：{ignored_keys}")

    # 4. 用筛选后的权重更新模型参数，并加载
    model_dict.update(filtered_dict)  # 用匹配的预训练权重更新模型字典
    model.load_state_dict(model_dict)  # 加载更新后的字典



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




if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_file = os.path.join(script_dir, "fgmodel")
    fgmodel = AutoModelForCausalLM.from_pretrained(
        model_file, trust_remote_code=True).to(device)
    fgmodel.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_file)
    image_processor = AutoImageProcessor.from_pretrained(model_file)

    imgsize = 512  # 448
    featuresize = imgsize//16
    maxnumpatches = int(featuresize*featuresize)
    epochs = 20

    dethead = AdaptedDetectHead().to(device)
    bestfile = 'dethead_yolo_best.pth'
    if os.path.exists(bestfile):
        load_partial_state_dict(dethead, bestfile)
    criterion = CustomYOLOLoss()
    optimizer = torch.optim.AdamW(
        dethead.parameters(), lr=1e-5, weight_decay=1e-4)
    # optimizer = torch.optim.SGD(
    #     dethead.parameters(), lr=1e-5, momentum=0.9)

    class_file = os.path.join(script_dir, "custom", "data.txt")
    custom_root = os.path.join(script_dir, "custom")
    train_dataset = YOLODataset(
        custom_root=custom_root, split="train", class_names_file=class_file,
        device=device, input_size=imgsize
    )
    val_dataset = YOLODataset(
        custom_root=custom_root, split="val", class_names_file=class_file,
        device=device, input_size=imgsize
    )

    # ====================== 文本特征缓存：加载 / 保存 ======================
    cache_path = "class_text_feat_cache.pth"
    text_feat_cache = {}

    if os.path.exists(cache_path):
        # 🔥 第二次运行：直接加载缓存，超快！
        print("✅ 找到缓存文件，直接加载文本特征...")
        text_feat_cache = torch.load(cache_path, map_location="cpu")
    else:
        # 🔥 第一次运行：计算 + 保存
        print("📌 未找到缓存，开始计算文本特征...")
        with torch.no_grad():
            for txt in train_dataset.global_class_set:
                cap_in = tokenizer(
                    txt, padding="max_length", max_length=64, truncation=True, return_tensors="pt"
                ).to(device)
                feat = fgmodel.get_text_features(**cap_in, walk_type="box")
                text_feat_cache[txt] = feat.cpu()
        # 保存到文件
        torch.save(text_feat_cache, cache_path)
        print(f"✅ 文本特征已保存到：{cache_path}")
    # ==================================================================================

    train_dataloader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=0, collate_fn=lambda x: x)
    val_dataloader = DataLoader(
        val_dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=lambda x: x)

    best_val_loss = float('inf')
    for epoch in range(epochs):
        dethead.train()
        total_train_loss = 0.0
        train_pos_num = 0

        # 训练进度条
        pbar = tqdm(
            train_dataloader, desc=f"Train Epoch {epoch}/{epochs-1}", total=len(train_dataloader), leave=True)
        for step, batch in enumerate(pbar):
            images = [x[0] for x in batch]
            gt_boxes_batch = [x[1] for x in batch]
            gt_cls_names_list = [x[2] for x in batch]

            B = len(images)
            optimizer.zero_grad()

            image_input = image_processor(
                images=images, max_num_patches=maxnumpatches, return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                dense_feature, last_hidden = fgmodel.get_image_dense_feature(
                    **image_input)
                text_feat_list = []
                for i in range(B):
                    txts = gt_cls_names_list[i]
                    tmplist = []
                    for txt_batch in txts:
                        feat = text_feat_cache.get(txt_batch)
                        tmplist.append(feat.to(device))
                    text_feat = torch.cat(tmplist, dim=0)
                    text_feat_list.append(text_feat)
                batch_text_feat = torch.stack(text_feat_list).to(device)

            pred_box, cls, cls_btm = dethead(
                last_hidden, dense_feature, batch_text_feat)
            loss = criterion(pred_box, gt_boxes_batch,
                             featuresize, featuresize, cls, cls_btm)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"[Train Summary] Epoch:{epoch} | Avg Loss:{avg_train_loss:.3f}")

        # 验证
        dethead.eval()
        total_val_loss = 0.0
        val_pos_num = 0

        # 验证进度条
        pbar_val = tqdm(
            val_dataloader, desc=f"Val Epoch {epoch}/{epochs-1}", total=len(val_dataloader), leave=True)
        with torch.no_grad():
            for val_step, batch in enumerate(pbar_val):
                images = [x[0] for x in batch]
                gt_boxes_batch = [x[1] for x in batch]
                gt_cls_names_list = [x[2] for x in batch]

                B = len(images)

                image_input = image_processor(
                    images=images, max_num_patches=maxnumpatches, return_tensors="pt"
                ).to(device)
                dense_feature, last_hidden = fgmodel.get_image_dense_feature(
                    **image_input)

                text_feat_list = []
                for i in range(B):
                    txts = gt_cls_names_list[i]
                    tmplist = []
                    for txt_batch in txts:
                        feat = text_feat_cache.get(txt_batch)
                        tmplist.append(feat.to(device))
                    text_feat = torch.cat(tmplist, dim=0)
                    text_feat_list.append(text_feat)
                batch_text_feat = torch.stack(text_feat_list).to(device)

                pred_box, cls, cls_btm = dethead(
                    last_hidden, dense_feature, batch_text_feat)

                val_loss = criterion(
                    pred_box, gt_boxes_batch, featuresize, featuresize, cls, cls_btm)
                total_val_loss += val_loss.item()

                # 实时更新验证进度条
                pbar_val.set_postfix({
                    "val_loss": f"{val_loss.item():.3f}"
                })

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(
            f"[Val Summary] Epoch:{epoch} | Avg Val Loss:{avg_val_loss:.3f}\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(dethead.state_dict(), "dethead_yolo_best.pth")
            print(f"✅ 最优模型已保存 | Best Val Loss: {best_val_loss:.3f}\n")

    torch.save(dethead.state_dict(), "dethead_yolo_final.pth")
    print("训练完成！")
