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
import random
from torchvision.transforms import ToTensor
from torch.amp import autocast, GradScaler
scaler = GradScaler()


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


class SELayer(nn.Module):
    """通道注意力层(Squeeze-and-Excitation)"""

    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, expansion=4):
        super().__init__()
        middleChannels = int(in_channels*expansion)
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            self.conv2 = nn.Conv2d(
                middleChannels, middleChannels, kernel_size=3, stride=2, padding=1, groups=middleChannels, bias=False)
        else:
            self.downsample = nn.Identity() if in_channels == out_channels else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            self.conv2 = nn.Conv2d(
                middleChannels, middleChannels, kernel_size=3, stride=1, padding=1, groups=middleChannels, bias=False)

        self.lay1 = nn.Identity() if (expansion == 1 and in_channels == out_channels) else nn.Sequential(
            nn.Conv2d(in_channels, middleChannels, kernel_size=1, bias=False),
            nn.BatchNorm2d(middleChannels),
            nn.GELU()
        )

        self.bn2 = nn.BatchNorm2d(middleChannels)
        self.conv3 = nn.Conv2d(
            middleChannels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.GELU()
        self.xattention = SELayer(out_channels)

    def forward(self, x):
        out = self.lay1(x)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.downsample(x)
        out = self.xattention(out)
        out = self.relu(out)
        return out


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, kernal_size // 2, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, p, n, c = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = qkv
        q = q.reshape(b, p, n, self.heads, -1).permute(0, 1, 3, 2, 4)
        k = k.reshape(b, p, n, self.heads, -1).permute(0, 1, 3, 2, 4)
        v = v.reshape(b, p, n, self.heads, -1).permute(0, 1, 3, 2, 4)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2, 4).reshape(b, p, n, -1)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size
        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)
        self.transformer = Transformer(dim, depth, 4, 64, mlp_dim, dropout)
        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)
        b, d, H, W = x.shape
        ph, pw = self.ph, self.pw
        nh = H // ph
        nw = W // pw
        x = x.reshape(b, d, nh, ph, nw, pw)
        x = x.permute(0, 3, 5, 2, 4, 1)
        x = x.reshape(b, ph * pw, nh * nw, d)
        x = self.transformer(x)
        x = x.reshape(b, ph, pw, nh, nw, d)
        x = x.permute(0, 5, 3, 1, 4, 2)
        x = x.reshape(b, d, H, W)
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class LightFPN(nn.Module):
    """轻量FPN，feat_s预先pool到80×80，显存降级版本，输出固定40×40,768"""

    def __init__(self, c_s: int, c_m: int, c_l: int, out_dim=768):
        super().__init__()
        self.proj_s = conv_1x1_bn(c_s, out_dim)   # 80x80
        self.proj_m = conv_1x1_bn(c_m, out_dim)   # 80x80
        self.proj_l = conv_1x1_bn(c_l, out_dim)   # 40x40

        self.fuse_s = conv_nxn_bn(out_dim, out_dim, kernal_size=3)
        self.fuse_m = conv_nxn_bn(out_dim, out_dim, kernal_size=3)
        self.fuse_out = conv_nxn_bn(out_dim, out_dim, kernal_size=3)

    def forward(self, feat_s_pool, feat_m, feat_l):
        """
        feat_s_pool: [B,C,80,80]
        feat_m:      [B,C,80,80]
        feat_l:      [B,C,40,40]
        return: fused [B,768,40,40]
        """
        ps = self.proj_s(feat_s_pool)
        pm = self.proj_m(feat_m)
        pl = self.proj_l(feat_l)

        # 高层40 → 上采样到80，融合中层pm
        pl_up = F.interpolate(pl, scale_factor=2,
                              mode="bilinear", align_corners=False)
        pm = pm + pl_up
        pm = self.fuse_m(pm)

        # 融合后的pm给到ps，ps也吸收高层语义（尺寸已经是80x80无需放大）
        pm_up = pm
        ps = ps + pm_up
        ps = self.fuse_s(ps)

        ps_down = F.interpolate(
            ps, size=pl.shape[-2:], mode="bilinear", align_corners=False)
        pm_down = F.interpolate(
            pm, size=pl.shape[-2:], mode="bilinear", align_corners=False)
        out = pl + pm_down + ps_down
        out = self.fuse_out(out)
        return out


class CustomVisionBackbone(nn.Module):
    """
    自定义CNN主干
    输入: [B,3,H,W] 图像像素 (0~1 / 归一化图像)
    输出: [B, num_patch, 768]  适配AdaptedDetectHead的last_hidden
    """

    def __init__(self, output_dim=768):
        super().__init__()
        self.output_dim = output_dim

        # 初始下采样
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

        self.layer1 = nn.Sequential(
            BasicConv2d(64, 64, expansion=1),
            BasicConv2d(64, 96, downsample=True, expansion=1.5),
            BasicConv2d(96, 96, expansion=1),
            BasicConv2d(96, 96, expansion=1)
        )

        self.layer2 = nn.Sequential(
            BasicConv2d(96, 192, downsample=True, expansion=2),
            BasicConv2d(192, 192, expansion=2),
            BasicConv2d(192, 192, expansion=2),
            BasicConv2d(192, 192, expansion=2),
            BasicConv2d(192, 192, expansion=2)
        )

        self.layer3 = nn.Sequential(
            BasicConv2d(192, 384, downsample=True, expansion=2),
            BasicConv2d(384, 384, expansion=2),
            BasicConv2d(384, 384, expansion=2),
            BasicConv2d(384, 384, expansion=2)
        )

        self.xlight_fpn = LightFPN(c_s=96, c_m=192, c_l=384, out_dim=768)

        patch_size = (2, 2)
        # 配置参数，你可以自由调 dims、L（transformer深度）
        self.dims = [160, 192]     # MobileViT内部transformer维度
        L = [1, 1]                 # 每个block内transformer深度
        in_ch = 768                # layer3输出通道固定768

        self.xmvit = nn.ModuleList()
        # Block 1
        self.xmvit.append(MobileViTBlock(
            self.dims[0], L[0], in_ch, 3, patch_size, int(self.dims[0] * 2)
        ))
        # Block 2
        self.xmvit.append(MobileViTBlock(
            self.dims[1], L[1], in_ch, 3, patch_size, int(self.dims[1] * 4)
        ))

        self.norm = nn.LayerNorm(output_dim)

    def forward(self, pixel_values):
        x = self.stem(pixel_values)
        x = self.layer1(x)
        feat_s = x.clone()   # stride4  160×160  ch=96
        feat_s_pool = F.avg_pool2d(feat_s, kernel_size=2, stride=2)

        x = self.layer2(x)
        feat_m = x.clone()   # stride8   80×80  ch=192

        x = self.layer3(x)
        feat_l = x.clone()   # stride16  40×40  ch=768

        # FPN融合三层 → [B,768,40,40]
        img_feat = self.xlight_fpn(feat_s_pool, feat_m, feat_l)

        x = img_feat
        # 顺序遍历ModuleList堆叠MobileViT
        for block in self.xmvit:
            x = block(x)

        B, C, H, W = x.shape
        last_hidden = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        img_feat = img_feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        img_feat = self.norm(img_feat)
        return img_feat, last_hidden


class AdaptedDetectHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones(1) * 2.6592)
        self.logit_bias = nn.Parameter(torch.zeros(1))

        self.ccclogit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))

        self.vision_backbone = CustomVisionBackbone(output_dim=768)

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

    def forward(self, img_feat, text_feat):
        multi_img_feat, last_hidden = self.vision_backbone(img_feat)
        B = multi_img_feat.shape[0]
        featsize = int(multi_img_feat.shape[1] ** 0.5)
        N = text_feat.size(1)

        fc_text_feat = F.normalize(text_feat, dim=-1)
        fc_img_feat = F.normalize(last_hidden, dim=-1)

        # cls_feat: [B,784,768]
        # text_feat: [B,30,768]
        # out:       [B,784,30]
        cls_sim = torch.matmul(fc_img_feat, fc_text_feat.transpose(-1, -2))
        logit_scale = self.logit_scale.exp()
        cls_sim = cls_sim * logit_scale + self.logit_bias

        cls_sim_sigm = torch.sigmoid(cls_sim)
        obj_by_text = torch.matmul(cls_sim_sigm.transpose(-1, -2), fc_img_feat)
        obj_by_text = F.normalize(obj_by_text, dim=-1)

        scale = self.ccclogit_scale.exp()
        sim_proto_text = torch.matmul(
            obj_by_text, fc_text_feat.transpose(-1, -2)) * scale

        cls_btm = cls_sim.view(
            # [B,n,48,48]
            B, featsize, featsize, -1).permute(0, 3, 1, 2).contiguous()

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

        mask_feat = multi_img_feat * mask  # [B, 784, 768]
        cls_feat = mask_feat.permute(
            0, 2, 1).reshape(B, -1, featsize, featsize)

        box = self.box_head(cls_feat)  # [B,4,48,48]
        box = box.flatten(2)           # [B,4,784]

        cls_map = self.cls_head(cls_feat)
        cls_map = cls_map.flatten(2)

        return box, cls_map, cls_btm, sim_proto_text


to_tensor = ToTensor()


def custom_collate(batch):
    """
    batch: list[ (img_tensor, boxes, class_names, cls_indices, featuresize) ]
    return:
        imgs: Tensor[B,3,H,W]
        boxes_list: list[list] 变长框，无法stack
        class_names_list: list[list]
        cls_indices_list: list[list]
        featuresize_list: list[int]
    """
    # 图片直接拼接成batch张量
    img_tensors = [item[0] for item in batch]

    # 下面都是变长数据，无法stack，保留原生list
    boxes_list = [item[1] for item in batch]
    class_names_list = [item[2] for item in batch]
    cls_indices_list = [item[3] for item in batch]
    featuresize_list = [item[4] for item in batch]

    return img_tensors, boxes_list, class_names_list, cls_indices_list, featuresize_list


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_file = os.path.join(script_dir, "fgmodel")
    fgmodel = AutoModelForCausalLM.from_pretrained(
        model_file, trust_remote_code=True).to(device)
    fgmodel.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_file)

    epochs = 35

    dethead = AdaptedDetectHead().to(device)
    bestfile = 'dethead_yolo_best.pth'
    if os.path.exists(bestfile):
        load_partial_state_dict(dethead, bestfile)
    criterion = CustomYOLOLoss()

    # for name, param in dethead.named_parameters():
    #     param.requires_grad = False

    # for name, param in dethead.vision_backbone.named_parameters():
    #     param.requires_grad = True

    optimizer = torch.optim.AdamW(
        dethead.parameters(), lr=5e-5, weight_decay=1e-4)
    # optimizer = torch.optim.SGD(
    #     dethead.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)

    class_file = os.path.join(script_dir, "custom", "data.txt")
    custom_root = os.path.join(script_dir, "custom")
    train_dataset = YOLODataset(
        custom_root=custom_root, split="train", class_names_file=class_file
    )
    val_dataset = YOLODataset(
        custom_root=custom_root, split="val", class_names_file=class_file,
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
                    txt, padding="max_length", max_length=196, truncation=True, return_tensors="pt"
                ).to(device)
                feat = fgmodel.get_text_features(
                    **cap_in, walk_type="long")
                text_feat_cache[txt] = feat.cpu()
        # 保存到文件
        torch.save(text_feat_cache, cache_path)
        print(f"✅ 文本特征已保存到：{cache_path}")
    # ==================================================================================

    train_dataloader = DataLoader(
        train_dataset, batch_size=16, num_workers=0, collate_fn=custom_collate)
    val_dataloader = DataLoader(
        val_dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=custom_collate)

    best_val_loss = float('inf')
    for epoch in range(epochs):
        dethead.train()
        total_train_loss = 0.0
        train_pos_num = 0

        # 训练进度条
        pbar = tqdm(
            train_dataloader, desc=f"Train Epoch {epoch}/{epochs-1}", total=len(train_dataloader), leave=True)
        for step, batch in enumerate(pbar):
            img_tensors, gt_boxes_batch, gt_cls_names_list, gt_box_cls_indices, featuresize_list = batch
            image_input = torch.stack(img_tensors, dim=0)
            image_input = image_input.to(device)

            featuresize = featuresize_list[0]
            maxnumpatches = int(featuresize*featuresize)
            B = image_input.shape[0]
            optimizer.zero_grad()

            with torch.no_grad():
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
            with autocast(device_type=device, dtype=torch.float16):
                pred_box, cls, cls_btm, txt_sim = dethead(
                    image_input, batch_text_feat)
                loss = criterion(pred_box, gt_boxes_batch,
                                 featuresize, featuresize, cls, cls_btm, gt_box_cls_indices, txt_sim)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

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
                img_tensors, gt_boxes_batch, gt_cls_names_list, gt_box_cls_indices, featuresize_list = batch
                image_input = torch.stack(img_tensors, dim=0)
                image_input = image_input.to(device)

                featuresize = featuresize_list[0]
                maxnumpatches = int(featuresize*featuresize)

                B = image_input.shape[0]

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

                pred_box, cls, cls_btm, txt_sim = dethead(
                    image_input, batch_text_feat)

                val_loss = criterion(
                    pred_box, gt_boxes_batch, featuresize, featuresize, cls, cls_btm, gt_box_cls_indices, txt_sim)
                total_val_loss += val_loss.item()

                # 实时更新验证进度条
                pbar_val.set_postfix({
                    "val_loss": f"{val_loss.item():.3f}"
                })

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(
            f"[Val Summary] Epoch:{epoch} | Avg Val Loss:{avg_val_loss:.3f}\n")

        with open("./best_val_loss.txt", "w", encoding="utf-8") as f:
            f.write(f"{avg_val_loss:.6f}")
            f.write(f"{avg_train_loss:.6f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(dethead.state_dict(), "dethead_yolo_best.pth")
            print(f"✅ 最优模型已保存 | Best Val Loss: {best_val_loss:.3f}\n")

    torch.save(dethead.state_dict(), "dethead_yolo_final.pth")
    print("训练完成！")
