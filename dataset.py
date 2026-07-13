import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image


class YOLODataset(Dataset):
    """
    适配你的数据集结构：
    custom/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
    """

    def __init__(self, custom_root, split, class_names_file, device):
        super().__init__()
        self.FIXED_BACKGROUND_CLASSES = [
            "空的", "未知的", "不知明的", "void_null_03", "void_null_04",
            "void_null_05", "void_null_06", "void_null_07", "void_null_08", "void_null_09",
            "blank_fill_10", "blank_fill_11", "blank_fill_12", "blank_fill_13", "blank_fill_14",
            "blank_fill_15", "blank_fill_16", "blank_fill_17", "blank_fill_18", "blank_fill_19",
            "empty_slot_20", "empty_slot_21", "empty_slot_22", "empty_slot_23", "empty_slot_24",
            "empty_slot_25", "empty_slot_26", "empty_slot_27", "无意义的词", "填充词",
            "未知事物", "空的3", "空的2", "空4", "空5", "空6", "空7", "空8", "空9", "空10"
        ]
        self.split = split
        self.device = device
        self.fixed_num_classes = 40
        self.input_size = 512

        self.img_dir = os.path.join(custom_root, "images", split)
        self.label_dir = os.path.join(custom_root, "labels", split)

        self.img_files = []
        all_img = [f for f in os.listdir(self.img_dir) if f.endswith(
            ('jpg', 'jpeg', 'png', 'bmp'))]

        for img_name in all_img:
            # 找到对应的标签
            txt_name = os.path.splitext(img_name)[0] + ".txt"
            txt_path = os.path.join(self.label_dir, txt_name)

            # 标签不存在 → 跳过
            if not os.path.exists(txt_path):
                continue

            # 标签为空 → 跳过
            with open(txt_path, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            if len(lines) == 0:
                continue

            # 只有 有标签、非空 的图片才加入数据集
            self.img_files.append(img_name)

        # 加载全局所有类别
        self.global_class_names = self._load_class_names(class_names_file)
        self.global_class_set = set(self.global_class_names)
        self.global_class_set.update(self.FIXED_BACKGROUND_CLASSES)

    def _load_class_names(self, class_names_file):
        class_list = []
        with open(class_names_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or ':' not in line:
                continue

            cls_id_str, cls_name = line.split(':', 1)
            cls_id = int(cls_id_str.strip())
            cls_name = cls_name.strip()

            while len(class_list) <= cls_id:
                class_list.append("")

            class_list[cls_id] = cls_name

        return class_list

    # 计算缩放填充参数（不返回image，只返回参数）
    def get_scale_params(self, w, h, iptsize):
        target_size = iptsize
        scale = target_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        paste_x = (target_size - new_w) // 2
        paste_y = (target_size - new_h) // 2
        return scale, paste_x, paste_y

    # 坐标修正（核心！返回已经对齐填充后的归一化坐标）
    def correct_boxes(self, boxes, scale, paste_x, paste_y, img_w, img_h, iptsize):
        corrected = []
        target = iptsize
        for box in boxes:
            cls_id, cxn, cyn, wn, hn = box
            # 原始坐标 → 缩放偏移后的新坐标
            cx = cxn * img_w * scale + paste_x
            cy = cyn * img_h * scale + paste_y
            w = wn * img_w * scale
            h = hn * img_h * scale
            # 新归一化坐标（完全对齐模型输入）
            cxn_new = cx / target
            cyn_new = cy / target
            wn_new = w / target
            hn_new = h / target
            corrected.append([cxn_new, cyn_new, wn_new, hn_new])
        return corrected

    def resize_and_pad(self, image, target_size=448, fill_color=(114, 114, 114)):
        w, h = image.size
        scale = target_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.BILINEAR)
        padded_img = Image.new("RGB", (target_size, target_size), fill_color)
        paste_x = (target_size - new_w) // 2
        paste_y = (target_size - new_h) // 2
        padded_img.paste(image, (paste_x, paste_y))
        return padded_img

    def __len__(self):
        return len(self.img_files)

    def change_size(self, input_size):
        self.input_size = input_size

    def __getitem__(self, idx):
        input_size = self.input_size
        featuresize = input_size // 16
        # 1. 路径
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # 2. 读取尺寸（不加载完整图片，节省内存）
        with Image.open(img_path) as img:
            w, h = img.size

        # 3. 计算缩放参数
        scale, paste_x, paste_y = self.get_scale_params(w, h, input_size)

        # 4. 读取标签
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        txt_path = os.path.join(self.label_dir, txt_name)
        raw_boxes = []
        cls_ids = []
        cls_names = []

        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = list(map(float, line.split()))
                cid, cx, cy, bw, bh = parts
                cid = int(cid)

                if cid < 0 or cid >= len(self.global_class_names):
                    print(f"⚠️  跳过无效标签 ID: {cid} (文件: {txt_path})")
                    continue  # 跳过错误ID，不崩溃

                raw_boxes.append([cid, cx, cy, bw, bh])
                cls_ids.append(cid)
                cls_names.append(self.global_class_names[cid])

        # ===================== 核心修改：固定 30 个类别名 =====================
        # 去重当前图片真实类别
        real_unique_names = list(set(cls_names)) if cls_names else []
        # real_name_set = set(real_unique_names)

        # 需要填充的数量
        need_fill = self.fixed_num_classes - len(real_unique_names)
        fill_names = []

        if need_fill > 0:
            fill_names = self.FIXED_BACKGROUND_CLASSES[:need_fill]

        # 最终固定 30 个类别名
        final_unique_names = real_unique_names + fill_names
        name_to_idx = {name: i for i, name in enumerate(final_unique_names)}

        # 真实框的索引
        box_cls_indices = []
        for box in raw_boxes:
            cid = int(box[0])
            cls_name = self.global_class_names[cid]
            box_cls_indices.append(name_to_idx[cls_name])

        # 5. 修正框坐标
        if raw_boxes:
            boxes = self.correct_boxes(
                raw_boxes, scale, paste_x, paste_y, w, h, input_size)
        else:
            boxes = []

        img = self.resize_and_pad(Image.open(img_path).convert(
            "RGB"), target_size=input_size)

        return img, boxes, final_unique_names, box_cls_indices, featuresize
