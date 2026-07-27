
结合多模态模型的开放词汇目标检测实现，通过图文特征深度对齐，实现零样本 / 小样本类别无关检测，无需重新训练主干网络即可快速适配新类别目标，兼顾检测精度与工程实用性。

本项目基于 **FG-CLIP** 多模态基础模型构建，主干网络与图文特征提取能力引用自：
https://github.com/360CVGroup/FG-CLIP


## 📌 环境配置

### 基础环境

numpy>=2.0.2
torch>=2.6.0
torchvision>=0.21.0
transformers==4.55.2

### 依赖安装

```bash
pip install torch torchvision pillow transformers tqdm
```


## 📊 数据准备

### 数据集结构

建议使用数据集LVIS。

采用自定义 YOLO 格式数据集，目录结构如下：

```plain
custom/
├─ images/          # 图像文件（train/val子目录，可选）
│  ├─ train/
│  └─ val/
├─ labels/          # 标签文件（与图像一一对应，txt格式）
│  ├─ train/
│  └─ val/
└─ data.txt         # 类别名称文件（每行一个类别，如：0: cat、1: dog、2: car）
```

## 🚀 模型训练

### 训练步骤

1. 从 **FG-CLIP** 项目获取预训练多模态模型（fgmodel），放入主脚本同级fgmodel名称的目录。modeling_fgclip2.py文件不要替换。

2. 配置数据集路径：在主脚本同级目录创建custom目录，将数据集按目录结构放入。

3. 运行训练脚本

```bash
python train.py
```


## 🔍 模型推理

运行脚本可测试模型的推理：

```bash
python test.py
```


## 🔧 核心代码解析

### 1\. 模型结构

- **AdaptedDetectHead**：自定义检测头，包含 box_head（边界框回归）和 cls_head（类别预测）双分支，通过图文相似度计算生成特征掩码，优化特征提取。

- **CustomYOLOLoss**：自定义损失函数，融合 CIoU 损失（边界框回归）和 Varifocal 损失（类别预测），适配开放词汇检测场景。

### 2\. 关键功能

- **文本特征缓存**：首次运行计算所有类别文本特征并缓存，避免重复计算，提升效率。

- **图文特征对齐**：通过矩阵乘法计算图像特征与文本特征的相似度，生成特征掩码，强化目标区域特征。


## 预览图片

识别文本”左边的猫“
<img width="1785" height="1346" alt="1" src="https://github.com/user-attachments/assets/d6cbf637-cc42-4f61-8975-3162c59f8c49" />


识别文本”书“
<img width="1785" height="1346" alt="2" src="https://github.com/user-attachments/assets/e8f2f3d6-4df3-4734-a449-d6ed972aacdb" />


识别文本”狗“
<img width="1785" height="1198" alt="img3" src="https://github.com/user-attachments/assets/638c5d9c-e16b-438b-9ec4-7823817ef23a" />

