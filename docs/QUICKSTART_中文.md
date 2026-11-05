# 训练与推理使用指南

[English](QUICKSTART.md)

## 1. 环境

安装兼容的 PyTorch 和 torchvision，在仓库根目录执行 `python -m pip install -e .`，然后在 VS Code 或 PyCharm 中选择对应解释器。参考环境为 Python 3.11、PyTorch 2.4.1+cu124、torchvision 0.19.1+cu124。学生推理支持 CPU，包含 SAM 的训练建议使用 CUDA。

## 2. 推理

打开 `demo.py`，修改顶部 `DEMO_SETTINGS` 后直接运行。默认输入为 `demo/shot_peening.jpg`，默认权重为 `weights/bipst/seed_3408.pt`。

| 参数 | 含义 |
|---|---|
| `input` | 图片文件或图片目录；目录只读取当前层 |
| `checkpoint` | 学生模型 `.pt` 文件 |
| `output` | 新目录或空目录；`None` 自动创建日期时间目录 |
| `device` | `auto`、`cuda` 或 `cpu` |
| `threshold` | 前景概率阈值，通常为 0.5 |
| `mixed_precision` | CUDA 混合精度；CPU 使用 FP32 |
| `open_preview` | 是否自动打开 HTML 预览 |
| `allow_file_dialog` | `input` 为 `None` 时是否弹出文件选择框 |

默认输出在 `outputs/demo/日期时间/`：`masks/` 保存与原图等尺寸的 0/255 掩膜，`overlays/` 保存叠加图，`predictions.json` 记录尺寸和前景面积比例，`inputs/` 与 `index.html` 用于并排预览。

命令行参数可覆盖 Python 设置：

```bash
python demo.py --input YOUR_IMAGE --device cpu
python demo.py --checkpoint YOUR_MODEL.pt --input YOUR_IMAGE_DIRECTORY
```

## 3. 训练输入

准备原图、验证掩膜、数据划分与辅助分组文件、通用 SAM ViT-H 权重及 torchvision MobileNetV3-Small 的 ImageNet 权重：

```text
data/images/                      原始图像
data/val_masks/                   与验证图像同名的 PNG 掩膜
data/all_gt/                      可选的评价参考掩膜
pretrained/sam_vit_h_4b8939.pth    SAM ViT-H 权重
pretrained/mobilenet_v3_small.pth ImageNet 编码器权重
```

参考掩膜使用原图尺寸、单通道 0/255 编码。学生参数使用 SAM 自动掩膜进行优化，验证参考用于选择检查点。

执行 `python train.py --setup` 创建 `configs/train.local.json` 和 `configs/splits.local.json`，已有文件会保留。把示例编号替换成自己的图像文件名（不含扩展名）。训练、验证与评价集合应互不重叠，评价集合可为空。每张训练图像只属于一个辅助分组，相关图像应归入同一组。

## 4. 训练参数与输出

`train.py` 中的 `TRAIN_SETTINGS` 设置基础配置、方法、种子和仅检查模式；`PARAMETER_OVERRIDES` 设置路径及训练参数。`seed` 设为 `"all"` 可运行配置中的全部种子，`check_only` 设为 `True` 可检查输入。

默认优先读取 `configs/train.local.json`，不存在时使用 `configs/train.example.json`。`configs/paper.json` 保存论文配置及原始图像哈希。使用自己的数据集时，采用 `train.example.json` 和自己的划分文件。

设置依次按基础 JSON、Python 中的明确值、命令行 `--set` 生效；Python 中为 `None` 的值沿用 JSON。数据、权重及输出路径相对于配置的 `base_dir`，模板中为仓库根目录；也支持绝对路径。

```bash
python train.py --check
python train.py
python train.py --seed all
python train.py --method probability_only --set training.learning_rate=0.0005
```

| `method` 值 | 训练路线 |
|---|---|
| `bipst` | 完整 BiPST 反馈 |
| `fixed` | 固定原始 DT-ZSAM 监督 |
| `one_update` | 从相同种子的 BiPST 第二阶段继续，固定第一次反馈监督 |
| `probability_only` | 采用概率排序负点增长的 BiPST |
| `refiner_student` | 从第一阶段起使用固定 SAMRefiner 监督 |
| `all` | 从 BiPST 开始依次运行全部路线 |

`one_update` 需要同一输出目录中已完成的对应 BiPST 路线。`refiner_student` 还需要 [SAMREFINER.md](SAMREFINER.md) 中的依赖。

| `paths.output` 下的路径 | 内容 |
|---|---|
| `initial/` | 初始 SAM 掩膜与正提示 |
| `common/seed_<seed>/` | 第一阶段学生与日志 |
| `common_aux/seed_<seed>/` | 第一次更新的辅助学生 |
| `runs/<method>/seed_<seed>/` | 后续训练及反馈状态 |
| `runs/<method>/seed_<seed>/best_student.pt` | 验证集选择的部署模型 |

输入签名一致时可复用已完成阶段。修改配置时使用新的输出目录。中断阶段需在新目录中重启，不支持 epoch 内断点续训。

## 5. 路径与 Windows 双击入口

Python 入口相对于仓库位置解析配置路径，从其他工作目录启动也可使用。移动整个文件夹时保持内部结构；更换电脑后更新外部绝对路径。

`TRAIN.bat` 和 `DEMO.bat` 是可选的双击入口，使用已激活的 Python 环境。也可把 `runtime.example.json` 复制为 `runtime.local.json`，填写 `python` 或 `python_wrapper`。在 IDE 中直接运行 Python 文件时，在 IDE 中选择解释器。
