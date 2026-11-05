# BiPST：面向喷丸图像分割的提示反馈蒸馏

[English](README.md) · [详细使用指南](docs/QUICKSTART_中文.md)

BiPST 利用自动提示 SAM 生成的掩膜训练轻量分割学生。学生预测用于修订历史提示并逐步构建负提示约束，冻结的 SAM 据此更新监督，学生继续学习。训练结束后，选定的学生独立完成图像分割。

本仓库提供训练与推理代码、15 个学生模型权重和一张喷丸示例图。完整数据集不公开。

## 安装

使用 Python 3.10 或更高版本，并安装相互兼容的 PyTorch 和 torchvision。参考环境为 Python 3.11、PyTorch 2.4.1+cu124、torchvision 0.19.1+cu124。

```bash
python -m pip install -e .
```

在 IDE 中选择该 Python 环境。学生推理支持 CUDA 和 CPU，包含 SAM 的训练建议使用 CUDA。

## 推理：运行 demo.py

直接打开 **[demo.py](demo.py)**，点击运行；也可以使用命令：

```bash
python demo.py
```

默认处理 `demo/shot_peening.jpg`，使用 `weights/bipst/seed_3408.pt`。在文件顶部的 `DEMO_SETTINGS` 中修改输入图片或图片目录、权重、输出目录和设备。

结果保存在 `outputs/demo/日期时间/`，包括二值掩膜、叠加图、预测前景面积比例和 `index.html` 预览页面。

## 训练：运行 train.py

打开 **[train.py](train.py)**：在 `PARAMETER_OVERRIDES` 中填写原图、验证掩膜和预训练权重路径，修改学习率、批大小及训练轮数；在 `TRAIN_SETTINGS` 中选择方法和随机种子。

首次使用可生成数据划分模板：

```bash
python train.py --setup
python train.py --check
python train.py
```

执行 `--setup` 后，先填写 `configs/splits.local.json` 中的图像编号和辅助分组，再设置 `train.py` 或 `configs/train.local.json` 中的路径，最后检查并启动训练。训练需要通用 SAM ViT-H 权重和 ImageNet MobileNetV3-Small 编码器权重。

选定的学生模型保存在 `<paths.output>/runs/<method>/seed_<seed>/best_student.pt`。把该路径填入 `demo.py` 的 `checkpoint` 即可用于推理。

参数优先级为：基础 JSON → Python 中的 `PARAMETER_OVERRIDES` → 命令行 `--set`。Python 参数值为 `None` 时沿用 JSON。输入格式和完整输出说明见[使用指南](docs/QUICKSTART_中文.md)。

## 文件结构

| 路径 | 作用 |
|---|---|
| `train.py` | 训练入口，顶部可修改训练参数 |
| `demo.py` | 推理入口，顶部可修改推理参数 |
| `bipst/` | 学生网络、训练、提示反馈、SAM 接入和评价代码 |
| `configs/` | 训练配置与数据划分模板 |
| `weights/` | 验证集选择的学生权重及模型信息 |
| `demo/` | 一张示例原图 |
| `docs/` | 使用指南、训练说明、代码说明和模型说明 |
| `tests/` | 程序测试 |
| `TRAIN.bat`、`DEMO.bat`、`scripts/` | Windows 双击启动入口 |
| `segment_anything/`、`licenses/` | SAM 实现与第三方许可证 |

入口中的相对路径以仓库所在位置为基准。保持内部结构即可整体迁移文件夹。Windows 双击启动器的解释器设置见使用指南。

## 进一步说明与测试

- [训练流程与参数](docs/TRAINING.md)
- [代码文件与 Python 接口](docs/CODE_MAP.md)
- [模型说明](docs/MODEL_CARD.md)
- [SAMRefiner 配置](docs/SAMREFINER.md)

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q
```

## 许可证

BiPST 代码和学生权重采用 [Apache-2.0](LICENSE)。第三方组件及致谢见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
