# BiPST: Prompt-Feedback Distillation for Shot-Peening Segmentation

[中文说明](README_中文.md) · [Quick start](docs/QUICKSTART.md)

BiPST trains a lightweight segmenter using automatically prompted SAM masks. Student predictions revise historical prompts and progressively construct negative constraints; frozen SAM generates updated supervision for subsequent learning. The selected student performs image-to-mask inference independently.

The repository provides training and inference code, 15 student checkpoints, and one shot-peening example image. The full dataset is not distributed.

## Installation

Use Python 3.10 or newer with a compatible PyTorch/torchvision pair. The reference environment uses Python 3.11, PyTorch 2.4.1+cu124 and torchvision 0.19.1+cu124.

```bash
python -m pip install -e .
```

Select this Python environment in your IDE. Student inference supports CUDA and CPU; CUDA is recommended for SAM-based training.

## Inference

Open **[demo.py](demo.py)** and run it, or execute:

```bash
python demo.py
```

The default input is `demo/shot_peening.jpg` and the default model is `weights/bipst/seed_3408.pt`. Edit `DEMO_SETTINGS` at the top of the file to change the image or image directory, checkpoint, output directory and device. Results are saved under `outputs/demo/<timestamp>/`: binary masks, overlays, foreground area fractions and an `index.html` preview.

## Training

Open **[train.py](train.py)**. Set image, validation-mask and pretrained-weight paths in `PARAMETER_OVERRIDES`; choose the method and seed in `TRAIN_SETTINGS`. Learning rate, batch size and epoch budgets are also editable in this file.

Prepare the dataset split and auxiliary groups using the templates:

```bash
python train.py --setup
python train.py --check
python train.py
```

After `--setup`, edit the image IDs and groups in `configs/splits.local.json` and the paths in `train.py` or `configs/train.local.json`. Training uses generic SAM ViT-H and ImageNet MobileNetV3-Small weights. The selected student is saved at `<paths.output>/runs/<method>/seed_<seed>/best_student.pt`.

Settings are applied in this order: base JSON configuration, Python `PARAMETER_OVERRIDES`, then optional command-line `--set` values. A Python value of `None` inherits the JSON setting. See the [quick start](docs/QUICKSTART.md) for input formats and output directories.

## Repository layout

| Path | Purpose |
|---|---|
| `train.py` | Training entry point and editable training parameters |
| `demo.py` | Inference entry point and editable inference parameters |
| `bipst/` | Student model, training, prompt feedback, SAM integration and evaluation |
| `configs/` | Training configurations and dataset split templates |
| `weights/` | Validation-selected student checkpoints and model metadata |
| `demo/` | Example input image |
| `docs/` | Usage, training protocol, code map and model documentation |
| `tests/` | Software tests |
| `TRAIN.bat`, `DEMO.bat`, `scripts/` | Optional Windows launchers |
| `segment_anything/`, `licenses/` | SAM implementation and third-party licenses |

Relative entry-point paths are based on the repository location. The folder can be moved while preserving its internal layout. `TRAIN.bat` and `DEMO.bat` provide optional double-click launchers; interpreter configuration is described in the quick start.

## Documentation and tests

- [Training protocol and settings](docs/TRAINING.md)
- [Code map and Python interfaces](docs/CODE_MAP.md)
- [Student model card](docs/MODEL_CARD.md)
- [SAMRefiner setup](docs/SAMREFINER.md)

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q
```

## License

BiPST code and student weights use [Apache-2.0](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md) for dependencies and acknowledgments.
