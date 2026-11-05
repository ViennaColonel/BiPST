# Training and inference guide

[中文指南](QUICKSTART_中文.md)

## 1. Environment

Install a compatible PyTorch/torchvision pair, then run `python -m pip install -e .` in the repository root. Select that interpreter in VS Code or PyCharm. Reference versions are Python 3.11, PyTorch 2.4.1+cu124 and torchvision 0.19.1+cu124. CPU supports student inference; CUDA is recommended for SAM-based training.

## 2. Inference

Open `demo.py`, edit `DEMO_SETTINGS` at the top, and run the file. The default input is `demo/shot_peening.jpg`; the default student is `weights/bipst/seed_3408.pt`.

| Setting | Meaning |
|---|---|
| `input` | Image file or a directory of images; directory scanning is nonrecursive |
| `checkpoint` | Student `.pt` file |
| `output` | New or empty directory; `None` creates a timestamped directory |
| `device` | `auto`, `cuda`, or `cpu` |
| `threshold` | Foreground probability cutoff, normally 0.5 |
| `mixed_precision` | CUDA autocast; CPU uses FP32 |
| `open_preview` | Open the generated HTML preview |
| `allow_file_dialog` | Show a file chooser when `input` is `None` |

The default output is `outputs/demo/<timestamp>/`. `masks/` contains original-size PNGs with foreground 255 and background 0; `overlays/` contains segmentation overlays; `predictions.json` records dimensions and foreground area fractions. `inputs/` and `index.html` provide a side-by-side preview.

Optional command-line arguments override Python settings:

```bash
python demo.py --input YOUR_IMAGE --device cpu
python demo.py --checkpoint YOUR_MODEL.pt --input YOUR_IMAGE_DIRECTORY
```

## 3. Training inputs

Prepare original images, validation masks, a dataset split/group file, generic SAM ViT-H weights and a torchvision MobileNetV3-Small ImageNet state dictionary:

```text
data/images/                      Original images
data/val_masks/                   Same-name validation PNGs
data/all_gt/                      Optional evaluation references
pretrained/sam_vit_h_4b8939.pth    SAM ViT-H checkpoint
pretrained/mobilenet_v3_small.pth ImageNet encoder state dictionary
```

Reference masks use the original image dimensions and single-channel 0/255 encoding. Student optimization uses SAM-generated masks; validation references select checkpoints.

Run `python train.py --setup` to create `configs/train.local.json` and `configs/splits.local.json`. Existing files are preserved. Replace the example IDs with image filename stems. Training, validation and evaluation partitions must be disjoint; evaluation may be empty. Each training image belongs to exactly one auxiliary group, with related images assigned to the same group.

## 4. Training parameters and output

In `train.py`, `TRAIN_SETTINGS` selects the configuration, method, seed and check-only mode. `PARAMETER_OVERRIDES` sets paths and training hyperparameters. Use `seed: "all"` for all configured seeds, or `check_only: True` to validate inputs.

The default configuration is `configs/train.local.json` when present, otherwise `configs/train.example.json`. `configs/paper.json` contains the paper protocol and original image hashes. For another dataset, use `train.example.json` and your own split file.

Settings resolve in this order: base JSON, explicit Python values, then CLI `--set` assignments. `None` inherits the JSON setting. Data/model/output paths resolve against the configuration's `base_dir`; the templates use the repository root. Absolute paths are also supported.

```bash
python train.py --check
python train.py
python train.py --seed all
python train.py --method probability_only --set training.learning_rate=0.0005
```

| Method value | Training route |
|---|---|
| `bipst` | Full BiPST feedback |
| `fixed` | Fixed original DT-ZSAM targets |
| `one_update` | Continue from the same-seed BiPST second stage with first-update masks fixed |
| `probability_only` | BiPST with probability-ranked negative growth |
| `refiner_student` | Fixed SAMRefiner supervision from the first training stage |
| `all` | Run all routes, starting with BiPST |

`one_update` requires the corresponding completed BiPST route in the same output directory. `refiner_student` additionally requires the dependencies in [SAMREFINER.md](SAMREFINER.md).

| Under `paths.output` | Contents |
|---|---|
| `initial/` | Initial SAM masks and positive prompts |
| `common/seed_<seed>/` | First-stage student and log |
| `common_aux/seed_<seed>/` | First-update auxiliary students |
| `runs/<method>/seed_<seed>/` | Subsequent training and feedback states |
| `runs/<method>/seed_<seed>/best_student.pt` | Validation-selected deployment model |

Completed stages can be reused when input signatures match. Use a new output directory for changed configurations. Interrupted stages restart in a fresh directory; continuation within an epoch is not supported.

## 5. Paths and Windows shortcuts

Python entry points resolve their configuration paths relative to the repository, so they also work when launched from another working directory. Keep the internal layout when moving the folder; update external absolute paths when changing computers.

`TRAIN.bat` and `DEMO.bat` are optional double-click launchers. They use the active Python environment, or an interpreter specified by copying `runtime.example.json` to `runtime.local.json` and setting `python` or `python_wrapper`. When running Python files in an IDE, select the interpreter in the IDE.
