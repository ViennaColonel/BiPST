# Code map

| File | Responsibility | Main interfaces |
|---|---|---|
| `train.py` / `TRAIN.bat` | Editable Python training settings, input checks and direct training workflow calls | `main`, `preflight`, `create_study` |
| `demo.py` / `DEMO.bat` | Editable Python demo settings, default real example and local HTML preview | `main`, `make_preview` |
| `scripts/runtime.ps1` | Windows interpreter selection and portable script launch | runtime settings |
| `bipst/config.py` | Typed training defaults, path resolution, split validation | `TrainingConfig`, `read_config` |
| `bipst/data.py` | Paired random crop augmentation | `PairedCropDataset` |
| `bipst/network.py` | Student architecture, BCE + Dice, padded native inference | `build_student_model`, `predict_native` |
| `bipst/training.py` | One stage with a fixed mask set | `train_stage`, `validate` |
| `bipst/prompt_policy.py` | Signed states, candidate locations, replacement, growth, response choice | `candidate_bank`, `choose` |
| `bipst/workflow.py` | Three-stage routes, exclusions, caching with provenance | `Study` |
| `bipst/dtzsam.py` | Original positive initialization and common mask cleanup | `positive_points`, `clean_mask` |
| `bipst/refiner.py` | Base SAMRefiner binary-task adapter | `BinarySAMRefiner` |
| `bipst/inference.py` | Student-only or direct SAM application | `infer_images`, `direct_images` |
| `bipst/evaluation.py` | Full-image Dice, IoU, coverage error | `evaluate_checkpoint`, `evaluate_study` |

Image arrays use OpenCV BGR at disk interfaces; SAM receives RGB. Student tensors use RGB/ImageNet normalization. Coordinates are (x,y) in the original image. Native inference pads reflectively, predicts and removes padding before thresholding. Cached probabilities do not contain manual references. Training and feedback are separated so changing a growth-pool rule does not silently alter the optimizer or supervision source.

`configs/paper.json` is the archived protocol. `configs/train.example.json` retains its learning settings but uses local data/group settings and a separate output directory. `configs/splits.example.json` contains illustrative filenames only. `runtime.example.json` configures the interpreter, not the algorithm. See [English quick start](QUICKSTART.md) or [Chinese quick start](QUICKSTART_中文.md) for outputs and first-run setup.

## Python interfaces

For direct SAM segmentation and evaluation, call the library functions from a Python script:

```python
from bipst.inference import direct_images
from bipst.evaluation import evaluate_checkpoint

direct_images("data/images", "outputs/dtzsam", "pretrained/sam_vit_h_4b8939.pth",
              method="dtzsam", device="cuda")
evaluate_checkpoint("weights/bipst/seed_3408.pt", "data/images", "data/all_gt",
                    "outputs/evaluation", device="auto")
```

Use `method="refiner"` with `samrefiner_repository` for direct SAMRefiner segmentation.
The `Study` class provides `initialize()`, `refine()` and `train_route(method, seed)`;
`bipst.evaluation.evaluate_study(study)` evaluates saved route checkpoints and masks.
