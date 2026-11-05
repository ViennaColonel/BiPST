# Configuration files

| File | Role |
|---|---|
| `paper.json` | Original paper protocol, pretrained-weight hashes and original image hashes; requires the unshared original dataset |
| `splits.json` | Original train/validation/evaluation IDs and auxiliary groups |
| `train.example.json` | Own-data template: same learning settings, local split file and separate output, no original image hashes |
| `splits.example.json` | Example IDs only; replace them with your image stems and actual groups |
| `train.local.json`, `splits.local.json` | Created by `python train.py --setup`; editable user settings |

Paths inside a training JSON resolve against `base_dir`, itself relative to that JSON. The templates use `base_dir: ".."`, meaning the repository root. Absolute paths are also accepted for data stored elsewhere. Changing the outer repository name does not break relative paths.

Edit the settings block in `train.py` and run it from your IDE after preparing inputs. BAT is optional. `python train.py --check` validates real inputs without fitting a model. It may write the input signature to the selected output folder. For parameter descriptions see [training](../docs/TRAINING.md); for Windows setup and outputs see [quick start](../docs/QUICKSTART_中文.md).
