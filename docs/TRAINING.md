# Training, crop counts and parameters

## What is trained

1. Generate original positive-only DT-ZSAM masks for all supplied images.
2. Fit the main student on training-image masks. The encoder loads generic ImageNet weights; the decoder starts randomly. Stage 1 uses at most 100 epochs and patience 10.
3. For BiPST, fit three group-excluded auxiliary students. Main and auxiliary predictions on original, flipped and brightened views propose prompt states. Frozen SAM produces candidate masks. Only candidates with actual negative growth are eligible; highest SAM-predicted IoU wins, then fewer negative points break ties. A complete tie retains generation order. The old state is retained when no growth candidate is eligible, not on the basis of an unimplemented improvement test.
4. Train stage 2 for 20 epochs on the selected masks, generate the second update using newly trained students, then train stage 3 for 20 epochs. Each stage resets AdamW and warm-starts model tensors from the preceding best-validation checkpoint. The highest validation Dice across all stages selects the deployment model.

Teacher and auxiliary models remain offline. No validation or historical-test mask enters gradient training. Auxiliary exclusion removes direct fitting to the queried image group, not all statistical dependence between predictions.

## Online crops are not new independent images

Each of 27 training originals contributes two newly sampled square views per epoch: **54 image-mask pairs**. Crop side is `min(H,W,int(384*u))` with `u` uniform in [0.8,1.2], followed by resizing to 384 square, shared 90-degree rotations and horizontal flips, and image-only brightness changes. Labels remain fixed inside each stage. Crops are not a permanent tiled dataset.

Auxiliary sampling separately uses `max(2,ceil(56/N_fold_train))` views per image. The fixed sampling floor 56 is not replaced with 54: fold-training sizes 17/18/19 yield 68/72/57 views per auxiliary epoch. Their actual stopping epochs are logged by new runs.

## Where to change settings

`train.py` exposes editable `TRAIN_SETTINGS` and `PARAMETER_OVERRIDES`; explicit Python values override the selected base JSON, followed by optional CLI overrides. `configs/paper.json` retains the original protocol. For example, set `PARAMETER_OVERRIDES["training.learning_rate"] = 0.0005` in `train.py`, or run `python train.py --set training.learning_rate=0.0005`. Use a fresh output path for changed settings. Unknown keys are rejected. An altered configuration is a new experiment, not the published protocol.

`bipst/prompt_policy.py` holds prompt-specific rules in `PromptConfig`, `TARGET` and `INCREMENTS`. The local replacement/growth pairs are (1,5), (3,10), (6,20), not all combinations and not a time schedule. Change these only for a separately documented experiment. The probability-only route changes growth-pool ranking alone.

## Reproducibility and reuse

Complete fits save an input signature, target hashes, model state, validation scores and a CSV training log. Reuse requires matching inputs. An interrupted fit is not silently treated as complete: preserve its directory and select a fresh output path. This implementation resumes completed stages, not the middle of an epoch.

The original fitting routine used `fit_seed+1` as its RNG seed; the refactor preserves this offset. Same-seed task routes create the network in the same order before loading pretrained encoder tensors. Main data order and augmentation draw order are retained. `one_update` starts from the same S2 and uses the same stage-3 seed/budget as BiPST. Numerical results may vary with hardware and library versions; the distributed checkpoints are the recorded validation-selected models.
