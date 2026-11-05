# Student models

The student uses a MobileNetV3-Small encoder and a U-Net-like decoder, with 1,972,369 trainable parameters. It accepts RGB images with ImageNet normalization and predicts shot-peening foreground probability. The default binary threshold is 0.5.

Fifteen checkpoints cover BiPST, fixed DT-ZSAM targets, one-update feedback, probability-only growth and fixed SAMRefiner supervision, each with seeds 3407, 3408 and 3409. Each checkpoint was selected by validation Dice across training stages. `weights/manifest.json` records file hashes, selected stages and model metadata. Default inference uses `weights/bipst/seed_3408.pt`.

Training used 27 images with SAM-generated supervision. Three validation images provided checkpoint selection, and four historical-evaluation images had also been used during development. Twelve images had manual references. The repository includes one demonstration image; the full dataset is private.

The encoder was initialized from torchvision ImageNet weights. Student checkpoints are provided under Apache-2.0; third-party components retain their original terms. The loader uses `torch.load(..., weights_only=True)`.
