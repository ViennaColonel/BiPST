# Student checkpoints

The folders `bipst`, `fixed`, `one_update`, `probability_only` and `refiner_student` each contain three validation-selected checkpoints: `seed_3407.pt`, `seed_3408.pt` and `seed_3409.pt`. All use the same 1,972,369-parameter student architecture. Default inference uses `bipst/seed_3408.pt`.

`manifest.json` records SHA-256 hashes, selected training stages and checkpoint metadata. Student weights use Apache-2.0; see the [model card](../docs/MODEL_CARD.md).

For training, supply generic SAM ViT-H and ImageNet MobileNetV3-Small weights separately. Student training initializes the encoder from ImageNet and the decoder randomly.
