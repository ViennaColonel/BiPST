# Optional base SAMRefiner comparison

This dependency is used by `python train.py --method refiner_student`, `Study.refine()` and `direct_images(..., method="refiner")`. The published comparison uses base SAMRefiner, not its separately adapted variant.

Obtain the authors' repository `https://github.com/linyq2117/SAMRefiner` at commit `4bb7f95738c7a2e805f189210fb09e02a2197557` and place it at `third_party/SAMRefiner` (or set `paths.samrefiner_repository`). Use `https://github.com/masadcv/FastGeodis` at commit `0f6cc1e6e104bee56e6eb9dfdd2cfd29c21a1a3e`, installing it in the same Python environment with its compiler/CUDA requirements, e.g. `python -m pip install PATH_TO_FASTGEODIS --no-build-isolation`. Relevant upstream file hashes are in `configs/paper.json` and checked during study refinement.

The adapter imports the upstream point, box and mask preparation, connected-region merging and SAM forward operators. It uses five internal refinement iterations, margin 0.1, gamma 4 and mask strength 15. Binary foreground components are merged by the upstream routine, processed separately and combined by union at original resolution. Empty input is retained. A per-image exception retains the coarse input and records the failure; failed images are not omitted. Inspect `refined/records.json` before using results.

Direct application starts from the image and original DT-ZSAM mask. The lightweight route trains a fresh same-architecture student on fixed refined masks from stage 1 through stage 3; it never inherits BiPST task weights. These are application comparisons assembled using the public refiner, not student pipelines claimed to originate in its paper.
