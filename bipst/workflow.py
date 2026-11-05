"""Three-stage learning and two prompt-feedback updates.

The state is a directory containing prompt_states.json and pseudo_labels/*.png.
Each fit consumes only training-image masks. References are read exclusively by
training.validate for checkpoint selection, or by the separate evaluation API.
"""

from dataclasses import asdict
from pathlib import Path
import gc
import hashlib
import json
import math
import shutil
import cv2
import numpy as np
import torch
from .config import read_config
from .image_io import read_image, read_mask, write_image
from . import prompt_policy as policy
from .network import build_student_model, predict_native
from .training import train_stage

METHODS = ("bipst", "fixed", "refiner_student", "one_update", "probability_only")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text("utf-8-sig"))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_student(path, device="cuda"):
    model = build_student_model(False).to(device)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state"], strict=True)
    return model.eval()


def image_paths(folder):
    result = {}
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            if p.stem in result:
                raise ValueError("Duplicate image stem: " + p.stem)
            result[p.stem] = p
    return result


class Study:
    """Own data partitions, immutable experiment settings and reusable stages."""

    def __init__(self, config_file, assignments=()):
        self.spec, self.cfg, self.paths, self.partition = read_config(config_file, assignments)
        requested = self.spec["device"]
        self.device = torch.device(
            "cuda"
            if requested == "auto" and torch.cuda.is_available()
            else "cpu" if requested == "auto" else requested
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA unavailable. Install matching torch/torchvision CUDA builds or use --set device=cpu."
            )
        self.splits, self.folds = self.partition["splits"], self.partition["folds"]
        self.train, self.val, self.test = [self.splits[k] for k in ("train", "val", "test")]
        self.ids = self.train + self.val + self.test
        available = image_paths(self.paths["images"])
        missing = set(self.ids) - set(available)
        if missing:
            raise FileNotFoundError("Missing dataset images: " + ", ".join(sorted(missing)))
        self.images = {k: available[k] for k in self.ids}
        self.out = self.paths["output"]
        torch.set_num_threads(self.cfg.torch_threads)
        cv2.setNumThreads(self.cfg.opencv_threads)
        for key in ["sam_weights", "encoder_weights"]:
            expected = self.spec.get("hashes", {}).get(key)
            if expected and sha(self.paths[key]) != expected:
                raise ValueError("Weight checksum differs: " + key)
        for k, expected in self.spec.get("image_hashes", {}).items():
            if k in self.images and sha(self.images[k]) != expected:
                raise ValueError("Image checksum differs: " + k)
        # References are hashed to prevent reusing a model selected on changed labels.
        signature = dict(
            images={k: sha(p) for k, p in self.images.items()},
            validation={k: sha(self.paths["validation_masks"] / (k + ".png")) for k in self.val},
            splits=self.partition,
            training=asdict(self.cfg),
            device=str(self.device),
            weights={k: sha(self.paths[k]) for k in ["sam_weights", "encoder_weights"]},
            code={p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
        )
        signature_file = self.out / "signature.json"
        if signature_file.exists() and load(signature_file) != signature:
            raise ValueError(
                "This output belongs to different inputs/settings/code; choose a new paths.output."
            )
        dump(signature_file, signature)
        dump(self.out / "resolved_settings.json", self.spec)

    def sam(self):
        # Lazy import keeps student-only inference independent of SAM.
        from segment_anything import sam_model_registry, SamPredictor

        model = (
            sam_model_registry["vit_h"](checkpoint=str(self.paths["sam_weights"]))
            .to(self.device)
            .eval()
        )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return SamPredictor(model)

    def set_image(self, predictor, k):
        file = self.out / "embeddings" / (k + ".pt")
        if file.exists():
            cache = torch.load(file, map_location=self.device, weights_only=True)
            if cache["image_sha256"] != sha(self.images[k]):
                raise ValueError("Cached image differs: " + k)
            predictor.reset_image()
            predictor.features = cache["features"]
            predictor.original_size = cache["original_size"]
            predictor.input_size = cache["input_size"]
            predictor.is_image_set = True
        else:
            predictor.set_image(cv2.cvtColor(read_image(self.images[k]), cv2.COLOR_BGR2RGB))
            file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                dict(
                    features=predictor.features.cpu(),
                    original_size=predictor.original_size,
                    input_size=predictor.input_size,
                    image_sha256=sha(self.images[k]),
                ),
                file,
            )

    def initialize(self):
        """Original automatic positive prompts; no initial negative points."""
        from .dtzsam import positive_points, clean_mask

        dest = self.out / "initial"
        if (dest / "complete.json").exists():
            return dest
        predictor, states = self.sam(), {}
        for k in self.ids:
            print("Initial SAM mask:", k, flush=True)
            points = positive_points(read_image(self.images[k]))
            if len(points) == 0:
                raise ValueError("No initial positive prompt: " + k)
            self.set_image(predictor, k)
            masks, scores, _ = predictor.predict(
                point_coords=points,
                point_labels=np.ones(len(points), np.int32),
                multimask_output=True,
            )
            write_image(
                dest / "pseudo_labels" / (k + ".png"),
                clean_mask(masks[int(scores.argmax())]).astype("uint8") * 255,
            )
            states[k] = dict(positive_points=points.tolist(), negative_points=[])
        dump(dest / "prompt_states.json", states)
        dump(dest / "complete.json", dict(images=self.ids))
        del predictor
        release()
        return dest

    def refine(self):
        """Optional base-SAMRefiner application route with fixed official settings."""
        from .refiner import BinarySAMRefiner

        initial = self.initialize()
        dest = self.out / "refined"
        if (dest / "complete.json").exists():
            return dest
        repo = self.paths["samrefiner_repository"]
        for name, expected in self.spec.get("samrefiner_source_sha256", {}).items():
            if sha(repo / name) != expected:
                raise ValueError("SAMRefiner source differs: " + name)
        predictor = self.sam()
        refiner = BinarySAMRefiner(predictor.model, repo)
        records = {}
        for k in self.ids:
            print("SAMRefiner mask:", k, flush=True)
            image = read_image(self.images[k])
            coarse = read_mask(initial / "pseudo_labels" / (k + ".png")) > 127
            try:
                self.set_image(predictor, k)
                mask, record = refiner(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB), coarse, predictor.features
                )
            except Exception as error:
                # Prespecified fallback: retain input and expose failure, never omit it.
                mask, record = coarse, dict(status="failed_retained_input", error=repr(error))
                release()
            write_image(dest / "pseudo_labels" / (k + ".png"), mask.astype("uint8") * 255)
            records[k] = record
        dump(dest / "records.json", records)
        dump(dest / "complete.json", dict(images=self.ids))
        del refiner, predictor
        release()
        return dest

    def fit(self, dest, seed, ids, labels, initial, epochs, patience, stage=1, auxiliary=False):
        """All actual training IDs must belong to the fixed training partition."""
        if not set(ids) <= set(self.train):
            raise ValueError("A validation/test image was requested for gradient training")
        repeats = (
            max(2, math.ceil(self.cfg.auxiliary_view_floor / len(ids)))
            if auxiliary
            else self.cfg.views_per_image
        )
        inputs = dict(
            images=ids,
            seed=seed,
            initial_sha256=sha(initial) if initial else None,
            targets={k: sha(labels / (k + ".png")) for k in ids},
            epochs=epochs,
            patience=patience,
            repeats=repeats,
            stage=stage,
            training=asdict(self.cfg),
        )
        dest = Path(dest)
        cp = dest / "model_best.pt"
        if (dest / "complete.json").exists():
            if load(dest / "inputs.json") != inputs or not cp.exists():
                raise ValueError("Completed fit does not match requested inputs: " + str(dest))
            return cp
        if (dest / "inputs.json").exists():
            raise RuntimeError(
                "Interrupted fit found; keep/archive its folder and use a fresh output directory: "
                + str(dest)
            )
        dump(dest / "inputs.json", inputs)
        cp, summary = train_stage(
            self.cfg,
            self.device,
            seed,
            [self.images[k] for k in ids],
            labels,
            [self.images[k] for k in self.val],
            self.paths["validation_masks"],
            self.paths["encoder_weights"],
            dest,
            initial,
            epochs,
            patience,
            repeats,
            stage,
        )
        dump(dest / "complete.json", summary)
        release()
        return cp

    def predictions(self, cp, ids, dest):
        """Aligned original, flipped and 1.08-brightened probability maps."""
        dest = Path(dest)
        cp_hash = sha(cp)
        missing = []
        for k in ids:
            expected = dict(checkpoint_sha256=cp_hash, image_sha256=sha(self.images[k]))
            if (dest / (k + ".npz")).exists():
                if (
                    not (dest / (k + ".source.json")).exists()
                    or load(dest / (k + ".source.json")) != expected
                ):
                    raise ValueError("Prediction inputs differ: " + k)
            else:
                missing.append(k)
        if not missing:
            return dest
        dest.mkdir(parents=True, exist_ok=True)
        model = load_student(cp, self.device)
        for k in missing:
            image = read_image(self.images[k])
            predict = lambda x: predict_native(model, x, mixed_precision=self.cfg.mixed_precision)
            views = np.stack(
                [
                    predict(image),
                    predict(np.ascontiguousarray(image[:, ::-1]))[:, ::-1],
                    predict(np.clip(image.astype(np.float32) * 1.08, 0, 255).astype("uint8")),
                ]
            )
            np.savez_compressed(dest / (k + ".npz"), views=views)
            dump(
                dest / (k + ".source.json"),
                dict(checkpoint_sha256=cp_hash, image_sha256=sha(self.images[k])),
            )
        del model
        release()
        return dest

    def auxiliary(self, seed, labels, dest):
        """Train without the queried group; external images use a fold ensemble."""
        for j, held in enumerate(self.folds):
            ids = [k for k in self.train if k not in held]
            cp = self.fit(
                dest / f"fold_{j}",
                seed + 100 * j,
                ids,
                labels,
                None,
                self.cfg.auxiliary_epochs,
                self.cfg.auxiliary_patience,
                auxiliary=True,
            )
            self.predictions(cp, held, dest / "predictions")
            self.predictions(cp, self.val + self.test, dest / f"external_{j}")
        for k in self.val + self.test:
            views = np.mean(
                [
                    np.load(dest / f"external_{j}" / (k + ".npz"))["views"]
                    for j in range(len(self.folds))
                ],
                axis=0,
            )
            np.savez_compressed(dest / "predictions" / (k + ".npz"), views=views)
        return dest / "predictions"

    def feedback(self, method, step, previous, pred, cross, dest):
        from .dtzsam import clean_mask

        if (dest / "complete.json").exists():
            return dest
        oldstates = load(previous / "prompt_states.json")
        states, events = {}, {}
        predictor = self.sam()
        for k in self.ids:
            old = policy.PromptState(**oldstates[k])
            mask = read_mask(previous / "pseudo_labels" / (k + ".png")) > 127
            main = np.load(pred / (k + ".npz"))["views"]
            cross_views = np.load(cross / (k + ".npz"))["views"]
            self.set_image(predictor, k)
            candidates = []
            pool = "probability" if method == "probability_only" else "structured"
            for name, state, details in policy.candidate_bank(
                old, mask, main, cross_views, policy.PromptConfig(), pool
            ):
                points, signs = state.arrays()
                masks, scores, _ = predictor.predict(
                    point_coords=points, point_labels=signs, multimask_output=True
                )
                index = int(scores.argmax())
                output = clean_mask(masks[index])
                write_image(dest / "candidates" / name / (k + ".png"), output.astype("uint8") * 255)
                candidates.append(
                    dict(
                        name=name,
                        state=state.as_dict(),
                        sam_predicted_iou=float(scores[index]),
                        negative_growth=details,
                    )
                )
            selected = policy.choose(candidates, "sam_ranked", step)
            source = (
                dest / "candidates" / selected["name"] / (k + ".png")
                if selected
                else previous / "pseudo_labels" / (k + ".png")
            )
            target = dest / "pseudo_labels" / (k + ".png")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            states[k] = selected["state"] if selected else oldstates[k]
            events[k] = dict(
                accepted=selected["name"] if selected else "retain",
                candidates=candidates,
                predecessor_negative_count=len(old.negative_points),
                adopted_negative_count=len(states[k]["negative_points"]),
                actual_queries=len(candidates),
            )
        dump(dest / "prompt_states.json", states)
        dump(dest / "events.json", events)
        dump(dest / "complete.json", dict(images=self.ids))
        del predictor
        release()
        return dest

    def train_route(self, method, seed):
        if method not in METHODS:
            raise ValueError("Unknown public method: " + method)
        folder = self.out / f"runs/{method}/seed_{seed}"
        if (folder / "complete.json").exists():
            return folder / "best_student.pt"
        if method == "one_update":
            base = self.out / f"runs/bipst/seed_{seed}"
            if not (base / "complete.json").exists():
                raise FileNotFoundError(
                    "Run BiPST for this seed first (or use train --method all)."
                )
            meta = load(base / "complete.json")
            cps = [self.out / p for p in meta["round_checkpoints"][:2]]
            # Same S2, RNG seed and 20-epoch budget as BiPST; targets remain M1.
            cps.append(
                self.fit(
                    folder / "student_round_3",
                    seed + 2,
                    self.train,
                    base / "update_1/pseudo_labels",
                    cps[-1],
                    self.cfg.later_epochs,
                    self.cfg.later_epochs + 1,
                    stage=3,
                )
            )
        else:
            previous = self.refine() if method == "refiner_student" else self.initialize()
            cps = []
            for stage in (1, 2, 3):
                shared = stage == 1 and method != "refiner_student"
                fitdir = (
                    self.out / f"common/seed_{seed}"
                    if shared
                    else folder / f"student_round_{stage}"
                )
                cp = self.fit(
                    fitdir,
                    seed + stage - 1,
                    self.train,
                    previous / "pseudo_labels",
                    cps[-1] if cps else None,
                    self.cfg.first_epochs if stage == 1 else self.cfg.later_epochs,
                    self.cfg.first_patience if stage == 1 else self.cfg.later_epochs + 1,
                    stage=stage,
                )
                cps.append(cp)
                if stage == 3 or method in ("fixed", "refiner_student"):
                    continue
                auxdir = (
                    self.out / f"common_aux/seed_{seed}"
                    if stage == 1
                    else folder / f"auxiliary_{stage}"
                )
                cross = self.auxiliary(seed, previous / "pseudo_labels", auxdir)
                pred = self.predictions(cp, self.ids, folder / f"predictions_{stage}")
                previous = self.feedback(
                    method, stage, previous, pred, cross, folder / f"update_{stage}"
                )
        scores = [
            torch.load(cp, map_location="cpu", weights_only=True)["validation"]["dice"]
            for cp in cps
        ]
        selected = int(np.argmax(scores))
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cps[selected], folder / "best_student.pt")
        dump(
            folder / "complete.json",
            dict(
                method=method,
                seed=seed,
                round_checkpoints=[p.relative_to(self.out).as_posix() for p in cps],
                selected_round=selected + 1,
                round_validation_dice=scores,
                checkpoint_sha256=sha(folder / "best_student.pt"),
            ),
        )
        return folder / "best_student.pt"
