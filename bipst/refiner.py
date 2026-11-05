"""Binary task adapter for the unmodified base SAMRefiner operators.

No reference mask is accepted by this interface. The upstream repository and
FastGeodis must be installed separately at the recorded commits.
"""

from pathlib import Path
import ast, importlib.util, types, typing
import numpy as np
import torch
import cv2
from segment_anything.utils.transforms import ResizeLongestSide

SETTINGS = dict(
    use_point=True, use_box=True, use_mask=True, add_neg=True, margin=0.1, gamma=4.0, strength=15.0
)
ITERATIONS = 5


def module_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def source_function(path, name, class_name=None, namespace=None):
    tree = ast.parse(Path(path).read_text("utf-8"))
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    fn = next(
        n
        for n in nodes
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )
    ns = dict(torch=torch, **vars(typing))
    ns.update(namespace or {})
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    return ns[name]


class BinarySAMRefiner:
    def __init__(self, sam, upstream):
        upstream = Path(upstream)
        self.sam = sam
        util = module_file("_bipst_official_refiner_utils", upstream / "utils.py")
        self.prepare = source_function(
            upstream / "sam_refiner.py", "sam_input_prepare", namespace=vars(util)
        )
        self.merge = module_file(
            "_bipst_official_merge", upstream / "sem2ins_mask.py"
        ).merge_regions
        forward = source_function(
            upstream / "segment-anything/segment_anything/modeling/sam.py",
            "forward_with_image_embeddings",
            "Sam",
        )
        sam.forward_with_image_embeddings = types.MethodType(forward, sam)
        self.transform = ResizeLongestSide(sam.image_encoder.img_size)

    @torch.inference_mode()
    def __call__(self, image_rgb, coarse, embedding=None):
        coarse = np.asarray(coarse, dtype=bool)
        assert coarse.shape == image_rgb.shape[:2]
        if not coarse.any():
            return coarse.copy(), dict(status="empty_input", components=0, queries=0)
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(
            coarse.astype("uint8"), connectivity=8
        )
        _, _, regions = self.merge(num - 1, stats[1:], centroids[1:], labels)
        if regions is None:
            regions = [labels == i for i in range(1, num)]
        image = (
            torch.as_tensor(self.transform.apply_image(image_rgb), device=self.sam.device)
            .permute(2, 0, 1)
            .contiguous()
        )
        if embedding is None:
            embedding = self.sam.image_encoder(self.sam.preprocess(image)[None])
        result = np.zeros_like(coarse)
        queries = 0
        traces = []
        # One region per batch avoids cross-region prompt coupling and bounds memory.
        for region in regions:
            pred = torch.as_tensor(np.asarray(region, dtype="uint8")[None], device=self.sam.device)
            scores = []
            for _ in range(ITERATIONS):
                inp, _ = self.prepare(image, pred, embedding, self.transform, **SETTINGS)
                out = self.sam.forward_with_image_embeddings(
                    embedding, [inp], multimask_output=True
                )[0]
                if (
                    not torch.isfinite(out["masks"]).all()
                    or not torch.isfinite(out["iou_predictions"]).all()
                ):
                    raise FloatingPointError("Nonfinite SAMRefiner output")
                idx = out["iou_predictions"][0].argmax()
                pred = (out["masks"][0, idx][None] > 0).to(torch.uint8)
                scores.append(float(out["iou_predictions"][0, idx]))
                queries += 1
            result |= pred[0].cpu().numpy().astype(bool)
            traces.append(scores)
        return result, dict(
            status="ok", components=len(regions), queries=queries, predicted_iou=traces
        )
