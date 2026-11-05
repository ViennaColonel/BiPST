"""Student-only inference demo. No SAM, references or training are needed.

Edit DEMO_SETTINGS below, then run this file in your IDE.
By default it segments the single bundled, author-selected example image.
Masks, overlays, coverage fractions and an HTML preview go to outputs/demo/.
Relative command-line paths are based on this repository, not the shell cwd.
"""

from datetime import datetime
from pathlib import Path
from urllib.parse import quote
import argparse
import html
import json
import sys
import webbrowser

ROOT = Path(__file__).resolve().parent

# ===================== EDIT YOUR DEMO SETTINGS HERE ========================
DEMO_SETTINGS = {
    "input": "demo/shot_peening.jpg",  # Change to your own image or folder.
    "checkpoint": "weights/bipst/seed_3408.pt",
    "output": None,  # None creates outputs/demo/<timestamp>/.
    "device": "auto",  # "auto", "cuda", or "cpu".
    "threshold": 0.5,  # Foreground probability threshold.
    "mixed_precision": True,  # Used on CUDA; CPU inference uses FP32.
    "open_preview": False,  # True opens the generated local HTML page.
    "allow_file_dialog": True,  # Used only if input is set to None.
}
# ======================= END OF USER SETTINGS =============================


def repository_path(value):
    return (ROOT / Path(value).expanduser()).resolve()


def select_input(value, allow_dialog=True):
    if value:
        return repository_path(value)
    if not allow_dialog:
        raise ValueError("No input selected. Set DEMO_SETTINGS['input'] or use --input PATH.")
    try:
        import tkinter as tk
        from tkinter import filedialog

        window = tk.Tk()
        window.withdraw()
        try:
            chosen = filedialog.askopenfilename(
                title="BiPST: select your shot-peening image",
                filetypes=[
                    ("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                    ("All files", "*.*"),
                ],
            )
        finally:
            window.destroy()
    except Exception as error:
        raise RuntimeError("File chooser unavailable. Use --input IMAGE_OR_FOLDER.") from error
    return Path(chosen).resolve() if chosen else None


def make_preview(output, files):
    """HTML uses only local files; user images are never uploaded."""
    from PIL import Image

    records = json.loads((output / "predictions.json").read_text("utf8"))
    originals = output / "inputs"
    originals.mkdir(parents=True, exist_ok=True)
    cards = []
    for path, record in zip(files, records):
        # PNG thumbnails also let browsers preview TIFF inputs. Saved masks stay native.
        target = originals / (path.stem + ".png")
        with Image.open(path) as image:
            image.convert("RGB").save(target)
        name = html.escape(path.name)
        uri = quote(path.stem + ".png")
        cards.append(
            f'<section><h2>{name}</h2><p>Predicted foreground area: {100*record["foreground_fraction"]:.2f}%</p><div class="row">'
            + "".join(
                f'<figure><img src="{folder}/{uri}"><figcaption>{label}</figcaption></figure>'
                for folder, label in [
                    ("inputs", "Input image"),
                    ("masks", "Predicted mask"),
                    ("overlays", "Prediction overlay"),
                ]
            )
            + "</div></section>"
        )
    document = """<!doctype html><html lang="en"><meta charset="utf-8"><title>BiPST inference</title>
<style>body{font:16px/1.5 system-ui,sans-serif;color:#1e466e;background:#fff;max-width:1280px;margin:32px auto;padding:0 24px}h1{font-size:28px}h2{font-size:18px}section{border-top:1px solid #aadce0;margin-top:24px}.row{display:flex;gap:16px}figure{margin:0;flex:1;min-width:0}img{width:100%;height:auto}figcaption{padding:8px 0;color:#52616b}p{color:#52616b}@media(max-width:700px){.row{display:block}}</style>
<h1>BiPST · Student inference</h1><p>Segmentation from the selected lightweight student. Foreground area is the fraction of image pixels classified as shot-peened.</p>"""
    (output / "index.html").write_text(document + "".join(cards) + "</html>", encoding="utf8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image", nargs="?", help="Optional image/folder; also accepts a file dropped onto DEMO.bat"
    )
    parser.add_argument("--input")
    parser.add_argument("--checkpoint", default=DEMO_SETTINGS["checkpoint"])
    parser.add_argument(
        "--output",
        default=DEMO_SETTINGS["output"],
        help="Default: a new timestamped directory under outputs/demo",
    )
    parser.add_argument("--device", default=DEMO_SETTINGS["device"])
    parser.add_argument("--threshold", type=float, default=DEMO_SETTINGS["threshold"])
    parser.add_argument("--fp32", action="store_true", default=not DEMO_SETTINGS["mixed_precision"])
    parser.add_argument(
        "--open",
        action=argparse.BooleanOptionalAction,
        default=DEMO_SETTINGS["open_preview"],
        help="Open the local HTML preview after inference",
    )
    parser.add_argument(
        "--no-dialog", action="store_true", default=not DEMO_SETTINGS["allow_file_dialog"]
    )
    args = parser.parse_args(argv)
    if args.image and args.input:
        parser.error("Use either a positional image or --input, not both")
    source = select_input(args.input or args.image or DEMO_SETTINGS["input"], not args.no_dialog)
    if source is None:
        print("Image selection cancelled; no inference performed.")
        return 0
    checkpoint = repository_path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Student checkpoint not found: {checkpoint}")
    if not source.exists():
        raise FileNotFoundError(f"Input not found: {source}")
    output = repository_path(
        args.output or ("outputs/demo/" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    )
    # Prevent overwriting inputs or mixing a previous demo's images with new results.
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output must be a new or empty directory: {output}")
    if output == source or (source.is_dir() and output in source.parents):
        raise ValueError("Output must not contain the input directory")
    from bipst.inference import infer_images, inputs

    files = inputs(source)
    infer_images(checkpoint, source, output, args.device, args.threshold, not args.fp32, True)
    make_preview(output, files)
    print("Preview:", output / "index.html")
    if args.open:
        webbrowser.open((output / "index.html").as_uri())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Demo: {error}", file=sys.stderr)
        raise SystemExit(2)
