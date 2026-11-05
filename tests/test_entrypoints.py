"""Training and inference entry-point behavior."""

from pathlib import Path
import json
import shutil
from types import SimpleNamespace
import pytest
import demo
import train

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def study_stub(tmp_path, monkeypatch):
    calls = []
    study = SimpleNamespace(
        spec={"seeds": [3407, 3408, 3409]},
        device="cpu",
        train=["a", "b"],
        val=["c"],
        test=[],
        folds=[["a"], ["b"]],
        cfg=SimpleNamespace(views_per_image=2),
        out=tmp_path,
        train_route=lambda method, seed: calls.append((method, seed)),
    )
    monkeypatch.setattr(train, "create_study", lambda *args: study)
    return study, calls


def test_setup_preserves_user_settings(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    for name in ("train", "splits"):
        shutil.copy2(
            ROOT / f"configs/{name}.example.json", tmp_path / f"configs/{name}.example.json"
        )
    monkeypatch.setattr(train, "ROOT", tmp_path)
    train.main(["--setup"])
    config = tmp_path / "configs/train.local.json"
    config.write_text('{"my_setting": true}', encoding="utf8")
    train.main(["--setup"])
    assert json.loads(config.read_text()) == {"my_setting": True}


def test_check_never_starts_training_and_seed_selection(tmp_path, monkeypatch, study_stub):
    config = tmp_path / "config.json"
    config.write_text("{}")
    monkeypatch.setattr(train, "preflight", lambda *args: ({}, {"output": tmp_path}))
    _, calls = study_stub
    train.main(["--config", str(config), "--check"])
    assert calls == []
    train.main(["--config", str(config), "--seed", "3409"])
    assert calls == [("bipst", 3409)]
    calls.clear()
    train.main(["--config", str(config), "--seed", "all"])
    assert calls == [("bipst", seed) for seed in [3407, 3408, 3409]]


def test_all_routes_run_bipst_before_one_update(tmp_path, monkeypatch, study_stub):
    config = tmp_path / "config.json"
    config.write_text("{}")
    monkeypatch.setattr(train, "preflight", lambda *args: ({}, {"output": tmp_path}))
    _, calls = study_stub
    train.main(["--config", str(config), "--method", "all", "--seed", "3408"])
    assert calls == [
        (name, 3408)
        for name in ["bipst", "fixed", "one_update", "probability_only", "refiner_student"]
    ]


def test_demo_requires_real_input_and_preserves_existing_output(tmp_path, monkeypatch):
    monkeypatch.setattr(demo, "ROOT", tmp_path)
    monkeypatch.setitem(demo.DEMO_SETTINGS, "input", None)
    with pytest.raises(ValueError, match="No input selected"):
        demo.main(["--no-dialog"])
    image = tmp_path / "image.png"
    image.write_bytes(b"placeholder")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"placeholder")
    output = tmp_path / "result"
    output.mkdir()
    (output / "existing.txt").write_text("keep")
    with pytest.raises(ValueError, match="new or empty"):
        demo.main(["--input", str(image), "--checkpoint", str(checkpoint), "--output", str(output)])
    assert (output / "existing.txt").read_text() == "keep"


def test_python_training_parameters_and_cli_precedence(tmp_path, monkeypatch, study_stub):
    from bipst.config import override

    config = tmp_path / "base.json"
    config.write_text("{}")
    monkeypatch.setattr(train, "PARAMETER_OVERRIDES", {"training.batch_size": 7})
    monkeypatch.setitem(train.TRAIN_SETTINGS, "seed", 3409)
    monkeypatch.setitem(train.TRAIN_SETTINGS, "check_only", True)
    captured = []

    def inspect(config, assignments):
        captured.append(override({"training": {"batch_size": 2}}, assignments))
        return {}, {"output": tmp_path}

    monkeypatch.setattr(train, "preflight", inspect)
    _, calls = study_stub
    train.main(["--config", str(config)])
    assert captured[-1]["training"]["batch_size"] == 7
    assert calls == []
    train.main(["--config", str(config), "--set", "training.batch_size=3"])
    assert captured[-1]["training"]["batch_size"] == 3


def test_dry_run_resolves_paths_without_loading_inputs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(train, "create_study", lambda *args: pytest.fail("Loaded study"))
    monkeypatch.setattr(train, "preflight", lambda *args: pytest.fail("Read training data"))
    monkeypatch.chdir(tmp_path.parent)
    train.main(["--config", "configs/missing.json", "--dry-run"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["config"] == str((tmp_path / "configs/missing.json").resolve())
    assert plan["methods"] == ["bipst"]


def test_training_stage_performs_optimizer_updates(tmp_path):
    # Small generated fixtures exercise actual forward/backward, validation and saving.
    # They are software checks, not shot-peening data or performance measurements.
    import numpy as np
    import torch
    from torchvision.models import mobilenet_v3_small
    from bipst.config import TrainingConfig
    from bipst.image_io import write_image
    from bipst.training import train_stage, fresh_student, seed_everything

    torch.set_num_threads(2)
    encoder = tmp_path / "encoder.pt"
    torch.save(mobilenet_v3_small(weights=None).state_dict(), encoder)
    image = np.zeros((64, 80, 3), dtype=np.uint8)
    image[16:48, 20:60] = 180
    path = tmp_path / "sample.png"
    write_image(path, image)
    masks = tmp_path / "masks"
    write_image(masks / "sample.png", (image[:, :, 0] > 0).astype("uint8") * 255)
    cfg = TrainingConfig(crop_size=64, mixed_precision=False, batch_size=2)
    seed_everything(3409)
    before = fresh_student(encoder, torch.device("cpu")).state_dict()
    checkpoint, summary = train_stage(
        cfg,
        torch.device("cpu"),
        3408,
        [path],
        masks,
        [path],
        masks,
        encoder,
        tmp_path / "fit",
        epochs=1,
        patience=2,
        repeats=2,
    )
    after = torch.load(checkpoint, weights_only=True)["model_state"]
    assert summary["optimizer_steps"] == 1 and summary["epochs_trained"] == 1
    assert np.isfinite(summary["validation"]["dice"])
    assert not torch.equal(before["head.1.weight"], after["head.1.weight"])
