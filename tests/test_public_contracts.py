"""Behavioral tests for prompt selection, augmentation and student inference."""

import numpy as np
import pytest
import torch
from bipst.config import TrainingConfig, override
from bipst.network import build_student_model, model_parameter_count, predict_native
from bipst.prompt_policy import choose, PromptState, PromptConfig, candidate_bank
from bipst.data import PairedCropDataset
from bipst.image_io import write_image


def candidate(name, score, count, growth):
    return dict(
        name=name,
        sam_predicted_iou=score,
        state={"negative_points": [[i, 0] for i in range(count)]},
        negative_growth={"actual_increment": growth},
    )


def test_response_eligibility_and_ties():
    rows = [
        candidate("no_growth", 0.99, 1, 0),
        candidate("large", 0.8, 20, 20),
        candidate("small", 0.8, 5, 5),
    ]
    assert choose(rows, "sam_ranked", 1)["name"] == "small"
    assert choose([candidate("old", 0.99, 5, 0)], "sam_ranked", 2) is None
    assert (
        choose([candidate("first", 0.7, 5, 5), candidate("second", 0.7, 5, 5)], "sam_ranked", 1)[
            "name"
        ]
        == "first"
    )


def test_candidate_growth_pairs_and_constraints():
    rng = np.random.default_rng(8)
    views = rng.uniform(0.01, 0.12, (3, 128, 256)).astype("float32")
    old = PromptState([[20, 20], [40, 20], [60, 20]], [])
    bank = candidate_bank(old, np.ones((128, 256), bool), views, views, PromptConfig())
    assert len(bank) <= 21
    for name, state, details in bank:
        assert details["requested_increment"] in [0, 5, 10, 20]
        if "_grow" in name:
            budget = int(name.split("_")[-2])
            assert details["requested_increment"] == {1: 5, 3: 10, 6: 20}[budget]
        assert not (set(map(tuple, state.positive_points)) & set(map(tuple, state.negative_points)))
        assert len(state.negative_points) == len(set(map(tuple, state.negative_points)))


def test_paired_geometry_and_new_random_views(tmp_path):
    # A binary image equals its mask, allowing paired geometry to be checked.
    im = np.zeros((130, 180, 3), np.uint8)
    im[25:100, 45:145] = 255
    write_image(tmp_path / "sample.png", im)
    labels = tmp_path / "labels"
    write_image(labels / "sample.png", im[:, :, 0])
    cfg = TrainingConfig(crop_size=64, brightness_scale=(1, 1), brightness_shift=(0, 0))
    ds = PairedCropDataset([tmp_path / "sample.png"], labels, cfg, repeats=2)
    assert len(ds) == 2
    np.random.seed(42)
    x, y, _ = ds[0]
    restored = x[0].numpy() * 0.229 + 0.485
    assert np.mean((restored > 0.5) == y[0].numpy()) > 0.97
    x2, _, _ = ds[0]
    assert not torch.equal(x, x2)


def test_model_native_shape_and_size():
    torch.set_num_threads(2)
    model = build_student_model(False)
    assert model_parameter_count(model) == 1972369
    output = predict_native(model, np.zeros((83, 157, 3), np.uint8))
    assert output.shape == (83, 157)
    assert np.isfinite(output).all()


def test_parameter_typos_are_not_silently_ignored():
    with pytest.raises(ValueError):
        override({"training": {"learning_rate": 0.001}}, ["training.learnig_rate=.01"])
    with pytest.raises(ValueError):
        TrainingConfig(batch_size=0).validate()
