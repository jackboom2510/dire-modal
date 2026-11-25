from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "platforms" / "hf-spaces"
SRC_DIR = ROOT / "src"
for entry in (str(APP_DIR), str(SRC_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from demolib import detector


def test_weight_suggestions_include_author_and_each_project_generator():
    assert detector.suggested_path("author") == r"F:\DiffusionForensics\checkpoints\lsun_adm.pth"
    for generator in ("gen-adm", "gen-iddpm", "gen-pndm", "gen-stylegan"):
        assert detector.suggested_path("project", generator).endswith(
            rf"train\{generator}\detector.pth"
        )
    assert detector.suggested_path("none") == ""


def test_legacy_project_threshold_is_preserved_from_declared_logit(tmp_path):
    weights = tmp_path / "detector.pth"
    weights.touch()
    (tmp_path / "train.json").write_text(json.dumps({
        "threshold_from_val": 0.0,
        "threshold_space": "logit",
        "threshold_source": "val",
    }))
    threshold, note = detector.threshold_from_train_json(weights, "project")
    assert threshold == pytest.approx(0.0)
    assert "legacy logit label" in note


def test_missing_or_untrusted_train_threshold_falls_back_to_half(tmp_path):
    weights = tmp_path / "detector.pth"
    weights.touch()
    (tmp_path / "train.json").write_text(json.dumps({
        "threshold_from_val": 0.1,
        "threshold_space": "logit",
        "threshold_source": "default (val too small)",
    }))
    assert detector.threshold_from_train_json(weights, "project")[0] == 0.5
    assert detector.threshold_from_train_json(weights, "author")[0] == 0.5


def test_prediction_scores_the_raw_three_channel_map_and_center_crops(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "detector.pth"
    checkpoint.write_bytes(b"weights")
    seen = {}

    class FakeModel:
        def logits(self, x):
            seen["shape"] = tuple(x.shape)
            seen["map"] = x.clone()
            return torch.tensor([1.3862944])

    monkeypatch.setattr(
        detector, "_load_detector_cached",
        lambda *args: (FakeModel(), 0.65, "unit test val threshold"),
    )
    arr = np.zeros((3, 256, 256), dtype=np.float32)
    arr[:, 16:240, 16:240] = 0.25
    result = detector.predict_map(arr, checkpoint, "project")
    assert seen["shape"] == (1, 3, 224, 224)
    assert float(seen["map"].mean()) == pytest.approx(0.25)
    assert result["p_fake"] == pytest.approx(0.8)
    assert result["logit"] == pytest.approx(1.3862944)
    assert result["prediction"] == "ẢNH SINH"
    assert result["threshold"] == pytest.approx(0.65)


def test_prediction_rejects_grayscale_display_png_values(tmp_path):
    checkpoint = tmp_path / "detector.pth"
    checkpoint.touch()
    with pytest.raises(ValueError, match=r"shape \(3,H,W\)"):
        detector.predict_map(np.zeros((224, 224), dtype=np.float32), checkpoint, "project")
