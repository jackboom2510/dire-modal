"""Đo và tiền — hai chỗ sai một cách im lặng.

* ACC/AP tính sai vẫn ra một con số trông hợp lý. Ở đây chúng được so với giá
  trị **tính tay** trên ví dụ nhỏ, và với một trường hợp biết trước đáp án
  (tách hoàn hảo ⇒ ACC = AP = 1; đoán mò ⇒ ACC ≈ 0,5).
* Tiền: công thức sai một hệ số là hoá đơn thật, không phải test đỏ.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dire_zero import budget, metrics


# --------------------------------------------------------------------------- #
# ACC / AP / AUC
# --------------------------------------------------------------------------- #

def test_accuracy_hand_computed():
    scores = [0.9, 0.1, 0.6, 0.4]
    labels = [1, 0, 1, 0]
    assert metrics.accuracy(scores, labels) == 1.0
    assert metrics.accuracy(scores, labels, threshold=0.7) == 0.75  # chỉ 0,6 sai
    assert metrics.confusion([0.9, 0.1], [1, 1]) == {"tp": 1, "fp": 0, "tn": 0, "fn": 1}


def test_average_precision_perfect_and_inverted():
    labels = [0, 0, 1, 1]
    assert metrics.average_precision([0.1, 0.2, 0.8, 0.9], labels) == pytest.approx(1.0)
    # xếp hạng ngược: AP = ½·⅓ + ½·½ = 5/12 (đúng định nghĩa không nội suy)
    assert metrics.average_precision([0.9, 0.8, 0.2, 0.1], labels) == pytest.approx(5 / 12)


def test_average_precision_single_class_is_nan_not_zero():
    """Không có ảnh giả ⇒ AP không xác định. Trả NaN, **không** trả 0 (0 là một
    tuyên bố "tệ nhất", và nó sẽ lọt vào bảng như một kết quả)."""
    assert metrics.average_precision([0.1, 0.2], [0, 0]) != metrics.average_precision([0.1, 0.2], [0, 0])


def test_roc_auc_is_mann_whitney():
    assert metrics.roc_auc([0.1, 0.9], [0, 1]) == 1.0
    assert metrics.roc_auc([0.9, 0.1], [0, 1]) == 0.0
    assert metrics.roc_auc([0.5, 0.5], [0, 1]) == 0.5
    assert metrics.roc_auc([0.1], [0]) != metrics.roc_auc([0.1], [0])   # NaN


def test_bootstrap_ci_brackets_the_point_and_shrinks_with_n():
    rng = __import__("random").Random(0)

    def sample(n: int):
        scores = [rng.random() for _ in range(n)]
        # nhãn có ~20 % nhiễu, nếu không ACC luôn bằng 1 và CI co lại thành 0
        labels = [int(s > 0.5) if i % 5 else int(s <= 0.5) for i, s in enumerate(scores)]
        return scores, labels

    small, y_small = sample(40)
    lo1, hi1 = metrics.bootstrap_ci(small, y_small, n=800)
    big, y_big = sample(400)
    lo2, hi2 = metrics.bootstrap_ci(big, y_big, n=800)
    assert lo1 <= hi1 and lo2 <= hi2
    assert (hi2 - lo2) < (hi1 - lo1)      # 10× dữ liệu ⇒ CI hẹp lại


def test_evaluate_reports_n_and_ci():
    out = metrics.evaluate([0.9, 0.2, 0.7, 0.1], [1, 0, 1, 0], ci=True)
    assert out["n"] == 4 and out["n_real"] == 2 and out["n_fake"] == 2
    assert out["acc"] == 1.0 and out["acc_ci95"][0] <= out["acc"] <= out["acc_ci95"][1]


def test_accuracy_rejects_empty():
    with pytest.raises(ValueError):
        metrics.accuracy([], [])
    with pytest.raises(ValueError):
        metrics.accuracy([0.1], [1, 0])


# --------------------------------------------------------------------------- #
# Tiền
# --------------------------------------------------------------------------- #

def test_cost_formula_hand_computed():
    # 1 000 ảnh × 40 NFE × 0,35 s = 14 000 s = 3,888… GPU-h
    assert budget.gpu_hours(1000 * 40, 0.35) == pytest.approx(3.8889, abs=1e-3)
    assert budget.usd(1000 * 40, 0.35, "A10G") == pytest.approx(4.2778, abs=1e-3)


def test_cost_scales_with_the_25x_bug():
    """Lỗi cũ (1020 NFE thay vì 40) phải đắt đúng 25,5 lần — nhìn thấy bằng số."""
    right = budget.usd(40, 0.35)
    wrong = budget.usd(1020, 0.35)
    assert wrong / right == pytest.approx(25.5)


def test_estimate_band_and_line():
    est = budget.Estimate(n_images=500, nfe_per_image=40, sec_per_step=0.35)
    assert est.nfe_total == 20_000
    lo, hi = est.band()
    assert lo < est.usd < hi
    assert "20 000 NFE" in est.line().replace("20000", "20 000")
    assert est.as_dict()["usd_band"][0] < est.as_dict()["usd"]


def test_max_usd_gate_refuses():
    est = budget.Estimate(1000, 40, 2.5)
    with pytest.raises(budget.GateError) as exc:
        budget.require_estimate(est, max_usd=1.0)
    assert "trần" in str(exc.value)
    budget.require_estimate(est, max_usd=None)      # không trần ⇒ đi qua


def test_sec_per_step_gate(tmp_path: Path):
    missing = tmp_path / "bench.json"
    with pytest.raises(budget.GateError) as exc:
        budget.load_sec_per_step(missing)
    assert "ĐO" in str(exc.value)

    budget.save_sec_per_step(tmp_path / "bench.json", 0.42, source="test", gpu="A10G")
    assert budget.load_sec_per_step(tmp_path / "bench.json") == pytest.approx(0.42)

    (tmp_path / "bad.json").write_text(json.dumps({"sec_per_step": 0}), encoding="utf-8")
    with pytest.raises(budget.GateError):
        budget.load_sec_per_step(tmp_path / "bad.json")


def test_price_table_has_the_gpus_we_use():
    assert budget.PRICES["A10G"] == pytest.approx(1.10)
    with pytest.raises(ValueError):
        budget.gpu_hours(10, 0)
