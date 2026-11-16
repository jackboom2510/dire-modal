"""Đo — ACC@0,5, AP, và khoảng tin cậy bootstrap.

Không dùng `sklearn`: một dòng import thêm 30 MB vào image chỉ để tính hai con
số 15 dòng code làm được, và ở đây điều quan trọng là **định nghĩa** hai con số
đó được viết ra rõ ràng.

Định nghĩa (khớp bài báo):

* ``P(fake) ≥ 0,5`` ⇒ đoán giả. ACC là tỉ lệ đoán đúng, nhãn ``fake = 1``.
* ``AP = Σ_n (R_n − R_{n−1}) · P_n`` trên ngưỡng quét giảm dần — diện tích dưới
  đường precision-recall, không nội suy.

**Khoảng tin cậy là bắt buộc, không phải tuỳ chọn.** Với 1 000 ảnh test, độ
lệch chuẩn của ACC quanh 0,95 là ~0,7 %; chênh lệch 1 % giữa hai cấu hình là
nhiễu, không phải kết quả. Báo cáo không có CI là báo cáo khó kiểm chứng.
"""

from __future__ import annotations

import random
from typing import Sequence

__all__ = [
    "accuracy",
    "average_precision",
    "roc_auc",
    "bootstrap_ci",
    "evaluate",
    "confusion",
    "balanced_accuracy",
    "pick_threshold",
]


def accuracy(scores: Sequence[float], labels: Sequence[int], threshold: float = 0.5) -> float:
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores/labels phải cùng độ dài và khác rỗng")
    hit = sum(1 for s, y in zip(scores, labels) if (s >= threshold) == bool(y))
    return hit / len(scores)


def confusion(scores: Sequence[float], labels: Sequence[int], threshold: float = 0.5) -> dict[str, int]:
    tp = fp = tn = fn = 0
    for s, y in zip(scores, labels):
        pred = s >= threshold
        if y and pred:
            tp += 1
        elif y and not pred:
            fn += 1
        elif not y and pred:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def balanced_accuracy(scores: Sequence[float], labels: Sequence[int], threshold: float = 0.5) -> float:
    """Trung bình của TPR và TNR — không bị lớp đông lấn át.

    Dùng để **chọn ngưỡng**: ACC thường trên val ít mẫu sẽ chọn ngưỡng cực đoan
    (ví dụ 0,02 khi mô hình còn non), và ngưỡng đó giết ACC trên test dù khả năng
    phân biệt (AP/AUC) vẫn hoàn hảo.
    """
    c = confusion(scores, labels, threshold)
    tpr = c["tp"] / max(c["tp"] + c["fn"], 1)
    tnr = c["tn"] / max(c["tn"] + c["fp"], 1)
    return 0.5 * (tpr + tnr)


def pick_threshold(
    scores: Sequence[float],
    labels: Sequence[int],
    min_samples: int = 20,
    default: float = 0.5,
) -> tuple[float, str]:
    """Chọn ngưỡng trên val theo **balanced accuracy**, quét tại điểm giữa các score.

    Trả ``(ngưỡng, nguồn)``. Nguồn là ``"val"`` hoặc ``"default"`` — chữ này được
    ghi vào `train.json`, vì "ngưỡng 0,5" và "ngưỡng chọn trên 200 ảnh val" là
    hai tuyên bố khác nhau và người đọc cần biết mình đang đọc cái nào.
    """
    if len(scores) < min_samples or len(set(labels)) < 2:
        return default, "default (val quá nhỏ hoặc một lớp)"
    uniq = sorted(set(scores))
    cand = [default]
    cand += [(a + b) / 2 for a, b in zip(uniq, uniq[1:])]
    best, best_bal = default, -1.0
    for thr in cand:
        bal = balanced_accuracy(scores, labels, thr)
        if bal > best_bal:
            best, best_bal = thr, bal
    return best, "val"


def average_precision(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AP không nội suy: quét ngưỡng theo điểm giảm dần, cộng dồn diện tích."""
    pairs = sorted(zip(scores, labels), key=lambda p: -p[0])
    pos = sum(1 for _, y in pairs if y)
    if pos == 0:
        return float("nan")
    tp = 0
    prev_recall = 0.0
    ap = 0.0
    for i, (_, y) in enumerate(pairs, start=1):
        if y:
            tp += 1
            recall = tp / pos
            precision = tp / i
            ap += (recall - prev_recall) * precision
            prev_recall = recall
    return ap


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Xác suất một ảnh giả được cho điểm cao hơn một ảnh thật (Mann–Whitney)."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def bootstrap_ci(
    scores: Sequence[float],
    labels: Sequence[float],
    metric=accuracy,
    n: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Khoảng tin cậy percentile bằng bootstrap (mặc định 95 %)."""
    idx = list(range(len(scores)))
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        pick = [rng.choice(idx) for _ in idx]
        s = [scores[i] for i in pick]
        y = [labels[i] for i in pick]
        v = metric(s, y)
        if v == v:            # bỏ NaN
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    vals.sort()
    lo = vals[int(alpha / 2 * (len(vals) - 1))]
    hi = vals[int((1 - alpha / 2) * (len(vals) - 1))]
    return (lo, hi)


def evaluate(
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float = 0.5,
    ci: bool = True,
    seed: int = 0,
) -> dict:
    """Gói mọi con số kèm số mẫu và CI — thứ được ghi vào `eval.json`."""
    out = {
        "n": len(scores),
        "n_real": sum(1 for y in labels if not y),
        "n_fake": sum(1 for y in labels if y),
        "threshold": threshold,
        "acc": accuracy(scores, labels, threshold),
        "acc_at_0.5": accuracy(scores, labels, 0.5),
        "ap": average_precision(scores, labels),
        "auc": roc_auc(scores, labels),
        "confusion": confusion(scores, labels, threshold),
    }
    if ci:
        lo, hi = bootstrap_ci(scores, labels, accuracy, seed=seed)
        out["acc_ci95"] = [lo, hi]
    return out
