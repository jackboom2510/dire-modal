"""Logit → ``P(fake)``. Toán Python thuần, **không** cần numpy/torch.

Vì sao không dùng ``torch.sigmoid``: hàm này từng sai âm thầm (đảo dấu, dùng
``softmax`` cho checkpoint 1 logit), và một bản sao trong mỗi module là cách
chắc chắn nhất để một bên đúng một bên sai. Tính bằng ``math`` cũng khiến nó
test được không cần torch — batch 64 nên chi phí không đáng kể.

Quy ước (đọc từ checkpoint, xem :func:`dire_core.detector.fc_out_features`):

* ``n_out == 1``: DIRE gốc, ``BCEWithLogitsLoss`` ⇒ ``sigmoid(z)``; ``1_fake``
  là lớp dương.
* ``n_out >= 2``: ``softmax(z)[FAKE]``.
"""

from __future__ import annotations

import math as _math

from .labels import FAKE


def sigmoid(z: float) -> float:
    """Sigmoid chịu được ``|z|`` lớn (``sigmoid(-800)`` = 0, không raise)."""
    z = float(z)
    if z >= 0:
        return 1.0 / (1.0 + _math.exp(-z))
    e = _math.exp(z)
    return e / (1.0 + e)


def prob_fake(rows, n_out: int = 1) -> list[float]:
    """Logit thô → ``P(fake)``, theo đúng quy ước mà checkpoint được train.

    Nhận list ``float`` (một logit/ảnh) hoặc list ``list[float]`` (nhiều logit).
    """
    out: list[float] = []
    for row in rows:
        if n_out == 1:
            out.append(sigmoid(row[0] if isinstance(row, (list, tuple)) else row))
            continue
        vals = list(row) if isinstance(row, (list, tuple)) else [float(row)]
        if len(vals) <= FAKE:
            # Không đủ cột để lấy lớp fake — thà báo lỗi còn hơn bịa điểm.
            raise ValueError(f"model trả {len(vals)} logit, cần > {FAKE}")
        m = max(vals)
        exps = [_math.exp(v - m) for v in vals]
        out.append(exps[FAKE] / sum(exps))
    return out
