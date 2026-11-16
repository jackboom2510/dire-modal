"""Test cho quy tắc N1 đã được nới (ADR-0004) — chống trôi về cả hai phía.

Vì sao có file này
------------------
N1 từng là **"không train lại detector"** tuyệt đối. Nó được nới thành quy tắc
mềm: train lại **được phép**, nhưng chỉ ở chế độ **đối chiếu** trọng số tác giả
(FR-8). Một quy tắc mềm rất dễ trôi theo hai hướng, và cả hai đều hỏng báo cáo:

  * trôi về **cấm tuyệt đối** ⇒ mất khả năng giải thích trọng số ResNet-50 mà
    tác giả đã train (lý do ADR-0004 tồn tại);
  * trôi về **cho phép không giới hạn** ⇒ số của mô hình tự train bị đem so với
    paper, và "cải thiện" biến thành "tự train lại rồi tự khen".

Test này khoá **cả hai phía**: spec phải còn FR-8, và phải còn ba điều kiện
( hai dòng cạnh nhau, chọn epoch trên val, có `provenance`).

Chạy::

    python3 -m pytest tests/test_n1_reconciliation.py -q
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = (ROOT / "spec.md").read_text(encoding="utf-8")
ADR4 = ROOT / "docs" / "adr-0004-retrain-for-reconciliation.md"


def _row(spec: str, key: str) -> str:
    """Trả về dòng bảng bắt đầu bằng ``| key |`` trong spec."""
    for line in spec.splitlines():
        if line.startswith(f"| {key} |"):
            return line
    raise AssertionError(f"không thấy dòng bảng {key!r} trong spec.md")


# ---------------------------------------------------------------------------
# 1. N1: nới, nhưng có điều kiện
# ---------------------------------------------------------------------------
def test_n1_is_no_longer_absolute():
    """N1 không còn là 'Train lại detector DIRE | không làm'."""
    line = _row(SPEC, "N1")
    assert "Train lại detector DIRE" not in line, line
    assert "đối chiếu" in line.lower(), line


def test_n1_still_forbids_using_retrained_weights_as_result():
    """Nới N1 KHÔNG có nghĩa dùng trọng số tự train làm số của paper."""
    line = _row(SPEC, "N1").lower()
    assert "tuyên bố" in line or "thay" in line, line
    assert "checkpoint tác giả" in line, line


def test_improvement_constraint_c3_survives_the_softening():
    """C3 phải còn 'training-free' — nếu không, mọi cải thiện thành train lại."""
    line = _row(SPEC, "C3")
    assert "training-free" in line, line
    assert "đối chiếu" in line, line


# ---------------------------------------------------------------------------
# 2. FR-8 tồn tại và có đủ ba điều kiện
# ---------------------------------------------------------------------------
def test_fr8_exists_as_its_own_requirement():
    assert "### FR-8 —" in SPEC
    assert "Train lại để ĐỐI CHIẾU" in SPEC


def test_fr8_requires_two_rows_on_the_same_test_set():
    """AC-8.3: không được chỉ in số của retrain rồi im lặng bỏ số tác giả."""
    assert "AC-8.3" in SPEC
    assert "dire_retrained" in SPEC
    # hai tên detector phải cùng xuất hiện trong khối FR-8
    fr8 = SPEC.split("### FR-8 —", 1)[1].split("## 8.", 1)[0]
    assert "`dire`" in fr8 and "`dire_retrained`" in fr8, fr8[:400]


def test_fr8_forbids_touching_test_while_choosing_epoch_or_threshold():
    fr8 = SPEC.split("### FR-8 —", 1)[1].split("## 8.", 1)[0]
    assert "AC-8.2" in fr8
    assert "trên `val`" in fr8, "phải nói rõ chọn epoch chỉ trên val"
    assert "test không được chạm" in fr8.lower(), "phải nói rõ test không bị chạm"


def test_fr8_requires_provenance_label():
    fr8 = SPEC.split("### FR-8 —", 1)[1].split("## 8.", 1)[0]
    assert "AC-8.5" in fr8
    assert "provenance" in fr8


def test_csv_contract_lists_dire_retrained():
    assert "`dire_retrained`" in SPEC
    assert re.search(r"`dire_retrained`[^\n]*bảng riêng", SPEC)


# ---------------------------------------------------------------------------
# 3. Hai đường không được lẫn vào nhau
# ---------------------------------------------------------------------------
def test_improve_pipeline_never_trains():
    """`02_improve.sh` là quy trình cải thiện ⇒ không được gọi train."""
    text = (ROOT / "scripts" / "02_improve.sh").read_text(encoding="utf-8")
    assert "train::main" not in text, "quy trình cải thiện không được train gì"


def test_train_py_states_it_is_fr8_not_the_report_path():
    text = (ROOT / "src" / "dire_modal" / "train.py").read_text(encoding="utf-8")
    assert "FR-8" in text
    assert "KHÔNG phải đường báo cáo chính" in text
    assert "NotImplementedError" in text, "chưa port thì phải nói thật là chưa port"


# ---------------------------------------------------------------------------
# 4. ADR-0004 là nơi ghi lý do — không được lặng lẽ xoá
# ---------------------------------------------------------------------------
def test_adr0004_records_the_three_conditions():
    assert ADR4.exists(), "thiếu ADR cho quyết định đảo N1"
    text = ADR4.read_text(encoding="utf-8")
    for needle in ("Chỉ để đối chiếu", "Đúng giao thức", "hai dòng"):
        assert needle.lower() in text.lower(), needle
    assert "ADR-0004" in text


def test_demo_train_reference_is_updated():
    """`demo_train` từng viện dẫn 'N1: không train lại' — phải trỏ sang ADR-0004."""
    text = (ROOT / "src" / "dire_modal" / "demo_train.py").read_text(encoding="utf-8")
    assert "adr-0004" in text
    assert "không train lại" not in text.lower()
