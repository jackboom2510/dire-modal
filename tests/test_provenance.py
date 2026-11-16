"""Mức tái lập — repo không được tự nhận đã tái lập khi còn dùng đồ của tác giả.

Vì sao có file này
------------------
Đường eval mặc định chạy ResNet-50 của tác giả trên DIRE map của tác giả rồi so
với bảng của tác giả. Kết quả khớp `32/32` rất dễ được đọc thành "đã tái lập
DIRE", trong khi nó chỉ chứng minh **harness eval** của mình đúng. Đây là loại
sai không có exception nào báo: câu chữ sai làm cho một bảng đúng bị hiểu sai.

Test khoá bốn thứ:

1. Nhãn suy ra từ **thành phần tự làm**, không khai bằng tay, và ba trục là
   độc lập (R2 mà không có R1 là hợp lệ — FR-8).
2. R0 **bị cấm** dùng từ "tái lập"; nhãn đầy đủ (R1+R2+R3) mới được.
3. `run.json` có khối `reproduction`, CSV có cột `data_source` — nếu thiếu thì
   một bảng số không tự nói được nó là loại bằng chứng gì.
4. Tài liệu không còn câu tuyên bố tái lập không kèm mức (README, spec, plan).

Chạy::

    python3 -m pytest tests/test_provenance.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from dire_core.provenance import COMPONENTS, banner, claim  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Mặc định: không tự làm gì ⇒ R0, và nói thẳng là "chấm lại"
# ---------------------------------------------------------------------------
def test_default_is_rescore_not_reproduction():
    c = claim()
    assert c.label == "R0"
    assert c.author_made == COMPONENTS
    assert c.self_made == ()
    assert c.data_source == ("authors:dire_maps+authors:detector+authors:images")
    assert not c.is_full


def test_r0_caveat_forbids_the_word_reproduction():
    """Câu duy nhất được nói ở R0 là "khớp số"; "tái lập DIRE" bị cấm."""
    c = claim()
    assert "CHẤM LẠI" in c.caveat
    assert "không được nói" in c.caveat
    assert "tái lập DIRE" in c.caveat  # xuất hiện trong câu CẤM
    assert 'tái lập thí nghiệm' not in c.caveat


def test_r0_does_not_claim_anything_about_the_method():
    """Thiếu ba bằng chứng phải hiện ra như ba câu phủ định, không phải im lặng."""
    c = claim()
    joined = " ".join(c.does_not_prove)
    assert "DIRE map có ĐÚNG không" in joined
    assert "train lại" in joined
    assert "sinh lại" in joined
    # Cái DUY NHẤT chứng minh được: đường đo khớp.
    assert len(c.proves) == 1
    assert "đường ĐO" in c.proves[0]


# ---------------------------------------------------------------------------
# 2. Ba trục độc lập — không phải một cái thang
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "flags,label,self_made",
    [
        ((True, False, False), "R1", ("dire_maps",)),
        ((False, True, False), "R2", ("detector",)),
        ((False, False, True), "R3", ("images",)),
        ((True, True, False), "R1+R2", ("dire_maps", "detector")),
        ((False, True, True), "R2+R3", ("detector", "images")),
        ((True, True, True), "R1+R2+R3", ("dire_maps", "detector", "images")),
    ],
)
def test_label_is_derived_from_components(flags, label, self_made):
    c = claim(*flags)
    assert c.label == label
    assert c.self_made == self_made


def test_retraining_alone_is_legitimate_and_not_ranked_below_r1():
    """FR-8 train lại được mà không cần tính lại map — R2 KHÔNG phải "R1 thiếu"."""
    c = claim(detector=True)
    assert c.label == "R2"
    assert "dire_maps" in c.author_made
    # Vẫn phải nói rõ map là của tác giả, không được để "R2" nghe như xong.
    assert "DIRE map" in c.caveat
    assert "train lại" in " ".join(c.proves).lower() or "công thức train" in " ".join(c.proves)


def test_full_claim_is_the_only_one_that_may_say_reproduction():
    c = claim(True, True, True)
    assert c.is_full
    assert c.author_made == ()
    assert "tái lập đầy đủ" in c.caveat
    assert "tái lập thí nghiệm" in c.caveat


def test_claim_is_frozen():
    with pytest.raises(Exception):
        claim().label = "R9"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 3. Ghi vào artefact: run.json + CSV
# ---------------------------------------------------------------------------
def test_as_dict_carries_label_source_and_what_is_still_missing():
    d = claim().as_dict()
    assert d["label"] == "R0"
    assert d["data_source"].startswith("authors:")
    assert d["full"] is False
    # run.json phải nói được "còn thiếu gì để lên mức" — không chỉ "đang ở đâu".
    assert set(d["needs"]) == set(COMPONENTS)


def test_banner_lists_both_columns():
    lines = banner(claim(detector=True))
    text = "\n".join(lines)
    assert "R2" in text
    assert "của tác giả:" in text and "tự làm" in text


# ---------------------------------------------------------------------------
# 4. Tài liệu: không còn câu "tái lập" không kèm mức
# ---------------------------------------------------------------------------
def test_readme_does_not_headline_full_reproduction():
    head = "\n".join((ROOT / "README.md").read_text(encoding="utf-8").splitlines()[:16])
    assert "chấm lại" in head.lower() or "R0" in head
    assert "Tái lập **DIRE" not in head


def test_spec_defines_the_levels():
    spec = (ROOT / "spec.md").read_text(encoding="utf-8")
    assert "### 2.1 Mức tái lập" in spec
    for token in ("R0", "R1", "R2", "R3", "data_source"):
        assert token in spec


def test_no_doc_claims_a_hundred_percent_reproduction():
    """Chuỗi này là câu khẳng định mạnh nhất mà repo KHÔNG được có."""
    for md in (ROOT / "README.md", ROOT / "spec.md", ROOT / "plan.md",
               ROOT / "docs" / "runbook-free.md"):
        text = md.read_text(encoding="utf-8")
        assert "tái lập 100%" not in text.lower(), f"{md.name} tự nhận 100%"


def test_self_declared_levels_say_they_are_self_declared():
    """Nhãn R1/R2/R3 do người chạy bật cờ — code không kiểm được, phải nói ra."""
    text = "\n".join(banner(claim(dire_maps=True)))
    assert "TỰ KHAI" in text
    # R0 thì không có gì phải khai, nên không được làm loãng cảnh báo.
    assert "TỰ KHAI" not in "\n".join(banner(claim()))
