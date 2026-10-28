"""Test cho bảng số THAM CHIẾU của paper (`src/dire_modal/paper_table2.tsv`).

Vì sao có file này
------------------
Quy trình tái lập (`scripts/01_repro.sh`) in Δ giữa số của chúng ta và số của
tác giả. Nếu bảng tham chiếu bị chép sai một ô, Δ sẽ "đúng" ở một ô sai — và
không ai phát hiện, vì tài liệu/dữ liệu tham chiếu không có test. Đây đúng loại
lỗi repo này đã gặp (con số "Tier A ≈ 40–70 GB" bị gán nhầm suốt nhiều tháng).

Test này kiểm:
  1. File parse được, đủ 4 detector × 8 generator của Table 2 (+ hàng `avg`).
  2. Hàng `avg` KHỚP trung bình 8 ô — nên không thể chép sai một ô mà vẫn khớp.
  3. Mọi generator trong bảng đều có thật trong tier `e2` của manifest, và mọi
     detector đều có thật trong tier `ckpts` — nếu không, quy trình sẽ in
     "thiếu dữ liệu" mãi mà không ai hiểu vì sao.
  4. `compare_to_paper` / `_delta` xử lý đúng ô trống và không in "-0.0".

Chạy::

    python3 -m pytest tests/test_paper_table.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dire_modal import eval as E  # noqa: E402
from dire_modal.manifest import TIERS, File  # noqa: E402

# 8 cột của Table 2 — thứ tự như paper in.
PAPER_COLS = ["adm", "ddpm", "iddpm", "pndm", "sdv1", "sdv2", "ldm", "vqdiffusion"]
# 4 hàng = 4 detector của tác giả.
PAPER_ROWS = ["lsun_adm", "lsun_pndm", "lsun_iddpm", "lsun_stylegan"]


@pytest.fixture(scope="module")
def paper():
    return E.load_paper_table()


def test_file_is_packaged():
    """Phải nằm trong package: Modal chỉ đóng gói thư mục module."""
    assert E.PAPER_TABLE.exists(), f"thiếu {E.PAPER_TABLE}"
    assert E.PAPER_TABLE.parent.name == "dire_modal"


def test_has_exactly_the_paper_shape(paper):
    assert sorted(paper) == sorted(PAPER_ROWS)
    for ckpt, gens in paper.items():
        assert sorted(gens) == sorted(PAPER_COLS + ["avg"]), ckpt


def test_avg_row_matches_mean_of_eight(paper):
    """Bắt lỗi chép sai một ô: paper in avg, nên avg phải khớp 8 ô.

    Paper in 1 chữ số thập phân ⇒ dung sai 0.05 là đúng độ phân giải của nó.
    """
    for ckpt, gens in paper.items():
        for metric in ("acc", "ap"):
            mean = sum(gens[g][metric] for g in PAPER_COLS) / len(PAPER_COLS)
            assert abs(mean - gens["avg"][metric]) <= 0.05, (
                f"{ckpt}.{metric}: trung bình 8 ô = {mean:.3f} "
                f"nhưng bảng ghi avg = {gens['avg'][metric]}"
            )


def test_generators_exist_in_e2_tier(paper):
    """Mỗi cột paper phải có tar tương ứng trong tier `e2` của manifest."""
    e2 = {f.local.rsplit("/", 1)[-1] for f in TIERS["e2"] if isinstance(f, File)}
    missing = [f"{g}.tar.gz" for g in PAPER_COLS if f"{g}.tar.gz" not in e2]
    assert not missing, f"tier e2 thiếu file cho cột paper: {missing}"


def test_detectors_exist_in_ckpts_tier(paper):
    ckpts = {
        f.local.rsplit("/", 1)[-1] for f in TIERS["ckpts"] if isinstance(f, File)
    }
    missing = [f"{c}.pth" for c in PAPER_ROWS if f"{c}.pth" not in ckpts]
    assert not missing, f"tier ckpts thiếu detector: {missing}"


def test_ckpt_stem_matches_table_keys(paper):
    for ckpt in paper:
        assert E.ckpt_stem(f"/raw/checkpoints/{ckpt}.pth") == ckpt


def test_compare_flags_wrong_cell(paper):
    """Ô lệch 5 điểm phải bị đánh dấu; ô khớp phải qua."""
    ours = {"lsun_adm": {g: {"acc": paper["lsun_adm"][g]["acc"],
                             "ap": paper["lsun_adm"][g]["ap"]} for g in PAPER_COLS}}
    ok = E.compare_to_paper(ours, paper)
    assert ok["n_cells"] == 8
    assert ok["n_ok"] == 8, ok["worst"]

    ours["lsun_adm"]["ldm"]["acc"] -= 5.0
    bad = E.compare_to_paper(ours, paper)
    assert bad["n_ok"] == 7
    assert bad["worst"]["generator"] == "ldm"
    assert bad["worst"]["d_acc"] == pytest.approx(-5.0)


def test_missing_cells_are_reported(paper):
    """Thiếu dữ liệu phải hiện ra, không được im lặng cho qua."""
    cmp = E.compare_to_paper({}, paper)
    assert cmp["n_cells"] == 0
    assert len(cmp["missing"]) == 4 * 8


def test_delta_never_prints_negative_zero():
    assert E._delta(99.94, 99.9).strip() == "+0.0"
    assert E._delta(None, 100.0).strip() == "-"
    assert E._delta(89.4, 100.0).strip() == "-10.6"


def test_report_renders_extra_generators_separately(paper):
    """8 generator mở rộng (stylegan, dalle2…) phải nằm NGOÀI phần so paper."""
    ours = {"lsun_adm": {g: {"acc": 100.0, "ap": 100.0} for g in PAPER_COLS}}
    ours["lsun_adm"]["dalle2"] = {"acc": 91.0, "ap": 99.0}
    lines = "\n".join(E.render_report(ours, paper))
    assert "MỞ RỘNG" in lines and "dalle2" in lines
    # Δ chỉ tính trên 8 cột paper, không tính generator mở rộng
    assert "8/8 ô" in lines


# ---------------------------------------------------------------------------
# Canh con số dễ sai nhất của repo: "16 generator"
# ---------------------------------------------------------------------------
def test_no_document_claims_sixteen_generators():
    """`e2` có 16 TAR, trong đó 1 tar là `real` ⇒ **15 generator**, 15 cặp.

    Con số "16 generator" từng nằm ở 12 chỗ (docs, spec, tiers.yaml, manifest,
    download, free_quota). Nó không sai kiểu gõ nhầm mà sai kiểu *đếm*: người
    đọc thấy 16 tar rồi gọi cả 16 là generator. Hệ quả thật: bảng eval bị mong
    đợi 16 dòng, và mọi ước lượng "16 job" đều lệch.

    Ngoại lệ duy nhất: `tools/manim/data.py` — chỗ đó *giải thích* lịch sử của
    lỗi này nên buộc phải nhắc lại cụm từ.
    """
    allowed = {ROOT / "tools" / "manim" / "data.py", Path(__file__).resolve()}
    bad: list[str] = []
    for pattern in ("*.md", "*.py", "*.yaml", "*.tsv", "*.sh"):
        for path in ROOT.rglob(pattern):
            if ".git" in path.parts or path in allowed or "vendor" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if "16 generator" in line or "16 gen " in line:
                    bad.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()[:70]}")
    assert not bad, (
        "còn chỗ ghi '16 generator' (đúng là 15: 16 tar, 1 tar là `real`):\n  "
        + "\n  ".join(bad)
    )
