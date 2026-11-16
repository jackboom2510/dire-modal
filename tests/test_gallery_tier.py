"""Test cho tier `gallery` + `tools/triplet_view.py`.

Vì sao có file này
------------------
`NEVER` là một trong những guard quan trọng nhất của repo: nó chặn 25 GB dữ liệu
không cần (`recons/` 44 tar, `*ddim*` rỗng, weight ADM 2.5 GB). Mở nó cho tier
`gallery` là một nới lỏng CÓ CHỦ ĐÍCH, nên phải có test canh đúng ranh giới:

1. `recons/` vẫn bị chặn ở mọi tier thường (`e2`, `e4`, `train`, …).
2. Chỉ file khai báo `allow_never` mới đi qua — và chỉ `recons/`, không phải
   `*ddim*` hay `lsun_bedroom.pt`.
3. Hai đường tải (Modal `download.py` và máy `local_fetch.py`) có **cùng** danh
   sách vùng được mở; lệch nhau là lỗi im lặng.
4. Bảng bộ ba (`GALLERY_TRIPLETS`) khớp tên file thật trên Baidu, kể cả hai ca
   lệch tên (`midjouney`, `stylegan_official`).
5. `tools/triplet_view.py` không import Pillow ở mức module (để test/CI chạy
   được khi máy không có Pillow).

Chạy::

    python3 -m pytest tests/test_gallery_tier.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from dire_modal import download as dl          # noqa: E402
from dire_modal import local_fetch as lf      # noqa: E402
from dire_modal import manifest as M          # noqa: E402


# ---------------------------------------------------------------------------
# 1. Ranh giới của NEVER
# ---------------------------------------------------------------------------
def test_recons_still_blocked_by_default():
    """Mọi tier thường vẫn KHÔNG được với tới recons/."""
    for tier in ("e2", "e2min", "e3", "e4", "e4x", "e5", "train", "trainx", "rawx", "ckpts"):
        assert M.is_never(f"recons/test/x/{tier}.tar.gz"), tier


def test_gallery_is_the_only_tier_that_carries_recons():
    tiers_with_recons = {
        name for name, ents in M.TIERS.items()
        if any(isinstance(e, M.File) and "recons/" in e.remote for e in ents)
    }
    assert tiers_with_recons == {"gallery"}, tiers_with_recons


def test_gallery_files_pass_never_but_no_other_file_does():
    files, _unresolved, blocked = M.expand(M.entries("gallery"))
    assert blocked == [], "tier gallery phải tự mang theo allow_never cho recons/"
    assert len(files) == len(M.GALLERY) == 9
    for f in files:
        if "/recons/" in f"/{f.remote}":
            assert f.allow_never, f
        else:
            assert not f.allow_never, f


def test_allow_never_does_not_open_ddim_or_adm_weights():
    """Ngoại lệ phải HẸP: hai loại bị chặn vì lý do khác không được mở kèm."""
    for path in (
        "images/test/lsun_bedroom/sdv1_new_ddim10s_reversed.tar.gz",
        "images/test/lsun_bedroom/sdv1_ddim5s_reversed.tar.gz",
        "checkpoints/lsun_bedroom.pt",
    ):
        assert M.is_never(path), path
        assert M.is_never(path, allow=False), path


def test_old_paths_behaviour_is_unchanged():
    """`--paths` không có cờ mới vẫn từ chối recons/ như trước."""
    with pytest.raises(SystemExit) as exc:
        dl._files_from_paths("recons/test/lsun_bedroom/lsun_bedroom/real.tar.gz")
    assert "NEVER" in str(exc.value)


def test_allow_never_opens_only_recons():
    ok = dl._files_from_paths(
        "recons/test/lsun_bedroom/lsun_bedroom/real.tar.gz", allow_never=True)
    assert ok[0].allow_never is True

    # …nhưng KHÔNG mở tar ablation DDIM hay weight ADM.
    for bad in ("images/test/lsun_bedroom/sdv1_new_ddim10s_reversed.tar.gz",
                "checkpoints/lsun_bedroom.pt"):
        with pytest.raises(SystemExit):
            dl._files_from_paths(bad, allow_never=True)


def test_two_download_paths_agree_on_the_open_area():
    """download.py và local_fetch.py phải mở CÙNG một vùng."""
    assert dl.ALLOW_NEVER_AREAS == lf.ALLOW_NEVER_AREAS == ("recons/",)


def test_local_fetch_honours_allow_never_flag(tmp_path):
    f_ok = M.File(remote="recons/x.tar.gz", local="recons/x.tar.gz", allow_never=True)
    f_no = M.File(remote="recons/x.tar.gz", local="recons/x.tar.gz")
    assert lf._never(f_ok) is False
    assert lf._never(f_no) is True


# ---------------------------------------------------------------------------
# 2. Bảng bộ ba
# ---------------------------------------------------------------------------
def test_every_gallery_triplet_names_three_branches():
    for gen, trip in M.GALLERY_TRIPLETS.items():
        assert set(trip) == {"images", "recons", "dire"}, gen
        for name in trip.values():
            assert name.endswith(".tar.gz"), (gen, name)


def test_triplet_names_carry_the_known_exceptions():
    """Hai ca tên LỆCH — nếu ai đó "dọn" chúng cho đẹp, test này đỏ."""
    assert M.GALLERY_TRIPLETS["midjourney"]["dire"] == "midjouney.tar.gz"
    assert M.GALLERY_TRIPLETS["stylegan"]["images"] == "stylegan_official.tar.gz"
    assert M.GALLERY_TRIPLETS["stylegan"]["recons"] == "stylegan.tar.gz"


def test_gallery_does_not_claim_adm_has_recons():
    """`adm` KHÔNG có recons/ trên Baidu ⇒ không được có mặt trong bảng."""
    assert "adm" not in M.GALLERY_TRIPLETS, (
        "adm không có recons/ — thêm vào bảng là hứa một file không tồn tại"
    )
    assert "adm" not in M.GALLERY_GENS


def test_gallery_dir_layout_matches_audit():
    """Đường dẫn phải khớp cây thật: dire/ lồng 2 tầng, images/ 1 tầng."""
    assert M.GALLERY_DIRS["images"] == "images/test/lsun_bedroom"
    assert M.GALLERY_DIRS["recons"] == "recons/test/lsun_bedroom/lsun_bedroom"
    assert M.GALLERY_DIRS["dire"] == "dire/test/lsun_bedroom/lsun_bedroom"


def test_gallery_default_is_cheap():
    """Mặc định 3 generator rẻ nhất — mục đích là bằng chứng, không phải benchmark."""
    assert M.GALLERY_GENS == ("real", "pndm", "ldm")
    assert len(M.GALLERY) == 3 * 3


def test_unknown_generator_is_rejected_with_a_hint():
    with pytest.raises(KeyError):
        M.gallery_files(["adm"])


# ---------------------------------------------------------------------------
# 3. Tool xem triplet
# ---------------------------------------------------------------------------
def test_tool_imports_without_pillow():
    """Module không được `import PIL` ở mức module (CI không có Pillow)."""
    src = (ROOT / "tools" / "triplet_view.py").read_text(encoding="utf-8")
    top = [ln for ln in src.splitlines() if ln.startswith(("import ", "from "))]
    assert not any("PIL" in ln for ln in top), top


def test_common_stems_is_deterministic_and_strict():
    import triplet_view as tv  # noqa: PLC0415

    a, b, c = {"x", "y", "z"}, {"y", "z", "w"}, {"z", "y", "q"}
    assert tv.common_stems(a, b, c) == ["y", "z"]


def test_pearson_handles_degenerate_input():
    import triplet_view as tv  # noqa: PLC0415

    assert tv.pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0
    assert tv.pearson([], []) == 0.0
    assert tv.pearson([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_check_published_detects_a_clear_mismatch():
    import triplet_view as tv  # noqa: PLC0415

    same = tv.check_published([([0, 0, 0], [0, 0, 0]), ([0, 0, 0], [0, 0, 0])])
    assert same["mean_abs_error"] == 0.0

    off = tv.check_published([([0, 0, 0], [50, 50, 50]), ([0, 0, 0], [50, 50, 50])])
    assert off["mean_abs_error"] == 50.0


def test_tool_reports_missing_tars_instead_of_crashing(tmp_path):
    import triplet_view as tv  # noqa: PLC0415

    rc = tv.main(["--raw", str(tmp_path), "--gen", "pndm", "--check"])
    assert rc == 2


def test_tool_rejects_unknown_generator(tmp_path):
    import triplet_view as tv  # noqa: PLC0415

    with pytest.raises(SystemExit):
        tv.paths_for(tmp_path, "adm")
