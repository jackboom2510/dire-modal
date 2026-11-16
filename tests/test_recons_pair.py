"""Ghép cặp `images/` ↔ `recons/`: bảng tên và phép so member.

Vì sao test này quan trọng hơn vẻ ngoài của nó
----------------------------------------------
Bốn ca tên lệch nhánh (`midjouney` typo, `stylegan_official`, `sdv1_new`,
`sdv1_test`) là chỗ mà code ghép theo pattern sẽ **trượt im lặng**: không thấy
file ⇒ in "chưa tải" thay vì "sai tên", và người dùng đi tải lại thứ đã có.

Test thứ hai canh việc bảng trong `dire_core` không lệch với bảng trong
`dire_modal.manifest` — hai bản sao là hai bản sẽ lệch nhau sau ba tháng.
"""

from __future__ import annotations

import pytest

from dire_core.recons_pair import (
    HAS_RECONS,
    NO_RECONS,
    TRIPLETS,
    check_pair,
    dire_rel,
    images_rel,
    member_diff,
    pair,
    recons_rel,
)


def test_every_generator_with_recons_has_three_branches():
    for gen, trip in TRIPLETS.items():
        assert set(trip) == {"images", "recons", "dire"}, gen
        assert trip["images"].endswith(".tar.gz")
        assert trip["recons"].endswith(".tar.gz")
        assert trip["dire"].endswith(".tar.gz")


def test_typo_names_are_preserved():
    assert TRIPLETS["midjourney"]["dire"] == "midjouney.tar.gz"      # thiếu 'r'
    assert TRIPLETS["midjourney"]["images"] == "midjourney.tar.gz"
    assert TRIPLETS["stylegan"]["images"] == "stylegan_official.tar.gz"
    assert TRIPLETS["stylegan"]["recons"] == "stylegan.tar.gz"


def test_paths_are_under_the_right_branch():
    assert images_rel("pndm") == "images/test/lsun_bedroom/pndm.tar.gz"
    assert recons_rel("pndm") == "recons/test/lsun_bedroom/lsun_bedroom/pndm.tar.gz"
    assert dire_rel("midjourney") == "dire/test/lsun_bedroom/lsun_bedroom/midjouney.tar.gz"


def test_adm_has_no_recons_and_says_so():
    assert "adm" in NO_RECONS
    assert "adm" not in HAS_RECONS
    with pytest.raises(KeyError, match="recons"):
        pair("adm")


def test_member_diff_identical_is_ok():
    names = ["0_real/a.jpg", "0_real/b.jpg"]
    res = member_diff(names, list(names))
    assert res["ok"] and res["n_images"] == 2 and res["n_only_images"] == 0


def test_member_diff_catches_missing_member():
    res = member_diff(["a.jpg", "b.jpg"], ["a.jpg"])
    assert not res["ok"]
    assert res["only_images"] == ["b.jpg"]


def test_member_diff_catches_wrong_batch():
    """Cùng số lượng nhưng khác tên = tar lệch lô, không phải tar thiếu."""
    res = member_diff(["adm/1.jpg", "adm/2.jpg"], ["pndm/1.jpg", "pndm/2.jpg"])
    assert not res["ok"]
    assert res["n_only_images"] == 2 and res["n_only_recons"] == 2


def test_member_diff_empty_tar_is_not_ok():
    assert not member_diff([], [])["ok"]


def test_check_pair_uses_lister_and_keeps_paths():
    res = check_pair("ldm", lambda rel: ["x.jpg"] if "images" in rel else ["x.jpg"])
    assert res["ok"] and res["generator"] == "ldm"
    assert res["images"].startswith("images/") and res["recons"].startswith("recons/")


# --------------------------------------------------------------------------- #
# Hai bảng phải khớp nhau (dire_core ↔ dire_modal.manifest)
# --------------------------------------------------------------------------- #

def test_core_table_matches_manifest_gallery_table():
    from dire_modal.manifest import GALLERY_TRIPLETS

    assert set(TRIPLETS) == set(GALLERY_TRIPLETS)
    for gen, trip in TRIPLETS.items():
        assert trip == GALLERY_TRIPLETS[gen], f"{gen} lệch giữa hai bảng"


def test_manifest_gallery_default_stays_subset_of_recons_generators():
    from dire_modal.manifest import GALLERY_GENS

    assert set(GALLERY_GENS) <= set(TRIPLETS)
