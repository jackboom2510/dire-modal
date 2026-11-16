"""Ghép cặp `images/` ↔ `recons/` để kiểm chứng bước tái tạo.

Vì sao cần bảng tên, không phải `f"{gen}.tar.gz"`
-------------------------------------------------
Bốn ca tên **lệch giữa các nhánh** trong DiffusionForensics (đã kiểm bằng
`audit` 2026-10-21):

    generator   images/                  recons/             dire/
    midjourney  midjourney.tar.gz        midjourney.tar.gz   midjouney.tar.gz  ← typo gốc
    stylegan    stylegan_official.tar.gz stylegan.tar.gz     stylegan.tar.gz
    sdv1        sdv1_new.tar.gz          KHÔNG CÓ            sdv1.tar.gz

Ghép theo pattern sẽ trượt đúng những ca này, và trượt **im lặng**: file không
thấy ⇒ báo "chưa tải" thay vì "sai tên". Bảng dưới đây là nguồn sự thật; test
`test_recons_pair.py` đối chiếu nó với `dire_modal.manifest.GALLERY_TRIPLETS`
để hai bên không lệch nhau.

Vì sao đặt ở `dire_core`
------------------------
Thuần Python, không torch: ghép cặp và so tập tên member là phép kiểm **rẻ** —
làm trước khi tốn GPU. Chạy được trên máy không có CUDA, trong CI, mỗi commit.
"""

from __future__ import annotations

#: Thư mục các nhánh (đường dẫn trong dataset).
DIRS = {
    "images": "images/test/lsun_bedroom",
    "recons": "recons/test/lsun_bedroom/lsun_bedroom",
    "dire": "dire/test/lsun_bedroom/lsun_bedroom",
}

#: Tên file THEO TỪNG NHÁNH. Chỉ generator có đủ cả `images` + `recons` mới
#: vào đây; `adm`/`ddpm`/`iddpm` cố tình vắng vì tác giả **không phát hành**
#: `recons/` cho chúng (xem giáo trình / `docs/download-manual.md`).
TRIPLETS: dict[str, dict[str, str]] = {
    "real": {"images": "real.tar.gz", "recons": "real.tar.gz", "dire": "real.tar.gz"},
    "pndm": {"images": "pndm.tar.gz", "recons": "pndm.tar.gz", "dire": "pndm.tar.gz"},
    "ldm": {"images": "ldm.tar.gz", "recons": "ldm.tar.gz", "dire": "ldm.tar.gz"},
    "sdv2": {"images": "sdv2.tar.gz", "recons": "sdv2.tar.gz", "dire": "sdv2.tar.gz"},
    "vqdiffusion": {
        "images": "vqdiffusion.tar.gz",
        "recons": "vqdiffusion.tar.gz",
        "dire": "vqdiffusion.tar.gz",
    },
    "projectedgan": {
        "images": "projectedgan.tar.gz",
        "recons": "projectedgan.tar.gz",
        "dire": "projectedgan.tar.gz",
    },
    "midjourney": {
        "images": "midjourney.tar.gz",
        "recons": "midjourney.tar.gz",
        "dire": "midjouney.tar.gz",   # ⚠ typo của dataset, KHÔNG sửa
    },
    "stylegan": {
        "images": "stylegan_official.tar.gz",   # ⚠ khác tên ở hai nhánh kia
        "recons": "stylegan.tar.gz",
        "dire": "stylegan.tar.gz",
    },
}

#: Generator CÓ `recons/` của tác giả (để đối chiếu được số học DDIM).
HAS_RECONS: tuple[str, ...] = tuple(TRIPLETS)

#: Generator có trong `dire/` nhưng KHÔNG có `recons/` — không đối chiếu được.
NO_RECONS: tuple[str, ...] = ("adm", "ddpm", "iddpm", "sdv1", "dalle2", "if",
                              "diff-projectedgan", "diff-stylegan")


def images_rel(gen: str) -> str:
    """Đường dẫn tương đối của ảnh gốc (dùng cho `File.remote`)."""
    return f"{DIRS['images']}/{_name(gen, 'images')}"


def recons_rel(gen: str) -> str:
    return f"{DIRS['recons']}/{_name(gen, 'recons')}"


def dire_rel(gen: str) -> str:
    return f"{DIRS['dire']}/{_name(gen, 'dire')}"


def _name(gen: str, branch: str) -> str:
    try:
        return TRIPLETS[gen][branch]
    except KeyError:
        raise KeyError(
            f"{gen!r} không có nhánh {branch!r}. Có recons/: {sorted(TRIPLETS)}; "
            f"không có recons/: {list(NO_RECONS)}"
        ) from None


def pair(gen: str) -> tuple[str, str]:
    """`(ảnh gốc, reconstruction của tác giả)` — hai đường dẫn remote."""
    return images_rel(gen), recons_rel(gen)


def member_diff(images_members: list[str], recons_members: list[str]) -> dict:
    """So tập tên member giữa hai tar.

    Phép kiểm này bắt được: tar hỏng, tải thiếu (tar cắt cụt), và **lệch lô**
    (ảnh của generator này ghép với reconstruction của generator khác — cùng
    số lượng nhưng khác tên). Không cần GPU, không cần mở ảnh.

    Trả dict `{only_images, only_recons, n_images, n_recons, ok}`.
    """
    a, b = set(images_members), set(recons_members)
    only_i = sorted(a - b)
    only_r = sorted(b - a)
    return {
        "n_images": len(a),
        "n_recons": len(b),
        "only_images": only_i[:20],
        "only_recons": only_r[:20],
        "n_only_images": len(only_i),
        "n_only_recons": len(only_r),
        "ok": not only_i and not only_r and len(a) > 0,
    }


def check_pair(gen: str, member_lister) -> dict:
    """`member_lister(rel_path) -> list[str]` để không phụ thuộc cách đọc tar.

    Cho phép test truyền hàm giả; production truyền hàm đọc tar thật.
    """
    img_rel, rec_rel = pair(gen)
    imgs = member_lister(img_rel)
    recs = member_lister(rec_rel)
    res = member_diff(imgs, recs)
    res.update({"generator": gen, "images": img_rel, "recons": rec_rel})
    return res
