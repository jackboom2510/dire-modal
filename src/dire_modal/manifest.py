"""Tier definitions. Tránh tải thừa TB.
Sau khi dò cây Baidu bằng `download.py::ls_remote`, điền BAIDU_ROOT.
"""

from dataclasses import dataclass, field

# Root trên Baidu/Rec (CẦN ĐIỀN sau khi dò)
BAIDU_ROOT = "/apps/AirExplorer"


@dataclass(frozen=True)
class File:
    remote: str  # path tương đối tính từ BAIDU_ROOT
    local: str  # path trên Volume /raw
    sha256: str = ""  # điền sau lần tải đầu (manifests/*.tsv)


# ========== Checkpoints ==========
CKPT = [
    File("checkpoints/lsun_adm.pth", "checkpoints/lsun_adm.pth"),
    File("checkpoints/imagenet_adm.pth", "checkpoints/imagenet_adm.pth"),
    File("checkpoints/celebahq_sdv2.pth", "checkpoints/celebahq_sdv2.pth"),
    File("checkpoints/lsun_iddpm.pth", "checkpoints/lsun_iddpm.pth"),
    File("checkpoints/lsun_pndm.pth", "checkpoints/lsun_pndm.pth"),
    File("checkpoints/lsun_stylegan.pth", "checkpoints/lsun_stylegan.pth"),
]

# ========== TIER A: reproduce chính (LSUN-Bedroom ADM) ==========
# Chỉ nhánh dire/ → không cần images/ hay recons/
_LSUN_TEST_GENS = [
    "adm",
    "ddpm",
    "iddpm",
    "pndm",
    "ldm",
    "sdv1",
    "sdv2",
    "vqdiffusion",
    "stylegan",
    "projectedgan",
    "diff-projectedgan",
    "diff-stylegan",
    "dalle2",
    "if",
    "midjouney",  # ⚠️ typo gốc — KHÔNG sửa thành midjourney
    "real",
]

TIER_A = [
    # train
    File("dire/train/lsun_bedroom/real.tar.gz", "dire/train/lsun_bedroom/real.tar.gz"),
    File("dire/train/lsun_bedroom/adm.tar.gz", "dire/train/lsun_bedroom/adm.tar.gz"),
    # val
    File("dire/val/lsun_bedroom/real.tar.gz", "dire/val/lsun_bedroom/real.tar.gz"),
    File("dire/val/lsun_bedroom/adm.tar.gz", "dire/val/lsun_bedroom/adm.tar.gz"),
    # test (tầng A=lsun_bedroom, B=lsun_bedroom)
    *[
        File(
            f"dire/test/lsun_bedroom/lsun_bedroom/{g}.tar.gz",
            f"dire/test/lsun_bedroom/lsun_bedroom/{g}.tar.gz",
        )
        for g in _LSUN_TEST_GENS
    ],
]

# ========== TIER B: thêm cross-domain (ImageNet + CelebA-HQ) ==========
TIER_B = [
    # ImageNet
    File("dire/train/imagenet/real.tar.gz", "dire/train/imagenet/real.tar.gz"),
    File("dire/train/imagenet/adm.tar.gz", "dire/train/imagenet/adm.tar.gz"),
    File("dire/val/imagenet/real.tar.gz", "dire/val/imagenet/real.tar.gz"),
    File("dire/val/imagenet/adm.tar.gz", "dire/val/imagenet/adm.tar.gz"),
    File("dire/val/imagenet/sdv1.tar.gz", "dire/val/imagenet/sdv1.tar.gz"),
    # ImageNet test (hai tầng)
    *[
        File(
            f"dire/test/imagenet/imagenet/{g}.tar.gz",
            f"dire/test/imagenet/imagenet/{g}.tar.gz",
        )
        for g in ["adm", "real", "sdv1"]
    ],
    *[
        File(
            f"dire/test/imagenet/lsun_bedroom/{g}.tar.gz",
            f"dire/test/imagenet/lsun_bedroom/{g}.tar.gz",
        )
        for g in ["adm", "real", "sdv1_test"]
    ],
    # CelebA-HQ
    File("dire/train/celebahq/real.tar.gz", "dire/train/celebahq/real.tar.gz"),
    File("dire/train/celebahq/sdv2.tar.gz", "dire/train/celebahq/sdv2.tar.gz"),
    *[
        File(f"dire/test/celebahq/{g}.tar.gz", f"dire/test/celebahq/{g}.tar.gz")
        for g in ["dalle2", "if", "midjourney", "real", "sdv2", "stargan"]
    ],
]

# ========== TIER C: images/ để tự tính lại DIRE (Giai đoạn 2) ==========
TIER_C = [
    File(
        "images/train/lsun_bedroom/real.tar.gz", "images/train/lsun_bedroom/real.tar.gz"
    ),
    File(
        "images/train/lsun_bedroom/adm.tar.gz", "images/train/lsun_bedroom/adm.tar.gz"
    ),
    File("images/val/lsun_bedroom/real.tar.gz", "images/val/lsun_bedroom/real.tar.gz"),
    File("images/val/lsun_bedroom/adm.tar.gz", "images/val/lsun_bedroom/adm.tar.gz"),
    # Thêm test generators khi cần
]

# ========== TIER D: ablation số bước DDIM (sẵn trong dataset gốc) ==========
TIER_D = [
    File(
        f"images/test/lsun_bedroom/{name}.tar.gz",
        f"images/test/lsun_bedroom/{name}.tar.gz",
    )
    for name in [
        "sdv1_ddim5s_reversed",
        "sdv1_ddim5s_reversed_res",
        "sdv1_newddim5s_reversed",
        "sdv1_newddim5s_reversed_res",
        "sdv1_new_ddim10s_reversed",
        "sdv1_new_ddim10s_reversed_res",
        "sdv1_new_ddim50s_reversed",
        "sdv1_new_ddim50s_reversed_res",
    ]
]

TIERS: dict[str, list[File]] = {
    "ckpt": CKPT,
    "A": TIER_A,
    "B": TIER_B,
    "C": TIER_C,
    "D": TIER_D,
}


def resolve(tier: str) -> list[File]:
    if tier not in TIERS:
        raise KeyError(f"Unknown tier {tier!r}. Có: {list(TIERS)}")
    return TIERS[tier]
