"""Tier definitions — chọn dữ liệu theo MỤC TIÊU, không theo giai đoạn.

Hai loại nguồn sự thật, tách bạch:
  * INVENTORY (file nào tồn tại, đường dẫn gì) → module này.
  * POLICY   (tier mặc định, trần GB, số container) → ``tiers.yaml`` cạnh module.

Vì sao tổ chức theo mục tiêu
---------------------------
Detector DIRE (``vendor/networks/resnet.py``: ``conv1=Conv2d(3,64)``,
``num_classes=2``) đọc dữ liệu qua ``torchvision.ImageFolder`` trên nhánh
``dire/`` với hai lớp ``0_real`` / ``1_fake`` (xem ``vendor/utils/datasets.py``).
Nghĩa là ``recons/`` (44 tar) và phần lớn ``images/`` (49 tar) **không nằm trên
đường train/eval** của detector. Manifest cũ gộp tất cả vào "Tier A" nên tải
thừa ~86% số file.

Ngoại lệ duy nhất: AEROBLADE (tier ``e4``) cần **ảnh gốc** trong ``images/``,
vì nó đo lỗi tái tạo qua autoencoder chứ không đọc ảnh DIRE.

Đo dung lượng trước khi tải
--------------------------
Đừng tin số GB ghi trong docs. Chạy::

    modal run -m dire_modal.download::plan --tiers e2,e4

``plan`` hỏi Baidu ``ls -l`` một lần cho mỗi thư mục và in bảng size thật,
không tải gì cả.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

# Root trên Baidu/Rec. Xác nhận lại bằng:
#   modal run -m dire_modal.download::ls_remote --path /apps
BAIDU_ROOT = "/apps/AirExplorer"


# ============================================================================
# Entry types
# ============================================================================
@dataclass(frozen=True)
class File:
    remote: str  # path tương đối tính từ BAIDU_ROOT
    local: str  # path trên Volume /raw
    sha256: str = ""  # điền sau lần tải đầu (manifests/*_sha256.tsv)
    note: str = ""  # vì sao cần file này


@dataclass(frozen=True)
class DirSpec:
    """Thư mục cần liệt kê ĐỘNG từ cây remote thay vì đoán tên file.

    Tồn tại vì manifest cũ có bug: ``TIER_A`` ghi chú "tầng B=lsun_bedroom"
    trong khi cây thật là ``dire/test/lsun_bedroom/imagenet/`` (~10 file) và
    không ai biết chính xác 10 file đó tên gì. Đoán tên file rồi hardcode sẽ
    tái tạo lại đúng bug đó; ``DirSpec`` để ``download.expand`` hỏi remote.
    """

    remote_dir: str
    note: str = ""
    only_suffix: str = ".tar.gz"


Entry = File | DirSpec


# ============================================================================
# Tên generator / dataset (giữ nguyên chính tả gốc của dataset, kể cả typo)
# ============================================================================
# 16 generator cho test LSUN-Bedroom — đây là bảng eval headline của DIRE.
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
    "midjouney",  # ⚠️ TYPO GỐC của dataset — KHÔNG sửa thành "midjourney"
    "real",
]

# CelebA-HQ dùng chính tả ĐÚNG ("midjourney") — khác LSUN. Giữ nguyên cả hai.
_CELEBAHQ_TEST_GENS = ["dalle2", "if", "midjourney", "real", "sdv2", "stargan"]

# AEROBLADE chỉ cần vài generator đại diện: 1 real + 3 fake phủ 3 họ model
# (LDM cũ / LDM mới / model đóng). Thêm generator chỉ tốn băng thông Baidu.
_AEROBLADE_GENS = ["real", "sdv1", "sdv2", "midjouney"]


def _f(remote: str, note: str = "", local: str | None = None) -> File:
    return File(remote=remote, local=local or remote, note=note)


def _tars(prefix: str, names: list[str], note: str) -> list[File]:
    return [_f(f"{prefix}/{n}.tar.gz", note) for n in names]


# ============================================================================
# INVENTORY theo tier
# ============================================================================

# --- e0 / e1: không tải gì từ Baidu -------------------------------------
# e0 = code, tooling, docs. e1 = demo AEROBLADE: chỉ cần VAE của SD1/SD2/
# Kandinsky (~1 GB) kéo thẳng từ HF Hub trong Space, không qua Volume.
E0: list[Entry] = []
E1: list[Entry] = []

# --- e2: eval headline DIRE trên LSUN-Bedroom (17 file) -----------------
# Chỉ 1 checkpoint: DIRE train trên ADM rồi test cross-generator, nên
# lsun_adm.pth là đủ cho bảng headline. 5 checkpoint còn lại thuộc e5.
E2: list[Entry] = [
    _f("checkpoints/lsun_adm.pth", "detector DIRE train trên LSUN-ADM"),
    *_tars(
        "dire/test/lsun_bedroom/lsun_bedroom",
        _LSUN_TEST_GENS,
        "eval headline: 16 generator, nhánh dire/ (0_real/1_fake)",
    ),
]

# --- gap: thư mục manifest cũ bỏ sót ------------------------------------
# Cây thật có dire/test/lsun_bedroom/imagenet/ (~10 file) mà TIER_A không kê.
# Liệt kê động thay vì đoán tên.
GAP: list[Entry] = [
    DirSpec(
        "dire/test/lsun_bedroom/imagenet",
        "manifest cũ bỏ sót; cần audit để biết tên file thật trước khi dùng",
    ),
]

# --- e3: eval mở rộng cross-domain (ImageNet + CelebA-HQ) ---------------
E3: list[Entry] = [
    _f("checkpoints/imagenet_adm.pth", "detector cho domain ImageNet"),
    _f("checkpoints/celebahq_sdv2.pth", "detector cho domain CelebA-HQ"),
    *_tars("dire/test/imagenet/imagenet", ["adm", "real", "sdv1"], "ImageNet→ImageNet"),
    *_tars(
        "dire/test/imagenet/lsun_bedroom",
        ["adm", "real", "sdv1_test"],
        "ImageNet→LSUN (cross-dataset)",
    ),
    *_tars("dire/test/celebahq", _CELEBAHQ_TEST_GENS, "CelebA-HQ (1 tầng, không lồng)"),
]

# --- e4: eval AEROBLADE trên ẢNH GỐC (4 file) ---------------------------
# AEROBLADE KHÔNG đọc dire/. Nó cần ảnh thật để chạy x̂ = D(E(x)) rồi đo LPIPS.
# Nên tier này trỏ vào nhánh images/ — thứ manifest cũ loại hoàn toàn.
# Đường dẫn 2 tầng (images/test/lsun_bedroom/lsun_bedroom/) suy ra từ cấu trúc
# dire/; nếu audit báo không tồn tại thì cây thật là 1 tầng → sửa ở đây.
E4: list[Entry] = _tars(
    "images/test/lsun_bedroom/lsun_bedroom",
    _AEROBLADE_GENS,
    "AEROBLADE: ảnh gốc để đo lỗi tái tạo qua autoencoder",
)

# --- e5: ensemble + hiệu chỉnh ngưỡng (8 file) --------------------------
E5: list[Entry] = [
    _f("checkpoints/lsun_iddpm.pth", "ensemble: detector train trên iDDPM"),
    _f("checkpoints/lsun_pndm.pth", "ensemble: detector train trên PNDM"),
    _f("checkpoints/lsun_stylegan.pth", "ensemble: detector train trên StyleGAN"),
    *_tars("dire/val/lsun_bedroom", ["real", "adm"], "hiệu chỉnh ngưỡng trên val"),
    *_tars("dire/val/imagenet", ["real", "adm", "sdv1"], "hiệu chỉnh ngưỡng cross-domain"),
]

# --- train: train lại detector headline (4 file) ------------------------
TRAIN: list[Entry] = _tars(
    "dire/train/lsun_bedroom", ["real", "adm"], "train lại detector LSUN-ADM"
) + _tars("dire/val/lsun_bedroom", ["real", "adm"], "val trong lúc train")

# --- trainx: train lại cross-domain (5 file) ----------------------------
TRAINX: list[Entry] = _tars(
    "dire/train/imagenet", ["real", "adm"], "train detector ImageNet"
) + _tars("dire/val/imagenet", ["real", "adm", "sdv1"], "val ImageNet") + _tars(
    "dire/train/celebahq", ["real", "sdv2"], "train detector CelebA-HQ"
)

# --- rawx: ảnh gốc để TỰ tính lại DIRE (4 file) -------------------------
# Chỉ cần khi muốn reproduce từ đầu bằng DDIM inversion. Đòi GPU + model ADM
# 2.5 GB (lsun_bedroom.pt) mà cây Baidu KHÔNG có → mặc định bỏ.
RAWX: list[Entry] = _tars(
    "images/train/lsun_bedroom", ["real", "adm"], "ảnh gốc train (tự tính DIRE)"
) + _tars("images/val/lsun_bedroom", ["real", "adm"], "ảnh gốc val (tự tính DIRE)")

# --- ckpts: toàn bộ 6 checkpoint detector -------------------------------
CKPTS: list[Entry] = [
    _f(f"checkpoints/{n}.pth", "checkpoint detector DIRE")
    for n in [
        "lsun_adm",
        "imagenet_adm",
        "celebahq_sdv2",
        "lsun_iddpm",
        "lsun_pndm",
        "lsun_stylegan",
    ]
]


TIERS: dict[str, list[Entry]] = {
    "e0": E0,
    "e1": E1,
    "e2": E2,
    "e3": E3,
    "e4": E4,
    "e5": E5,
    "train": TRAIN,
    "trainx": TRAINX,
    "rawx": RAWX,
    "ckpts": CKPTS,
    "gap": GAP,
}

# Alias cho lệnh cũ (--tiers ckpt,A / pack::main --tiers A) để không gãy script.
# "D" cố ý trỏ tới rỗng: toàn bộ nội dung TIER_D cũ nằm trong NEVER.
LEGACY_ALIASES: dict[str, str] = {
    "ckpt": "ckpts",
    "A": "e2,train",
    "B": "e3",
    "C": "rawx",
    "D": "",
}


# ============================================================================
# NEVER — chặn cứng, không tier nào được với tới
# ============================================================================
# (substring trong đường dẫn remote, lý do). Kiểm tra bằng is_never().
NEVER: list[tuple[str, str]] = [
    (
        "recons/",
        "44 tar ảnh reconstruction. Detector đọc dire/ có sẵn; recons/ chỉ cần "
        "nếu TỰ tính lại DIRE (GPU + ADM 2.5 GB không có trong cây).",
    ),
    (
        "ddim",
        "10 tar ablation số bước DDIM (images/test/lsun_bedroom/sdv1_*ddim*). "
        "Không có bản dire/ tương ứng ⇒ vô dụng trừ khi tự tính DIRE.",
    ),
    (
        "lsun_bedroom.pt",
        "Weight ADM generator 2.5 GB của OpenAI guided-diffusion. KHÔNG có trong "
        "cây Baidu — tải từ nguồn khác nếu thật sự cần DDIM inversion.",
    ),
]


def is_never(remote: str) -> str | None:
    """Trả về lý do chặn nếu path rơi vào NEVER, ngược lại None."""
    low = remote.lower()
    for pat, why in NEVER:
        if pat.lower() in low:
            return why
    return None


# ============================================================================
# POLICY — đọc tiers.yaml cạnh module, fallback cứng nếu thiếu
# ============================================================================
@dataclass(frozen=True)
class Policy:
    default_tiers: str = "e2,e4"
    max_gb: float = 25.0
    max_containers: int = 2
    fetch_attempts: int = 3
    subprocess_timeout: int = 5400
    notes: dict[str, str] | None = None


_POLICY_FALLBACK = Policy(notes={})


def load_policy() -> Policy:
    """Đọc ``tiers.yaml`` cạnh module. Không bao giờ raise."""
    path = Path(__file__).with_name("tiers.yaml")
    try:
        import yaml  # pyyaml có trong cpu_image

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return Policy(
            default_tiers=str(raw.get("default_tiers", _POLICY_FALLBACK.default_tiers)),
            max_gb=float(raw.get("max_gb", _POLICY_FALLBACK.max_gb)),
            max_containers=int(
                raw.get("max_containers", _POLICY_FALLBACK.max_containers)
            ),
            fetch_attempts=int(
                raw.get("fetch_attempts", _POLICY_FALLBACK.fetch_attempts)
            ),
            subprocess_timeout=int(
                raw.get("subprocess_timeout", _POLICY_FALLBACK.subprocess_timeout)
            ),
            notes={k: str(v) for k, v in (raw.get("tiers") or {}).items()},
        )
    except Exception as exc:  # thiếu file / thiếu pyyaml / YAML hỏng
        warnings.warn(f"không đọc được {path} ({exc!r}) → dùng policy fallback")
        return _POLICY_FALLBACK


POLICY = load_policy()


# ============================================================================
# Resolve
# ============================================================================
def _key(e: Entry) -> str:
    return e.remote if isinstance(e, File) else e.remote_dir


def entries(tiers: str) -> list[Entry]:
    """Entry thô của một HOẶC nhiều tier (phân cách dấu phẩy), khử trùng lặp.

    Đệ quy vì alias cũ ánh xạ sang *danh sách* tier (``"A" → "e2,train"``) chứ
    không phải một tier đơn — bản đầu chỉ nhận tier đơn nên ``resolve("A")``
    chết ``KeyError: 'e2,train'``.
    """
    out: list[Entry] = []
    seen: set[str] = set()
    for raw in tiers.split(","):
        t = raw.strip()
        if not t:
            continue
        if t in LEGACY_ALIASES:
            target = LEGACY_ALIASES[t]
            if not target:
                warnings.warn(
                    f"tier cũ {t!r} bị chặn hoàn toàn (nội dung nằm trong NEVER) → rỗng"
                )
                continue
            warnings.warn(f"tier cũ {t!r} → {target!r} (tên mới theo mục tiêu)")
            sub = entries(target)
        elif t in TIERS:
            sub = TIERS[t]
        else:
            raise KeyError(
                f"Không biết tier {t!r}. Có: {sorted(TIERS)} "
                f"(alias cũ: {sorted(LEGACY_ALIASES)})"
            )
        for e in sub:
            k = _key(e)
            if k in seen:
                continue
            seen.add(k)
            out.append(e)
    return out


def resolve_many(tiers: str) -> list[Entry]:
    """Bí danh của :func:`entries` — giữ tên cũ cho code/script đã viết."""
    return entries(tiers)


def expand(
    ents: list[Entry], lister=None
) -> tuple[list[File], list[DirSpec], list[tuple[File, str]]]:
    """Biến Entry → File tải được.

    Args:
        ents: kết quả của :func:`resolve_many`.
        lister: callable ``(remote_dir) -> list[str]`` liệt kê tên file thật.
            ``None`` (chạy offline) ⇒ DirSpec giữ nguyên, báo lại cho người dùng.

    Returns:
        ``(files, unresolved_dirs, blocked)`` — ``blocked`` là cặp
        ``(File, lý do)`` rơi vào NEVER và đã bị loại.
    """
    files: list[File] = []
    unresolved: list[DirSpec] = []
    blocked: list[tuple[File, str]] = []

    def _push(f: File) -> None:
        why = is_never(f.remote)
        if why:
            blocked.append((f, why))
        else:
            files.append(f)

    for e in ents:
        if isinstance(e, File):
            _push(e)
            continue
        # DirSpec
        if lister is None:
            unresolved.append(e)
            continue
        try:
            names = lister(e.remote_dir)
        except Exception as exc:
            warnings.warn(f"không liệt kê được {e.remote_dir}: {exc!r}")
            unresolved.append(e)
            continue
        hits = [n for n in names if n.endswith(e.only_suffix)]
        if not hits:
            warnings.warn(f"{e.remote_dir} không có {e.only_suffix} nào — kiểm tra lại cây")
            unresolved.append(e)
            continue
        for n in sorted(hits):
            # _push tự ghi vào `blocked` nếu dính NEVER, nên file bị loại vẫn
            # được BÁO CÁO chứ không biến mất âm thầm.
            _push(_f(f"{e.remote_dir}/{n}", e.note))

    return files, unresolved, blocked


def resolve(tier: str) -> list[File]:
    """Tương thích ngược: trả về **File** của một tier, bỏ qua DirSpec.

    ``pack.py`` và các script cũ gọi hàm này. DirSpec không expand được khi
    không có kết nối Baidu nên bị bỏ kèm cảnh báo — dùng :func:`expand` trong
    ``download.py`` nếu cần chúng.
    """
    files, unresolved, _ = expand(entries(tier))
    for d in unresolved:
        warnings.warn(
            f"bỏ qua {d.remote_dir} (cần liệt kê từ remote — dùng download::plan)"
        )
    return files
