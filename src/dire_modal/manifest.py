"""Tier definitions — chọn dữ liệu theo MỤC TIÊU, không theo giai đoạn.

Hai loại nguồn sự thật, tách bạch:
  * INVENTORY (file nào tồn tại, đường dẫn gì) → module này.
  * POLICY   (tier mặc định, trần GB, số container) → ``tiers.yaml`` cạnh module.

Vì sao tổ chức theo mục tiêu
---------------------------
Detector DIRE (``vendor/networks/resnet.py``: ``conv1=Conv2d(3,64)``,
``num_classes=2``) đọc dữ liệu qua ``torchvision.ImageFolder`` trên nhánh
``dire/`` với hai lớp ``0_real`` / ``1_fake``.
Nghĩa là ``recons/`` và phần lớn ``images/`` **không nằm trên đường train/eval**
của detector. Manifest cũ gộp tất cả vào "Tier A" nên tải thừa.

Ngoại lệ duy nhất: AEROBLADE (tier ``e4``) cần **ảnh gốc** trong ``images/``,
vì nó đo lỗi tái tạo qua autoencoder chứ không đọc ảnh DIRE.

Dung lượng THẬT (audit 2026-10-21, không còn là ước lượng)
-----------------------------------------------------------
Toàn bộ cây Baidu đã audit: **61.40 GB** / 15 thư mục, trong đó **47 GB (77%)
là dữ liệu train**. Tier A cũ (20 file) thực tế chỉ **7.42 GB** — con số
"40–70 GB" từng ghi trong docs là của CẢ CÂY, gán nhầm cho Tier A.

======================  ======  ========
tier (mặc định: e2+e4)   file    size
======================  ======  ========
e2  eval headline LSUN      17   1.94 GB
e4  AEROBLADE ảnh gốc        5   1.45 GB
**e2+e4 mặc định**      **22**  **3.39 GB**
e3  cross-domain            24   4.85 GB
e5  ensemble+calibration    11   2.11 GB
e2+e3+e4+e5 (eval đủ)       57  10.35 GB
e4x AEROBLADE mọi gen       17   3.47 GB
train / trainx / trainens    4/7/3   5.74 / 11.24 / 14.60 GB
rawx / rawxens               4/6     4.48 / 12.10 GB
======================  ======  ========

Vì 77% cây là dữ liệu train mà ràng buộc dự án là "chỉ cải thiện, không train
lại", phần bỏ được là rất lớn: eval đầy đủ chỉ 10.35 GB so với 61.40 GB cả cây.

Muốn đo lại (dataset có thể đổi)::

    modal run -m dire_modal.download::audit          # cây remote ↔ manifest
    modal run -m dire_modal.download::plan --tiers e2,e4

``plan`` hỏi Baidu ``ls -l`` một lần cho mỗi thư mục và in bảng size thật,
không tải gì cả. Số GB trong docstring này chỉ là ảnh chụp ngày 2026-10-21.
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

# dire/test/lsun_bedroom/imagenet/ — audit 2026-10-21 xác nhận ĐÚNG 10 tên.
# Ít hơn danh sách 16 của .../lsun_bedroom/: KHÔNG có projectedgan,
# diff-projectedgan, diff-stylegan, dalle2, if, midjouney.
_LSUN_IMAGENET_TEST_GENS = [
    "adm",
    "ddpm",
    "iddpm",
    "ldm",
    "pndm",
    "real",
    "sdv1",
    "sdv2",
    "stylegan",
    "vqdiffusion",
]

# AEROBLADE cần ẢNH GỐC (nhánh images/), không đọc dire/.
# Audit 2026-10-21 sửa ba chỗ so với bản đoán:
#   * nhánh images/ là MỘT tầng (images/test/lsun_bedroom/), không lồng
#     thêm lsun_bedroom/ như nhánh dire/
#   * chính tả ĐÚNG "midjourney" — typo "midjouney" chỉ có ở dire/test/lsun_bedroom/lsun_bedroom/
#   * KHÔNG có sdv1.tar.gz; chỉ có sdv1_new (362.7MB) và sdv1_new2 (361.1MB,
#     biến thể chưa rõ khác gì — không lấy cả hai để khỏi tốn 361MB vô ích)
# "adm" được thêm làm CHỨNG ÂM: ADM không phải latent diffusion nên AEROBLADE
# dự đoán sẽ fail — đó là giới hạn cần ĐO, không phải lỗi cần tránh.
_AEROBLADE_GENS = ["real", "adm", "sdv1_new", "sdv2", "midjourney"]


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

# --- e2: eval headline DIRE trên LSUN-Bedroom (17 file · 1.94 GB) --------
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

# --- gap: ĐÃ RESOLVE bởi audit 2026-10-21 -------------------------------
# Thư mục dire/test/lsun_bedroom/imagenet/ từng không ai kê tên file. Nó được
# khai báo bằng DirSpec để LIỆT KÊ ĐỘNG thay vì đoán — và audit đã trả về đúng
# 10 tên thật. Nay đã kê khai tĩnh trong E3 (_LSUN_IMAGENET_TEST_GENS).
# Cơ chế DirSpec vẫn giữ và đang được E4X dùng thật.

# --- e2min: 3 file để có CON SỐ ĐẦU TIÊN (≈ 400 MB · 1.5 giờ ở 76.6 KiB/s) --
# `e2` là 1.94 GB = 7.4 giờ ở tốc độ đo được. Nhưng để trả lời "đường eval có
# đúng không" chỉ cần MỘT generator (đúng tuyên bố chính của paper: ADM-LSUN):
#   * detector DIRE train trên LSUN-ADM
#   * DIRE map của ảnh real LSUN (nhãn 0)
#   * DIRE map của ADM-LSUN (nhãn 1)  ← generator mạnh nhất của DIRE
# Không có `real.tar.gz` thì `pairs_from_raw` bỏ qua cả nhóm, nên nó là bắt buộc.
# Cổng S2 của SPEC-001 (ACC ≥ 0.99 trên `adm`) kiểm được với đúng 3 file này.
E2MIN: list[Entry] = [
    _f("checkpoints/lsun_adm.pth", "detector DIRE (pre-trained, KHÔNG train lại)"),
    _f("dire/test/lsun_bedroom/lsun_bedroom/real.tar.gz",
       "con số đầu tiên: DIRE map ảnh real LSUN (nhãn 0)"),
    _f("dire/test/lsun_bedroom/lsun_bedroom/adm.tar.gz",
       "con số đầu tiên: DIRE map ADM-LSUN (nhãn 1)"),
]

# --- e3: eval mở rộng cross-domain (24 file · 4.85 GB) ------------------
E3: list[Entry] = [
    _f("checkpoints/imagenet_adm.pth", "detector cho domain ImageNet"),
    _f("checkpoints/celebahq_sdv2.pth", "detector cho domain CelebA-HQ"),
    # Audit 2026-10-21 xác nhận 10 tên thật (trước đó là DirSpec `gap` vì
    # manifest cũ bỏ sót cả thư mục). Dùng CHUNG checkpoint lsun_adm.pth của
    # e2 ⇒ có thêm kết quả mà KHÔNG phải train gì. Chỉ 1.1 GB.
    #
    # ✅ NGỮ NGHĨA ĐÃ XÁC NHẬN bằng probe_pair (2026-10-21, chạy thật):
    #   real.tar.gz ở .../lsun_bedroom/lsun_bedroom/ và .../lsun_bedroom/imagenet/
    #   là CÙNG 1000 ảnh (trùng tên + trùng byte từng file); sha256 container
    #   khác chỉ vì tar chứa mtime/owner khác nhau.
    #   ⇒ Tầng 1 = nguồn ảnh REAL (và do đó là domain của detector).
    #     Tầng 2 = nguồn của nhánh ảnh GIẢ.
    #   ⇒ Nhánh này là CROSS-GENERATOR, KHÔNG phải cross-domain: detector LSUN,
    #     ảnh real LSUN, chỉ đổi model sinh ảnh giả.
    #
    # ⚠ Hệ quả khi báo cáo số:
    #   - KHÔNG được cộng nhánh này vào bảng eval như một "test domain" riêng:
    #     làm vậy là đếm CÙNG 1000 ảnh real hai lần và kéo mean AP về phía LSUN.
    #   - Báo cáo riêng, nhãn "cross-generator" (real LSUN dùng chung với e2).
    *_tars(
        "dire/test/lsun_bedroom/imagenet",
        _LSUN_IMAGENET_TEST_GENS,
        "cross-generator (real DÙNG CHUNG với lsun_bedroom/lsun_bedroom/, "
        "probe_pair 2026-10-21)",
    ),
    *_tars("dire/test/imagenet/imagenet", ["adm", "real", "sdv1"], "ImageNet→ImageNet"),
    *_tars(
        "dire/test/imagenet/lsun_bedroom",
        ["adm", "real", "sdv1_test"],
        "ImageNet→LSUN (cross-dataset)",
    ),
    *_tars("dire/test/celebahq", _CELEBAHQ_TEST_GENS, "CelebA-HQ (1 tầng, không lồng)"),
]

# --- e4: eval AEROBLADE trên ẢNH GỐC (5 file · 1.45 GB) -----------------
# AEROBLADE KHÔNG đọc dire/. Nó cần ảnh thật để chạy x̂ = D(E(x)) rồi đo LPIPS.
# Nên tier này trỏ vào nhánh images/ — thứ manifest cũ loại hoàn toàn.
# Audit 2026-10-21: nhánh images/ là MỘT tầng. Đường dẫn 2 tầng mà bản đầu
# suy ra từ cấu trúc dire/ KHÔNG tồn tại — chính `audit` đã bắt được lỗi này
# trước khi `--mode fetch` kịp fail cả 4 file.
E4: list[Entry] = _tars(
    "images/test/lsun_bedroom",
    _AEROBLADE_GENS,
    "AEROBLADE: ảnh gốc để đo lỗi tái tạo qua autoencoder",
)

# --- e4x: AEROBLADE trên MỌI generator (17 file · 3.47 GB) --------------
# DirSpec liệt kê ĐỘNG: thư mục này có 25 tar, trong đó 8 file *ddim* bị NEVER
# loại (2 file trong số đó chỉ 130 B — rỗng ngay trong dataset gốc) ⇒ còn 17.
# Dùng DirSpec thay vì kê tay 17 tên để dataset có thêm generator mới thì tự
# bắt được, và để cơ chế này không thành code chết sau khi `gap` đã resolve.
E4X: list[Entry] = [
    DirSpec(
        "images/test/lsun_bedroom",
        "AEROBLADE mọi generator; NEVER tự loại 8 file *ddim* → 17/25 file",
    ),
]

# --- e5: ensemble + hiệu chỉnh ngưỡng (11 file · 2.11 GB) ---------------
# Audit 2026-10-21: dire/val/lsun_bedroom/ có 5 file chứ không phải 2 — thêm
# iddpm (119.2MB), pndm (118.4MB), stylegan (137.9MB). Ba file này khớp ĐÚNG
# với ba checkpoint ensemble bên dưới; thiếu chúng thì không hiệu chỉnh ngưỡng
# được cho 3/4 detector. Thêm vào chỉ tốn 375 MB.
E5: list[Entry] = [
    _f("checkpoints/lsun_iddpm.pth", "ensemble: detector train trên iDDPM"),
    _f("checkpoints/lsun_pndm.pth", "ensemble: detector train trên PNDM"),
    _f("checkpoints/lsun_stylegan.pth", "ensemble: detector train trên StyleGAN"),
    *_tars(
        "dire/val/lsun_bedroom",
        ["real", "adm", "iddpm", "pndm", "stylegan"],
        "hiệu chỉnh ngưỡng trên val — 1 tar cho mỗi detector ensemble",
    ),
    *_tars("dire/val/imagenet", ["real", "adm", "sdv1"], "hiệu chỉnh ngưỡng cross-domain"),
]

# --- train: train lại detector headline (4 file · 5.74 GB) --------------
TRAIN: list[Entry] = _tars(
    "dire/train/lsun_bedroom", ["real", "adm"], "train lại detector LSUN-ADM"
) + _tars("dire/val/lsun_bedroom", ["real", "adm"], "val trong lúc train")

# --- trainx: train lại cross-domain (7 file · 11.24 GB) -----------------
TRAINX: list[Entry] = _tars(
    "dire/train/imagenet", ["real", "adm"], "train detector ImageNet"
) + _tars("dire/val/imagenet", ["real", "adm", "sdv1"], "val ImageNet") + _tars(
    "dire/train/celebahq", ["real", "sdv2"], "train detector CelebA-HQ"
)

# --- trainens: train 3 detector ensemble (3 file · 14.60 GB) ------------
# Audit 2026-10-21: dire/train/lsun_bedroom/ có 5 file (20.3 GB) chứ không phải
# 2. Ba file ngoài real+adm là dữ liệu train cho lsun_iddpm/pndm/stylegan.pth
# mà e5 dùng làm ensemble. Kê ra đây để audit hết báo "không ai kê".
# KHÔNG mặc định: ràng buộc của dự án là "chỉ cải thiện, không train lại".
TRAINENS: list[Entry] = _tars(
    "dire/train/lsun_bedroom",
    ["iddpm", "pndm", "stylegan"],
    "train detector ensemble — 14.6 GB, chỉ khi thật sự train lại",
)

# --- rawx: ảnh gốc để TỰ tính lại DIRE (4 file · 4.48 GB) ---------------
# Chỉ cần khi muốn reproduce từ đầu bằng DDIM inversion. Đòi GPU + model ADM
# 2.5 GB (lsun_bedroom.pt) mà cây Baidu KHÔNG có → mặc định bỏ.
RAWX: list[Entry] = _tars(
    "images/train/lsun_bedroom", ["real", "adm"], "ảnh gốc train (tự tính DIRE)"
) + _tars("images/val/lsun_bedroom", ["real", "adm"], "ảnh gốc val (tự tính DIRE)")

# --- rawxens: ảnh gốc cho 3 detector ensemble (6 file · 12.10 GB) -------
# Audit 2026-10-21: images/train|val/lsun_bedroom/ cũng có iddpm, pndm,
# stylegan (ngoài real+adm mà rawx kê). Kê ra đây để audit không còn báo
# "file không ai kê" — nhưng KHÔNG mặc định, vì tự tính lại DIRE cần GPU.
RAWXENS: list[Entry] = _tars(
    "images/train/lsun_bedroom", ["iddpm", "pndm", "stylegan"], "ảnh gốc ensemble (train)"
) + _tars(
    "images/val/lsun_bedroom", ["iddpm", "pndm", "stylegan"], "ảnh gốc ensemble (val)"
)

# --- ckpts: toàn bộ 6 checkpoint detector (6 file · 1.58 GB) ------------
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
    "e2min": E2MIN,
    "e3": E3,
    "e4": E4,
    "e4x": E4X,
    "e5": E5,
    "train": TRAIN,
    "trainx": TRAINX,
    "trainens": TRAINENS,
    "rawx": RAWX,
    "rawxens": RAWXENS,
    "ckpts": CKPTS,
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
        "8 tar ablation số bước DDIM (images/test/lsun_bedroom/sdv1_*ddim*; "
        "audit đếm được 8, trong đó 2 tar chỉ 130 B — rỗng ngay trong dataset gốc). "
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
    # Trần cho MỘT attempt. Hạ từ 5400 → 1800 sau khi log thật cho thấy băng
    # thông chỉ 24–48 KB/s: ở tốc độ đó không attempt nào tải xong file 692 MB,
    # nên resume là cơ chế chính và attempt ngắn = nhiều lần lấy phiên mới hơn.
    subprocess_timeout: int = 1800
    # --- tải: tốc độ + resume xuyên container ------------------------------
    # 0 = KHÔNG truyền cờ -p (mặc định của BaiduPCS-Go). >0 = số kết nối song
    # song CHO MỘT FILE. Baidu throttle theo phiên, nên nhiều kết nối có thể
    # nhanh hơn, nhưng cũng dễ bị cắt hơn — đổi sau khi đo.
    # Số kết nối cho MỘT file. Tài liệu của BaiduPCS-Go (fork qjfoidnh) nói rõ:
    # user free phải để 1; tăng thread chỉ nhanh một lúc rồi **kích throttle**
    # khiến tài khoản về gần 0 tốc độ trong nhiều giờ. `0B/s in 42m48s` của lần
    # chạy 2026-10-25 đúng là chữ ký đó. Nên đặt TƯỜNG MINH 1.
    baidu_parallel: int = 1
    # Tên thư mục staging trong /raw. Đổi qua env DIRE_STAGING_DIR nếu muốn.
    dire_staging: str = ".staging"
    # Chu kỳ commit Volume khi đang tải (giây). Manifest dở nằm trong Volume nên
    # phải commit định kỳ, nếu không container chết là mất tiến độ.
    staging_commit_s: int = 45
    # `--mode` của BaiduPCS-Go: rỗng = không truyền (dùng mặc định `locate`).
    # Ba giá trị hợp lệ: locate | pcs | stream — ba API khác nhau của Baidu.
    # Thử `pcs` khi tốc độ vẫn tệ: cùng tài khoản, khác đường.
    baidu_mode: str = ""
    # Giết BaiduPCS-Go nếu file trong staging không lớn thêm trong bao nhiêu giây.
    # 0 = tắt watchdog. Đo thật 2026-10-25: `0B/s in 42m48s` — không có watchdog
    # thì BaiduPCS-Go ngồi chờ tới hết `subprocess_timeout` (90 phút) và đốt
    # sạch ngân sách thời gian của lần chạy.
    stall_timeout_s: int = 600
    # --- guard chi phí GPU (compute_dire.py) -------------------------------
    # Đây là phần TIÊU TIỀN THẬT. Bản trước để cứng 40 container × timeout
    # 24 h trên A100 ≈ $2 400 trần xấu nhất cho một lần chạy hỏng.
    dire_max_containers: int = 4
    dire_timeout_s: int = 1800
    dire_max_usd: float = 5.0
    dire_max_images: int = 500
    dire_usd_per_gpu_hour: float = 2.5
    dire_sec_per_image: float = 5.0
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
            baidu_parallel=int(
                raw.get("baidu_parallel", _POLICY_FALLBACK.baidu_parallel)
            ),
            dire_staging=str(
                raw.get("dire_staging", _POLICY_FALLBACK.dire_staging)
            ),
            staging_commit_s=int(
                raw.get("staging_commit_s", _POLICY_FALLBACK.staging_commit_s)
            ),
            dire_max_containers=int(
                raw.get("dire_max_containers", _POLICY_FALLBACK.dire_max_containers)
            ),
            dire_timeout_s=int(
                raw.get("dire_timeout_s", _POLICY_FALLBACK.dire_timeout_s)
            ),
            dire_max_usd=float(
                raw.get("dire_max_usd", _POLICY_FALLBACK.dire_max_usd)
            ),
            dire_max_images=int(
                raw.get("dire_max_images", _POLICY_FALLBACK.dire_max_images)
            ),
            dire_usd_per_gpu_hour=float(
                raw.get("dire_usd_per_gpu_hour", _POLICY_FALLBACK.dire_usd_per_gpu_hour)
            ),
            dire_sec_per_image=float(
                raw.get("dire_sec_per_image", _POLICY_FALLBACK.dire_sec_per_image)
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
        ``(File, lý do)`` rơi vào NEVER và đã bị loại. ``files`` đã khử trùng
        lặp theo đường dẫn remote.
    """
    files: list[File] = []
    unresolved: list[DirSpec] = []
    blocked: list[tuple[File, str]] = []
    seen: set[str] = set()

    def _push(f: File) -> None:
        # Khử trùng lặp theo remote, KHÔNG theo object: cùng một file có thể đến
        # từ entry tĩnh (e4) VÀ từ DirSpec (e4x) — hai object File khác nhau
        # nhưng cùng đường dẫn. entries() chỉ khử trùng ở tầng Entry nên không
        # bắt được ca này. Không khử thì --tiers e4,e4x báo 22 file / 4.91 GB
        # thay vì 17 file / 3.47 GB, và guard ước lượng sai.
        if f.remote in seen:
            return
        seen.add(f.remote)
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
