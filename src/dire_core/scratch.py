"""Đường tái lập **from scratch**: ba thành phần tự làm, và các cổng chặn.

Khác gì đường `01_repro.sh`
---------------------------
`01_repro.sh` nạp DIRE map **đã phát hành** (thư mục `dire/` của tác giả) rồi chấm bằng
detector **đã phát hành**. Đó là *chấm lại* (R0) — cần, nhưng chỉ chứng minh
đường đo của mình khớp đường đo của tác giả.

Đường này làm cả ba (`R1+R2+R3`) và **không đọc một byte nào** từ `dire/**`:

===============  ==================================================
thành phần       tự làm bằng
===============  ==================================================
`dire_maps` R1   ADM LSUN-Bedroom + DDIM S=20, trên ảnh **mình** sinh
`detector`  R2   train ResNet-50 theo siêu tham số của tác giả
`images`    R3   sinh ảnh từ checkpoint công khai, seed mình chọn
===============  ==================================================

Cái KHÔNG nằm trong "scratch" — phải ghi rõ trong báo cáo
--------------------------------------------------------
* **Trọng số diffusion** (ADM, và sau này iDDPM/PNDM/SD) vẫn của người khác.
  Train ADM-LSUN từ đầu là ~250k step × batch 128 trên 8×A100 — hàng nghìn
  GPU-giờ, ngoài ngân sách 0đ. Đây là *công cụ*, không phải kết quả.
* **Dữ liệu thật** là LSUN-Bedroom: giấy phép chỉ cho **nghiên cứu**.
* **Kiến trúc** ResNet-50/U-Net lấy từ torchvision/guided-diffusion.

Ba cổng dưới đây là thứ giữ cho câu "from scratch" không trôi thành khẩu hiệu.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from .provenance import claim

#: Thư mục chứa DIRE map **do tác giả tính**. Đường scratch không được đọc.
#: Luật là "có một thành phần đường dẫn tên đúng `dire`" — bắt được cả
#: `raw/dire/...`, `D:/data/dire/train/x.tar.gz` và `./dire/x`, mà vẫn cho qua
#: `raw/scratch/maps/...` và `D:/dire-data/...` (tên khác, không phải map tác giả).
PUBLISHED_DIR = "dire"

def normalise_rel(path: str | Path) -> str:
    """Bỏ tiền tố volume (`raw/`) và dấu `./` để so sánh theo thành phần.

    `raw/dire/x.tar.gz` → `dire/x.tar.gz`; `D:/data/dire/x` → `D:/data/dire/x`.
    """
    p = str(path).replace("\\", "/").strip().lstrip("/")
    while p.startswith("./"):
        p = p[2:]
    if p.startswith("raw/"):
        p = p[4:]
    return p


def is_published_path(path: str | Path) -> bool:
    """True nếu đường dẫn đi qua một thư mục tên `dire` (map của tác giả)."""
    parts = [p for p in normalise_rel(path).split("/") if p not in ("", ".")]
    return PUBLISHED_DIR in parts


def assert_no_published(paths) -> None:
    """Cổng G0: chặn **trước** khi mở file, không phải sau khi tính xong.

    Đặt ở đây (không ở bash) để cả ba entrypoint — sinh ảnh, tính map, train —
    dùng chung một luật. Một luật viết hai lần là một luật sẽ lệch.
    """
    bad = [str(p) for p in paths if is_published_path(p)]
    if bad:
        raise PermissionError(
            "Đường from-scratch không được đọc DIRE map của tác giả, nhưng nhận: "
            + ", ".join(bad[:5])
            + (f" … (+{len(bad) - 5})" if len(bad) > 5 else "")
            + ". Dùng `dire/**` là chấm lại (R0), không phải tái lập."
        )


# --------------------------------------------------------------------------- #
# Chia seed — để ảnh sinh không rò từ tập train sang tập test
# --------------------------------------------------------------------------- #

def split_seeds(n_train: int, n_test: int, base: int = 0) -> dict[str, list[int]]:
    """Seed **rời nhau** cho train/val/test, suy ra tất định từ `base`.

    Vì sao không dùng `range(n)`: bộ sinh ảnh nhận seed làm đầu vào, nên nếu
    train và test trùng seed thì hai tập có ảnh **giống nhau từng byte** — mô
    hình đạt 100% và ta không biết vì nó giỏi hay vì nó đã thấy ảnh đó.
    Đây đúng loại lỗi đã có tiền lệ trong repo (`val_cfg` trỏ vào `train/`).
    """
    if n_train < 1 or n_test < 1:
        raise ValueError("n_train và n_test phải ≥ 1")
    if base < 0:
        raise ValueError("base phải ≥ 0")
    total = n_train + n_test
    # Bước nhảy cố định 1_000_003 (số nguyên tố) ⇒ ba tập không chồng nhau
    # ngay cả khi ai đó đổi n_train sau này.
    seeds = [base + i for i in range(total)]
    random.Random(base).shuffle(seeds)
    return {
        "train": seeds[:n_train],
        "test": seeds[n_train:n_train + n_test],
    }


def assert_disjoint(*groups: list[int]) -> None:
    """Không seed nào được xuất hiện ở hai tập."""
    seen: dict[int, str] = {}
    for name, group in zip(("train", "val", "test"), groups):
        for s in group:
            if s in seen and seen[s] != name:
                raise ValueError(f"seed {s} nằm ở cả {seen[s]} và {name}")
            seen[s] = name


# --------------------------------------------------------------------------- #
# Cổng vàng
# --------------------------------------------------------------------------- #

#: Mỗi cổng: (id, mô tả ngắn, vì sao nó chặn). Thứ tự = thứ tự phải xanh.
GATES: tuple[tuple[str, str, str], ...] = (
    ("G0", "không đọc dire/**",
     "đọc map tác giả là chấm lại (R0), không phải tái lập"),
    ("G1", "ảnh thật + checkpoint có sha256 khớp manifest",
     "tải hỏng giữa đường cho ra ảnh cắt cụt, DIRE map sai mà không báo"),
    ("G2", "seed train/test rời nhau",
     "trùng seed ⇒ ảnh test đã nằm trong train, ACC vô nghĩa"),
    ("G3", "gen.json đủ trường cho từng lô ảnh sinh",
     "thiếu seed/sampler/steps thì không ai sinh lại được"),
    ("G4", "DIRE map đúng giao thức (S=20, mốc cuối 950, |x−x̂|)",
     "lệch mốc ⇒ lệch biểu diễn, số không so sánh được với bài"),
    ("G5", "NFE/ảnh = 2S (đếm thật, không giả định)",
     "đường cũ chạy 1020 NFE: sai 25× và không có exception"),
    ("G6", "ngưỡng chọn trên val, test chấm một lần",
     "chọn ngưỡng trên test là tự lừa mình, ACC tăng giả tạo"),
    ("G7", "guard chi phí chạy trước khi bật container",
     "một timeout sai = tiền thật, không sửa được sau"),
    ("G8", "sec/step là số ĐO, không phải giả định",
     "0,35 s/step là giả định; mọi ước lượng tiền dựa vào nó"),
)


def gate_report(ok: dict[str, bool]) -> list[dict]:
    """Ghép trạng thái thực tế với mô tả cổng để in ra / ghi run.json."""
    out = []
    for gid, title, why in GATES:
        out.append({
            "id": gid,
            "title": title,
            "why": why,
            "ok": bool(ok.get(gid, False)),
        })
    return out


def first_blocking(report: list[dict]) -> dict | None:
    """Cổng đầu tiên còn đỏ — dừng ở đây, không chạy tiếp cho đủ bộ."""
    return next((g for g in report if not g["ok"]), None)


# --------------------------------------------------------------------------- #
# Metadata của một lần chạy từ đầu
# --------------------------------------------------------------------------- #

def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


#: Những trường **bắt buộc** trong `gen.json`; thiếu một trường là G3 đỏ.
GEN_REQUIRED = (
    "generator", "ckpt_sha256", "sampler", "steps", "seeds", "n_images",
    "resolution", "prompt", "created_utc",
)


def check_gen_meta(meta: dict) -> list[str]:
    """Trả danh sách trường thiếu (rỗng = đạt G3)."""
    # `prompt` được phép là None: ADM LSUN là unconditional, không có prompt.
    # Thiếu **khoá** mới là lỗi; giá trị rỗng thì không.
    missing = [k for k in GEN_REQUIRED
               if k not in meta or (k != "prompt" and meta[k] in (None, "", []))]
    if meta.get("seeds") and len(meta["seeds"]) != meta.get("n_images"):
        missing.append("seeds (độ dài khác n_images)")
    return missing


def scratch_claim():
    """Nhãn được phép nói: cả ba thành phần đều tự làm ⇒ `R1+R2+R3`."""
    return claim(dire_maps=True, detector=True, images=True)


def run_meta(
    *,
    generator: str,
    ckpt_sha256: str,
    dir_steps: int,
    n_train: int,
    n_test: int,
    seed_base: int,
    sec_per_step: float | None,
    gates: dict[str, bool],
    git_sha: str = "",
) -> dict:
    """Toàn bộ dữ kiện cần để người khác dựng lại đúng lần chạy này."""
    c = scratch_claim()
    return {
        "mode": "scratch",
        "claim": c.as_dict(),
        "label": c.label,
        "data_source": c.data_source,
        "generator": generator,
        "ckpt_sha256": ckpt_sha256,
        "ddim_steps": dir_steps,
        "nfe_per_image": 2 * dir_steps,
        "n_train": n_train,
        "n_test": n_test,
        "seed_base": seed_base,
        "sec_per_step": sec_per_step,
        "sec_per_step_is_measured": sec_per_step is not None,
        "gates": gate_report(gates),
        "git_sha": git_sha,
    }


def write_json(path: str | Path, doc: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return p
