"""Bố cục Volume chuẩn của đường clean-room — **một nguồn sự thật duy nhất**.

Vì sao phải có file này
-----------------------
Trước khi có nó, cùng một checkpoint ADM được nhắc tới bằng **sáu** đường dẫn
khác nhau trong chính repo này::

    /raw/checkpoints/lsun_adm.pth        src/dire_modal/api.py
    /ckpt/pretrained/lsun_adm.pth        src/dire_modal/compute_dire.py
    /raw/ckpt/adm_lsun.pt                thông báo lỗi của modal_app
    /raw/zero/ckpt/adm_lsun.pt           src/dire_zero/modal_app.py
    zero/ckpt/adm_lsun.pt                lệnh `modal volume put` trong tài liệu
    ckpt/adm_lsun.pt                     docs/scratch-reproduction.md

Không đường nào sai về mặt cú pháp, nên không có lỗi nào báo. Người dùng đẩy
2,5 GB lên một đường, chương trình tìm ở đường khác, và thông báo lỗi lại đưa ra
đường thứ ba. Đây là kiểu hỏng đắt nhất: mất một buổi tối và một lượt upload.

Bố cục mới, một volume, một gốc::

    dire-zero  →  /zero
    ├── ckpt/adm_lsun.pt          ADM 256 LSUN-Bedroom (~2,2 GB, đẩy tay)
    ├── ckpt/detector.pth         detector R2 của mình
    ├── ckpt/author/*.pth         detector của **tác giả** — chỉ để chấm lại R0
    ├── data/<split>/<gen>/*.png  kết quả `ingest`
    ├── data/manifest.tsv
    ├── data/data_card.json
    ├── maps/<split>/<gen>/*.npy  DIRE map
    ├── gen/<split>/<gen>/*.png   R3: ảnh tự sinh
    ├── runs/*.json               gen/dire/train/eval/bench/report
    ├── runs/author/*.json        số của tác giả (`results/repro_*`) để đối chiếu R0
    └── _migration/receipt.json   biên lai: cái gì đã chuyển từ đâu, sha nào

Vì sao **một** volume thay vì hai (`dire-raw` + `dire-ckpt`)
------------------------------------------------------------
Hai volume nghĩa là hai lần `commit()`, hai chỗ để quên commit, và một lớp lỗi
"file có trên Volume nhưng container khác không thấy". Cũng không còn lý do tách:
`dire-ckpt` từng tách để giữ checkpoint khỏi bị xoá khi dọn scratch — nay việc
dọn nằm ở đường dẫn (`_staging`), không ở volume.

Volume cũ **không bị xoá, không bị sửa**. `dire_zero.migrate` chỉ **đọc** nó và
**copy** sang volume mới. Không có hàm nào trong repo này xoá dữ liệu trên
Volume cũ; muốn dọn thì làm tay, sau khi `verify` xanh.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "VOLUME_NAME",
    "MOUNT",
    "LAYOUT_VERSION",
    "CKPT",
    "DETECTOR",
    "CKPT_AUTHOR",
    "DATA",
    "MANIFEST",
    "DATA_CARD",
    "MAPS",
    "GEN",
    "RUNS",
    "RUNS_AUTHOR",
    "MIGRATION",
    "OLD_VOLUMES",
    "LAYOUT",
    "targets",
]

#: Tên volume (Modal) và điểm mount trong container.
VOLUME_NAME = "dire-zero"
MOUNT = "/zero"

#: Tăng số này khi bố cục đổi theo cách **không** tương thích. Biên lai migration
#: ghi lại số này, nên đọc biên lai là biết dữ liệu được xếp theo bố cục nào.
LAYOUT_VERSION = 2

#: Vì sao 2: thêm `ckpt/author/` và `runs/author/`. Không đổi chỗ cũ nào, nhưng
#: biên lai phải nói được "dữ liệu này đã có chỗ cho detector tác giả hay chưa".

# ---- đường dẫn tuyệt đối trong container ---------------------------------- #
CKPT = f"{MOUNT}/ckpt/adm_lsun.pt"
DETECTOR = f"{MOUNT}/ckpt/detector.pth"
#: Detector của **tác giả** (ResNet-50, 23,5 M tham số, ~283 MB/file). Chỉ dùng để
#: chấm lại (R0) trên map có sẵn — không bao giờ dùng làm detector R2.
CKPT_AUTHOR = f"{MOUNT}/ckpt/author"
DATA = f"{MOUNT}/data"                        # <split>/<generator>/*.png
MANIFEST = f"{DATA}/manifest.tsv"
DATA_CARD = f"{DATA}/data_card.json"
MAPS = f"{MOUNT}/maps"                        # <split>/<generator>/*.npy
GEN = f"{MOUNT}/gen"                          # R3: ảnh mình sinh
RUNS = f"{MOUNT}/runs"                        # *.json của từng bước
#: Số của tác giả (`results/repro_*` trên volume cũ) — đích để đối chiếu R0.
RUNS_AUTHOR = f"{MOUNT}/runs/author"
MIGRATION = f"{MOUNT}/_migration"             # biên lai + log

#: Volume cũ và điểm mount của chúng — **chỉ đọc** trong quá trình migration.
OLD_VOLUMES = {
    "dire-raw": "/raw",
    "dire-ckpt": "/ckpt",
}

#: Bảng tra cho người đọc và cho test: tên logic → đường dẫn tuyệt đối.
LAYOUT: dict[str, str] = {
    "ckpt": CKPT,
    "detector": DETECTOR,
    "author_ckpt": CKPT_AUTHOR,
    "data": DATA,
    "manifest": MANIFEST,
    "data_card": DATA_CARD,
    "maps": MAPS,
    "gen": GEN,
    "runs": RUNS,
    "author_runs": RUNS_AUTHOR,
    "migration": MIGRATION,
}


def targets(root: str | Path = MOUNT) -> dict[str, str]:
    """Cùng bố cục nhưng dưới một gốc khác — dùng cho test và cho ``--root``."""
    base = str(root).rstrip("/")
    if base in ("", "/"):
        base = ""
    return {
        "ckpt": f"{base}/ckpt/adm_lsun.pt",
        "detector": f"{base}/ckpt/detector.pth",
        "author_ckpt": f"{base}/ckpt/author",
        "data": f"{base}/data",
        "manifest": f"{base}/data/manifest.tsv",
        "data_card": f"{base}/data/data_card.json",
        "maps": f"{base}/maps",
        "gen": f"{base}/gen",
        "runs": f"{base}/runs",
        "author_runs": f"{base}/runs/author",
        "migration": f"{base}/_migration",
    }
