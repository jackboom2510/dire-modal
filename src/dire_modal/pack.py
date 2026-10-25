"""Giải nén .tar.gz từ /raw thành WebDataset shards trên /shards.
2000 ảnh/shard → 100k ảnh = 50 inode (so với 100k inode).
"""

import io
import json
import re
import tarfile
from pathlib import Path

import modal

from .common import app, cpu_image, vol_raw, vol_shards

SHARD_SIZE = 2000  # ảnh/shard


def _label_from_path(p: str) -> int:
    """Convention: parent folder == 'real' → 0, còn lại → 1 (fake)."""
    return 0 if "/real/" in p or p.endswith("/real") else 1


def shard_group(src_rel: str, split: str) -> str:
    """Tên nhóm shard, GIỮ NHÁNH — chỗ từng gây đếm trùng 1000 ảnh real.

    Bản cũ dùng ``out_dir = /shards/{split}/{tên tar}``, nên::

        dire/test/lsun_bedroom/lsun_bedroom/real.tar.gz → /shards/test/real/
        dire/test/lsun_bedroom/imagenet/real.tar.gz     → /shards/test/real/  ← trùng
        dire/test/imagenet/lsun_bedroom/real.tar.gz     → /shards/test/real/  ← trùng

    Ba nhánh khác nhau ghi vào CÙNG một thư mục, và ``_DONE`` của nhánh đầu làm
    hai nhánh sau bị ``[SKIP]`` vĩnh viễn. Ai đọc ``/shards/test/*`` sẽ trộn
    nhánh cross-generator với nhánh chính — đúng kiểu đếm trùng mà ``probe_pair``
    đã cảnh báo.

    Nhóm mới = phần đường dẫn giữa ``<split>/`` và tên file, nối bằng ``__``::

        dire/test/lsun_bedroom/imagenet/real.tar.gz → "lsun_bedroom__imagenet__real"

    Đây là thay đổi PHÁ VỠ tương thích của ``/shards`` cũ: shard đã pack bằng
    bản trước nằm ở thư mục khác tên nên sẽ pack lại (không xoá gì, chỉ tốn
    thời gian). Eval thì không bị ảnh hưởng — nó đọc tar gốc trong ``/raw``.
    """
    rel = src_rel.replace("\\", "/").lstrip("/")
    parts = [p for p in rel.split("/") if p]
    try:
        i = parts.index(split)
    except ValueError:
        i = 0
    mid = parts[i + 1: -1]          # bỏ "<split>/" ở đầu, "<name>.tar.gz" ở cuối
    name = parts[-1] if parts else ""
    for suffix in (".tar.gz", ".tgz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return "__".join([*mid, name]) if mid else name


@app.function(
    image=cpu_image,
    volumes={"/raw": vol_raw, "/shards": vol_shards},
    timeout=6 * 3600,
    cpu=2.0,
)
def pack_one(src_rel: str, split: str) -> dict:
    """Pack 1 file .tar.gz → nhiều shard trong /shards/{split}/{nhóm}/."""
    import webdataset as wds

    src = Path("/raw") / src_rel
    stem = shard_group(src_rel, split)
    out_dir = Path("/shards") / split / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Nếu đã có shards → skip (idempotent)
    done_marker = out_dir / "_DONE"
    if done_marker.exists():
        print(f"[SKIP] {src_rel}: đã pack")
        return {"src": src_rel, "status": "skipped"}

    pattern = str(out_dir / "shard-%06d.tar")
    count = 0
    label = _label_from_path(src_rel)
    # "r:*" chứ KHÔNG phải "r:gz": dữ liệu DIRE trên Baidu là tar *không nén*
    # dù tên file là .tar.gz. Với "r:gz", member đầu tiên ném
    # "ReadError: not a gzip file" và cả stage pack chết — không có guard nào ở
    # đây, khác với `_tar_profile` bên download.py.
    with wds.ShardWriter(pattern, maxcount=SHARD_SIZE) as sink, tarfile.open(
        src, "r:*"
    ) as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = member.name.lower()
            if not name.endswith((".png", ".jpg", ".jpeg")):
                continue
            data = tf.extractfile(member).read()
            ext = name.rsplit(".", 1)[-1]
            sink.write(
                {
                    "__key__": f"{stem}/{count:08d}",
                    ext: data,
                    "cls": str(label).encode(),
                }
            )
            count += 1

    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "src": src_rel,
                "count": count,
                "label": label,
                "shard_size": SHARD_SIZE,
            },
            indent=2,
        )
    )
    done_marker.touch()
    vol_shards.commit()
    print(f"[OK] {src_rel}: {count} ảnh → {out_dir}")
    return {"src": src_rel, "status": "ok", "count": count}


@app.function(
    image=cpu_image,
    volumes={"/raw": vol_raw, "/shards": vol_shards},
    timeout=12 * 3600,
)
def main(tiers: str = "train") -> None:
    """Pack /raw → WebDataset shards.

    ``modal run -m dire_modal.pack::main --tiers train``

    Mặc định đổi từ ``A`` sang ``train`` vì pack chỉ phục vụ huấn luyện: tier
    ``train`` = 2 tar train + 2 tar val của LSUN. Tier cũ ``A`` vẫn chạy được
    (alias → ``e2,train``) nhưng sẽ pack thêm 16 tar test — chỉ cần khi muốn
    eval trên shards thay vì đọc tar gốc.

    ``resolve()`` bỏ qua DirSpec (thư mục liệt kê động) kèm cảnh báo; pack chỉ
    làm việc với file đã tải sẵn trong /raw nên điều đó vô hại.
    """
    from .manifest import resolve

    jobs = []
    for t in tiers.split(","):
        for f in resolve(t.strip()):
            if not f.local.endswith(".tar.gz"):
                continue
            # Infer split: dire/train/... → train
            m = re.match(r"dire/(train|val|test)/", f.local)
            if not m:
                continue
            split = m.group(1)
            jobs.append({"src_rel": f.local, "split": split})
    print(f"[plan] pack {len(jobs)} archives")
    list(
        pack_one.map(
            [j["src_rel"] for j in jobs],
            [j["split"] for j in jobs],
            order_outputs=False,
        )
    )


@app.function(
    image=cpu_image,
    volumes={"/shards": vol_shards},
    timeout=600,
)
def stats() -> None:
    """modal run -m dire_modal.pack::stats"""
    for meta in sorted(Path("/shards").rglob("meta.json")):
        d = json.loads(meta.read_text())
        print(
            f"{meta.parent.relative_to('/shards')}: {d['count']:>7} ảnh  "
            f"(label={d['label']})"
        )
