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


@app.function(
    image=cpu_image,
    volumes={"/raw": vol_raw, "/shards": vol_shards},
    timeout=6 * 3600,
    cpu=2.0,
)
def pack_one(src_rel: str, split: str) -> dict:
    """Pack 1 file .tar.gz → nhiều shard trong /shards/{split}/{src_stem}/."""
    import webdataset as wds

    src = Path("/raw") / src_rel
    stem = src.name.replace(".tar.gz", "")
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
    with wds.ShardWriter(pattern, maxcount=SHARD_SIZE) as sink, tarfile.open(
        src, "r:gz"
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
def main(tiers: str = "A") -> None:
    """modal run -m dire_modal.pack::main --tiers A"""
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
