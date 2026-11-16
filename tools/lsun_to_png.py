#!/usr/bin/env python3
"""LSUN LMDB → PNG 256×256, tất định, có manifest.

Vì sao cần bước này
-------------------
`bedroom_val_lmdb.zip` là LMDB (key = `000abc...`, value = JPEG). Muốn dùng cho
DIRE thì phải ra file ảnh, và **phải ghi lại mình đã lấy ảnh nào** — nếu chỉ
copy cả nghìn ảnh vào một thư mục mà không có danh sách, lần chạy sau không ai
biết tập test gồm những gì, và việc "chọn lại cho đẹp" trở thành bình thường.

Cách dùng::

    python tools/lsun_to_png.py --lmdb D:/dire-data/sources/lsun/bedroom_val_lmdb \\
        --out D:/dire-data/sources/lsun/png --n 1000 --manifest manifests/lsun_val.tsv

Lọc ảnh: chỉ giữ ảnh RGB 256×256 đúng, bỏ ảnh xám/khác cỡ (LSUN có lẫn). Bỏ
ảnh là hành động **được ghi lại** trong manifest, không im lặng.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", required=True, help="thư mục LMDB đã giải nén")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--manifest", default="manifests/lsun_val.tsv")
    a = ap.parse_args()

    try:
        import lmdb  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        print("[!] cần: pip install lmdb pillow", file=sys.stderr)
        return 2

    src = Path(a.lmdb)
    if not (src / "data.mdb").exists():
        print(f"[!] không thấy {src/'data.mdb'} — giải nén zip trước", file=sys.stderr)
        return 2

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man = Path(a.manifest)
    man.parent.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(str(src), readonly=True, lock=False, readahead=False, meminit=False)
    rows, kept, skipped = [], 0, 0
    with env.begin(write=False) as txn:
        for key, val in txn.cursor():
            if kept >= a.n:
                break
            try:
                img = Image.open(io.BytesIO(val)).convert("RGB")
            except Exception:  # noqa: BLE001 — một ảnh hỏng không được giết cả lô
                skipped += 1
                continue
            if img.size != (256, 256):
                skipped += 1
                continue
            name = key.decode("ascii", "replace").replace("/", "_")
            path = out / f"{name}.png"
            img.save(path, format="PNG")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(f"{name}\t{img.size[0]}x{img.size[1]}\t{digest}\t{path.name}")
            kept += 1
    env.close()

    man.write_text(
        "# nguồn: LSUN-Bedroom val (dl.yf.io/lsun/objects/bedroom_val_lmdb.zip)\n"
        "# cột: key\\t size\\t sha256\\t file\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    print(f"[xong] {kept} ảnh → {out} · bỏ {skipped} ảnh (cỡ/kênh khác 256² RGB)")
    print(f"[manifest] {man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
