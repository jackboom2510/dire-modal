#!/usr/bin/env python3
"""Đẩy cây dữ liệu lên Volume theo **lô nhỏ** — dừng ở đâu, chạy lại từ đó.

Vì sao không dùng `modal volume put` cho 200 nghìn file
------------------------------------------------------
`modal volume put <vol> <dir> <remote>` **không** ghi dần từng file. Trong
``modal/volume.py`` (1.6.x), mọi file được đăng ký vào một danh sách trong bộ
nhớ, rồi ở cuối context manager mới gửi **một** request duy nhất::

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not exc_val:                       # có exception ⇒ KHÔNG gửi gì
            ...upload từng file...
            await self._client._stub.VolumePutFiles(request)   # một request cho cả lô

Hệ quả đã xảy ra thật: 217 579/217 585 file (99,997 %) rồi treo ⇒ **không file
nào** vào được FS của Volume. Thanh tiến trình đếm byte đã gửi, không đếm file
đã ghi; lần chạy lại dừng đúng cùng con số. Một file hỏng làm mất cả 22 GB.

Cách làm ở đây
--------------
1. Liệt kê file **đã có** trên Volume (`Volume.listdir`) — chỉ metadata, rẻ.
2. Chỉ đẩy file còn thiếu hoặc **khác kích thước** (bắt được cả file đẩy dở).
3. Mỗi lô là một `batch_upload` riêng ⇒ lô xong là bền; lô lỗi chỉ mất chính nó.
4. `manifest.tsv` + `data_card.json` ở **gốc** cây local được đẩy **sau cùng**,
   để chúng chỉ xuất hiện khi ảnh đã đủ. Đây chính là hai file từng bị bỏ sót
   khi đẩy thủ công từng thư mục con.
5. In bảng đối chiếu cuối (local vs remote) để không phải tin lời ai.

Dùng (Git Bash, đứng ở gốc repo)::

    python tools/put_resume.py --local F:/dire-zero-norm --remote data --dry-run
    python tools/put_resume.py --local F:/dire-zero-norm --remote data
    python tools/put_resume.py --local F:/dire-zero-norm --remote data --verify

Khoanh vùng file cứng đầu (mỗi file một request)::

    python tools/put_resume.py --local F:/dire-zero-norm --remote data --only train/pndm --chunk 1

Xong thì::

    modal run -m dire_zero.modal_app::checkdata      # sha256 từng PNG, 0 đồng GPU
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

#: `modal.types.FileEntryType.FILE` (IntEnum) — so sánh bằng int để không phụ
#: thuộc vào việc Modal đổi tên enum.
FILE_TYPE = 1

#: Hai file ở **gốc** cây local. `manifest.tsv` là thứ `checkdata` đọc để biết
#: ảnh nào có nhãn gì; thiếu nó thì mọi bước sau đều vô nghĩa.
ROOT_FILES = ("manifest.tsv", "data_card.json")


# --------------------------------------------------------------------------- #
# Phần thuần: không gọi mạng, test được bằng thư mục tạm + Volume giả
# --------------------------------------------------------------------------- #

def leaves(root: str | Path, only: str | None = None) -> list[tuple[str, Path]]:
    """``[("train/pndm", <thư mục local>)]`` — bỏ qua thư mục rỗng.

    `only` lọc theo chuỗi con, để khoanh vùng một nhánh khi cần soi từng file.
    """
    root = Path(root)
    out: list[tuple[str, Path]] = []
    for split in sorted(p for p in root.iterdir() if p.is_dir()):
        for gen in sorted(p for p in split.iterdir() if p.is_dir()):
            rel = f"{split.name}/{gen.name}"
            if only and only not in rel:
                continue
            if any(p.is_file() for p in gen.iterdir()):
                out.append((rel, gen))
    return out


def missing_in(files: list[Path], have: dict[str, int]) -> list[Path]:
    """File cần đẩy: chưa có trên Volume **hoặc** có nhưng khác kích thước.

    So kích thước, không chỉ so tên: file đẩy dở (đúng tên, thiếu byte) là kiểu
    hỏng mà `checkdata` sẽ báo "sha256 lệch" ở bước sau — muốn bắt nó ở đây.
    """
    return [f for f in files if have.get(f.name) != f.stat().st_size]


def root_files_to_push(root: str | Path, have: dict[str, int]) -> list[Path]:
    """`manifest.tsv` / `data_card.json` cần đẩy (theo cùng luật tên + size)."""
    root = Path(root)
    return [p for name in ROOT_FILES if (p := root / name).is_file()
            and have.get(name) != p.stat().st_size]


def remote_files(vol, prefix: str) -> dict[str, int]:
    """``{tên file: size}`` trong một thư mục trên Volume. Lỗi ⇒ coi như rỗng."""
    try:
        entries = vol.listdir(prefix)
    except Exception as exc:              # noqa: BLE001 — thư mục chưa có là bình thường
        print(f"    (chưa liệt kê được {prefix}: {type(exc).__name__}: {exc})")
        return {}
    return {
        str(e.path).rstrip("/").rsplit("/", 1)[-1]: int(getattr(e, "size", 0))
        for e in entries
        if int(getattr(e, "type", 0)) == FILE_TYPE
    }


def push(vol, files: list[Path], remote_dir: str, chunk: int, retries: int) -> tuple[int, list[dict]]:
    """Đẩy theo lô; trả `(số file đã đẩy, danh sách lô hỏng)`.

    Mỗi lô là một `batch_upload` riêng — đây là toàn bộ lý do công cụ này tồn tại.
    """
    uploaded, failed = 0, []
    n_batches = -(-len(files) // chunk)
    for i in range(0, len(files), chunk):
        batch = files[i:i + chunk]
        ok = False
        for attempt in range(1, retries + 2):
            try:
                with vol.batch_upload(force=True) as b:
                    for f in batch:
                        b.put_file(f, f"{remote_dir}/{f.name}")
                ok = True
                break
            except Exception as exc:      # noqa: BLE001 — lô lỗi là dữ liệu, không phải bug
                print(f"    ! lô {i // chunk + 1} lần {attempt}: {type(exc).__name__}: {exc}")
                if attempt <= retries:
                    time.sleep(2 * attempt)
        if ok:
            uploaded += len(batch)
        else:
            failed.append({"dir": remote_dir, "files": [f.name for f in batch],
                           "error": "hết số lần thử"})
        print(f"    → lô {i // chunk + 1}/{n_batches}: {min(i + chunk, len(files))}/{len(files)}")
    return uploaded, failed


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", required=True, help="thư mục local, vd F:/dire-zero-norm")
    ap.add_argument("--volume", default="dire-zero")
    ap.add_argument("--remote", default="data", help="gốc trên Volume, mặc định data")
    ap.add_argument("--chunk", type=int, default=500, help="số file mỗi lô (mặc định 500)")
    ap.add_argument("--only", default=None, help="chỉ một nhánh, vd train/pndm")
    ap.add_argument("--dry-run", action="store_true", help="chỉ in kế hoạch, không ghi")
    ap.add_argument("--verify", action="store_true", help="chỉ đối chiếu, không đẩy")
    ap.add_argument("--strict", action="store_true", help="lô lỗi thì dừng hẳn")
    ap.add_argument("--retries", type=int, default=2, help="số lần thử lại mỗi lô")
    args = ap.parse_args(argv)

    root = Path(args.local)
    if not root.is_dir():
        print(f"[CHẶN] không thấy {root}. Windows: dùng F:/… chứ không phải /f/… khi đã "
              "đặt MSYS_NO_PATHCONV=1 (biến đó tắt đúng phép đổi mà /f/… cần).",
              file=sys.stderr)
        return 2

    import modal  # noqa: PLC0415 — chỉ cần khi thật sự đẩy

    vol = modal.Volume.from_name(args.volume, create_if_missing=False)
    groups = leaves(root, args.only)
    total_local = sum(1 for _rel, d in groups for p in d.iterdir() if p.is_file())
    mode = "DRY-RUN" if args.dry_run else ("VERIFY" if args.verify else "ĐẨY")

    print(f"Volume {args.volume!r} · gốc {args.remote!r} · {len(groups)} nhánh · "
          f"{total_local} file local · lô {args.chunk} · {mode}\n")

    uploaded, failed, missing_total = 0, [], 0
    t0 = time.time()

    for rel, d in groups:
        prefix = f"{args.remote}/{rel}"
        have = remote_files(vol, prefix)
        files = sorted((p for p in d.iterdir() if p.is_file()), key=lambda p: p.name)
        gaps = missing_in(files, have)
        missing_total += len(gaps)
        print(f"  {rel:24s} local {len(files):6d} · remote {len(have):6d} · "
              f"{'OK' if not gaps else f'thiếu {len(gaps)}'}")
        if args.dry_run or args.verify or not gaps:
            continue
        up, bad = push(vol, gaps, prefix, args.chunk, args.retries)
        uploaded += up
        failed += bad
        if bad and args.strict:
            print(json.dumps({"failed": failed}, ensure_ascii=False, indent=2))
            return 3

    # Hai file ở gốc: đẩy sau cùng. Đẩy thủ công từng thư mục con là cách chúng
    # bị bỏ sót, và `checkdata` sau đó báo "manifest rỗng" — thông báo đúng nhưng
    # dẫn người dùng đi sai hướng.
    root_gaps = root_files_to_push(root, remote_files(vol, args.remote))
    print(f"  {args.remote + '/':24s} metadata thiếu: "
          f"{[p.name for p in root_gaps] if root_gaps else 'không'}")
    if root_gaps and not (args.dry_run or args.verify):
        up, bad = push(vol, root_gaps, args.remote, chunk=max(1, len(root_gaps)), retries=args.retries)
        uploaded += up
        failed += bad

    print(f"\n[đối chiếu] {len(groups)} nhánh · {missing_total} file thiếu lúc đầu · "
          f"{uploaded} đã đẩy · {len(failed)} lô lỗi · {time.time() - t0:.0f}s")
    if failed:
        print(json.dumps({"failed": failed}, ensure_ascii=False, indent=2))
        print("\nChạy lại đúng lệnh này để thử tiếp; soi từng file: thêm --chunk 1.")
        return 3
    if args.dry_run or args.verify:
        return 0 if not missing_total and not root_gaps else 1
    print("Xong. Bước tiếp:  modal run -m dire_zero.modal_app::checkdata")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
