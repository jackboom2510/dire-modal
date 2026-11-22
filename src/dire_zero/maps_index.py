"""Sửa nhãn generator cho DIRE map đã có — đổi tên, **không** tính lại.

Vì sao tồn tại: bản `::dire` cũ dựng item 3 phần tử `(path, label, split)`, nên
`_as_item` gán `generator="fake"` cho **mọi** ảnh sinh. Map của `adm` và `ours_adm`
rơi vào cùng `maps/<split>/fake/`; `::train --generator adm` báo "chưa có map" dù
map đã nằm trên đĩa và đã trả tiền.

`dire.json` của lần chạy đó vẫn ghi `source` cho từng map — đường dẫn ảnh gốc, mang
tên generator. Nên sửa nhãn là **đổi tên thư mục**: 0 USD, thay vì 2S NFE/ảnh.

Ba nguyên tắc:

* mặc định **chỉ lập kế hoạch** (`apply=False`); tool sửa dữ liệu đã trả tiền để
  tạo nên mặc định phải là "xem trước";
* nguồn sự thật là `source` trong `dire.json`, **không** phải tên file (tên `.npy`
  là sha của ảnh, không mang thông tin);
* đích đã tồn tại ⇒ **không ghi đè**, đưa vào `unresolved`; đoán bừa một generator
  là gắn nhãn sai cho dữ liệu train.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

GENERATOR_BUCKET = "fake"


def relabel_log_path(runs_root: str | Path) -> Path:
    """Tạo đường dẫn mới, duy nhất cho audit log dưới `<runs>/relabel/`."""
    log_dir = Path(runs_root) / "relabel"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return log_dir / f"relabel-{stamp}-{uuid4().hex[:8]}.json"


def write_relabel_log(path: str | Path, report: dict) -> None:
    """Ghi toàn bộ kế hoạch/kết quả (gồm danh sách conflict) thành JSON riêng."""
    Path(path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _generator_from_source(source: str | None) -> str | None:
    """`…/data/<split>/<generator>/<name>.png` → `<generator>`; không đoán được ⇒ None."""
    if not source:
        return None
    parts = Path(str(source).replace("\\", "/")).parts
    if len(parts) < 2:
        return None
    gen = parts[-2]
    if not gen or gen in (".", "..", GENERATOR_BUCKET):
        return None
    return gen


def plan(maps_root: str | Path) -> dict:
    """Kế hoạch: danh sách `(src_abs, dst_abs, generator)` + chỗ không sửa được.

    Không đụng vào đĩa. `plan["moves"]` chỉ chứa item **đủ tin** để đổi tên.
    """
    root = Path(maps_root)
    index_path = root / "dire.json"
    if not index_path.exists():
        raise FileNotFoundError(f"{index_path} không có — không biết map nào từ ảnh nào")
    index = _read(index_path)

    moves: list[dict] = []
    unknown: list[dict] = []
    already_ok = 0
    for item in index.get("items") or []:
        if not item.get("path"):
            continue
        rel = Path(str(item["path"]))
        if item.get("generator") != GENERATOR_BUCKET:
            already_ok += 1
            continue

        gen = _generator_from_source(item.get("source"))
        base = {
            "rel_src": rel.as_posix(),
            "split": item.get("split") or (rel.parts[0] if len(rel.parts) >= 3 else ""),
            "source": item.get("source"),
        }
        if gen in (None, "real", GENERATOR_BUCKET):
            unknown.append({**base, "problem": "không suy được generator từ `source` — để nguyên"})
            continue
        dst = rel.parent.parent / gen / rel.name if len(rel.parts) >= 3 else Path(gen) / rel.name
        entry = {
            **base,
            "src": str(root / rel),
            "dst": str(root / dst),
            "rel_dst": dst.as_posix(),
            "generator": gen,
        }
        if not (root / rel).exists():
            unknown.append({**entry, "problem": "map không có trên đĩa"})
        elif (root / dst).exists():
            unknown.append({**entry, "problem": "đích đã tồn tại — không ghi đè"})
        else:
            moves.append(entry)

    by_gen: dict[str, int] = {}
    for m in moves:
        by_gen[m["generator"]] = by_gen.get(m["generator"], 0) + 1
    return {
        "maps_root": str(root),
        "moves": moves,
        "unresolved": unknown,
        "already_ok": already_ok,
        "n_moves": len(moves),
        "by_generator": dict(sorted(by_gen.items())),
    }


def apply(plan_dict: dict) -> dict:
    """Thực hiện `plan()` rồi cập nhật `dire.json` (giữ nguyên `runs`)."""
    root = Path(plan_dict["maps_root"])
    done: list[dict] = []
    for m in plan_dict["moves"]:
        src, dst = Path(m["src"]), Path(m["dst"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        done.append(m)

    index_path = root / "dire.json"
    index = _read(index_path)
    moved = {m["rel_src"]: m for m in done}
    for item in index.get("items") or []:
        m = moved.get(item.get("path"))
        if m:
            item["path"] = m["rel_dst"]
            item["abs"] = m["dst"]
            item["generator"] = m["generator"]
            item["relabelled_from"] = m["rel_src"]
    index["generators"] = sorted({it.get("generator") for it in index.get("items") or [] if it.get("generator")})
    index["relabel"] = {
        "n_moves": len(done),
        "by_generator": plan_dict["by_generator"],
        "note": "đổi tên từ maps/<split>/fake/ theo `source` của từng item; không tính lại map",
    }
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2, default=str) + "\n",
                          encoding="utf-8")

    # dọn thư mục `fake/` nếu đã rỗng — để report không còn thấy một generator ma
    for d in root.glob("*" + os.sep + GENERATOR_BUCKET):
        try:
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        except OSError:
            pass
    return {"n_moves": len(done), "moved": done}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--maps", required=True, help="thư mục maps (có dire.json), ví dụ /zero/maps")
    ap.add_argument("--apply", action="store_true", help="đổi tên thật (mặc định chỉ in kế hoạch)")
    args = ap.parse_args(argv)

    p = plan(args.maps)
    print(json.dumps({k: v for k, v in p.items() if k != "moves"}, ensure_ascii=False, indent=2))
    for m in p["moves"]:
        print(f"  {m['rel_src']}  ->  {m['rel_dst']}", file=sys.stderr)
    if not p["moves"]:
        print("không có gì để đổi tên", file=sys.stderr)
        return 0 if not p["unresolved"] else 2
    if not args.apply:
        print(f"[dry-run] {p['n_moves']} map sẽ được đổi tên; thêm --apply để làm thật", file=sys.stderr)
        return 0
    out = apply(p)
    print(json.dumps({"applied": out["n_moves"]}, ensure_ascii=False))
    return 0 if not p["unresolved"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
