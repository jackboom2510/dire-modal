#!/usr/bin/env python3
"""Dựng `platforms/hf-spaces/gallery/` từ DIRE map đã tính.

Vì sao cần một tool riêng
-------------------------
Space chạy trên CPU free, nên nó **không** tính DIRE lúc chạy: map phải có sẵn.
Nhưng "có sẵn" không có nghĩa là "có ảnh" — `::dire` lưu `.npy` float16, thang
`[0, 2]`, shape **`(3, 256, 256)`** (ba kênh của `|x − x̂|`, không phải ảnh xám),
thứ mà trình duyệt không đọc được. Tool này làm đúng ba việc:

1. chọn mẫu **tất định** (sort rồi lấy đều, không `random`) để chạy lại cho ra
   cùng một gallery;
2. vẽ `.npy` thành PNG chỉ để **xem**, và ghi rõ trong `.json` rằng PNG không
   phải dữ liệu — bản gốc là `.npy`, có sha256;
3. ghi `.json` nguồn gốc đủ để `platforms/hf-spaces/demolib/provenance.py` chấp nhận.

Đường dẫn truyền **tường minh**, không đoán
-------------------------------------------
Không có mặc định kiểu `F:/DiffusionForensics/...`. Cây dữ liệu của mỗi người một
khác, và một tool đoán sai rồi vẫn chạy là cách nhanh nhất để sinh ra gallery
gắn nhãn sai.

Ví dụ
-----
::

    python tools/make_gallery.py \\
        --maps  /vol/maps/test \\
        --orig  /vol/data/test \\
        --out   platforms/hf-spaces/gallery \\
        --n 8 --steps 20 --label R1+R2 \\
        --map-source "maps/test/*.npy · ::dire --steps 20" \\
        --dry-run

`--scores` là tuỳ chọn: một JSON `{"<tên mẫu>": {"p_fake": 0.97, …}}`. Không có
nó thì gallery vẫn hợp lệ — chỉ là app sẽ hiện "chưa có điểm", đúng như phải thế.

Thang hiển thị
--------------
`::dire` lưu map trong `[0, 2]` (xem `docs/from-zero-clean-room.md`). PNG dùng
``v/2`` rồi ``**gamma`` với `gamma = 0.5` để nhìn thấy vùng sai số nhỏ; **gamma chỉ
là hiển thị**, mọi phép đo dùng `.npy`. Đổi `--gamma 1.0` nếu cần trung thực tuyến tính.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "platforms" / "hf-spaces"))

try:
    from demolib import provenance  # noqa: E402
except ModuleNotFoundError as exc:  # chỉ xảy ra khi tệp đi một mình, không có thư viện
    if exc.name != "demolib":
        raise
    raise SystemExit(
        f"không import được `demolib` — tìm ở {REPO / 'platforms' / 'hf-spaces' / 'demolib'}\n"
        "  Thư viện này ghi/đọc định dạng `.json` của gallery; thiếu nó thì không có gì\n"
        "  để tự kiểm. Chạy từ repo thì đã có sẵn; trong container thì `demolib/` phải\n"
        "  được mount cạnh `tools/make_gallery.py` (`::gallery` trong\n"
        "  `src/dire_zero/modal_app.py` làm việc đó)."
    ) from None

MAP_MAX = 2.0          # thang lưu của `::dire`, không phải thang hiển thị
SIZE = 256


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pick(items: list[Path], n: int) -> list[Path]:
    """Chọn `n` phần tử rải đều trên danh sách đã sort — tất định, không random."""
    items = sorted(items)
    if n >= len(items):
        return items
    step = len(items) / n
    return [items[min(len(items) - 1, int(i * step))] for i in range(n)]


#: Những bố cục `map_to_png` chấp nhận, sau khi chuẩn hoá.
#: `::dire` ghi ``dm.squeeze(0)``, tức **(3, 256, 256)** cho ảnh RGB — không phải
#: ảnh xám. Bản đầu của tool chỉ nhận `(256, 256)` và vì thế chết trên dữ liệu
#: thật của chính repo:
#:
#:     ValueError: 004006349103115c.npy: shape (3, 256, 256), cần (256, 256)
ACCEPTED = "(256, 256) hoặc (3, 256, 256) [::dire ghi (3, 256, 256) cho ảnh RGB]"


def _canonical_map(arr):
    """Đưa map về ``(H, W)`` hoặc ``(3, H, W)``; **không** trộn kênh.

    Không lấy trung bình ba kênh: PNG là để *xem*, và ba kênh của ``|x − x̂|``
    là ba phép đo riêng. Trộn chúng thành ảnh xám là bịa ra một đại lượng không
    có trong `.npy`. Hỗ trợ thêm ``(H, W, C)`` vì một số tool khác lưu kênh cuối.
    """
    import numpy as np

    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 3 and arr.shape[-1] in (1, 3) and arr.shape[:2] == (SIZE, SIZE):
        return arr[..., 0] if arr.shape[-1] == 1 else np.moveaxis(arr, -1, 0)
    return arr


def map_to_png(npy: Path, png: Path, gamma: float) -> str:
    """Vẽ map thành PNG **chỉ để xem**; trả ``"L"`` hoặc ``"RGB"`` (mode đã ghi)."""
    import numpy as np
    from PIL import Image

    raw = np.load(npy)
    arr = _canonical_map(raw)
    if arr.shape == (SIZE, SIZE):
        mode = "L"
    elif arr.shape == (3, SIZE, SIZE):
        mode = "RGB"
    else:
        raise ValueError(
            f"{npy.name}: shape {raw.shape} — cần {ACCEPTED}.\n"
            "  `.npy` do `::dire` sinh ra có 3 kênh; nếu đây là map tự tạo, kiểm"
            " lại trục kênh (channel-first) và kích thước 256×256."
        )
    v = np.clip(arr.astype("float32") / MAP_MAX, 0.0, 1.0)
    if gamma != 1.0:
        v = v ** gamma
    data = (v * 255.0 + 0.5).astype("uint8")
    if mode == "RGB":
        # PIL nhận ảnh màu ở dạng (H, W, 3); map lưu channel-first như torch.
        data = np.moveaxis(data, 0, -1)
    Image.fromarray(data, mode=mode).save(png, optimize=True)
    return mode


def copy_orig(src: Path, dst: Path) -> None:
    from PIL import Image

    with Image.open(src) as im:
        if im.size != (SIZE, SIZE):
            im = im.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
        else:
            im = im.convert("RGB")
        im.save(dst, optimize=True)


def _hint_dir(kind: str, path: Path) -> str:
    """Thông báo cho thư mục không tồn tại — phải nói **đường dẫn được tra từ đâu**.

    `--maps không phải thư mục: maps\test` không cho người dùng biết cái gì sai:
    đường dẫn tương đối được tra từ thư mục hiện tại (khác với nơi họ tưởng), hoặc
    map chưa được tải từ Volume xuống. Cả hai đều là lỗi phổ biến, và cả hai đều
    sửa được trong một dòng — nếu thông báo chịu nói ra.
    """
    absolute = path.resolve()
    if kind == "maps":
        why = (
            "DIRE map nằm trên Volume Modal, không nằm trong repo. Tải về đúng thư mục:\n"
            "    modal volume get dire-zero maps ./vol/maps\n"
            "  rồi trỏ `--maps ./vol/maps/test` (mỗi lớp là một thư mục con).\n"
            "  Hoặc dựng thẳng trên Modal, khỏi tải 800 MB:\n"
            "    modal run -m dire_zero.modal_app::gallery --n 8 --steps 20\n"
            "    modal volume get dire-zero gallery ./platforms/hf-spaces/gallery"
        )
    else:
        why = (
            "Ảnh gốc nằm trên Volume Modal. Tải về:\n"
            "    modal volume get dire-zero data/test ./vol/data/test\n"
            "  rồi trỏ `--orig ./vol/data/test`."
        )
    eaten = ""
    shown = str(path).replace("\\", "/")
    # Dấu vết của một placeholder bị **shell** ăn mất: `--maps <volume>/maps/test`
    # không phải là đường dẫn có dấu ngoặc — bash biến `<volume>` thành chuyển
    # hướng stdin và để lại `/maps/test`. Trên Git Bash/MSYS đường dẫn đó lại được
    # dịch thành `C:/Program Files/Git/maps/test`, nên đây là thứ nhìn thấy được.
    if "<" in shown or ">" in shown:
        eaten = ("  Có vẻ bạn gõ nguyên `<...>` của tài liệu. Trong bash, `<` là chuyển\n"
                 "  hướng stdin: lệnh sẽ chết với \"volume: No such file or directory\" và\n"
                 "  python không bao giờ chạy. Đặt đường dẫn vào biến: `VOL=/duong/dan`.\n")
    elif shown.startswith("/maps") or shown.startswith("/data") or "Program Files/Git/" in shown:
        eaten = ("  Đường dẫn này là dấu vết của một placeholder bị shell ăn: `--maps\n"
                 "  <volume>/maps/test` để lại `/maps/test` (Git Bash còn dịch thành\n"
                 "  `C:/Program Files/Git/maps/test`). Đặt đường dẫn vào biến, đừng\n"
                 "  dùng `<...>`.\n"
                 "  Cách không cần đường dẫn nào: `modal run -m dire_zero.modal_app::gallery`.\n")
    return (
        f"--{kind} không phải thư mục: {path}\n"
        f"  (đang tra từ {Path.cwd()}; đường dẫn tuyệt đối sẽ là {absolute})\n"
        + eaten +
        f"  {why}"
    )


def build(args) -> list[dict]:
    maps_root, orig_root, out = Path(args.maps), Path(args.orig), Path(args.out)
    if not maps_root.is_dir():
        raise SystemExit(_hint_dir("maps", maps_root))
    if not orig_root.is_dir():
        raise SystemExit(_hint_dir("orig", orig_root))
    # `--dry-run` không ghi gì nên không cần `--force` để xem thử trên gallery có sẵn.
    if not args.dry_run and out.exists() and any(out.glob("*.json")) and not args.force:
        raise SystemExit(f"{out} đã có gallery; dùng --force để ghi đè")

    scores = {}
    if args.scores:
        sp = Path(args.scores)
        if not sp.is_file():
            raise SystemExit(
                f"--scores không phải tệp: {sp}\n"
                f"  (đang tra từ {Path.cwd()})\n"
                "  `scores.json` do `::evaluate` sinh ra trên Volume:\n"
                "    modal volume get dire-zero runs/eval/gen-adm ./tmp/eval\n"
                "  rồi trỏ `--scores ./tmp/eval/scores.json`. Bỏ `--scores` thì gallery "
                "vẫn hợp lệ — chỉ là mọi mẫu hiện \"chưa có điểm\"."
            )
        raw = json.loads(sp.read_text(encoding="utf-8"))
        scores = raw.get("samples", raw) if isinstance(raw, dict) else {}
        if not isinstance(scores, dict):
            raise SystemExit("--scores phải là object {tên mẫu: {...}}")

    classes = sorted(p.name for p in maps_root.iterdir() if p.is_dir())
    if not classes:
        raise SystemExit(f"không thấy lớp nào trong {maps_root} (cần các thư mục con)")

    written, skipped = [], []
    for cls in classes:
        npys = pick(list((maps_root / cls).glob("*.npy")), args.n)
        if not npys:
            skipped.append(f"{cls}: không có .npy")
            continue
        kind = "real" if cls in ("real", "0_real", "lsun_real") else "fake"
        for npy in npys:
            stem = npy.stem.replace("_dire", "")
            cands = [orig_root / cls / f"{stem}.png", orig_root / cls / f"{stem}.jpg",
                     orig_root / cls / f"{stem}.jpeg"]
            src = next((c for c in cands if c.is_file()), None)
            if src is None:
                skipped.append(f"{cls}/{npy.stem}: không có ảnh gốc ({cands[0].name})")
                continue

            name = f"{args.prefix}{cls}_{npy.stem}"
            orig_png = out / f"{name}_orig.png"
            diff_png = out / f"{name}_diff.png"
            diff_mode = "L"
            if not args.dry_run:
                out.mkdir(parents=True, exist_ok=True)
                copy_orig(src, orig_png)
                diff_mode = map_to_png(npy, diff_png, args.gamma)

            meta = {
                "name": name,
                "generator": "real" if kind == "real" else cls,
                "split": args.split,
                "label": args.label,
                "map_source": args.map_source or f"{npy}",
                "sha256": {
                    "orig": sha256_file(src),
                    "map": sha256_file(npy),
                    **({} if args.dry_run else {"png_orig": sha256_file(orig_png),
                                               "png_diff": sha256_file(diff_png)}),
                },
                "S": args.steps,
                "display": {
                    "png_is_display_only": True,
                    "scale": f"[0, {MAP_MAX}] → uint8, gamma {args.gamma}",
                    "channels": 3 if diff_mode == "RGB" else 1,
                    "channel_order": "RGB" if diff_mode == "RGB" else "",
                    "note": ("PNG giữ nguyên từng kênh của |x − x̂|; không trộn thành ảnh xám"
                             if diff_mode == "RGB" else "map một kênh"),
                    "authoritative": ".npy",
                },
                "source_image": str(src),
            }
            if name in scores:
                meta["detector"] = {
                    "gate": args.gate,
                    "threshold": args.threshold,
                    "threshold_split": "val",
                    **scores[name],
                }
            written.append(meta)
            if not args.dry_run:
                (out / f"{name}.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # `--scores` được đưa vào nhưng **không khớp tên mẫu nào** là lỗi im lặng nguy
    # hiểm: gallery dựng ra vẫn đẹp, mọi mẫu hiện "chưa có điểm", và người dựng
    # tưởng detector chưa qua cổng. Tên khoá phải là `f"{prefix}{cls}_{stem}"` —
    # đúng thứ `::evaluate` ghi.
    if scores and written:
        matched = sum(1 for r in written if "detector" in r)
        if matched == 0:
            sample = sorted(scores)[:3]
            raise SystemExit(
                f"--scores có {len(scores)} khoá nhưng KHÔNG khớp mẫu nào "
                f"({len(written)} mẫu).\n"
                f"  khoá cần dạng `<lớp>_<stem>` (ví dụ `adm_000123`), đang có: {sample}\n"
                "  `--prefix` phải rỗng nếu khoá không có tiền tố.\n"
                "  Bỏ `--scores` nếu chỉ muốn gallery không kèm điểm."
            )
        print(f"[gallery] {matched}/{len(written)} mẫu có điểm từ {args.scores}")

    return written + [{"_skipped": skipped}] if skipped else written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="dựng gallery cho HF Space từ DIRE map")
    ap.add_argument("--maps", required=True, help="thư mục chứa <lớp>/*.npy, thang [0, 2]")
    ap.add_argument("--orig", required=True, help="thư mục chứa <lớp>/*.png ảnh gốc 256²")
    ap.add_argument("--out", default=str(REPO / "platforms" / "hf-spaces" / "gallery"))
    ap.add_argument("--n", type=int, default=8, help="số mẫu mỗi lớp (rải đều, tất định)")
    ap.add_argument("--steps", type=int, default=20, help="S đã dùng khi tính map")
    ap.add_argument("--split", default="test")
    ap.add_argument("--label", default="R1+R2", choices=["R0", "R1+R2", "R1+R2+R3"])
    ap.add_argument("--map-source", default="", help="câu mô tả nguồn map, hiện thẳng trên UI")
    ap.add_argument("--scores", default="", help="JSON {tên mẫu: {p_fake,…}} — tuỳ chọn")
    ap.add_argument("--gate", default="failed", choices=["passed", "failed", "missing"],
                    help="cổng 2 của docs/plans/demos/demo-web-plan.md §5; mặc định `failed` để KHÔNG hiện điểm")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.5, help="chỉ ảnh hưởng PNG hiển thị")
    ap.add_argument("--prefix", default="", help="tiền tố tên mẫu (tránh trùng giữa các split)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if not 0.0 < args.gamma <= 1.0:
        raise SystemExit("--gamma phải trong (0, 1]")
    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold phải trong [0, 1]")

    rows = build(args)
    skipped = [r["_skipped"] for r in rows if "_skipped" in r]
    made = [r for r in rows if "_skipped" not in r]
    print(f"[gallery] {len(made)} mẫu" + (" (dry-run)" if args.dry_run else f" → {args.out}"))
    for r in made:
        has = "có điểm" if "detector" in r else "chưa có điểm"
        print(f"  {r['name']:<32} {r['generator']:<10} {has}")
    for group in skipped:
        for s in group:
            print(f"  [bỏ] {s}", file=sys.stderr)

    # Tự kiểm bằng chính bộ đọc của app: nếu `.json` mình vừa ghi không qua được
    # `provenance.load_sample` thì gallery sẽ hiện "0 mẫu" mà không ai hiểu vì sao.
    if not args.dry_run and made:
        samples, problems = provenance.discover(args.out)
        if problems:
            for p in problems:
                print(f"  [tự kiểm] {p}", file=sys.stderr)
            return 2
        print(f"[gallery] tự kiểm: {len(samples)} mẫu qua `provenance.load_sample`")
    if not made:
        print("[gallery] không dựng được mẫu nào — kiểm lại --maps/--orig", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
