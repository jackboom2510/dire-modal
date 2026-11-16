#!/usr/bin/env python3
"""Xem bộ ba ảnh gốc | ảnh tái tạo | DIRE map — và KIỂM CHỨNG chúng khớp nhau.

Vì sao có file này
------------------
Paper nói mỗi ảnh trong DiffusionForensics là **triplet**: `x` (ảnh gốc),
`x̂` (ảnh tái tạo qua ADM) và `DIRE = |x − x̂|`. Đó là một lời khẳng định *kiểm
được*, và repo này chỉ dùng `dire/` — tức tin vào `DIRE` mà không bao giờ nhìn
hai cái còn lại.

File này làm hai việc:

1. **Xem**: ghi ra ảnh ghép `orig | recon | dire (tác giả) | |x−x̂| (tự tính)`
   để đưa vào báo cáo/slide.
2. **Đối chiếu**: tính `|x − x̂|` từ hai nhánh đầu rồi so với DIRE map tác giả
   phát hành. Nếu khớp ⇒ dữ liệu là triplet thật; nếu lệch ⇒ phải biết TRƯỚC
   khi trích dẫn bất cứ con số nào từ `dire/`.

Không cần GPU, không cần weight ADM 2.5 GB — chỉ cần 3 tar của tier `gallery`.

Chạy
----
    # 1. lấy bộ ba (674 MB cho real/pndm/ldm)
    modal run -m dire_modal.download::main --tiers gallery --mode fetch

    # 2. xem + kiểm chứng
    python3 tools/triplet_view.py --raw data/raw --gen pndm --n 4
    python3 tools/triplet_view.py --raw data/raw --gen real --check   # chỉ số, không ghi ảnh

Đầu ra: `docs/media/triplets/<gen>_<i>_panel.jpg` (bị .gitignore chừa chỗ).

Giới hạn đã biết
----------------
* DIRE map tác giả lưu ở uint8 ⇒ `|x − x̂|` của mình cũng phải làm tròn/clip
  như họ. Vì vậy so sánh dùng **tương quan** và **sai khác trung bình**, không
  đòi khớp từng pixel.
* Chỉ 8 generator có đủ bộ ba; `adm` (headline của paper) **không có** `recons/`
  ⇒ không xem được bằng đường này (phải tự tính, cần weight ADM).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dire_modal.manifest import GALLERY_DIRS, GALLERY_TRIPLETS  # noqa: E402

IMAGE_EXT = (".png", ".jpg", ".jpeg")
#: Hệ số khuếch đại khi hiển thị DIRE map. Paper hiển thị residual rất mờ nếu
#: để nguyên; ×4 làm cấu trúc hiện ra mà vẫn không đổi thứ tự sáng/tối.
AMP = 4


# ---------------------------------------------------------------------------
# Đọc tar — thuần stdlib, test được không cần Pillow
# ---------------------------------------------------------------------------
def tar_members(path: Path) -> dict[str, tuple[str, bytes]]:
    """``{stem: (tên member, bytes)}`` cho ảnh trong tar.

    Khớp theo **stem** (tên không phần mở rộng) vì ba nhánh có thể lưu định dạng
    khác nhau: `images/` là JPEG, `dire/` là PNG.
    """
    out: dict[str, tuple[str, bytes]] = {}
    with tarfile.open(path, "r:*") as tf:      # "r:*": tar KHÔNG nén dù tên .gz
        for m in tf:
            if not m.isfile():
                continue
            low = m.name.lower()
            if not low.endswith(IMAGE_EXT):
                continue
            fh = tf.extractfile(m)
            if fh is None:
                continue
            stem = Path(m.name).stem
            out[stem] = (m.name, fh.read())
    return out


def common_stems(a: set[str], b: set[str], c: set[str]) -> list[str]:
    """Stem có mặt ở CẢ BA nhánh, sắp xếp để kết quả tất định."""
    return sorted(a & b & c)


def paths_for(raw: Path, gen: str) -> dict[str, Path]:
    """Ba đường dẫn tar của ``gen`` — dùng chung bảng với tier `gallery`."""
    trip = GALLERY_TRIPLETS.get(gen)
    if trip is None:
        raise SystemExit(
            f"[!] {gen!r} không có bộ ba. Có: {', '.join(sorted(GALLERY_TRIPLETS))}\n"
            "    (`adm` không có recons/ ⇒ phải tự tính DIRE, cần weight ADM 2.5 GB)"
        )
    return {b: raw / GALLERY_DIRS[b] / name for b, name in trip.items()}


# ---------------------------------------------------------------------------
# Số học — thuần Python, test được
# ---------------------------------------------------------------------------
def diff_pixels(orig, recon) -> list[int]:
    """``|x − x̂|`` theo từng pixel (0..255), làm phẳng thành list.

    Nhận hai ảnh PIL cùng kích thước; trả list int để phần thống kê không phụ
    thuộc numpy (repo không có numpy trong phụ thuộc bắt buộc).
    """
    a = orig.convert("RGB")
    b = recon.convert("RGB")
    if a.size != b.size:
        b = b.resize(a.size)
    pa, pb = a.tobytes(), b.tobytes()
    return [abs(x - y) for x, y in zip(pa, pb)]


def summarize(values: list[int]) -> dict:
    """Thống kê gọn của một residual."""
    if not values:
        return {"n": 0, "mean": 0.0, "p95": 0.0, "max": 0}
    s = sorted(values)
    return {
        "n": len(values),
        "mean": round(sum(s) / len(s), 3),
        "p95": float(s[min(len(s) - 1, int(len(s) * 0.95))]),
        "max": s[-1],
    }


def pearson(a: list[float], b: list[float]) -> float:
    """Tương quan Pearson; trả ``0.0`` khi một bên không đổi (tránh chia 0)."""
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    a, b = a[:n], b[:n]
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va == 0 or vb == 0:
        return 0.0
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    return cov / (va ** 0.5 * vb ** 0.5)


def check_published(pairs: list[tuple[list[int], list[int]]]) -> dict:
    """So residual TỰ TÍNH với DIRE map TÁC GIẢ trên nhiều mẫu.

    ``pairs``: ``[(diff_tự_tính, diff_tác_giả), ...]``. Trả:
      * ``mean_abs_error`` — trung bình |khác biệt trung bình| giữa hai map;
      * ``corr`` — Pearson của giá trị trung bình từng ảnh (đây mới là tín hiệu
        detector dùng, xem `spec.md` AC-2.2);
      * ``sign_ok`` — dấu của ``mean(real) − mean(fake)`` có giống nhau không.
    """
    ours = [statistics.fmean(d) for d, _ in pairs]
    theirs = [statistics.fmean(t) for _, t in pairs]
    err = [abs(o - t) for o, t in zip(ours, theirs)]
    # Tương quan pixel giữa hai map, lấy trung bình trên các mẫu.
    pix = [pearson([float(v) for v in d], [float(v) for v in t]) for d, t in pairs]
    return {
        "samples": len(pairs),
        "mean_ours": round(statistics.fmean(ours), 3) if ours else 0.0,
        "mean_theirs": round(statistics.fmean(theirs), 3) if theirs else 0.0,
        "mean_abs_error": round(statistics.fmean(err), 3) if err else 0.0,
        "corr_means": round(pearson(ours, theirs), 4) if len(ours) > 1 else 0.0,
        "corr_pixels": round(statistics.fmean(pix), 4) if pix else 0.0,
    }


# ---------------------------------------------------------------------------
# Ghi ảnh ghép (cần Pillow, chỉ khi không --check)
# ---------------------------------------------------------------------------
def _require_pil():
    try:
        from PIL import Image, ImageDraw  # noqa: PLC0415
    except ImportError:  # pragma: no cover - phụ thuộc môi trường
        raise SystemExit(
            "[!] cần Pillow để ghi ảnh ghép:  pip install pillow\n"
            "    (--check chỉ in số, không cần Pillow… trừ khi đọc PNG/JPEG)"
        )
    return Image, ImageDraw


def panel(images: list[tuple[str, object]], out: Path, size: int = 256) -> None:
    """Ghép ``[(nhãn, ảnh PIL), ...]`` thành một hàng ngang có nhãn."""
    Image, ImageDraw = _require_pil()
    pad, bar = 6, 22
    w = size * len(images) + pad * (len(images) + 1)
    canvas = Image.new("RGB", (w, size + bar + pad * 2), (16, 18, 24))
    draw = ImageDraw.Draw(canvas)
    for i, (label, im) in enumerate(images):
        x = pad + i * (size + pad)
        canvas.paste(im.convert("RGB").resize((size, size)), (x, bar + pad))
        draw.text((x, 6), label, fill=(230, 230, 230))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=92)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", default="data/raw", help="gốc dữ liệu (mặc định data/raw)")
    ap.add_argument("--gen", default="pndm", help=f"một trong {sorted(GALLERY_TRIPLETS)}")
    ap.add_argument("--n", type=int, default=4, help="số mẫu mỗi lớp (mặc định 4)")
    ap.add_argument("--out", default="docs/media/triplets", help="thư mục ghi panel")
    ap.add_argument("--check", action="store_true", help="chỉ in số, không ghi ảnh")
    ap.add_argument("--compare-with", default="",
                    help="generator thứ hai để so mean|x−x̂| (ví dụ `real` so với `pndm`); "
                         "dùng để kiểm tuyên bố gốc của paper mà KHÔNG cần weight ADM")
    args = ap.parse_args(argv)

    raw = Path(args.raw).expanduser()
    paths = paths_for(raw, args.gen)
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        print("[!] thiếu tar — chạy trước:")
        print("    modal run -m dire_modal.download::main --tiers gallery --mode fetch")
        for p in missing:
            print(f"    - {p}")
        return 2

    ours, theirs = tar_members(paths["images"]), tar_members(paths["recons"])
    published = tar_members(paths["dire"])
    stems = common_stems(set(ours), set(theirs), set(published))
    print(f"[{args.gen}] images={len(ours)} recons={len(theirs)} dire={len(published)}"
          f" → khớp cả ba: {len(stems)}")
    if not stems:
        print("[!] không có ảnh nào khớp tên giữa ba nhánh.\n"
              "    Kiểm lại bảng GALLERY_TRIPLETS — tên file có thể lệch "
              "(midjouney/midjourney, stylegan/stylegan_official).")
        return 1

    from io import BytesIO  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    picked = stems[: max(1, args.n)]
    pairs: list[tuple[list[int], list[int]]] = []
    print(f"\n{'ảnh':<28} {'mean|x-x̂|':>10} {'mean(DIRE)':>11} {'pearson':>8}")
    print("-" * 60)
    for i, stem in enumerate(picked):
        orig = Image.open(BytesIO(ours[stem][1])).convert("RGB")
        recon = Image.open(BytesIO(theirs[stem][1])).convert("RGB")
        dire = Image.open(BytesIO(published[stem][1])).convert("RGB")
        d = diff_pixels(orig, recon)
        t = list(dire.tobytes())
        pairs.append((d, t))
        print(f"{stem[:28]:<28} {statistics.fmean(d):>10.2f} "
              f"{statistics.fmean(t):>11.2f} {pearson([float(v) for v in d], [float(v) for v in t]):>8.3f}")

        if not args.check:
            from PIL import ImageChops  # noqa: PLC0415

            # Dựng lại residual bằng chính phép trừ, không dựng lại từ list `d`:
            # `d` là 3 kênh đã làm phẳng, ghép lại theo pixel rất dễ lệch một
            # bậc và ảnh sẽ trông "gần đúng" — sai kiểu không ai phát hiện.
            ours_img = ImageChops.difference(orig, recon).convert("RGB")
            amp_theirs = Image.eval(dire, lambda v: min(255, v * AMP))
            amp_ours = Image.eval(ours_img, lambda v: min(255, v * AMP))
            panel(
                [("ảnh gốc x", orig), ("tái tạo x̂", recon),
                 (f"DIRE tác giả (×{AMP})", amp_theirs),
                 (f"|x−x̂| tự tính (×{AMP})", amp_ours)],
                Path(args.out) / f"{args.gen}_{i:02d}_{stem[:10]}_panel.jpg",
            )

    rep = check_published(pairs)
    print("\nĐỐI CHIẾU residual TỰ TÍNH ↔ DIRE map TÁC GIẢ")
    for k, v in rep.items():
        print(f"  {k:<16} {v}")
    # Tiêu chí: (a) thang độ của residual khớp — `mean_abs_error` nhỏ; (b) cấu
    # trúc khớp — `corr_pixels`. KHÔNG dùng `corr_means` để kết luận: nó chỉ có
    # nghĩa khi các ảnh có residual trung bình KHÁC nhau; một lô ảnh cùng
    # generator thường gần bằng nhau ⇒ phương sai ~0 ⇒ tương quan vô nghĩa
    # (đã thấy = 0.0 trên dữ liệu tổng hợp đều nhau). Vẫn in ra để tham khảo.
    ok = rep["mean_abs_error"] <= 8 and rep["corr_pixels"] >= 0.80
    print(f"\n  → {'KHỚP' if ok else 'LỆCH'} (ngưỡng: mean_abs_error ≤ 8/255 "
          f"và corr_pixels ≥ 0.80)")
    if not ok:
        print("    Nếu LỆCH: đừng trích dẫn số từ `dire/` cho tới khi hiểu vì sao.\n"
              "    Hai nguyên nhân thường gặp: (1) ảnh gốc bị JPEG hoá lại sau khi\n"
              "    tính DIRE, (2) `dire/` là map đã qua chuẩn hoá/hiển thị chứ không\n"
              "    phải hiệu thô.")
    if not args.check:
        print(f"\n[ok] panel → {Path(args.out)}")

    if args.compare_with:
        other = _mean_residual(raw, args.compare_with, max(1, args.n))
        if other is None:
            return 1
        mine = statistics.fmean([statistics.fmean(d) for d, _ in pairs])
        print("\nSO HAI LỚP — tuyên bố của paper: ảnh THẬT có residual LỚN hơn ảnh sinh")
        print(f"  {args.gen:<12} {mine:6.2f}/255")
        print(f"  {args.compare_with:<12} {other:6.2f}/255")
        hi, lo = (args.gen, args.compare_with) if mine > other else (args.compare_with, args.gen)
        print(f"  → lớp `{hi}` có residual lớn hơn `{lo}`")
        if "real" in (args.gen, args.compare_with):
            real_is_higher = mine > other if args.gen == "real" else other > mine
            print(f"  → {'PHÙ HỢP' if real_is_higher else 'NGƯỢC'} với paper"
                  f" (real > fake mới đúng chiều)")
            return 0 if (ok and real_is_higher) else 1
    return 0 if ok else 1


def _mean_residual(raw: Path, gen: str, n: int) -> float | None:
    """``mean|x−x̂|`` trung bình của ``gen`` — dùng để so hai lớp."""
    from io import BytesIO  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    paths = paths_for(raw, gen)
    if any(not p.is_file() for p in paths.values()):
        print(f"[!] thiếu tar cho {gen} — bỏ qua so sánh")
        return None
    a, b = tar_members(paths["images"]), tar_members(paths["recons"])
    stems = common_stems(set(a), set(b), set(b))
    if not stems:
        print(f"[!] {gen}: không khớp tên giữa images/ và recons/")
        return None
    vals = []
    for stem in stems[:n]:
        orig = Image.open(BytesIO(a[stem][1])).convert("RGB")
        recon = Image.open(BytesIO(b[stem][1])).convert("RGB")
        vals.append(statistics.fmean(diff_pixels(orig, recon)))
    return round(statistics.fmean(vals), 2) if vals else None


if __name__ == "__main__":
    raise SystemExit(main())
