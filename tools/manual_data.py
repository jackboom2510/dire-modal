#!/usr/bin/env python3
"""Tải dữ liệu **bằng tay** rồi kiểm + đẩy lên Modal — không dùng BaiduPCS-Go.

Vì sao có công cụ này
--------------------
`local_fetch.py` giả định có BaiduPCS-Go để lấy size và resume từng byte. Khi
tải bằng tay (GUI BaiduDrive, trình duyệt, hoặc copy từ máy khác) thì:

* không có `ls -l` của remote để biết "đúng chưa" ⇒ phải kiểm **cấu trúc**
  (gzip/tar hợp lệ, đọc được header, đếm được member);
* tên file hay bị thêm `(1)`, `.crdownload`, `.part` ⇒ phải phát hiện;
* người dùng cần một **danh sách tải** chính xác tới từng đường dẫn, không
  phải "tải tier e2" rồi tự đoán file nào nằm ở đâu.

Ba lệnh::

    python tools/manual_data.py list --need a,gallery,e2min,e2,ckpts
    python tools/manual_data.py verify --dest data/raw
    python tools/manual_data.py push --dest data/raw --tiers e2min --execute

`list` đọc `src/dire_modal/manifest.py` (nguồn sự thật về tên file, kể cả bốn
ca tên lệch nhánh), không chép tay danh sách — chép tay là cách chắc chắn nhất
để nó lệch với code sau ba tháng.

Quy ước vị trí: **đường dẫn trong dataset = đường dẫn dưới `data/raw/`**.
`checkpoints/lsun_adm.pth` → `data/raw/checkpoints/lsun_adm.pth`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dire_core.recons_pair import DIRS as RECONS_DIRS  # noqa: E402

# Nhóm "cần tải", mỗi nhóm là một câu hỏi cụ thể của người dùng.
NEEDS: dict[str, tuple[str, str]] = {
    "a": ("A · đường từ đầu (KHÔNG lấy từ dataset)",
          "2 file từ nguồn ngoài: ADM của OpenAI + LSUN val"),
    "gallery": ("B · kiểm chứng reconstruction — bộ tối thiểu",
                "3 generator × 3 nhánh (images/recons/dire)"),
    "recons-more": ("B · mở rộng lên 7 generator có recons/",
                    "thêm images+recons cho 5 generator còn lại"),
    "e2min": ("C · chấm lại R0 — con số đầu tiên", "3 file, ~0,40 GB"),
    "e2": ("C · chấm lại R0 — hàng ADM (8 cột)", "17 file, ~1,94 GB"),
    "ckpts": ("C · chấm lại R0 — đủ 32 ô", "6 checkpoint, ~1,58 GB"),
}

# Nguồn NGOÀI dataset (A). `size` để trống khi repo chưa đo được — không bịa.
OUTSIDE: list[dict] = [
    {"need": "a", "remote": "256x256_diffusion_uncond.pt",
     "url": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt",
     "local": "sources/ckpt/adm_lsun.pt", "size": "2,5 GB"},
    {"need": "a", "remote": "bedroom_val_lmdb.zip",
     "url": "http://dl.yf.io/lsun/objects/bedroom_val_lmdb.zip",
     "local": "sources/lsun/bedroom_val_lmdb.zip", "size": "— (chưa đo)"},
]

#: Size đã biết cho vài nhóm (số ĐO, không ước). Thiếu thì để "—".
KNOWN_SIZE = {
    "checkpoints/lsun_adm.pth": "269,4 MB",
    "checkpoints/lsun_iddpm.pth": "269,4 MB",
    "checkpoints/lsun_pndm.pth": "269,4 MB",
    "checkpoints/lsun_stylegan.pth": "269,4 MB",
    "checkpoints/imagenet_adm.pth": "269,4 MB",
    "checkpoints/celebahq_sdv2.pth": "269,4 MB",
}

TIER_TOTAL = {
    "e2min": "3 file · ~0,40 GB",
    "e2": "17 file · ~1,94 GB",
    "ckpts": "6 file · ~1,58 GB",
    "gallery": "9 file · ~0,67 GB",
}


def _manifest():
    from dire_modal import manifest as mf  # noqa: PLC0415

    return mf


def _tier_files(tier: str):
    mf = _manifest()
    files, unresolved, blocked = mf.expand(mf.resolve_many(tier), lister=None)
    if unresolved:
        raise SystemExit(
            f"[!] tier {tier} có DirSpec cần liệt kê online: "
            f"{[d.remote_dir for d in unresolved]}. Dùng tier khác cho tải tay."
        )
    return files, blocked


def _need_files(need: str) -> list:
    """File của một nhóm nhu cầu (đã bỏ file ngoài dataset)."""
    mf = _manifest()
    if need in {"a", "recons-more"}:
        if need == "a":
            return []
        # 5 generator có recons nhưng chưa nằm trong `gallery`.
        out = []
        for gen in ("sdv2", "vqdiffusion", "projectedgan", "midjourney", "stylegan"):
            for branch in ("images", "recons"):
                remote = f"{RECONS_DIRS[branch]}/{mf.GALLERY_TRIPLETS[gen][branch]}"
                out.append(mf.File(remote=remote, local=remote,
                                   note=f"{need}: {branch} của {gen}",
                                   allow_never=(branch == "recons")))
        return out
    return _tier_files(need)[0]


def cmd_list(args) -> int:
    needs = [n.strip() for n in args.need.split(",") if n.strip()]
    unknown = [n for n in needs if n not in NEEDS]
    if unknown:
        raise SystemExit(f"[!] nhóm lạ: {unknown}. Có: {sorted(NEEDS)}")

    print("=" * 100)
    print("DANH SÁCH TẢI TAY — đặt mọi file vào:", Path(args.dest).as_posix())
    print("Quy ước: đường dẫn trong dataset = đường dẫn dưới", Path(args.dest).as_posix())
    print("=" * 100)

    rows: list[tuple[str, str, str, str, str]] = []
    for need in needs:
        title, sub = NEEDS[need]
        print(f"\n### {title}\n    {sub}")
        if need == "a":
            print(f"    {'nguồn':<62} {'đặt vào':<40} size")
            for item in OUTSIDE:
                print(f"    {item['url']:<62} {args.dest}/{item['local']:<32} {item['size']}")
                rows.append((need, item["url"], f"{args.dest}/{item['local']}",
                             item["size"], "tải bằng trình duyệt"))
            print("\n    ⚠  KHÔNG có trong dataset DiffusionForensics — tải từ nguồn trên.")
            continue

        files = _need_files(need)
        print(f"    tổng: {TIER_TOTAL.get(need, f'{len(files)} file')}")
        print(f"    {'#':>3}  {'đường dẫn trong dataset':<66} {'đặt vào':<34} size")
        for i, f in enumerate(files, 1):
            size = KNOWN_SIZE.get(f.remote, "—")
            print(f"    {i:>3}  {f.remote:<66} {args.dest}/{f.local:<28} {size}")
            rows.append((need, f.remote, f"{args.dest}/{f.local}", size, f.note))

    print("\n" + "=" * 100)
    print("TÊN FILE DỄ SAI — đừng đổi tên khi tải tay:")
    print("  • images/test/lsun_bedroom/midjourney.tar.gz      (ĐÚNG chính tả)")
    print("  • recons/test/lsun_bedroom/lsun_bedroom/midjourney.tar.gz  (ĐÚNG)")
    print("  • dire/test/lsun_bedroom/lsun_bedroom/midjouney.tar.gz     (THIẾU chữ 'r' — typo GỐC)")
    print("  • images/test/lsun_bedroom/stylegan_official.tar.gz  (khác tên hai nhánh kia)")
    print("  Không có recons/ cho: adm, ddpm, iddpm, sdv1, dalle2, if, diff-*gan.")
    print("=" * 100)

    if args.tsv:
        out = Path(args.tsv)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# need\tremote\tlocal\tsize\tnote"] + [
            "\t".join(r) for r in rows
        ]
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[ghi] {out}  ({len(rows)} dòng)")
    return 0


# --------------------------------------------------------------------------- #
# verify — không cần Baidu, không cần Modal
# --------------------------------------------------------------------------- #

def _sniff_bad(p: Path) -> str | None:
    """Phát hiện file không phải dữ liệu thật (trang lỗi HTML/JSON)."""
    try:
        head = p.open("rb").read(400)
    except OSError as exc:
        return f"không đọc được: {exc}"
    low = head.lstrip().lower()
    if low.startswith(b"<!doctype") or low.startswith(b"<html"):
        return "nội dung là TRANG HTML (tải bằng trình duyệt bị chặn/đăng nhập)"
    if low.startswith(b"{") and b"\"error\"" in low:
        return "nội dung là JSON lỗi của API"
    return None


def _check_tar_gz(p: Path) -> tuple[bool, str, int]:
    try:
        with tarfile.open(p, "r:gz") as tf:
            n = sum(1 for _ in tf)
    except (tarfile.TarError, OSError, EOFError) as exc:
        return False, f"tar.gz hỏng: {exc}", 0
    if n == 0:
        return False, "tar.gz rỗng (0 member)", 0
    return True, "ok", n


def _check_pth(p: Path) -> tuple[bool, str]:
    try:
        head = p.open("rb").read(4)
    except OSError as exc:
        return False, f"không đọc được: {exc}"
    if head[:4] == b"PK\x03\x04":
        return True, "ok (zip/pytorch ≥1.6)"
    return False, f"không phải zip checkpoint (magic={head!r}) — có thể là trang HTML"


def _check_file(p: Path) -> tuple[bool, str, dict]:
    bad = _sniff_bad(p)
    if bad:
        return False, bad, {}
    size = p.stat().st_size
    if size < 256:
        # Ngưỡng thấp có chủ ý: dataset gốc có tar 130 B (rỗng thật). File rỗng
        # bị bắt bởi phép thử tar bên dưới, chứ không bởi kích thước.
        return False, f"quá nhỏ ({size} B) — chắc chắn là file lỗi", {}
    if p.name.endswith((".tar.gz", ".tgz")):
        ok, why, n = _check_tar_gz(p)
        return ok, why, {"members": n}
    if p.suffix == ".pth":
        ok, why = _check_pth(p)
        return ok, why, {}
    return True, "ok (không có phép kiểm riêng)", {}


def _find(dest: Path, rel: str) -> tuple[Path | None, str]:
    """Tìm đúng đường dẫn; nếu không, tìm theo TÊN và nói rõ đã tìm thấy ở đâu."""
    exact = dest / rel
    if exact.is_file():
        return exact, "exact"
    hits = [p for p in dest.rglob(Path(rel).name)
            if p.is_file() and not any(x in p.parts for x in (".staging", "_held"))]
    if len(hits) == 1:
        return hits[0], f"theo tên → {hits[0].relative_to(dest).as_posix()}"
    if len(hits) > 1:
        return None, f"mơ hồ: {len(hits)} file cùng tên ({[h.relative_to(dest).as_posix() for h in hits[:3]]})"
    return None, "không thấy"


def cmd_verify(args) -> int:
    dest = Path(args.dest)
    needs = [n.strip() for n in args.need.split(",") if n.strip()]
    files = []
    for need in needs:
        if need == "a":
            continue
        files.extend(_need_files(need))
    # khử trùng theo local
    seen, uniq = set(), []
    for f in files:
        if f.local not in seen:
            seen.add(f.local)
            uniq.append(f)

    print("=" * 100)
    print(f"KIỂM CHỨNG OFFLINE — {len(uniq)} file trong {dest} (không gọi mạng)")
    print("=" * 100)

    ok, bad, missing = [], [], []
    warn = []
    junk = [p for p in dest.rglob("*")
            if p.is_file() and (p.suffix in {".part", ".crdownload"}
                                or p.name.endswith(".tmp"))] if dest.is_dir() else []
    for f in uniq:
        p, how = _find(dest, f.local)
        if p is None:
            print(f"  ✗ THIẾU  {f.local}   ({how})")
            missing.append(f)
            continue
        good, why, extra = _check_file(p)
        size = p.stat().st_size
        if good:
            m = f" · {extra['members']} member" if "members" in extra else ""
            print(f"  ✓ {size/1e6:9.1f} MB  {f.local}{m}"
                  + ("" if how == "exact" else f"   [{how}]"))
            ok.append(f)
        else:
            print(f"  ✗ HỎNG   {size/1e6:9.1f} MB  {f.local}\n            → {why}")
            bad.append((f, why))
    for p in junk:
        warn.append(p)
        print(f"  ! RÁC    {p.relative_to(dest).as_posix()} — file tải dở, xoá được")

    print("-" * 100)
    print(f"  {len(ok)} đạt · {len(bad)} hỏng · {len(missing)} thiếu")
    if bad:
        print("\n  File hỏng: tải lại (đừng đổi tên). Nếu là HTML: phiên đăng nhập Baidu đã hết.")
    if missing:
        print("\n  Còn thiếu:")
        for f in missing:
            print(f"    - {f.remote}  →  {args.dest}/{f.local}")
    if args.strict and (bad or missing):
        return 3
    return 0


# --------------------------------------------------------------------------- #
# push — đưa dữ liệu tay tải lên Volume /raw
# --------------------------------------------------------------------------- #

def cmd_push(args) -> int:
    dest = Path(args.dest)
    files = []
    for need in [n.strip() for n in args.need.split(",") if n.strip()]:
        if need == "a":
            continue
        files.extend(_need_files(need))
    seen, uniq = set(), []
    for f in files:
        if f.local not in seen:
            seen.add(f.local)
            uniq.append(f)

    cmds = []
    for f in uniq:
        p, _how = _find(dest, f.local)
        if p is None:
            print(f"  [bỏ] thiếu {f.local}")
            continue
        cmds.append(["modal", "volume", "put", args.volume, str(p), f.local])

    print(f"\n{len(cmds)} lệnh `modal volume put {args.volume}`:")
    for c in cmds:
        print("  " + " ".join(f'"{x}"' if " " in x else x for x in c))
    if not args.execute:
        print("\n(đây là DANH SÁCH; thêm --execute để chạy thật)")
        return 0
    if not shutil.which("modal"):
        raise SystemExit("[!] không thấy lệnh `modal` trong PATH")
    for c in cmds:
        print("→", " ".join(c))
        r = subprocess.run(c, check=False)
        if r.returncode != 0:
            raise SystemExit(f"[!] lệnh thất bại: {c}")
    print("\n[xong] đã đẩy lên Volume. Kiểm lại bằng `modal volume ls` nếu cần.")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Tải tay (không BaiduPCS-Go): in danh sách, kiểm, đẩy lên Volume.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = os.environ.get("DIRE_LOCAL_ROOT", "data/raw")

    p = sub.add_parser("list", help="in danh sách file cần tải + vị trí dưới data/raw/")
    p.add_argument("--need", default="a,gallery,e2min,e2,ckpts")
    p.add_argument("--dest", default=d)
    p.add_argument("--tsv", default="manifests/manual-download.tsv")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("verify", help="kiểm cấu trúc file đã tải tay (offline)")
    p.add_argument("--need", default="a,gallery,e2min,e2,ckpts")
    p.add_argument("--dest", default=d)
    p.add_argument("--strict", action="store_true", help="thiếu/hỏng ⇒ exit 3")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("push", help="in (hoặc chạy) modal volume put cho từng file")
    p.add_argument("--need", default="e2min")
    p.add_argument("--dest", default=d)
    p.add_argument("--volume", default="dire-raw")
    p.add_argument("--execute", action="store_true")
    p.set_defaults(func=cmd_push)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = _build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
