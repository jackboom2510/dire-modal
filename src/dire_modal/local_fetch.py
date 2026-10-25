"""Tải dữ liệu DIRE về **máy local** — không dùng Modal, không dùng Volume.

    python -m dire_modal.local_fetch --tiers e2min --dest ./data/raw            # kế hoạch
    python -m dire_modal.local_fetch --tiers e2min --dest ./data/raw --mode fetch

Vì sao có đường này
-------------------
Đường Modal (``download.py``) đúng nhưng bị ba trần của nền tảng: mỗi attempt
1800 s, mỗi lần gọi ``fetch_one`` 7200 s, mỗi app 20 h ngân sách. Ở 76.6 KiB/s
(trần của TÀI KHOẢN Baidu, không phải của Modal) thì ``e2`` 1.94 GB = 7.4 h, nên
một file 269 MB luôn bị cắt ít nhất một lần.

Ở máy local không có ba trần đó: bật terminal, để chạy, Ctrl+C lúc nào cũng
được — mảnh dở nằm trên đĩa và lần sau resume từ đúng byte đã có.

Nguyên tắc giữ nguyên (đây là những bài học đã trả giá, xem ``fetchcore``):

* **Một nguồn sự thật cho "file đã đúng chưa?"** — dùng đúng ``fetchcore``,
  không viết lại phép kiểm. ``ls -l`` in size đã làm tròn ⇒ phải có ``slack``.
* **Không bao giờ xoá byte đã tải**: file không qua kiểm đi vào
  ``<dest>/.staging/_held/``.
* **Staging tất định theo đường dẫn remote** ⇒ BaiduPCS-Go resume được.
* **Không tải file thừa**: chỉ tải file đã kê khai trong tier; ``NEVER`` bị chặn.
* **Không tăng song song**: trần ~76.6 KiB/s là của tài khoản; thêm phiên chỉ
  làm Baidu throttle.

Khác biệt so với đường Modal (có chủ đích):

| | Modal | Local |
|---|---|---|
| thời gian mỗi attempt | 1800 s | ``--stall-s`` (mặc định 900 s không ghi byte nào thì bỏ) |
| số attempt | 3 | ``--attempts`` (mặc định 5) |
| ngân sách | 20 h | ``--max-hours`` (mặc định 0 = không giới hạn) |
| song song | 1 container | tuần tự (đúng 1 phiên Baidu) |
| Ctrl+C | SIGTERM → commit Volume | ``KeyboardInterrupt`` → ``finally`` giữ staging |
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .fetchcore import (
    SizeEst,
    _bytes_of,
    _demangle,
    _demangle_list,
    _human,
    _parse_ls_estimates,
    _sha256,
    _slack_of,
    _validate_ex,
    hold_into,
    staging_dir_for,
)
from .manifest import BAIDU_ROOT, POLICY, TIERS, File, expand, is_never, resolve_many

# ---------------------------------------------------------------------------
# Hằng số
# ---------------------------------------------------------------------------
POLL_S = 5.0            # nhịp kiểm tiến độ (chỉ `stat` thư mục, rất rẻ)
DEFAULT_STALL_S = 900.0  # 15 phút không thêm byte nào ⇒ bỏ attempt, resume sau
DEFAULT_ATTEMPTS = 5
DEFAULT_DEST = os.environ.get("DIRE_LOCAL_ROOT", "data/raw")
MANIFEST = "_manifest.tsv"


def _abs(remote_rel: str) -> str:
    return f"{BAIDU_ROOT.rstrip('/')}/{remote_rel.lstrip('/')}"


def _find_baidu(explicit: str = "") -> str:
    """Đường dẫn tới ``BaiduPCS-Go``. ``.exe`` trên Windows được ``which`` lo."""
    for cand in (explicit, os.environ.get("BAIDU_BIN", ""), "BaiduPCS-Go",
                 "BaiduPCS-Go.exe", "baidupcs-go"):
        if not cand:
            continue
        hit = shutil.which(cand)
        if hit:
            return hit
        p = Path(cand)
        if p.is_file():
            return str(p)
    raise SystemExit(
        "[!] không tìm thấy BaiduPCS-Go trong PATH.\n"
        "    Tải: https://github.com/qjfoidnh/BaiduPCS-Go/releases\n"
        "    Rồi đăng nhập MỘT lần:  BaiduPCS-Go login -bduss=<BDUSS>\n"
        "    Hoặc chỉ định:  --baidu-bin /path/to/BaiduPCS-Go"
    )


def _run(bin_: str, args: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run([bin_, *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def _dir_bytes(d: Path) -> int:
    try:
        return sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
    except OSError:
        return 0


def _preflight(bin_: str) -> None:
    """Đăng nhập chưa? In dung lượng tài khoản. Không raise nếu không đọc được."""
    try:
        cp = _run(bin_, ["quota"], timeout=60)
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"[warn] không chạy được `BaiduPCS-Go quota`: {exc}")
        return
    text = ((cp.stdout or "") + (cp.stderr or "")).strip()
    if cp.returncode != 0 or "未登录" in text or "not logged" in text.lower():
        raise SystemExit(
            "[!] BaiduPCS-Go chưa đăng nhập (hoặc BDUSS hết hạn).\n"
            "    BaiduPCS-Go login -bduss=<BDUSS>   # lấy BDUSS từ cookie trình duyệt\n"
            f"    output: {text[:300]}"
        )
    print(f"[baidu] {text.splitlines()[0] if text else 'ok'}")


# ---------------------------------------------------------------------------
# Hỏi size remote — một lệnh `ls -l` cho MỖI thư mục
# ---------------------------------------------------------------------------
def _ls_estimates(bin_: str, remote_dir_rel: str, timeout: float = 300) -> dict[str, SizeEst]:
    try:
        cp = _run(bin_, ["ls", "-l", _abs(remote_dir_rel)], timeout=timeout)
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"[warn] ls {remote_dir_rel} thất bại: {exc!r}")
        return {}
    if cp.returncode != 0:
        return {}
    return _parse_ls_estimates(cp.stdout or "")


def _size_map(bin_: str, files: list[File]) -> dict[str, SizeEst | None]:
    by_dir: dict[str, list[File]] = {}
    for f in files:
        by_dir.setdefault(str(Path(f.remote).parent), []).append(f)
    sizes: dict[str, SizeEst | None] = {}
    for d, group in sorted(by_dir.items()):
        listing = _ls_estimates(bin_, d)
        for f in group:
            sizes[f.remote] = listing.get(Path(f.remote).name)
    return sizes


# ---------------------------------------------------------------------------
# Tải một file
# ---------------------------------------------------------------------------
def _download_with_watchdog(bin_: str, remote: str, saveto: Path,
                            stall_s: float, max_s: float = 0) -> None:
    """Chạy ``BaiduPCS-Go d``; giết nếu **không ghi thêm byte nào** trong ``stall_s``.

    Không dùng ``subprocess.run(timeout=)``: nó chỉ biết *tổng* thời gian. File
    269 MB ở 76.6 KiB/s cần ~1 h — hợp lý; 1 h ở 0 B/s là mất trắng. Chỉ
    watchdog phân biệt được hai ca đó.
    """
    proc = subprocess.Popen([bin_, "d", "--saveto", str(saveto), remote])
    t0 = time.monotonic()
    last_size, last_change = -1, t0
    try:
        while True:
            try:
                rc = proc.wait(timeout=POLL_S)
                if rc != 0:
                    raise subprocess.CalledProcessError(rc, "BaiduPCS-Go")
                return
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            size = _dir_bytes(saveto)
            if size != last_size:
                last_size, last_change = size, now
            if stall_s > 0 and now - last_change >= stall_s:
                raise TimeoutError(
                    f"không ghi thêm byte nào trong {stall_s:.0f}s "
                    f"(giữ {_human(size)})")
            if max_s > 0 and now - t0 >= max_s:
                raise TimeoutError(f"vượt --max-hours cho một file ({max_s:.0f}s)")
    finally:
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass


def _find_downloaded(tmpdir: Path, want: str) -> Path | None:
    exact = tmpdir / want
    if exact.exists() and exact.stat().st_size > 0:
        return exact
    cands = [p for p in tmpdir.rglob("*") if p.is_file() and p.stat().st_size > 0]
    return cands[0] if len(cands) == 1 else None


def _load_state(dest: Path) -> dict[str, dict]:
    try:
        return json.loads((dest / ".state" / "validated.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_state(dest: Path, rel: str, path: Path, digest: str, fmt: str) -> None:
    state = _load_state(dest)
    st = path.stat()
    state[rel] = {
        "size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": digest,
        "fmt": fmt,
        "checked": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        out = dest / ".state" / "validated.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, out)
    except OSError as exc:
        print(f"[warn] không ghi được cache kiểm tra: {exc}")


def _append_manifest(dest: Path, row: dict) -> None:
    out = dest / MANIFEST
    header = not out.exists()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        with out.open("a", encoding="utf-8", newline="") as fh:
            if header:
                fh.write("timestamp\tremote\tlocal\tsize_bytes\tsha256\tstatus\n")
            fh.write(f"{now}\t{row['remote']}\t{row['local']}\t{row['size']}\t"
                     f"{row.get('sha256', '')}\t{row['status']}\n")
    except OSError as exc:
        print(f"[warn] không ghi được manifest: {exc}")


def _sleep(seconds: float) -> None:
    """Nghỉ giữa hai attempt. Tách ra để test không phải chờ thật."""
    time.sleep(seconds)


def _snapshot(d: Path) -> dict[str, tuple[int, int]]:
    """``{relpath: (size, mtime_ns)}`` của mọi file trong staging, TRƯỚC attempt."""
    out: dict[str, tuple[int, int]] = {}
    try:
        for p in d.rglob("*"):
            if p.is_file():
                st = p.stat()
                out[p.relative_to(d).as_posix()] = (st.st_size, st.st_mtime_ns)
    except OSError:
        pass
    return out


def _untouched(before: dict[str, tuple[int, int]], staging: Path, got: Path) -> bool:
    """``True`` nếu attempt này KHÔNG đụng gì tới ``got``.

    Đó là dấu hiệu BaiduPCS-Go từ chối làm lại vì nó thấy file đích đã tồn tại
    (``文件已经存在 … 跳过``, ``数据总量: 0B``). So cả ``mtime_ns`` chứ không chỉ
    size: ghi lại đúng số byte vẫn là "có ghi", và mtime đổi.
    """
    try:
        rel = got.relative_to(staging).as_posix()
        st = got.stat()
    except (OSError, ValueError):
        return False
    return before.get(rel) == (st.st_size, st.st_mtime_ns)


def fetch_one(bin_: str, f: File, dest: Path, expected: SizeEst | None,
              stall_s: float, attempts: int) -> dict:
    """Tải MỘT file. Không bao giờ raise — trả dict như đường Modal."""
    remote_rel = f.remote
    local = dest / f.local
    exp, slack = _bytes_of(expected), _slack_of(expected)
    staging = staging_dir_for(dest / ".staging", remote_rel)
    held_dir = dest / ".staging" / "_held" / staging.name

    # --- đã có & hợp lệ → skip ----------------------------------------------
    if local.exists() and local.stat().st_size > 0:
        v = _validate_ex(local, exp, slack=slack)
        if v.ok:
            print(f"[SKIP] {remote_rel} đã có & hợp lệ")
            return {"remote": remote_rel, "local": f.local, "status": "skipped",
                    "size": local.stat().st_size, "sha256": ""}
        print(f"[REDO] {remote_rel}: {v.reason} → chuyển bản cũ sang _held")
        hold_into(local, held_dir)

    local.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)
    last = "chưa thử"
    held_once = False
    attempt = 0

    while attempt < attempts:
        attempt += 1
        before = _snapshot(staging)
        try:
            _download_with_watchdog(bin_, _abs(remote_rel), staging, stall_s)
        except subprocess.CalledProcessError as exc:
            last = f"BaiduPCS-Go exit {exc.returncode}"
            print(f"[RETRY {attempt}/{attempts}] {remote_rel}: {last}")
            _sleep(min(30 * attempt, 120))
            continue
        except (TimeoutError, OSError) as exc:
            last = f"{type(exc).__name__}: {exc} → resume sau"
            print(f"[RETRY {attempt}/{attempts}] {remote_rel}: {last}")
            _sleep(min(30 * attempt, 120))
            continue

        got = _find_downloaded(staging, Path(remote_rel).name)
        if got is None:
            zeros = [p for p in staging.rglob("*") if p.is_file() and p.stat().st_size == 0]
            last = ("0 byte — BaiduPCS-Go exit 0 nhưng không tải gì "
                    "(BDUSS hết hạn, hoặc bị throttle)" if zeros else
                    f"không tạo file nào — đường dẫn remote sai? ({remote_rel})")
            print(f"[RETRY {attempt}/{attempts}] {remote_rel}: {last}")
            _sleep(min(30 * attempt, 120))
            continue

        v = _validate_ex(got, exp, slack=slack)
        if v.ok:
            size = got.stat().st_size
            digest = _sha256(got)
            if local.exists():
                hold_into(local, held_dir)
            os.replace(got, local)
            _save_state(dest, f.local, local, digest, v.fmt or "gzip")
            print(f"[OK] {remote_rel}  {_human(size)}  sha256={digest[:12]}…")
            return {"remote": remote_rel, "local": f.local, "status": "ok",
                    "size": size, "sha256": digest, "attempts": attempt}

        last = v.reason

        # Baidu không đụng tới file đã có ⇒ thử lại y nguyên là vô nghĩa. Nhưng
        # KHÔNG xoá (byte đã tải là tài sản): chuyển sang `_held`, rồi tải lại
        # từ 0 B đúng MỘT lần.
        if _untouched(before, staging, got):
            if not held_once:
                held_once = True
                held = hold_into(got, held_dir)
                print(f"[HOLD] {remote_rel}: Baidu không ghi thêm byte nào mà file "
                      f"vẫn không qua kiểm ({v.reason}) → giữ ở {held}, tải lại từ 0 B")
                attempt -= 1           # lần "Baidu bỏ qua" không tiêu suất tải
                continue
            last = f"tải lại từ 0 B vẫn không qua kiểm: {v.reason}"
            break

        if not v.retryable:
            held = hold_into(got, held_dir)
            print(f"[STOP] {remote_rel}: {v.reason} — giữ ở {held}")
            return {"remote": remote_rel, "local": f.local, "status": "error",
                    "error": v.reason, "size": 0, "held": str(held),
                    "deterministic": True, "attempts": attempt}

        print(f"[RETRY {attempt}/{attempts}] {remote_rel}: {last}")
        _sleep(min(30 * attempt, 120))

    kept = f" — bản đã tải nằm ở {staging}" if staging.exists() else ""
    return {"remote": remote_rel, "local": f.local, "status": "error",
            "error": last + kept, "size": 0, "attempts": attempts}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m dire_modal.local_fetch",
        description="Tải dữ liệu DIRE về máy local (không Modal, không Volume).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""ví dụ:
  python -m dire_modal.local_fetch --tiers e2min --dest ./data/raw
  python -m dire_modal.local_fetch --tiers e2min --dest ./data/raw --mode fetch
  python -m dire_modal.local_fetch --tiers e2,ckpts --mode fetch   # 3.25 GB, ~12 h
  python -m dire_modal.local_fetch --paths checkpoints/lsun_adm.pth --mode fetch
""")
    ap.add_argument("--tiers", default="e2min",
                    help=f"tier cần tải, phân cách bằng dấu phẩy. Có: {sorted(TIERS)}")
    ap.add_argument("--dest", default=DEFAULT_DEST,
                    help=f"thư mục gốc trên máy (mặc định {DEFAULT_DEST})")
    ap.add_argument("--mode", default="plan", choices=["plan", "fetch"],
                    help="plan (mặc định — KHÔNG tải gì) hoặc fetch")
    ap.add_argument("--paths", default="",
                    help="danh sách đường dẫn remote, phân cách bằng dấu phẩy")
    ap.add_argument("--max-gb", type=float, default=POLICY.max_gb,
                    help=f"trần dung lượng (mặc định {POLICY.max_gb:g} GB)")
    ap.add_argument("--force", action="store_true",
                    help="tải lại cả file đã có (bỏ qua cache)")
    ap.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    ap.add_argument("--stall-s", type=float, default=DEFAULT_STALL_S,
                    help="bỏ attempt nếu không ghi thêm byte nào trong ngần này giây")
    ap.add_argument("--max-hours", type=float, default=0,
                    help="0 = không giới hạn (mặc định)")
    ap.add_argument("--baidu-bin", default="", help="đường dẫn tới BaiduPCS-Go")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # Git Bash: "/raw/x" tới Python dưới dạng "C:/.../Git/raw/x" — xem fetchcore
    dest = Path(_demangle(args.dest)).expanduser()
    paths = _demangle_list(args.paths)

    bin_ = _find_baidu(args.baidu_bin)
    _preflight(bin_)

    print(f"[local] đích      : {dest.resolve()}")
    print(f"[local] baidu     : {bin_}")

    # --- chọn file -----------------------------------------------------------
    unresolved = []
    if paths.strip():
        files = [File(remote=p.strip(), local=p.strip()) for p in paths.split(",") if p.strip()]
        files = [f for f in files if not _never(f)]
    else:
        ents = resolve_many(args.tiers)
        if not ents:
            raise SystemExit(f"[!] tier {args.tiers!r} không có file nào. Có: {sorted(TIERS)}")
        files, unresolved, blocked = expand(
            ents, lister=lambda d: list(_ls_estimates(bin_, d)))
        for f, why in blocked:
            print(f"  ⛔ NEVER chặn: {why}")

    sizes = _size_map(bin_, files)
    print(f"\n{'='*78}\nKẾ HOẠCH TẢI — {len(files)} file → {dest}\n{'='*78}")
    known = unknown = largest = 0
    for f in files:
        est = sizes.get(f.remote)
        if est is None:
            unknown += 1
            cell = "? (ls -l không thấy)"
        else:
            known += est.nbytes
            largest = max(largest, est.nbytes)
            cell = _human(est.nbytes)
        note = f"  # {f.note}" if f.note else ""
        print(f"  {cell:>22}  {f.remote}{note}")
    est = known + unknown * largest
    print(f"{'-'*78}\n  Tổng (đã biết): {_human(known)}"
          + (f"  + {unknown} file chưa rõ size" if unknown else ""))
    if unknown:
        print(f"  ⚠ ước lượng bảo thủ: {_human(est)}")

    if est > args.max_gb * (1024 ** 3):
        raise SystemExit(
            f"\n[GUARD] {_human(est)} > trần {args.max_gb:g} GB → từ chối tải.\n"
            f"  Chọn ít tier hơn, hoặc nâng trần có chủ đích: --max-gb")

    if unresolved:
        print(f"\n  ⚠ {len(unresolved)} thư mục chưa expand được: "
              f"{[d.remote_dir for d in unresolved[:3]]}")

    if args.mode != "fetch":
        print("\n[DRY-RUN] Chưa tải gì. Tải thật: thêm --mode fetch")
        return 0

    # --- tải -----------------------------------------------------------------
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results: list[dict] = []
    try:
        for i, f in enumerate(files, 1):
            if args.max_hours and (time.time() - t0) / 3600 >= args.max_hours:
                print(f"\n[budget] hết --max-hours {args.max_hours:g} — dừng, chạy lại để resume")
                break
            print(f"\n[{i}/{len(files)}] {f.remote}")
            r = fetch_one(bin_, f, dest, sizes.get(f.remote),
                          args.stall_s, max(1, args.attempts))
            results.append(r)
            _append_manifest(dest, r)
    except KeyboardInterrupt:
        print("\n[stop] Ctrl+C — phần đã tải nằm trong "
              f"{dest / '.staging'} và KHÔNG bị xoá. Chạy lại đúng lệnh này để resume.")
        return 130

    # --- tổng kết ------------------------------------------------------------
    ok = [r for r in results if r["status"] == "ok"]
    skip = [r for r in results if r["status"] == "skipped"]
    err = [r for r in results if r["status"] == "error"]
    dt = time.time() - t0
    print(f"\n{'='*78}\nKẾT QUẢ ({dt/60:.1f} phút)\n{'='*78}")
    print(f"  ok      : {len(ok)}")
    print(f"  skipped : {len(skip)}  (đã có sẵn & hợp lệ)")
    print(f"  error   : {len(err)}")
    print(f"  còn lại : {len(files) - len(ok) - len(skip)} / {len(files)}")
    for r in err:
        print(f"    {r['remote']}\n       → {r['error']}")
    if err:
        print("\nChạy lại đúng lệnh: file đã xong sẽ [SKIP], file dở resume từ byte đã có.")
    print(f"\n[manifest] → {dest / MANIFEST}")
    return 0 if not err else 1


def _never(f: File) -> bool:
    """``True`` nếu file nằm trong vùng ``NEVER`` (recons/, weight ADM, ...)."""
    why = is_never(f.remote)
    if why:
        print(f"  ⛔ NEVER chặn {f.remote}: {why}")
    return bool(why)


if __name__ == "__main__":
    sys.exit(main())
