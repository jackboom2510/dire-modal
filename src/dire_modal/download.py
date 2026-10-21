"""Tải .tar.gz từ Baidu/Rec vào Volume ``/raw`` — có dry-run, guard dung lượng,
và **không tải file thừa**.

Luồng chuẩn (3 bước, bước 1–2 không tải gì)::

    modal run -m dire_modal.download::audit                 # cây remote ↔ manifest
    modal run -m dire_modal.download::plan --tiers e2,e4    # bảng file + size thật
    modal run -m dire_modal.download::main --tiers e2,e4 --mode fetch

``main`` mặc định là **plan** (dry-run). Muốn tải thật phải nói rõ
``--mode fetch``. Vượt trần ``max_gb`` trong ``tiers.yaml`` ⇒ từ chối.

Sáu lỗi đã vá so với bản đầu (đều thấy trong log chạy thật 2026-10-21):

1. ``Path.rename()`` từ ``/tmp`` sang ``/raw`` ⇒ ``OSError errno 18 EXDEV``.
   ``rename(2)`` không qua được khác filesystem; ``/raw`` là Volume mạng.
   → dùng ``shutil.move()`` (= ``mv`` của shell: copy + unlink).
2. BaiduPCS-Go **exit 0 dù tải 0 byte** (``下载结束, 数据总量: 0B``) ⇒ ``check=True``
   vô dụng. → validate size + ``gzip -t`` sau khi tải, và đối chiếu size remote.
3. Tham số ``--concurrency`` bị bỏ qua hoàn toàn và ``fetch_one`` không đặt
   ``max_containers`` ⇒ ~26 container bắn Baidu cùng lúc ⇒ throttle ``0B/s``
   hàng loạt. → ``max_containers=POLICY.max_containers`` (mặc định 2).
4. Mọi file đều tải về ``/tmp/<basename>`` mà ``adm.tar.gz`` / ``real.tar.gz``
   xuất hiện 3 lần trong cùng Tier A ⇒ BaiduPCS-Go resume nhầm mảnh của file
   khác ⇒ tar lai, hỏng âm thầm. → thư mục tmp riêng theo hash đường dẫn remote.
5. Một task raise ⇒ ``.map()`` huỷ toàn bộ đợt ⇒ "Runner terminated" ×15 và mất
   trắng mọi file dở (``/tmp`` ephemeral, ``commit()`` chưa chạy). → không bao giờ
   raise ra khỏi ``fetch_one``; retry thủ công bên trong và trả status dict.
6. ``subprocess.run`` không có ``timeout`` ⇒ phiên treo 0B/s ngồi đó tới 6 giờ.
   → ``timeout=POLICY.subprocess_timeout`` mỗi lệnh.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import modal

from .common import app, baidu_secret, cpu_image, vol_raw
from .manifest import (
    BAIDU_ROOT,
    NEVER,
    POLICY,
    TIERS,
    DirSpec,
    File,
    expand,
    is_never,
    resolve_many,
)

RAW = Path("/raw")

_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB)\b", re.I)
_KNOWN_SUFFIX = (".tar.gz", ".pth", ".pt")

# Thư mục đáng quan tâm khi audit. Đã đối chiếu với audit thật 2026-10-21:
# bỏ "images/test/lsun_bedroom/lsun_bedroom" (KHÔNG tồn tại — nhánh images/ chỉ
# có một tầng), thêm các nhánh recons/ và images/test/ chưa quét để con số
# "NEVER chặn được X GB" hết là cận dưới.
# Thư mục không tồn tại chỉ in "⚠ không liệt kê được" — vô hại, và chính nó là
# cách dò cây: thêm tên phỏng đoán vào đây rồi đọc output.
AUDIT_DIRS = [
    "checkpoints",
    "dire/test/lsun_bedroom/lsun_bedroom",
    "dire/test/lsun_bedroom/imagenet",
    "dire/test/imagenet/imagenet",
    "dire/test/imagenet/lsun_bedroom",
    "dire/test/celebahq",
    "dire/val/lsun_bedroom",
    "dire/val/imagenet",
    "dire/val/celebahq",
    "dire/train/lsun_bedroom",
    "dire/train/imagenet",
    "dire/train/celebahq",
    "images/test/lsun_bedroom",
    "images/test/imagenet/imagenet",
    "images/test/imagenet/lsun_bedroom",
    "images/test/celebahq",
    "images/train/lsun_bedroom",
    "images/val/lsun_bedroom",
    "recons/test/lsun_bedroom/lsun_bedroom",
    "recons/test/lsun_bedroom/imagenet",
    "recons/test/imagenet/imagenet",
    "recons/test/imagenet/lsun_bedroom",
    "recons/test/celebahq",
    "recons/val/lsun_bedroom",
    "recons/train/lsun_bedroom",
]


# ============================================================================
# Helper thuần (không phải Modal function)
# ============================================================================
def _sha256(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1024
    return f"{n:.1f}TB"


def _baidu_login() -> None:
    """Login mỗi container (BDUSS/STOKEN inject qua Secret ``baidu-cred``)."""
    bduss = os.environ["BDUSS"]
    stoken = os.environ.get("STOKEN", "")
    args = ["login", f"-bduss={bduss}"]
    if stoken:
        args.append(f"-stoken={stoken}")
    _baidu(args, timeout=120, check=True)


def _baidu(
    args: list[str], timeout: int, check: bool = False, capture: bool = True
) -> subprocess.CompletedProcess:
    """Một lệnh BaiduPCS-Go. LUÔN có timeout — bản cũ không có nên treo 6 giờ."""
    return subprocess.run(
        ["BaiduPCS-Go", *args],
        check=check,
        capture_output=capture,
        text=capture,
        timeout=timeout,
    )


def _abs(remote_rel: str) -> str:
    return f"{BAIDU_ROOT.rstrip('/')}/{remote_rel.lstrip('/')}"


def _parse_ls_long(text: str) -> dict[str, int]:
    """Parse ``BaiduPCS-Go ls -l`` → ``{basename: size_bytes}``.

    Tolerant có chủ đích: output của BaiduPCS-Go trộn tiếng Trung, cột có thể
    xê dịch theo phiên bản. Chỉ nhận dòng vừa có token kích thước vừa có token
    cuối trông như đường dẫn file đã biết đuôi; dòng lạ thì bỏ qua chứ không
    đoán bừa. Size không parse được ⇒ "unknown", không phải 0.
    """
    out: dict[str, int] = {}
    for line in text.splitlines():
        m = _SIZE_RE.search(line)
        if not m:
            continue
        toks = line.split()
        if not toks:
            continue
        base = toks[-1].rstrip("/").rsplit("/", 1)[-1]
        if not base.endswith(_KNOWN_SUFFIX):
            continue
        size = int(float(m.group(1)) * _UNITS[m.group(2).upper()])
        out[base] = max(out.get(base, 0), size)
    return out


def _ls_sizes(remote_dir_rel: str, timeout: int = 300) -> dict[str, int]:
    """``ls -l`` MỘT thư mục. Trả {} nếu thư mục không tồn tại / lỗi."""
    try:
        cp = _baidu(["ls", "-l", _abs(remote_dir_rel)], timeout=timeout)
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"[warn] ls {remote_dir_rel} thất bại: {exc!r}")
        return {}
    if cp.returncode != 0:
        return {}
    return _parse_ls_long(cp.stdout)


def _size_map(files: list[File]) -> dict[str, int | None]:
    """Size remote cho từng file, hỏi Baidu **một lần mỗi thư mục** (không phải
    mỗi file) để không nhân số phiên lên — chính số phiên đồng thời là thứ làm
    Baidu throttle."""
    by_dir: dict[str, list[File]] = {}
    for f in files:
        by_dir.setdefault(str(Path(f.remote).parent), []).append(f)

    sizes: dict[str, int | None] = {}
    for d, group in sorted(by_dir.items()):
        listing = _ls_sizes(d)
        for f in group:
            sizes[f.remote] = listing.get(Path(f.remote).name)
    return sizes


def _validate(path: Path, expected: int = 0) -> tuple[bool, str]:
    """Kiểm tra file đã tải. Đây là chốt chặn lỗi #2: BaiduPCS-Go exit 0 dù 0 byte."""
    if not path.exists():
        return False, "file không tồn tại"
    size = path.stat().st_size
    if size == 0:
        return False, "0 byte — BaiduPCS-Go exit 0 nhưng không tải gì (throttle/BDUSS hết hạn)"
    if expected and size != expected:
        return False, f"size {size} ≠ remote {expected} → tải dở dang"
    if path.name.endswith(".tar.gz"):
        chk = subprocess.run(["gzip", "-t", str(path)], capture_output=True)
        if chk.returncode != 0:
            return False, "gzip -t FAIL — file cụt/hỏng"
    return True, "ok"


def _tmpdir_for(remote_rel: str) -> Path:
    """Thư mục tmp RIÊNG cho mỗi đường dẫn remote (sửa lỗi #4).

    Hash theo đường dẫn nên (a) không bao giờ trùng giữa ``train/adm.tar.gz`` và
    ``val/adm.tar.gz``, và (b) ỔN ĐỊNH giữa các attempt ⇒ BaiduPCS-Go resume
    được mảnh dở thay vì tải lại từ đầu.
    """
    digest = hashlib.sha1(remote_rel.encode()).hexdigest()[:12]
    return Path("/tmp") / f"baidu_{digest}"


def _find_downloaded(tmpdir: Path, want: str) -> Path | None:
    exact = tmpdir / want
    if exact.exists() and exact.stat().st_size > 0:
        return exact
    # BaiduPCS-Go có thể đổi tên khi trùng (adm(1).tar.gz) → lấy file duy nhất
    cands = [p for p in tmpdir.rglob("*") if p.is_file() and p.stat().st_size > 0]
    return cands[0] if len(cands) == 1 else None


def _tar_profile(path: Path, cap: int = 200_000) -> dict:
    """Đếm thành viên tar.gz mà KHÔNG giải nén — rẻ, không tốn đĩa.

    Trả về số file và số ảnh theo lớp ``0_real``/``1_fake`` (cấu trúc
    ImageFolder mà DIRE dùng). ``cap`` để một tar khổng lồ không làm treo hàm.
    """
    import tarfile

    n = 0
    classes: dict[str, int] = {}
    truncated = False
    with tarfile.open(path, "r:gz") as t:
        for mem in t:
            if not mem.isfile():
                continue
            n += 1
            if n > cap:
                truncated = True
                break
            for part in mem.name.split("/"):
                if part in ("0_real", "1_fake"):
                    classes[part] = classes.get(part, 0) + 1
    return {"files": n, "classes": classes, "truncated": truncated}


def _all_expected() -> dict[str, list[str]]:
    """``{remote_rel: [tier, ...]}`` cho mọi File kê khai tĩnh trong manifest."""
    out: dict[str, list[str]] = {}
    for tier, ents in TIERS.items():
        for e in ents:
            if isinstance(e, File):
                out.setdefault(e.remote, []).append(tier)
    return out


def _report(
    files: list[File],
    sizes: dict[str, int | None],
    blocked: list[tuple[File, str]],
    unresolved: list[DirSpec],
) -> tuple[int, int, int]:
    """In bảng kế hoạch.

    Returns:
        ``(known_bytes, unknown_count, largest_known)`` — cần cả ba vì guard
        dung lượng không được âm thầm bỏ qua file chưa rõ size.
    """
    print(f"\n{'='*78}\nKẾ HOẠCH TẢI — {len(files)} file\n{'='*78}")
    known = 0
    unknown = 0
    largest = 0
    for f in files:
        s = sizes.get(f.remote)
        if s is None:
            unknown += 1
            cell = "? (ls -l không thấy)"
        else:
            known += s
            largest = max(largest, s)
            cell = _human(s)
        note = f"  # {f.note}" if f.note else ""
        print(f"  {cell:>22}  {f.remote}{note}")

    print(f"{'-'*78}")
    print(f"  Tổng (đã biết size): {_human(known)}"
          + (f"  + {unknown} file chưa rõ size" if unknown else ""))
    print(f"  Số file             : {len(files)}")

    if unresolved:
        print(f"\n  ⚠ {len(unresolved)} thư mục chưa expand được (cần remote ls):")
        for d in unresolved:
            print(f"      {d.remote_dir}  — {d.note}")

    if blocked:
        print(f"\n  ⛔ {len(blocked)} file bị NEVER chặn (đã loại):")
        seen: set[str] = set()
        for f, why in blocked:
            if why in seen:
                continue
            seen.add(why)
            print(f"      {why}")
    return known, unknown, largest


def _execute(tiers: str, max_gb: float, do_fetch: bool) -> None:
    """Ruột của cả ``plan`` và ``main`` — tránh hai bản logic lệch nhau."""
    ents = resolve_many(tiers)
    if not ents:
        if any(t.strip() in {"e0", "e1"} for t in tiers.split(",")):
            raise SystemExit(
                f"[plan] tier {tiers!r} cố ý KHÔNG tải gì từ Baidu:\n"
                "  e0 = code/tooling/docs · e1 = demo AEROBLADE (VAE ~1 GB kéo từ HF Hub)\n"
                "  Không có gì để làm."
            )
        raise SystemExit(
            f"[plan] tier {tiers!r} không có file nào. Các tier: {sorted(TIERS)}"
        )

    _baidu_login()
    # DirSpec (tier `gap`) cần hỏi remote để biết tên file thật — không đoán.
    files, unresolved, blocked = expand(ents, lister=lambda d: list(_ls_sizes(d)))
    sizes = _size_map(files)
    known, unknown, largest = _report(files, sizes, blocked, unresolved)

    # Đếm chính xác số file bỏ qua: chỉ File (lister=None để không gọi Baidu
    # thêm lần nào). Bản đầu đếm cả DirSpec nên báo sai 1 đơn vị.
    all_files, _, _ = expand(resolve_many(",".join(TIERS)))
    chosen = {f.remote for f in files}
    skipped_n = len({f.remote for f in all_files} - chosen)
    print(f"\n  Bỏ qua {skipped_n} file kê khai trong tier khác → KHÔNG tải.")

    # ---- guard dung lượng ----
    # KHÔNG chỉ so `known`: nếu `ls -l` không parse được size thì known là cận
    # DƯỚI, và guard sẽ thả qua một đợt tải lớn hơn trần — tức là guard vô
    # hiệu đúng lúc cần nó nhất. Ước lượng bảo thủ: mỗi file chưa rõ size được
    # tính bằng file LỚN NHẤT đã biết.
    est = known + unknown * largest
    limit = max_gb * (1024**3)
    if unknown:
        print(f"  ⚠ {unknown} file chưa rõ size → ước lượng bảo thủ {_human(est)} "
              f"(mỗi file tính bằng {_human(largest)}; size đã biết: {_human(known)})")
    if est > limit:
        raise SystemExit(
            f"\n[GUARD] {_human(est)} > trần {max_gb:g} GB → từ chối tải.\n"
            f"  Chọn ít tier hơn: {sorted(TIERS)}\n"
            f"  Hoặc nâng trần có chủ đích: sửa max_gb trong src/dire_modal/tiers.yaml\n"
            f"  (trần này tồn tại để chặn tai nạn kiểu tải cả recons/ — 44 tar)"
        )

    if not do_fetch:
        print("\n[DRY-RUN] Chưa tải gì. Tải thật: thêm --mode fetch")
        return

    payloads = [
        {"remote": f.remote, "local": f.local, "expected_size": sizes.get(f.remote) or 0}
        for f in files
    ]
    print(f"\n[fetch] {len(payloads)} file · tối đa {POLICY.max_containers} container "
          f"song song · {POLICY.fetch_attempts} attempt/file")
    t0 = time.time()
    results = list(fetch_one.map(payloads, order_outputs=False))
    dt = time.time() - t0

    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errs = [r for r in results if r["status"] == "error"]

    _write_manifest(results)

    print(f"\n{'='*78}\nKẾT QUẢ  ({dt/60:.1f} phút)\n{'='*78}")
    print(f"  ok      : {len(ok)}")
    print(f"  skipped : {len(skipped)}  (đã có sẵn & hợp lệ)")
    print(f"  error   : {len(errs)}")
    for r in errs:
        print(f"      {r['remote']}\n          → {r['error']}")

    if errs:
        # KHÔNG raise: để người dùng chạy lại và fetch_one sẽ skip file đã ok.
        print("\nChạy lại đúng lệnh trên — file đã tải thành công sẽ được skip.")
        print("Nếu vẫn 0 byte: BDUSS hết hạn, hoặc giảm max_containers xuống 1.")


def _write_manifest(results: list[dict]) -> None:
    out = RAW / "_manifest.tsv"
    write_header = not out.exists()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with out.open("a", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        if write_header:
            w.writerow(["timestamp", "remote", "local", "size_bytes", "sha256", "status"])
        for r in results:
            w.writerow(
                [now, r.get("remote", ""), r["local"], r["size"], r.get("sha256", ""),
                 r["status"]]
            )
    vol_raw.commit()
    print(f"[manifest] → {out}")


# ============================================================================
# Modal functions
# ============================================================================
@app.function(
    image=cpu_image,
    secrets=[baidu_secret],
    volumes={"/raw": vol_raw},
    timeout=600,
)
def ls_remote(path: str = "/") -> list[str]:
    """Dò cây remote. ``modal run -m dire_modal.download::ls_remote --path /apps``"""
    _baidu_login()
    cp = _baidu(["ls", path], timeout=300, check=True)
    print(cp.stdout)
    return cp.stdout.splitlines()


@app.function(
    image=cpu_image,
    secrets=[baidu_secret],
    timeout=3600,
)
def audit(dirs: str = "") -> None:
    """Đối chiếu cây remote thật ↔ manifest. **Không tải gì.**

    Việc này phải chạy TRƯỚC khi tin bất kỳ con số GB nào, vì nó là thứ phát
    hiện ra ba lớp sai: (a) manifest kê file không tồn tại trên Baidu, (b) file
    có thật nhưng không tier nào kê — ví dụ ``dire/test/lsun_bedroom/imagenet/``
    mà ``TIER_A`` cũ bỏ sót, (c) file rơi vào NEVER và dung lượng tiết kiệm được.

    Args:
        dirs: thư mục cần quét, cách nhau dấu phẩy. Rỗng ⇒ dùng ``AUDIT_DIRS``.
            Ví dụ ``--dirs recons/train/lsun_bedroom,images/test/celebahq``.

    Đã chứng minh giá trị: audit 2026-10-21 bắt được tier ``e4`` trỏ vào đường
    dẫn 2 tầng KHÔNG tồn tại. Nếu chạy ``--mode fetch`` trước thì cả 4 file đã
    fail và tốn một vòng chờ Baidu.
    """
    _baidu_login()
    expected = _all_expected()
    targets = [d.strip() for d in dirs.split(",") if d.strip()] or AUDIT_DIRS
    if dirs:
        print(f"[audit] quét {len(targets)} thư mục từ --dirs (bỏ qua AUDIT_DIRS)")

    tot_never = 0
    tot_unknown: list[str] = []
    tot_missing: list[str] = []
    seen_remote: set[str] = set()

    for d in targets:
        listing = _ls_sizes(d)
        if not listing:
            print(f"\n[{d}]  ⚠ không liệt kê được (thư mục không tồn tại?)")
            continue
        print(f"\n[{d}]  {len(listing)} file · {_human(sum(listing.values()))}")
        for name, size in sorted(listing.items()):
            rel = f"{d}/{name}"
            seen_remote.add(rel)
            if is_never(rel):
                tot_never += size
                print(f"    ⛔ {_human(size):>10}  {name}   (NEVER)")
            elif rel in expected:
                print(f"    ✅ {_human(size):>10}  {name}   tier={','.join(expected[rel])}")
            else:
                tot_unknown.append(rel)
                print(f"    ❓ {_human(size):>10}  {name}   KHÔNG có trong manifest")

    # Chỉ kết luận "kê nhưng không có" cho những thư mục THẬT SỰ đã quét.
    # Khi người dùng thu hẹp bằng --dirs, mọi entry ngoài phạm vi đó sẽ vắng
    # mặt trong seen_remote — nếu vẫn báo "đường dẫn sai" thì sẽ xui người ta
    # đi sửa những path vốn đúng (bug này do chính tham số --dirs sinh ra).
    scanned = set(targets)
    for rel in expected:
        if rel in seen_remote:
            continue
        if dirs and str(Path(rel).parent) not in scanned:
            continue
        tot_missing.append(rel)

    print(f"\n{'='*78}\nTÓM TẮT AUDIT\n{'='*78}")
    print(f"  dung lượng NEVER chặn được : {_human(tot_never)}"
          + ("  (chỉ trong phạm vi --dirs)" if dirs else ""))
    print(f"  file remote không ai kê    : {len(tot_unknown)}")
    for r in tot_unknown:
        print(f"      {r}")
    print(f"  manifest kê nhưng KHÔNG có : {len(tot_missing)}"
          + (f"  (chỉ xét {len(scanned)} thư mục đã quét)" if dirs else ""))
    for r in tot_missing:
        print(f"      ⚠ {r}  ← đường dẫn sai, sửa trong manifest.py")

    if tot_missing:
        print("\n→ Có entry kê sai. `fetch_one` sẽ báo lỗi cho từng file đó;")
        print("  đối chiếu tên thật ở bảng trên rồi sửa manifest.py.")


@app.function(
    image=cpu_image,
    secrets=[baidu_secret],
    volumes={"/raw": vol_raw},
    # Tải HAI file TUẦN TỰ ⇒ timeout phải gấp đôi của một file. Bản đầu dùng
    # timeout của MỘT file nên có thể bị Modal kill giữa file thứ hai và báo
    # lỗi khó hiểu hơn chính nguyên nhân.
    timeout=2 * POLICY.fetch_attempts * POLICY.subprocess_timeout + 1800,
    cpu=1.0,
)
def probe_pair(
    a: str = "dire/test/lsun_bedroom/lsun_bedroom/real.tar.gz",
    b: str = "dire/test/lsun_bedroom/imagenet/real.tar.gz",
) -> None:
    """Chốt ngữ nghĩa ``dire/test/<A>/<B>/`` BẰNG BẰNG CHỨNG, không đoán.

    Vấn đề: hai file ``real.tar.gz`` này có size **bằng nhau y hệt** (26.6 MB)
    dù nằm dưới hai nhánh ``<B>`` khác nhau, trong khi hai nhánh dưới
    ``dire/test/imagenet/`` lại lệch nhau (53.3 vs 177.4 MB). Nếu ``<B>`` là
    "test domain" thì ảnh real phải khác nhau giữa LSUN và ImageNet. Không giả
    thuyết nào khớp cả bốn số liệu, và ``vendor/utils/datasets.py`` KHÔNG được
    vendor nên không xác nhận từ code được.

    Tải cả hai (~53 MB) rồi so sha256 + đếm ảnh:

    * sha256 **trùng** → cùng một tập ảnh ⇒ ``<B>`` KHÔNG phải test domain theo
      nghĩa tập ảnh khác nhau ⇒ chưa được diễn giải nhánh imagenet là
      cross-domain.
    * sha256 **khác**, số ảnh bằng nhau → đọc ``<B>`` là test domain: hợp lý.
    * số ảnh **lệch** → cỡ mẫu không đồng đều; mean AP gộp chung các nhánh sẽ
      lệch, phải báo cáo riêng từng nhánh.

    Việc TẢI 10 file ``dire/test/lsun_bedroom/imagenet/`` không phụ thuộc kết
    quả này; việc GHI SỐ vào bảng eval thì có. Chạy trước khi báo cáo.
    """
    res: dict[str, dict] = {}
    for p in (a, b):
        # Gọi thân thật, KHÔNG gọi `fetch_one` — nó là modal.Function, gọi trực
        # tiếp từ trong container sẽ ra "TypeError: 'Function' object is not
        # callable" (chính lỗi làm lệnh này chết ngay lần chạy đầu tiên).
        r = _fetch_one_impl({"remote": p, "local": p, "expected_size": 0})
        res[p] = r
        if r["status"] == "error":
            print(f"[probe] không tải được {p}: {r['error']} → không kết luận gì")
            return

    prof: dict[str, dict] = {}
    dig: dict[str, str] = {}
    print(f"\n{'='*78}\nPROBE: ngữ nghĩa dire/test/<A>/<B>/\n{'='*78}")
    for p in (a, b):
        path = RAW / p
        prof[p] = _tar_profile(path)
        dig[p] = res[p].get("sha256") or _sha256(path)
        print(f"\n  {p}")
        print(f"    size   : {_human(path.stat().st_size)}")
        print(f"    sha256 : {dig[p]}")
        cls = prof[p]["classes"]
        print(f"    ảnh    : {prof[p]['files']}  {cls or '(không thấy 0_real/1_fake)'}")
        if prof[p]["truncated"]:
            print("    ⚠ dừng đếm sớm (tar lớn hơn cap)")

    na, nb = prof[a]["files"], prof[b]["files"]
    print(f"\n{'-'*78}")
    if dig[a] == dig[b]:
        print("  KẾT LUẬN: sha256 TRÙNG ⇒ hai đường dẫn chứa CÙNG một tập ảnh.")
        print("  ⇒ <B> KHÔNG phải 'test domain' theo nghĩa tập ảnh khác nhau.")
        print("  ⇒ CHƯA được diễn giải dire/test/lsun_bedroom/imagenet/ là cross-domain.")
        print("    Cần đọc bảng cross-dataset trong paper DIRE hoặc datasets.py gốc.")
    else:
        print("  KẾT LUẬN: sha256 KHÁC ⇒ hai tập ảnh thật sự khác nhau.")
        if na == nb:
            print(f"  Số ảnh bằng nhau ({na}) ⇒ đọc <B> là test domain: HỢP LÝ.")
            print("  ⇒ dire/test/lsun_bedroom/imagenet/ đo detector LSUN trên ảnh ImageNet,")
            print("    dùng chung checkpoint lsun_adm.pth của e2 → cross-domain không cần train.")
        else:
            print(f"  ⚠ Số ảnh LỆCH: {na} vs {nb} ⇒ cỡ mẫu không đồng đều.")
            print("    mean AP gộp chung các nhánh sẽ bị lệch — báo cáo riêng từng nhánh.")


@app.function(
    image=cpu_image,
    secrets=[baidu_secret],
    volumes={"/raw": vol_raw},
    timeout=1800,
)
def plan(tiers: str = POLICY.default_tiers, max_gb: float = POLICY.max_gb) -> None:
    """In bảng file + size thật từ Baidu. **Không tải gì.**

    ``modal run -m dire_modal.download::plan --tiers e2,e4``
    """
    _execute(tiers, max_gb, do_fetch=False)


@app.function(
    image=cpu_image,
    secrets=[baidu_secret],
    volumes={"/raw": vol_raw},
    # 3 attempt × timeout mỗi lệnh + dư 30 phút cho sha256 / gzip -t
    timeout=POLICY.fetch_attempts * POLICY.subprocess_timeout + 1800,
)
def main(
    tiers: str = POLICY.default_tiers,
    max_gb: float = POLICY.max_gb,
    mode: str = "plan",
) -> None:
    """Lên kế hoạch rồi (tuỳ chọn) tải.

    ``mode`` nhận ``plan`` (mặc định — KHÔNG tải) hoặc ``fetch`` / ``commit``
    (tải thật). Dùng chuỗi thay vì bool vì CLI của Modal parse ``--flag`` kiểu
    bool không nhất quán giữa các phiên bản; ``--mode fetch`` không mơ hồ.

    Ví dụ::

        modal run -m dire_modal.download::main --tiers e2,e4            # dry-run
        modal run -m dire_modal.download::main --tiers e2,e4 --mode fetch
    """
    _execute(tiers, max_gb, do_fetch=mode.strip().lower() in {"fetch", "commit", "yes"})


def _fetch_one_impl(f: dict) -> dict:
    """Thân thật của ``fetch_one`` — hàm Python THƯỜNG, gọi được trong container.

    Tách khỏi decorator vì ở phạm vi module, ``fetch_one`` sau ``@app.function``
    là một ``modal.Function``, **không gọi được như hàm thường**:

        TypeError: 'Function' object is not callable

    ``.map()`` cần đối tượng Function để chạy song song; code chạy TRONG một
    container (``probe_pair``) cần gọi thẳng. Cả hai dùng chung thân này để chỉ
    có MỘT bản logic — trước đây ``probe_pair`` gọi ``fetch_one(...)`` nên chưa
    từng chạy được lần nào.

    Tải MỘT file. Resumable, tự retry, và **không bao giờ raise**.
    """
    remote_rel = f["remote"]
    remote = _abs(remote_rel)
    local = RAW / f["local"]
    expected = int(f.get("expected_size") or 0)

    # --- đã có & hợp lệ → skip (idempotent, chạy lại bao nhiêu lần cũng được)
    if local.exists() and local.stat().st_size > 0:
        ok, why = _validate(local, expected)
        if ok:
            print(f"[SKIP] {remote_rel} đã có & hợp lệ")
            return {
                "remote": remote_rel, "local": str(local), "status": "skipped",
                "size": local.stat().st_size, "sha256": _sha256(local),
            }
        print(f"[REDO] {remote_rel}: {why}")
        local.unlink(missing_ok=True)

    local.parent.mkdir(parents=True, exist_ok=True)
    tmpdir = _tmpdir_for(remote_rel)  # riêng + ổn định ⇒ resume được
    tmpdir.mkdir(parents=True, exist_ok=True)
    last = "chưa thử"

    try:
        for attempt in range(1, POLICY.fetch_attempts + 1):
            try:
                _baidu_login()
                # KHÔNG capture output: để progress bar chảy ra log của Modal.
                # Chính nhờ log đó mà chẩn đoán ra throttle 0B/s.
                cp = _baidu(
                    ["d", "--saveto", str(tmpdir), remote],
                    timeout=POLICY.subprocess_timeout,
                    capture=False,
                )
                got = _find_downloaded(tmpdir, Path(remote).name)
                if got is None:
                    # Phân biệt hai ca, vì cách chữa khác nhau hoàn toàn:
                    # (a) file 0 byte ĐƯỢC TẠO — đúng chữ trong log thật
                    #     "下载结束, 数据总量: 0B" ⇒ throttle hoặc BDUSS hết hạn.
                    # (b) KHÔNG có file nào ⇒ thường là đường dẫn remote sai.
                    zeros = [
                        p for p in tmpdir.rglob("*") if p.is_file() and p.stat().st_size == 0
                    ]
                    if zeros:
                        last = (
                            "0 byte — BaiduPCS-Go exit 0 nhưng không tải gì "
                            "(throttle theo số phiên đồng thời, hoặc BDUSS/STOKEN hết hạn)"
                        )
                    else:
                        last = (
                            f"BaiduPCS-Go không tạo file nào (exit {cp.returncode}) — "
                            "đường dẫn remote sai, hoặc BDUSS/STOKEN hết hạn"
                        )
                else:
                    # Lỗi #1: /tmp là overlayfs của container, /raw là Volume mạng.
                    # Path.rename() gọi rename(2) ⇒ EXDEV errno 18, thất bại TUYỆT
                    # ĐỐI chứ không phải do mạng. shutil.move() = `mv` của shell:
                    # khác filesystem thì copy + unlink ⇒ qua được.
                    if local.exists():
                        local.unlink()
                    shutil.move(str(got), str(local))

                    ok, why = _validate(local, expected)
                    if ok:
                        size = local.stat().st_size
                        digest = _sha256(local)
                        vol_raw.commit()  # bắt buộc để file sống qua container khác
                        print(f"[OK] {remote_rel}  {_human(size)}  sha256={digest[:12]}…")
                        return {
                            "remote": remote_rel, "local": str(local), "status": "ok",
                            "size": size, "sha256": digest, "attempts": attempt,
                        }
                    # file hỏng đã move sang Volume → dọn để lần sau không skip nhầm
                    local.unlink(missing_ok=True)
                    last = why
            except subprocess.TimeoutExpired:
                last = f"BaiduPCS-Go treo quá {POLICY.subprocess_timeout}s (0B/s)"
            except subprocess.CalledProcessError as exc:
                last = f"BaiduPCS-Go exit {exc.returncode}"
            except Exception as exc:  # kể cả OSError/EXDEV nếu còn sót đường nào
                last = f"{type(exc).__name__}: {exc}"

            print(f"[RETRY {attempt}/{POLICY.fetch_attempts}] {remote_rel}: {last}")
            if attempt < POLICY.fetch_attempts:
                time.sleep(30 * attempt)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Lỗi #5: KHÔNG raise. Exception lọt ra .map() sẽ cancel mọi task anh em
    # ("Runner terminated" ×15 trong log) và vứt hết file đang tải dở.
    return {"remote": remote_rel, "local": str(local), "status": "error",
            "error": last, "size": 0}


@app.function(
    image=cpu_image,
    secrets=[baidu_secret],
    volumes={"/raw": vol_raw},
    # max_containers là fix của lỗi #3: bản cũ không đặt nên .map() bung ~26
    # container, tự làm Baidu throttle xuống 0B/s. retries=0 vì retry nằm trong
    # vòng lặp thủ công — exception lọt ra .map() sẽ huỷ cả đợt (lỗi #5).
    timeout=POLICY.fetch_attempts * POLICY.subprocess_timeout + 1800,
    cpu=0.25,
    max_containers=POLICY.max_containers,
    retries=modal.Retries(max_retries=0),
)
def fetch_one(f: dict) -> dict:
    """Một file trong MỘT container. Dùng qua ``.map()`` để chạy song song.

    KHÔNG gọi trực tiếp hàm này từ code khác: nó là ``modal.Function``, không
    phải hàm Python (``TypeError: 'Function' object is not callable``). Cần chạy
    inline trong container hiện tại thì gọi :func:`_fetch_one_impl`.
    """
    return _fetch_one_impl(f)


@app.function(
    image=cpu_image,
    volumes={"/raw": vol_raw},
    timeout=3600,
    cpu=1.0,
)
def prune(tiers: str = POLICY.default_tiers, mode: str = "dry") -> None:
    """Liệt kê file trong ``/raw`` KHÔNG thuộc tier đang chọn (tải thừa).

    ``mode=dry`` (mặc định) chỉ in. ``mode=apply`` mới xoá thật.

    ``modal run -m dire_modal.download::prune --tiers e2,e4``
    """
    keep = {f.local for f in expand(resolve_many(tiers))[0]}
    orphans: list[tuple[Path, int]] = []
    total = 0
    for p in sorted(RAW.rglob("*")):
        if not p.is_file() or p.name.startswith("_"):
            continue
        rel = str(p.relative_to(RAW))
        if rel in keep:
            continue
        size = p.stat().st_size
        total += size
        orphans.append((p, size))

    print(f"\n{'='*78}\nPRUNE — tier {tiers!r} giữ {len(keep)} file\n{'='*78}")
    if not orphans:
        print("  Không có file thừa.")
        return
    for p, size in orphans:
        why = is_never(str(p.relative_to(RAW)))
        tag = "  ⛔ NEVER" if why else ""
        print(f"  {_human(size):>10}  {p.relative_to(RAW)}{tag}")
    print(f"{'-'*78}\n  Tổng có thể thu hồi: {_human(total)} ({len(orphans)} file)")

    if mode.strip().lower() not in {"apply", "yes", "force"}:
        print("\n[DRY] Chưa xoá gì. Xoá thật: thêm --mode apply")
        print("  ⚠ file đã xoá phải tải lại từ Baidu (chậm, bị throttle).")
        return

    for p, _ in orphans:
        p.unlink(missing_ok=True)
    vol_raw.commit()
    print(f"\n[APPLIED] đã xoá {len(orphans)} file, thu hồi {_human(total)}")


@app.function(
    image=cpu_image,
    volumes={"/raw": vol_raw},
    timeout=2 * 3600,
    cpu=2.0,
)
def verify() -> None:
    """``gzip -t`` toàn bộ file trong ``/raw``.

    Không cần Baidu secret (chỉ đọc Volume) nên cố ý KHÔNG gọi ``_baidu_login``
    — bản nháp đầu có gọi và sẽ chết KeyError vì thiếu ``secrets=[baidu_secret]``.
    In đủ báo cáo rồi mới ``SystemExit(1)``, để một lần chạy thấy hết file hỏng.
    """
    bad: list[str] = []
    n = 0
    for p in sorted(RAW.rglob("*.tar.gz")):
        n += 1
        chk = subprocess.run(["gzip", "-t", str(p)], capture_output=True)
        tag = "OK  " if chk.returncode == 0 else "FAIL"
        print(f"[{tag}] {_human(p.stat().st_size):>10}  {p.relative_to(RAW)}")
        if chk.returncode != 0:
            bad.append(str(p.relative_to(RAW)))

    print(f"\nđã kiểm {n} file · hỏng {len(bad)}")
    if bad:
        for b in bad:
            print(f"  ⚠ {b}")
        print("\nTải lại: modal run -m dire_modal.download::main --mode fetch")
        print("  (fetch_one sẽ skip file hợp lệ, chỉ tải lại file hỏng)")
        raise SystemExit(1)
    print("All OK.")


@app.local_entrypoint()
def local_status() -> None:
    """Chạy LOCAL, không tốn credit: in tier + policy để soát trước khi vào Modal.

    ``modal run -m dire_modal.download::local_status``
    """
    print(f"BAIDU_ROOT       = {BAIDU_ROOT}")
    print(f"default_tiers    = {POLICY.default_tiers}")
    print(f"max_gb           = {POLICY.max_gb:g}")
    print(f"max_containers   = {POLICY.max_containers}   (Baidu throttle theo số phiên)")
    print(f"fetch_attempts   = {POLICY.fetch_attempts}")
    print(f"subproc timeout  = {POLICY.subprocess_timeout}s")
    print(f"\n{'TIER':<8} {'FILE':>5}  MỤC ĐÍCH")
    print("-" * 78)
    for tier in TIERS:
        ents = TIERS[tier]
        # Đếm File thật và DirSpec riêng: tier `e4x` chỉ có 1 entry nhưng nó
        # đại diện 17 file, in "1" sẽ gây hiểu nhầm là tier chỉ có 1 file.
        n_file = sum(1 for e in ents if isinstance(e, File))
        n_dir = sum(1 for e in ents if isinstance(e, DirSpec))
        cell = f"{n_file}" + (f"+{n_dir}?" if n_dir else "")
        note = (POLICY.notes or {}).get(tier, "")
        print(f"{tier:<8} {cell:>5}  {note}")
    print("-" * 78)
    for t in POLICY.default_tiers.split(","):
        t = t.strip()
        if t in TIERS:
            files, unresolved, _ = expand(TIERS[t])
            print(f"  {t}: {len(files)} file"
                  + (f" + {len(unresolved)} thư mục cần expand" if unresolved else ""))
    print("\nNEVER (chặn cứng):")
    for pat, why in NEVER:
        print(f"  {pat!r:<28} {why}")
