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
from collections import Counter
from pathlib import Path
from typing import NamedTuple

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


# Định dạng nhận ra qua magic bytes. Dùng khi file KHÔNG phải gzip để nói rõ
# nó thực sự là gì, thay vì chỉ báo "gzip -t FAIL".
# (offset, bytes, tên)
_MAGIC: list[tuple[int, bytes, str]] = [
    (0, b"\x1f\x8b", "gzip"),
    (0, b"PK\x03\x04", "zip"),
    (0, b"PK\x05\x06", "zip (rỗng)"),
    (0, b"Rar!\x1a\x07", "rar"),
    (0, b"7z\xbc\xaf\x27\x1c", "7z"),
    (0, b"BZh", "bzip2"),
    (0, b"\xfd7zXZ\x00", "xz"),
    (0, b"\x28\xb5\x2f\xfd", "zstd"),
    (0, b"\x04\x22\x4d\x18", "lz4"),
    (257, b"ustar", "tar (không nén)"),
]


def _diagnose(path: Path) -> str:
    """Đoán ĐỊNH DẠNG THẬT của file từ magic bytes. Không bao giờ raise.

    Ra đời từ một lần chạy thật: tải đủ đúng **27 863 040 byte** (khớp 100% con
    số Baidu báo) nhưng ``gzip -t`` fail. Nếu file đúng số byte mà vẫn không
    giải nén được thì không phải lỗi mạng — phải xem **nội dung** là gì. Thông
    báo "gzip -t FAIL — file cụt/hỏng" che mất sự thật đó.
    """
    try:
        with path.open("rb") as fh:
            head = fh.read(512)
    except OSError as exc:
        return f"không đọc được ({exc})"

    for off, sig, name in _MAGIC:
        if head[off : off + len(sig)] == sig:
            return name

    low = head[:200].lstrip().lower()
    if low.startswith((b"<!doctype", b"<html", b"<?xml")) or b"<html" in low:
        return "HTML (trang lỗi/đăng nhập của Baidu, KHÔNG phải dữ liệu)"
    if low.startswith(b"{") or low.startswith(b"["):
        return "JSON (phản hồi lỗi API, KHÔNG phải dữ liệu)"
    if not head:
        return "rỗng"
    printable = sum(1 for b in head if 32 <= b < 127 or b in (9, 10, 13))
    if printable > len(head) * 0.95:
        return f"text ({head[:40]!r})"
    return "không rõ (byte đầu: " + head[:8].hex(" ") + ")"


# Định dạng "hợp lệ nhưng sai tên": nếu gặp, việc tải lại CHẮC CHẮN vô ích vì
# nội dung đã đúng số byte và có cấu trúc của một archive thật.
# KHÔNG có "tar (không nén)" trong đây: tar không nén vẫn là dữ liệu DÙNG ĐƯỢC,
# xem ``_tar_ok`` — nó được chấp nhận, không bị coi là lỗi.
_KNOWN_ARCHIVES = {"zip", "zip (rỗng)", "rar", "7z", "bzip2", "xz", "zstd", "lz4"}


class Verdict(NamedTuple):
    """Kết quả kiểm tra + có nên thử lại hay không.

    ``retryable=False`` nghĩa là lỗi TẤT ĐỊNH: tải lại sẽ ra đúng những byte đó,
    chỉ tốn thêm hàng chục phút. Log thật cho thấy 3 attempt × 6 phút cho cùng
    một file hỏng nguồn = 19 phút vô ích.
    """

    ok: bool
    reason: str
    retryable: bool = True
    fmt: str = ""


def _validate_ex(
    path: Path, expected: int = 0, expect_archive: bool = True
) -> Verdict:
    """Kiểm tra file đã tải, trả về cả chẩn đoán và khả năng thử lại.

    Args:
        expect_archive: ``False`` ⇒ chỉ kiểm tra tồn tại/size, KHÔNG đòi phải là
            gzip/tar. Dùng cho ``probe_pair``: mục đích chính của nó là so
            **sha256** giữa hai file, việc đó đúng trên mọi chuỗi byte. Chặn nó
            lại chỉ vì file không giải nén được là bỏ mất câu trả lời đang cần.
    """
    if not path.exists():
        return Verdict(False, "file không tồn tại")
    size = path.stat().st_size
    if size == 0:
        return Verdict(
            False,
            "0 byte — BaiduPCS-Go exit 0 nhưng không tải gì (throttle/BDUSS hết hạn)",
        )
    if expected and size != expected:
        # Lệch size là lỗi mạng/throttle ⇒ thử lại có ý nghĩa.
        return Verdict(
            False, f"size {size} ≠ remote {expected} → tải dở dang", retryable=True
        )
    if not expect_archive or not path.name.endswith((".tar.gz", ".tgz")):
        return Verdict(True, "ok (không kiểm tra định dạng)")

    chk = subprocess.run(["gzip", "-t", str(path)], capture_output=True)
    if chk.returncode == 0:
        return Verdict(True, "ok", fmt="gzip")

    # --- gzip fail: chẩn đoán thay vì chỉ nói "hỏng" -----------------------
    fmt = _diagnose(path)
    detail = (chk.stderr or b"").decode("utf-8", "replace").strip()[:160]
    if fmt == "tar (không nén)":
        # Tên `.tar.gz` nhưng nội dung là tar KHÔNG nén. Đây không phải lỗi:
        # tar đọc được thì dữ liệu vẫn nguyên. Chỉ hỏng khi tar bị cụt
        # (mất block kết thúc toàn số 0) — khi đó mới đáng tải lại.
        tar_ok, why = _tar_ok(path)
        if tar_ok:
            return Verdict(
                True,
                f"ok — **tar KHÔNG nén** (tên .tar.gz sai nhưng dữ liệu hợp lệ)"
                f"; {why}",
                fmt=fmt,
            )
        return Verdict(
            False,
            f"tar không nén nhưng đọc không được: {why} → tải lại có thể hữu ích",
            retryable=True,
            fmt=fmt,
        )
    if fmt in _KNOWN_ARCHIVES:
        return Verdict(
            False,
            f"KHÔNG phải gzip mà là **{fmt}** (size đúng {size} B, khớp remote) "
            f"⇒ file nguồn sai định dạng hoặc đặt sai tên. Tải lại vô ích.",
            retryable=False,
            fmt=fmt,
        )
    if fmt.startswith("HTML") or fmt.startswith("JSON"):
        return Verdict(
            False,
            f"nội dung là {fmt} ⇒ Baidu trả trang lỗi thay vì dữ liệu "
            "(BDUSS hết hạn hoặc link bị chặn). Tải lại vô ích.",
            retryable=False,
            fmt=fmt,
        )
    return Verdict(
        False,
        f"gzip -t FAIL — {fmt}; gzip nói: {detail or 'không rõ'}"
        " ⇒ nội dung không ổn định, thử lại có thể hữu ích.",
        retryable=True,
        fmt=fmt,
    )


def _validate(path: Path, expected: int = 0) -> tuple[bool, str]:
    """Kiểm tra file đã tải. Đây là chốt chặn lỗi #2: BaiduPCS-Go exit 0 dù 0 byte."""
    v = _validate_ex(path, expected)
    return v.ok, v.reason


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


def _open_tar(path: Path):
    """Mở tar **nén hay không nén** đều được.

    ``"r:gz"`` hardcode là bug thật: dữ liệu DIRE trên Baidu là **tar không
    nén** dù tên file là ``.tar.gz`` (xác nhận 2026-10-21 trên
    ``dire/test/lsun_bedroom/{lsun_bedroom,imagenet}/real.tar.gz``: sha256 khác
    nhau nhưng cả hai đều là ``tar (không nén)``). Với ``"r:gz"``, mọi file như
    vậy đều ``ReadError: not a gzip file`` — kể cả ``pack.py``, nơi lỗi đó
    không có guard và sẽ giết cả stage.
    """
    import tarfile

    return tarfile.open(path, "r:*")


def _tar_ok(path: Path, cap: int = 200_000) -> tuple[bool, str]:
    """Tar có đọc được và có KẾT THÚC đúng không (tức là không bị cụt)?

    Kiểm tra rẻ trước: tar hợp lệ luôn kết thúc bằng ≥2 block 512 B toàn số 0
    (EOF marker). File bị cụt giữa chừng mất marker này ⇒ phát hiện bằng 2 lần
    đọc nhỏ, không phải quét cả archive nhiều GB.
    """
    import tarfile

    try:
        size = path.stat().st_size
        if size >= 1024:
            with path.open("rb") as fh:
                fh.seek(size - 1024)
                has_eof = fh.read(1024) == b"\x00" * 1024
            if has_eof:
                with _open_tar(path) as t:
                    for mem in t:  # header đầu tiên phải parse được
                        return True, f"tar mở được, có EOF marker, member đầu: {mem.name!r}"
    except (OSError, tarfile.TarError, EOFError) as exc:
        return False, f"tar không đọc được ({type(exc).__name__}: {exc})"

    # Không thấy EOF marker: có thể tar ghi kiểu append, hoặc bị cụt. Quét thử
    # tới `cap` member — quét hết mà không lỗi thì vẫn coi là dùng được.
    n = 0
    try:
        with _open_tar(path) as t:
            for _mem in t:
                n += 1
                if n > cap:
                    return True, f"không thấy EOF marker; đã quét {cap} member không lỗi"
    except (OSError, tarfile.TarError, EOFError) as exc:
        return False, f"tar cụt/hỏng ở member thứ {n} ({type(exc).__name__}: {exc})"
    return True, f"đọc hết {n} member không lỗi (không thấy EOF marker)"


def _tar_content(
    path: Path, cap: int = 200_000, byte_cap: int = 2 << 30
) -> dict:
    """Đọc tar (mọi kiểu nén) và trả về **nội dung từng member**.

    ``_tar_profile`` chỉ đếm; hàm này còn băm từng file để trả lời được câu
    "hai archive có cùng tập ảnh không" — câu mà sha256 của *container* không
    trả lời được (đổi mtime là sha256 đổi, size giữ nguyên).

    Returns:
        ``{files, classes, image_bytes, pairs: [(name, size, sha256)],
        truncated, error}``. ``error`` chỉ có khi không mở/đọc được.
    """
    n = 0
    classes: dict[str, int] = {}
    image_bytes = 0
    truncated = False
    pairs: list[tuple[str, int, str]] = []
    try:
        with _open_tar(path) as t:
            for mem in t:
                if not mem.isfile():
                    continue
                if n >= cap or image_bytes >= byte_cap:
                    truncated = True
                    break
                fh = t.extractfile(mem)
                if fh is None:
                    continue
                h = hashlib.sha256()
                size = 0
                while True:
                    chunk = fh.read(1 << 20)
                    if not chunk:
                        break
                    h.update(chunk)
                    size += len(chunk)
                n += 1
                image_bytes += size
                pairs.append((mem.name, size, h.hexdigest()))
                for part in mem.name.split("/"):
                    if part in ("0_real", "1_fake"):
                        classes[part] = classes.get(part, 0) + 1
    except Exception as exc:  # không phải tar, cụt, hoặc lỗi I/O
        return {
            "files": -1,
            "classes": {},
            "image_bytes": 0,
            "pairs": [],
            "truncated": truncated,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "files": n,
        "classes": classes,
        "image_bytes": image_bytes,
        "pairs": pairs,
        "truncated": truncated,
    }


def _tar_profile(path: Path, cap: int = 200_000) -> dict:
    """Đếm thành viên tar mà KHÔNG giải nén ra đĩa — rẻ, không tốn dung lượng.

    Trả về số file và số ảnh theo lớp ``0_real``/``1_fake`` (cấu trúc
    ImageFolder mà DIRE dùng). ``cap`` để một tar khổng lồ không làm treo hàm.
    Đọc được cả tar nén và không nén (``_open_tar``).
    """
    n = 0
    classes: dict[str, int] = {}
    truncated = False
    with _open_tar(path) as t:
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


def _content_compare(ca: dict, cb: dict) -> dict:
    """So hai archive theo **nội dung ảnh**, không theo byte của container.

    Vì sao cần: tar chứa metadata (mtime, owner, thứ tự member, padding). Cùng
    một tập ảnh, đóng gói lại với mtime khác ⇒ sha256 container KHÁC mà size
    giữ nguyên tuyệt đối. Nên ``sha256(container)`` không trả lời được câu
    "hai đường dẫn có cùng tập ảnh không" — phải so từng member.

    Returns:
        ``{status, na, nb, identical, overlap, frac}`` với ``status`` là một
        trong ``identical`` (trùng cả tên và byte), ``same_images`` (cùng byte
        ảnh nhưng khác tên/thứ tự), ``partial``, ``disjoint``, ``empty``,
        ``unreadable``.
    """
    if "error" in ca or "error" in cb:
        return {
            "status": "unreadable",
            "reason": ca.get("error") or cb.get("error"),
            "na": ca.get("files", -1),
            "nb": cb.get("files", -1),
        }

    na, nb = ca["files"], cb["files"]
    a_pairs, b_pairs = ca["pairs"], cb["pairs"]
    if na <= 0 or nb <= 0:
        return {"status": "empty", "na": na, "nb": nb, "identical": 0, "overlap": 0}

    ia = {name: (size, sha) for name, size, sha in a_pairs}
    ib = {name: (size, sha) for name, size, sha in b_pairs}
    identical = sum(1 for n in set(ia) & set(ib) if ia[n] == ib[n])

    # So theo đa tập sha để bắt ca "cùng ảnh nhưng khác tên/thứ tự".
    ca_cnt = Counter(sha for _, _, sha in a_pairs)
    cb_cnt = Counter(sha for _, _, sha in b_pairs)
    overlap = sum((ca_cnt & cb_cnt).values())

    if identical == na == nb:
        status = "identical"
    elif overlap == na == nb:
        status = "same_images"
    elif overlap == 0:
        status = "disjoint"
    else:
        status = "partial"
    return {
        "status": status,
        "na": na,
        "nb": nb,
        "identical": identical,
        "overlap": overlap,
        "frac": min(overlap / na, overlap / nb),
        "truncated": ca.get("truncated") or cb.get("truncated"),
    }


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
    # Hỏi size thật trước: có nó thì mới nói được câu quyết định "đủ đúng số
    # byte mà vẫn không giải nén được ⇒ lỗi nguồn". Không có thì chỉ biết "hỏng".
    want = {p: _ls_sizes(str(Path(p).parent)).get(Path(p).name, 0) for p in (a, b)}
    for p, sz in want.items():
        print(f"[probe] remote {p} = {_human(sz) if sz else '? (ls không thấy)'}")

    res: dict[str, dict] = {}
    for p in (a, b):
        # Gọi thân thật, KHÔNG gọi `fetch_one` — nó là modal.Function, gọi trực
        # tiếp từ trong container sẽ ra "TypeError: 'Function' object is not
        # callable" (chính lỗi làm lệnh này chết ngay lần chạy đầu tiên).
        #
        # expect_archive=False: mục đích chính của probe là so **sha256**, việc
        # đó đúng trên mọi chuỗi byte. Đòi file phải giải nén được sẽ chặn đúng
        # câu trả lời đang cần (đã xảy ra thật: file tải đủ 27 863 040 B nhưng
        # gzip -t fail ⇒ 3 attempt × 6 phút rồi bỏ cuộc, không so được gì).
        r = _fetch_one_impl(
            {
                "remote": p,
                "local": p,
                "expected_size": want[p],
                "expect_archive": False,
            }
        )
        res[p] = r
        if r["status"] == "error":
            print(f"[probe] không tải được {p}: {r['error']} → không kết luận gì")
            return

    content: dict[str, dict] = {}
    dig: dict[str, str] = {}
    print(f"\n{'='*78}\nPROBE: ngữ nghĩa dire/test/<A>/<B>/\n{'='*78}")
    for p in (a, b):
        path = RAW / p
        content[p] = _tar_content(path)
        dig[p] = res[p].get("sha256") or _sha256(path)
        print(f"\n  {p}")
        print(f"    size   : {_human(path.stat().st_size)}")
        print(f"    sha256 : {dig[p]}  ← của CONTAINER, không phải của ảnh")
        print(f"    định dạng: {_diagnose(path)}")
        c = content[p]
        if "error" in c:
            print(f"    ảnh    : KHÔNG đọc được — {c['error']}")
        else:
            cls = c["classes"]
            print(
                f"    ảnh    : {c['files']}  {cls or '(không thấy 0_real/1_fake)'}"
                f"  · {_human(c['image_bytes'])} dữ liệu ảnh"
            )
        if c.get("truncated"):
            print("    ⚠ dừng đọc sớm (archive lớn hơn cap) — số dưới đây là một phần")

    cmp = _content_compare(content[a], content[b])
    print(f"\n{'-'*78}")
    same_bytes = dig[a] == dig[b]
    print(f"  sha256 container : {'TRÙNG' if same_bytes else 'KHÁC'}")

    if same_bytes:
        # Byte y hệt thì chắc chắn cùng nội dung, khỏi cần đọc.
        print("  KẾT LUẬN: sha256 TRÙNG ⇒ hai đường dẫn chứa CÙNG một tập ảnh.")
        print("  ⇒ <B> KHÔNG phải 'test domain' theo nghĩa tập ảnh khác nhau.")
        print("  ⇒ CHƯA được diễn giải dire/test/lsun_bedroom/imagenet/ là cross-domain.")
        return

    st = cmp["status"]
    if st in ("unreadable", "empty"):
        why = cmp.get("reason") or "archive rỗng"
        print(f"  nội dung ảnh     : KHÔNG so được ({why})")
        print("\n  KẾT LUẬN: CHƯA KẾT LUẬN ĐƯỢC.")
        print("  ⇒ sha256 container KHÁC **không** chứng minh tập ảnh khác nhau:")
        print("    tar chứa metadata (mtime/thứ tự/padding) — đổi mtime là sha256 đổi,")
        print("    size giữ nguyên. Cần archive đọc được để so từng ảnh.")
        return

    na, nb = cmp["na"], cmp["nb"]
    if st == "identical":
        print(f"  nội dung ảnh     : CÙNG HOÀN TOÀN ({cmp['identical']}/{na} ảnh trùng tên + trùng byte)")
        print("\n  KẾT LUẬN: hai đường dẫn là CÙNG một tập ảnh, chỉ khác đóng gói.")
        print("  ⇒ sha256 container KHÁC nhưng nội dung GIỐNG ⇒ sha256 container là")
        print("    bằng chứng YẾU, đừng dùng nó để kết luận cross-domain.")
        print("  ⇒ <B> KHÔNG phải test domain khác ảnh.")
    elif st == "same_images":
        print(f"  nội dung ảnh     : CÙNG ảnh, KHÁC tên/thứ tự ({cmp['overlap']}/{na} ảnh trùng byte)")
        print("\n  KẾT LUẬN: cùng tập ảnh, chỉ đổi tên hoặc đổi thứ tự member.")
        print("  ⇒ Vẫn KHÔNG phải test domain khác ảnh.")
    elif st == "disjoint":
        print(f"  nội dung ảnh     : KHÁC HOÀN TOÀN (0/{na} ảnh trùng byte)")
        print("\n  KẾT LUẬN: hai tập ảnh thật sự KHÁC nhau.")
        if na == nb:
            print(f"  Số ảnh bằng nhau ({na}) ⇒ đọc <B> là test domain: HỢP LÝ.")
            print("  ⇒ dire/test/lsun_bedroom/imagenet/ đo detector LSUN trên ảnh ImageNet,")
            print("    dùng chung checkpoint lsun_adm.pth của e2 → cross-domain không cần train.")
        else:
            print(f"  ⚠ Số ảnh LỆCH: {na} vs {nb} ⇒ cỡ mẫu không đồng đều.")
            print("    mean AP gộp chung các nhánh sẽ bị lệch — báo cáo riêng từng nhánh.")
    else:
        print(
            f"  nội dung ảnh     : LỆCH MỘT PHẦN ({cmp['overlap']}/{na} và "
            f"{cmp['overlap']}/{nb} trùng byte, {cmp['frac']:.0%})"
        )
        print("\n  KẾT LUẬN: hai tập ảnh GIAO NHAU nhưng không trùng khít.")
        print("  ⇒ Không phải 'cùng tập'; cũng không phải hai tập rời nhau.")
        print("  ⇒ Báo cáo riêng từng nhánh, đừng gộp mean AP.")
    if cmp.get("truncated"):
        print("  ⚠ Một archive bị dừng đọc sớm ⇒ tỉ lệ trùng chỉ là cận dưới.")


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
    prev_sha = ""  # sha256 của lần tải trước, để phát hiện lỗi TẤT ĐỊNH
    expect_archive = bool(f.get("expect_archive", True))

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

                    # Tính verdict MỘT lần: gzip -t trên file 26 MB không rẻ.
                    v = _validate_ex(local, expected, expect_archive=expect_archive)
                    if v.ok:
                        size = local.stat().st_size
                        digest = _sha256(local)
                        vol_raw.commit()  # bắt buộc để file sống qua container khác
                        print(f"[OK] {remote_rel}  {_human(size)}  sha256={digest[:12]}…")
                        if v.fmt and v.fmt != "gzip":
                            # Dữ liệu dùng được nhưng tên đuôi nói dối — nêu ra
                            # để không ai đi tìm "file hỏng" ở chỗ khác.
                            print(f"       ↳ {v.fmt} — {v.reason}")
                        return {
                            "remote": remote_rel, "local": str(local), "status": "ok",
                            "size": size, "sha256": digest, "attempts": attempt,
                            "format": v.fmt or "gzip",
                        }

                    # --- thất bại: chia làm hai loại, vì cách chữa khác hẳn ---
                    digest = _sha256(local)
                    size_now = local.stat().st_size
                    # file hỏng đã move sang Volume → dọn để lần sau không skip nhầm
                    local.unlink(missing_ok=True)
                    last = v.reason

                    if not v.retryable:
                        # Lỗi TẤT ĐỊNH (file nguồn sai định dạng, hoặc Baidu trả
                        # trang lỗi). Log thật: 3 attempt × 6 phút cho cùng một
                        # file = 19 phút vô ích. Dừng ngay.
                        print(f"[STOP] {remote_rel}: {v.reason}")
                        return {
                            "remote": remote_rel, "local": str(local),
                            "status": "error", "error": v.reason,
                            "size": 0, "attempts": attempt,
                            "deterministic": True, "format": v.fmt,
                        }

                    # Cùng một chuỗi byte qua hai lần tải ⇒ tải lại cũng thế.
                    # Đây là bằng chứng mạnh nhất mà không cần hỏi thêm Baidu.
                    if digest == prev_sha:
                        msg = (
                            f"hai lần tải cho ra byte GIỐNG HỆT (sha256={digest[:12]}…, "
                            f"{_human(size_now)}) ⇒ lỗi tất định, dừng thử lại. {v.reason}"
                        )
                        print(f"[STOP] {remote_rel}: {msg}")
                        return {
                            "remote": remote_rel, "local": str(local),
                            "status": "error", "error": msg,
                            "size": 0, "attempts": attempt,
                            "deterministic": True, "format": v.fmt,
                        }
                    prev_sha = digest
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
    bad: list[tuple[str, str, bool]] = []
    n = 0
    for p in sorted(RAW.rglob("*.tar.gz")):
        n += 1
        v = _validate_ex(p)
        tag = "OK  " if v.ok else "FAIL"
        print(f"[{tag}] {_human(p.stat().st_size):>10}  {p.relative_to(RAW)}")
        if v.ok:
            continue
        # Chẩn đoán định dạng: "gzip FAIL" không nói được gì; "là rar/HTML"
        # mới cho biết phải làm gì.
        print(f"        ↳ {v.reason}")
        bad.append((str(p.relative_to(RAW)), v.reason, v.retryable))

    print(f"\nđã kiểm {n} file · hỏng {len(bad)}")
    if bad:
        det = [b for b in bad if not b[2]]
        ret = [b for b in bad if b[2]]
        for b, _why, _r in bad:
            print(f"  ⚠ {b}")
        if ret:
            print(
                f"\n{len(ret)} file lỗi TẠM THỜI (lệch size / nội dung không ổn định)"
                " → tải lại có ý nghĩa:"
                "\n  modal run -m dire_modal.download::main --mode fetch"
                "\n  (fetch_one skip file hợp lệ, chỉ tải lại file hỏng)"
            )
        if det:
            print(
                f"\n{len(det)} file lỗi TẤT ĐỊNH (nguồn sai định dạng / Baidu trả"
                " trang lỗi) → TẢI LẠI VÔ ÍCH, đừng mất thêm thời gian:"
            )
            for b in det:
                print(f"  ✗ {b}")
            print(
                "  Cách xử lý: đối chiếu tên/định dạng với bản gốc của tác giả,"
                "\n  hoặc để tier tương ứng ra khỏi phạm vi."
            )
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
