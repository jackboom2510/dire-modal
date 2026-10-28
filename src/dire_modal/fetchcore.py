"""Lõi DÙNG CHUNG giữa đường tải trên Modal (``download.py``) và trên máy local
(``local_fetch.py``) — không import ``modal``, không đụng Volume.

Vì sao tách ra
--------------
Cùng một câu hỏi "file này đã tải đúng chưa?" phải có ĐÚNG MỘT câu trả lời, dù
byte nằm trong Modal Volume hay trong ``D:/dire-raw``. Cả một buổi sửa lỗi của
dự án này là hệ quả của việc hai bên trả lời khác nhau:

* ``ls -l`` in size ĐÃ LÀM TRÒN 2 chữ số thập phân ⇒ đòi khớp từng byte là bắt
  mọi file đúng phải "tải lại" (sự cố 2026-10-25, xem :class:`SizeEst`).
* ``probe_pair`` gọi ``fetch_one`` như một hàm thường ⇒ ``TypeError`` vì nó là
  ``modal.Function``.
* ``gzip -t`` phải được xét TRƯỚC size, vì nó là bằng chứng về nội dung.

Module này chỉ chứa hàm THUẦN (đọc file, chạy ``gzip -t``, parse output của
BaiduPCS-Go). Không có gì phụ thuộc Modal, nên ``python -m
dire_modal.local_fetch`` chạy được trên máy cá nhân mà không cần token.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import NamedTuple

_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB)\b", re.I)
_KNOWN_SUFFIX = (".tar.gz", ".pth", ".pt")
#: File KHÔNG phải archive: không kiểm được bằng `gzip -t`, nên bằng chứng duy
#: nhất là size + sha256 đối chiếu manifest (xem :func:`check_artifact`).
ARTIFACT_SUFFIX = (".pth", ".pt")


# ============================================================================
# Helper thuần (không phải Modal function)
# ============================================================================
def _sha256(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def _md5(path: Path, buf: int = 1 << 20) -> str:
    """md5 của file — chỉ dùng làm ý kiến thứ hai khi ``ls -l`` không đủ tin."""
    h = hashlib.md5()  # noqa: S324 — không phải bảo mật, chỉ đối chiếu nguồn
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


class SizeEst(NamedTuple):
    """Size đọc từ ``ls -l`` + SAI SỐ của nó.

    ``ls -l`` của BaiduPCS-Go in size người-đọc-được, làm tròn 2 chữ số thập
    phân: ``119.22MB``. Quy đổi ngược ra byte được ``125011230``, còn file thật
    là ``125015131`` — lệch 3 901 B, tức 0.003 % nhưng **không bằng nhau**.

    Đòi khớp từng byte với con số đó là bắt mọi file tải ĐÚNG phải báo "tải dở
    dang": đúng thứ xảy ra trong lần chạy thật 2026-10-25 (iddpm, pndm, ldm
    tải xong 100 % rồi bị loại). Vì vậy size từ ``ls -l`` chỉ là **ước lượng
    có sai số**, không bao giờ là bằng chứng về nội dung.
    """

    nbytes: int
    slack: int = 0


def _bytes_of(v: "SizeEst | int | None") -> int:
    """Bytes của một ước lượng (chấp nhận cả ``int`` trần cho code/test cũ)."""
    if v is None:
        return 0
    return v.nbytes if isinstance(v, SizeEst) else int(v)


def _slack_of(v: "SizeEst | int | None") -> int:
    """Sai số cho phép; ``int`` trần ⇒ 0 (khớp tuyệt đối, như trước)."""
    return v.slack if isinstance(v, SizeEst) else 0


# Git Bash / MSYS2 "chuyển đổi tham số": mọi argv bắt đầu bằng "/" bị đổi thành
# đường dẫn Windows TRƯỚC khi tiến trình Python nhìn thấy. Trong container Linux
# thì không có ổ C:, nên lệnh chết ngay:
#
#     ./tools/tree.sh
#     [!] C:/Program Files/Git/raw không tồn tại. Volume mount: [...]
#
# (xảy ra thật 2026-10-25 trên Windows.) Sửa ở hai lớp: script export
# MSYS_NO_PATHCONV=1 (gốc rễ), và hàm dưới đây cho người gõ `modal run` trực tiếp.
_MSYS_PATH_RE = re.compile(r"^[A-Za-z]:[\\/].*?[\\/]Git[\\/](?P<rest>.*)$", re.I)


def _demangle(path: str) -> str:
    r"""``C:/Program Files/Git/raw/checkpoints`` → ``/raw/checkpoints``.

    Chỉ sửa khi chuỗi ĐÚNG là dạng ``<ổ>:\...\Git\...`` — dạng mà MSYS2 sinh
    ra. Đường dẫn Linux hợp lệ không bao giờ có tiền tố ổ đĩa, nên không có ca
    sửa nhầm.
    """
    if not path:
        return path
    m = _MSYS_PATH_RE.match(path.strip())
    if not m:
        return path
    return "/" + m.group("rest").replace("\\", "/").lstrip("/")


def _demangle_list(spec: str) -> str:
    """Như :func:`_demangle` nhưng cho danh sách phân cách bằng dấu phẩy."""
    if not spec:
        return spec
    return ",".join(_demangle(p) for p in spec.split(","))


def _parse_ls_estimates(text: str) -> dict[str, SizeEst]:
    """Parse ``BaiduPCS-Go ls -l`` → ``{basename: SizeEst}``.

    Tolerant có chủ đích: output của BaiduPCS-Go trộn tiếng Trung, cột có thể
    xê dịch theo phiên bản. Chỉ nhận dòng vừa có token kích thước vừa có token
    cuối trông như đường dẫn file đã biết đuôi; dòng lạ thì bỏ qua chứ không
    đoán bừa. Size không parse được ⇒ "unknown", không phải 0.

    Sai số = nửa đơn vị của chữ số thập phân cuối (làm tròn ⇒ ±0.5 ulp):
    ``119.22MB`` ⇒ ±0.005 MB = ±5 243 B. Token không có phần thập phân
    (``125011230B``) ⇒ sai số 0: đó là số nguyên chính xác.
    """
    out: dict[str, SizeEst] = {}
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
        num, unit = m.group(1), m.group(2).upper()
        mult = _UNITS[unit]
        dec = len(num.split(".", 1)[1]) if "." in num else 0
        slack = 0 if dec == 0 else int(round(0.5 * 10 ** (-dec) * mult)) + 1
        est = SizeEst(int(float(num) * mult), slack)
        prev = out.get(base)
        if prev is None or est.nbytes > prev.nbytes:
            out[base] = est
    return out


def _parse_ls_long(text: str) -> dict[str, int]:
    """``{basename: size_bytes}`` — bản chỉ-lấy-số, cho chỗ chỉ cần in/liệt kê.

    KHÔNG dùng hàm này để kiểm tính toàn vẹn: nó đã ném mất sai số. Dùng
    :func:`_parse_ls_estimates` ở mọi chỗ so size.
    """
    return {k: v.nbytes for k, v in _parse_ls_estimates(text).items()}


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
    path: Path, expected: int = 0, expect_archive: bool = True, slack: int = 0
) -> Verdict:
    """Kiểm tra file đã tải, trả về cả chẩn đoán và khả năng thử lại.

    Args:
        expected: size remote — nhưng là ước lượng từ ``ls -l``, tức ĐÃ LÀM
            TRÒN (xem :class:`SizeEst`).
        slack: sai số cho phép của ``expected``. Size lệch trong khoảng này là
            chuyện của phép làm tròn, không phải của lần tải.
        expect_archive: ``False`` ⇒ chỉ kiểm tra tồn tại/size, KHÔNG đòi phải là
            gzip/tar. Dùng cho ``probe_pair``: mục đích chính của nó là so
            **sha256** giữa hai file, việc đó đúng trên mọi chuỗi byte. Chặn nó
            lại chỉ vì file không giải nén được là bỏ mất câu trả lời đang cần.

    Thứ tự kiểm là có chủ đích: **bằng chứng về nội dung (gzip -t/tar) đứng
    trước** con số size đã làm tròn. Bản trước kiểm size trước, nên một file
    tải đủ 119 MB, gzip -t qua, vẫn bị loại và bị tải lại từ 0 B.
    """
    if not path.exists():
        return Verdict(False, "file không tồn tại")
    size = path.stat().st_size
    if size == 0:
        return Verdict(
            False,
            "0 byte — BaiduPCS-Go exit 0 nhưng không tải gì (throttle/BDUSS hết hạn)",
        )
    tol = max(0, int(slack))
    size_bad = bool(expected) and abs(size - expected) > tol
    size_note = ""
    if size_bad:
        size_note = (
            f"size {size} ≠ ls -l {expected}"
            + (f" (sai số cho phép ±{tol} B)" if tol else "")
            + f", lệch {size - expected:+d} B"
        )

    if not expect_archive or not path.name.endswith((".tar.gz", ".tgz")):
        if size_bad:
            return Verdict(False, f"{size_note} → tải dở dang", retryable=True)
        return Verdict(True, "ok (không kiểm tra định dạng)")

    chk = subprocess.run(["gzip", "-t", str(path)], capture_output=True)
    if chk.returncode == 0:
        if size_bad:
            # gzip -t đọc HẾT file và kiểm CRC + độ dài trailer, nên nó là
            # bằng chứng mạnh hơn hẳn một con số in ra từ `ls -l`.
            return Verdict(
                True,
                f"ok — gzip -t qua toàn bộ file dù {size_note} "
                "(ls -l chỉ in size làm tròn; tin gzip)",
                fmt="gzip",
            )
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
            f"KHÔNG phải gzip mà là **{fmt}** (size {size} B"
            f"{'' if size_bad else ', khớp remote'}) "
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
        " ⇒ nội dung không ổn định, thử lại có thể hữu ích."
        + (f" [{size_note}]" if size_bad else ""),
        retryable=True,
        fmt=fmt,
    )


def _validate(path: Path, expected: int = 0) -> tuple[bool, str]:
    """Kiểm tra file đã tải. Đây là chốt chặn lỗi #2: BaiduPCS-Go exit 0 dù 0 byte."""
    v = _validate_ex(path, expected)
    return v.ok, v.reason


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


def hold_into(path: Path, dest_dir: Path) -> Path:
    """Chuyển ``path`` vào ``dest_dir`` (không ghi đè) — **KHÔNG xoá byte nào**.

    Dùng chung cho đường Modal (``download._hold``) và đường local
    (``local_fetch``): cả hai đều phải giữ lại thứ mình *chưa chứng minh được là
    hỏng*, thay vì ``unlink()``.

    ``os.replace`` cùng filesystem ⇒ O(1), 0 byte đọc/ghi. Đây là lý do phải
    *chuyển* chứ không *copy*: copy vừa chậm hơn theo định nghĩa, vừa để lại file
    trong thư mục tải — mà BaiduPCS-Go thấy file đã có thì ``跳过`` mãi, tức là
    bế tắc vĩnh viễn.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    n = 1
    while dest.exists():
        dest = dest_dir / f"{path.name}.{n}"
        n += 1
    os.replace(path, dest)
    return dest


def staging_dir_for(root: Path, remote_rel: str) -> Path:
    """Thư mục staging RIÊNG và **ổn định** cho một đường dẫn remote.

    Ổn định là điều kiện của resume: BaiduPCS-Go lưu ``.BaiduPCS-Go-downloading``
    cạnh file đích, nên lần chạy sau phải hạ vào ĐÚNG thư mục đó.

    Dùng ``sha1`` chứ KHÔNG dùng ``hash()`` của Python: ``hash()`` đổi theo
    ``PYTHONHASHSEED`` mỗi tiến trình ⇒ mỗi lần chạy một thư mục khác ⇒ resume
    im lặng mất tác dụng và tải lại từ 0 B. Hash theo đường dẫn cũng bảo đảm
    ``train/adm.tar.gz`` và ``val/adm.tar.gz`` không trộn mảnh dở của nhau.
    """
    digest = hashlib.sha1(remote_rel.encode()).hexdigest()[:12]
    return root / f"baidu_{digest}"


# ---------------------------------------------------------------------------
# Checkpoint (.pth/.pt): `gzip -t` không áp dụng được
# ---------------------------------------------------------------------------
def parse_sha256_manifest(text: str) -> dict[tuple[str, str], tuple[int, str]]:
    """Parse ``_manifest_sha256.tsv`` → ``{(source, path): (size, sha256)}``.

    Cùng định dạng với ``archive::make_manifest`` và ``local_fetch --mode
    manifest``: ``source<TAB>path<TAB>size_bytes<TAB>sha256``. Header và dòng
    hỏng bị bỏ qua chứ không raise: manifest là *bằng chứng thêm*, thiếu nó thì
    vẫn kiểm được size — ném lỗi ở đây sẽ biến một file manifest cũ thành
    "không verify được gì".
    """
    rows: dict[tuple[str, str], tuple[int, str]] = {}
    for line in text.splitlines():
        line = line.rstrip("\r")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        source, rel, size, digest = (x.strip() for x in parts[:4])
        if source == "source" or not rel:      # header
            continue
        try:
            nbytes = int(size)
        except ValueError:
            continue
        rows[(source, rel.strip("/"))] = (nbytes, digest.lower())
    return rows


def check_artifact(path: Path, expected: "tuple[int, str] | None" = None,
                   digest_of=None) -> Verdict:
    """Kiểm một file **không phải archive** (``.pth``/``.pt``).

    Vì sao cần riêng: ``download::verify`` trước đây chỉ quét ``*.tar.gz``, nên
    6 checkpoint ``.pth`` (269.4 MB/file) nằm ngoài mọi phép kiểm — CỔNG 7 của
    runbook ("ckpt tồn tại và sha256 khớp manifest") không có code. Một file
    ``.pth`` bị cụt vẫn "tồn tại", vẫn đúng tên, và chỉ nổ ở ``torch.load``
    giữa lúc eval đã tiêu tiền GPU.

    Args:
        expected: ``(size, sha256)`` từ manifest; ``None`` ⇒ chỉ kiểm tồn tại +
            size > 0, và **nói rõ** là không có tham chiếu (không được im lặng
            gọi đó là "khớp manifest").
        digest_of: hàm băm, cho phép dùng cache ``(size, mtime_ns)`` của caller.
    """
    if not path.exists():
        return Verdict(False, "file không tồn tại")
    size = path.stat().st_size
    if size == 0:
        return Verdict(False, "0 byte")
    if expected is None:
        return Verdict(
            True,
            f"ok ({_human(size)}) — KHÔNG có tham chiếu trong manifest nên chưa "
            "chứng minh được nội dung",
        )
    want_size, want_sha = expected
    if want_size and size != want_size:
        return Verdict(
            False,
            f"size {size} ≠ manifest {want_size} (lệch {size - want_size:+d} B) "
            "→ tải lại",
            retryable=True,
        )
    if want_sha:
        got = digest_of(path) if callable(digest_of) else _sha256(path)
        if got != want_sha:
            return Verdict(
                False,
                f"sha256 {got[:12]}… ≠ manifest {want_sha[:12]}… → nội dung khác "
                "(cùng size, khác byte)",
                retryable=True,
            )
        return Verdict(True, f"ok — sha256 khớp manifest ({_human(size)})",
                       fmt="sha256")
    return Verdict(True, f"ok — size khớp manifest ({_human(size)}), manifest "
                         "không có sha256", fmt="size")
