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
7. Size remote lấy từ ``ls -l`` **đã bị làm tròn** (2 chữ số thập phân MB), mà
   phép kiểm lại đòi khớp từng byte. Lần chạy 2026-10-25: ``iddpm.tar.gz`` tải
   đủ 125 015 131 B, ``ls -l`` in ``119.22MB`` → quy đổi ngược 125 011 230 B
   → báo "tải dở dang"; retry thì Baidu thấy file đã có nên bỏ qua, và code cũ
   gọi đó là "lỗi tất định" rồi **xoá**. Ba file × 26 phút bị vứt.
   → :class:`SizeEst` mang theo sai số; gzip -t/tar (bằng chứng nội dung) được
   xét TRƯỚC size; khi vẫn nghi ngờ thì hỏi ``meta`` (size chính xác + md5);
   file không qua kiểm được ``_hold()`` **chứ không xoá**.
"""

from __future__ import annotations

import csv
import datetime
import errno
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

import modal

from .common import (
    RAW_VOLUME,
    app,
    baidu_secret,
    cpu_image,
    vol_ckpt,
    vol_raw,
    vol_shards,
)
from .fetchcore import (  # noqa: F401 — re-export, xem `__all__` bên dưới
    _KNOWN_SUFFIX,
    SizeEst,
    Verdict,
    _bytes_of,
    _content_compare,
    _demangle,
    _demangle_list,
    _diagnose,
    _human,
    hold_into,
    staging_dir_for,
    _open_tar,
    _parse_ls_long,
    _tar_ok,
    _tar_profile,
    _validate,
    _md5,
    _parse_ls_estimates,
    _sha256,
    _slack_of,
    _tar_content,
    _validate_ex,
)
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

# `fetchcore` mới là nơi ĐỊNH NGHĨA các hàm thuần dùng chung với `local_fetch`;
# ở đây chỉ re-export để `download._validate_ex` (và test) vẫn là CÙNG một
# object, không phải bản sao — hai bản sao là đúng thứ đã sinh ra lỗi trước đây.
__all__ = [
    "SizeEst", "Verdict",
    "_bytes_of", "_content_compare", "_demangle", "_demangle_list", "_diagnose",
    "_human", "_md5", "_open_tar", "_parse_ls_estimates", "_parse_ls_long",
    "_sha256", "_slack_of", "_tar_content", "_tar_ok", "_tar_profile",
    "_validate", "_validate_ex",
]

# ---------------------------------------------------------------------------
# Ngân sách thời gian — lần chạy thật 2026-10-25 chết vì chỗ này
# ---------------------------------------------------------------------------
# `fetch_one` có timeout của Modal = `fetch_attempts × subprocess_timeout + 1800`
# = 3 × 5400 + 1800 = **18 000s**. Nhưng bên trong KHÔNG có deadline: cả 3
# attempt đều được phép chạy đủ 5400s, tức 16 200s, chỉ còn 1 800s để chứa
#   cold start + 3× `_baidu_login` + 2× `_sleep(30·n)` (90s)
#   + 3× join thread commit (105s) + 3× đọc CẢ file để `gzip -t`/`sha256`
#
# Log thật cho thấy đúng hệ quả: một task đốt 3 attempt rồi bị Modal giết ở
# 18 000s với `FunctionTimeoutError`, và vì `_execute` dùng `list(fn.map(...))`
# nên kết quả của các file ĐÃ XONG cũng mất sạch.
#
# `FunctionTimeoutError` KHÔNG bắt được trong hàm — nó giết container. Nên cách
# duy nhất đúng là không bao giờ để attempt vượt ngân sách còn lại.
FETCH_CALL_TIMEOUT_S = POLICY.fetch_attempts * POLICY.subprocess_timeout + 1800

# Chừa cho phần KHÔNG phải tải: cold start (pull image + mount Volume), 3×
# `_baidu_login`, 2× backoff (90s), 3× join thread commit (3×105s), 3× đọc cả
# file để `gzip -t`/`sha256`, và shutdown. Đo được ~1 000s trong lần chạy hỏng;
# 1 800 là mức có dư.
#
# Con số này KHÔNG làm mất attempt nào: 18 000 − 1 800 = 16 200 = 3 × 5 400, đúng
# `fetch_attempts × subprocess_timeout`. Attempt thứ 4 (nếu có) tự bị chặn.
_FETCH_RESERVE_S = 1800

# Dưới mức này thì một attempt không kịp làm gì — nhưng resume vẫn giữ tiến độ,
# nên dừng sạch rồi chạy lại tốt hơn là bị giết giữa chừng.
_MIN_ATTEMPT_S = 600

# Orchestrator `main` KHÔNG được chết trước các task con: nếu nó bị Modal giết
# thì không ai kịp in tổng kết. 24h là trần của Modal; thời gian chờ ở đây gần
# như miễn phí (function chỉ ngồi chờ `.map`).
ORCHESTRATOR_TIMEOUT_S = 24 * 3600

# Ngân sách của MỘT lần `main --mode fetch` khi nó tự chạy nhiều vòng.
#
# Vì sao cần vòng lặp: đo 2026-10-25 cho 76.6 KiB/s cho cả tài khoản. `e2` là
# 1.94 GB ⇒ **7.4 giờ**, `e2,e4` ⇒ 12.9 giờ. Trước đây mỗi lần gọi chỉ tải được
# `fetch_attempts × subprocess_timeout × 76.6 KiB/s` ≈ 400 MB, tức phải gõ lại
# lệnh ~9 lần. Vòng lặp biến 9 lần gõ thành 1 lần.
#
# 20h < timeout 24h của function: chừa 4h để in tổng kết và commit Volume.
ORCHESTRATOR_BUDGET_S = 20 * 3600

# Dưới mức này thì mở thêm một vòng không kịp tải xong file nhỏ nhất — dừng
# sạch, để lần chạy sau resume (staging đã nằm trong Volume).
_MIN_WAVE_S = 600

# Chặn trên số vòng. Mỗi vòng đã có chốt "không tiến bộ thì dừng", nhưng một
# chốt cứng thì rẻ và bảo vệ được cả trường hợp logic tiến bộ bị tính sai.
_MAX_WAVES = 40


def _attempt_timeout(started: float, now: float | None = None) -> float:
    """Số giây còn được phép dùng cho MỘT lệnh ``BaiduPCS-Go``.

    Trả ``0`` khi đã hết ngân sách ⇒ caller phải dừng, giữ mảnh dở trong
    staging, và trả ``status="partial"`` để lần chạy sau resume.
    """
    now = time.monotonic() if now is None else now
    left = FETCH_CALL_TIMEOUT_S - (now - started) - _FETCH_RESERVE_S
    if left < _MIN_ATTEMPT_S:
        return 0.0
    return min(float(POLICY.subprocess_timeout), float(left))

# Móc sleep để test thay được MÀ KHÔNG patch `time.sleep` toàn cục.
# (Test cũ gán `download.time.sleep = lambda _s: None` — vì `download.time` là
# module `time` của stdlib, việc đó vô hiệu hoá sleep của cả process, kể cả
# pytest, và làm test về thread/commit chạy sai một cách khó hiểu.)
_sleep = time.sleep

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
# Liệt kê Volume dưới dạng CÂY — `modal volume ls` chỉ đọc được MỘT tầng
# ============================================================================
# Vì sao phải có hàm riêng: CLI của Modal không có `--recursive`
# (`modal volume ls [OPTIONS] VOLUME_NAME [PATH]` chỉ nhận --json/-e/--profile),
# nên muốn ra cây thì phải tự đệ quy. Gọi CLI trong vòng lặp bash được, nhưng mỗi
# lần là một round-trip gRPC (~1 s) — cây `dire-raw` có hơn trăm thư mục.
# Chạy một container đọc mount FUSE thì rẻ hơn nhiều và đúng thứ tự.
#
# ⚠ `modal volume ls` còn một cái bẫy: khi stdout không phải TTY nó CHỈ in đường
# dẫn (legacy behavior trong `modal/cli/volume.py`), mất hết cột size. Đó là lý do
# hàm này tự in size thay vì để người dùng `| head`.


def _tree_lines(
    root: Path,
    max_depth: int = 0,
    show_hidden: bool = True,
) -> tuple[list[str], int, int, int]:
    """Dựng cây kiểu ``tree(1)``. Trả ``(dòng, n_dir, n_file, tổng_byte)``.

    Tách khỏi Modal để test được trên thư mục tạm.

    Thứ tự: **thư mục trước, file sau, mỗi loại theo alphabet** — để hai lần chạy
    cho cùng kết quả (``scandir`` không bảo đảm thứ tự). ``max_depth=0`` = vô hạn.
    """
    lines: list[str] = []
    counts = {"dir": 0, "file": 0, "bytes": 0}

    def walk(d: Path, prefix: str, level: int) -> None:
        try:
            entries = list(os.scandir(d))
        except OSError as exc:  # FUSE lỗi / mất quyền: in ra, đừng làm sập cả cây
            lines.append(f"{prefix}└── [!] {d.name}: {type(exc).__name__}")
            return
        if not show_hidden:
            entries = [e for e in entries if not e.name.startswith(".")]
        items = sorted(
            entries, key=lambda e: (not e.is_dir(follow_symlinks=False), e.name)
        )
        for i, e in enumerate(items):
            last = i == len(items) - 1
            branch = "└── " if last else "├── "
            is_dir = e.is_dir(follow_symlinks=False)
            if is_dir:
                lines.append(f"{prefix}{branch}{e.name}/")
                counts["dir"] += 1
                if max_depth and level + 1 >= max_depth:
                    continue
                walk(Path(e.path), prefix + ("    " if last else "│   "), level + 1)
            else:
                try:
                    size = e.stat(follow_symlinks=False).st_size
                except OSError:
                    size = 0
                lines.append(f"{prefix}{branch}{e.name}  {_human(size)}")
                counts["file"] += 1
                counts["bytes"] += size

    head = root.as_posix().rstrip("/") or "/"
    lines.append(head + "/")
    if root.is_dir():
        walk(root, "", 0)
    return lines, counts["dir"], counts["file"], counts["bytes"]


def _flat_lines(root: Path, show_hidden: bool = True) -> list[tuple[int, str]]:
    """Danh sách phẳng ``[(size, đường_dẫn)]`` — để sort, grep, đối chiếu manifest."""
    out: list[tuple[int, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if not show_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            if not show_hidden and fn.startswith("."):
                continue
            fp = Path(dirpath) / fn
            try:
                size = fp.stat().st_size
            except OSError:
                size = 0
            out.append((size, fp.as_posix()))
    return sorted(out, key=lambda t: t[1])


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


# ---------------------------------------------------------------------------
# Watchdog chống treo 0B/s
# ---------------------------------------------------------------------------
# Log thật 2026-10-25: `↓ 153.69MB/154.19MB 0B/s in 42m48s` — Baidu cắt băng
# thông giữa chừng và BaiduPCS-Go **không tự bỏ cuộc**, nó ngồi chờ tới hết
# `subprocess_timeout` (90 phút). Ba attempt như vậy là 4.5 giờ đốt vào 0 byte.
#
# Watchdog nhìn kích thước file trong staging: không lớn thêm trong
# `stall_timeout_s` giây thì giết, để attempt sau resume từ đúng chỗ đó.
class _Stalled(RuntimeError):
    """BaiduPCS-Go không ghi thêm byte nào trong `stall_timeout_s` giây."""


# Nhịp kiểm tiến độ. Test hạ xuống rất nhỏ; production 10s là đủ mịn so với
# một file cần hàng chục phút, và mỗi lần kiểm chỉ là `stat` trên thư mục staging.
_POLL_S = 10.0


class _StopRequested(RuntimeError):
    """Có yêu cầu dừng (SIGTERM/SIGINT) — dừng sạch, không phải lỗi tải."""


# ---------------------------------------------------------------------------
# Dừng êm: SIGTERM (modal app stop / client mất kết nối) không được làm mất byte
# ---------------------------------------------------------------------------
# Ba đường dừng, ba hậu quả khác nhau:
#
#   Ctrl+C trong container  → ``KeyboardInterrupt`` (BaseException) ⇒ ``finally``
#                             của ``_fetch_one_impl`` CHẠY ⇒ có commit.
#   SIGTERM                 → mặc định Python **chết ngay**, ``finally`` KHÔNG
#                             chạy. Trước đây chỉ còn lớp bảo hiểm 45 s của
#                             ``_PeriodicCommit``.
#   SIGKILL                 → không chặn được; vẫn chỉ mất tối đa
#                             ``staging_commit_s`` (45 s) nhờ commit định kỳ.
#
# Handler dưới đây biến SIGTERM thành ca thứ nhất: giết BaiduPCS-Go con, commit
# Volume, trả ``status="partial"`` (chạy lại là resume). Không đặt handler cho
# orchestrator: nó đang chờ ``.map()``, chặn SIGTERM ở đó chỉ làm chậm tắt máy.
_STOP = threading.Event()
_STOP_HANDLERS: set[int] = set()


def _request_stop(signum=None, frame=None) -> None:
    """Ghi nhận yêu cầu dừng. Idempotent; gọi được từ test."""
    if not _STOP.is_set():
        _STOP.set()
        print(f"[stop] nhận tín hiệu dừng ({signum}) → ngừng tải, commit Volume, "
              "giữ nguyên phần đã tải trong staging")


def _install_stop_handlers() -> None:
    """Cài handler cho SIGTERM/SIGINT — chỉ trong worker, chỉ một lần."""
    for sig in (signal.SIGTERM, signal.SIGINT):
        if sig in _STOP_HANDLERS:
            continue
        try:
            prev = signal.getsignal(sig)
            signal.signal(sig, _request_stop)
        except (ValueError, OSError):
            # Không phải main thread (hoặc nền tảng không cho) ⇒ bỏ qua: vẫn
            # còn `_PeriodicCommit` làm lưới an toàn.
            continue
        _STOP_HANDLERS.add(sig)
        if prev is signal.SIG_DFL or prev is None:
            print(f"[stop] đã cài handler dừng êm cho {sig.name}")


def _dir_bytes(d: Path) -> int:
    """Tổng byte trong thư mục staging (lỗi đọc ⇒ 0, không được giết việc tải)."""
    try:
        return sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
    except OSError:
        return 0


def _baidu_download(args: list[str], timeout: float, stall_s: float, watch: Path) -> int:
    """Chạy ``BaiduPCS-Go d``, trả exit code; giết nếu treo hoặc quá hạn.

    Vì sao không dùng ``subprocess.run(timeout=...)``: nó chỉ biết *tổng* thời
    gian, không biết "đang treo". Với file 154 MB ở 24 KB/s thì 90 phút là
    *hợp lý*, còn 90 phút ở 0 B/s là mất trắng — chỉ watchdog phân biệt được.
    """
    proc = subprocess.Popen(["BaiduPCS-Go", *args])
    t0 = time.monotonic()
    last_size = -1
    last_change = t0
    try:
        while True:
            # Kiểm TRƯỚC `wait`: dừng ngay, không phải đợi hết nhịp poll 10 s.
            if _STOP.is_set():
                raise _StopRequested("có yêu cầu dừng")
            try:
                return proc.wait(timeout=_POLL_S)
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            size = _dir_bytes(watch)
            if size != last_size:
                last_size, last_change = size, now
            if stall_s > 0 and now - last_change >= stall_s:
                raise _Stalled(
                    f"không ghi thêm byte nào trong {stall_s:.0f}s "
                    f"(giữ {size} B trong staging)"
                )
            if now - t0 >= timeout:
                raise subprocess.TimeoutExpired(cmd="BaiduPCS-Go", timeout=timeout)
    finally:
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass


# ---------------------------------------------------------------------------
# Commit định kỳ: thứ làm staging trong Volume có nghĩa
# ---------------------------------------------------------------------------
# Modal chỉ bền hoá thay đổi của Volume khi ``vol_raw.commit()`` chạy xong. File
# dở nằm trong Volume mà chưa commit thì container chết là mất — tức là staging
# trong Volume **không tự động** cho resume xuyên container; phải commit giữa
# chừng. Thread dưới đây làm việc đó trong lúc BaiduPCS-Go đang chạy.
_COMMIT_LOCK = threading.Lock()


def _commit_volume(reason: str = "") -> bool:
    """``vol_raw.commit()`` có khoá + không bao giờ raise (commit lỗi ≠ mất file)."""
    with _COMMIT_LOCK:
        try:
            vol_raw.commit()
            return True
        except Exception as exc:  # noqa: BLE001 — commit lỗi không được giết việc tải
            print(f"[warn] commit Volume thất bại{f' ({reason})' if reason else ''}: {exc}")
            return False


class _PeriodicCommit:
    """Commit Volume mỗi ``interval`` giây trong lúc đang tải.

    ``interval <= 0`` ⇒ tắt (chỉ commit sau khi file xong). Dùng làm context
    manager để thread chắc chắn dừng kể cả khi subprocess raise.
    """

    def __init__(self, interval: float):
        self.interval = float(interval or 0)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.commits = 0

    def __enter__(self):
        if self.interval <= 0:
            return self
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="dire-volume-commit")
        self._thread.start()
        return self

    def _run(self) -> None:
        # wait() trả True khi stop() được gọi ⇒ thoát ngay, không phải chờ hết
        # một chu kỳ nữa.
        while not self._stop.wait(self.interval):
            if _commit_volume("định kỳ khi đang tải"):
                self.commits += 1

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 60)


def _commit_interval() -> float:
    """Chu kỳ commit, clamp [0, 600]. 0 = tắt commit định kỳ."""
    raw = int(POLICY.staging_commit_s or 0)
    return float(max(0, min(raw, 600)))


def _abs(remote_rel: str) -> str:
    return f"{BAIDU_ROOT.rstrip('/')}/{remote_rel.lstrip('/')}"


def _ls_sizes(remote_dir_rel: str, timeout: int = 300) -> dict[str, int]:
    """``ls -l`` MỘT thư mục → bytes (KHÔNG kèm sai số). In/liệt kê thì dùng."""
    return {k: v.nbytes for k, v in _ls_estimates(remote_dir_rel, timeout).items()}


def _ls_estimates(remote_dir_rel: str, timeout: int = 300) -> dict[str, SizeEst]:
    """``ls -l`` MỘT thư mục → ``SizeEst`` (bytes + sai số làm tròn)."""
    try:
        cp = _baidu(["ls", "-l", _abs(remote_dir_rel)], timeout=timeout)
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"[warn] ls {remote_dir_rel} thất bại: {exc!r}")
        return {}
    if cp.returncode != 0:
        return {}
    return _parse_ls_estimates(cp.stdout)


def _size_map(files: list[File]) -> dict[str, SizeEst | None]:
    """Size remote cho từng file, hỏi Baidu **một lần mỗi thư mục** (không phải
    mỗi file) để không nhân số phiên lên — chính số phiên đồng thời là thứ làm
    Baidu throttle."""
    by_dir: dict[str, list[File]] = {}
    for f in files:
        by_dir.setdefault(str(Path(f.remote).parent), []).append(f)

    sizes: dict[str, SizeEst | None] = {}
    for d, group in sorted(by_dir.items()):
        listing = _ls_estimates(d)
        for f in group:
            sizes[f.remote] = listing.get(Path(f.remote).name)
    return sizes


# ---------------------------------------------------------------------------
# `meta`: nguồn DUY NHẤT cho size chính xác + md5
# ---------------------------------------------------------------------------
# `ls -l` chỉ in size đã làm tròn 2 chữ số thập phân MB (xem `SizeEst`), nên nó
# không đủ tư cách để kết luận "file hỏng". `meta` in đúng khối thông tin mà
# `d` in trước khi tải:
#
#     文件大小          125015131, 119.223720MB
#     md5 (截图请打码)  5a6fb52a2bf9a4c1abf0d50bf30955d4
#
# Chỉ gọi khi kiểm tra đã thất bại (đường bình thường không tốn thêm phiên).
_META_SIZE_RE = re.compile(r"(\d{4,})\s*,\s*[\d.]+\s*(?:B|KB|MB|GB|TB)", re.I)
_META_MD5_RE = re.compile(r"\b([0-9a-fA-F]{32})\b")


def _parse_meta(text: str) -> tuple[int | None, str]:
    """``(size_bytes, md5)`` từ output ``BaiduPCS-Go meta``; thiếu thì ``None``/``""``."""
    size: int | None = None
    m = _META_SIZE_RE.search(text)
    if m:
        size = int(m.group(1))
    md5 = ""
    for line in text.splitlines():
        if "md5" in line.lower():
            h = _META_MD5_RE.search(line)
            if h:
                md5 = h.group(1).lower()
                break
    return size, md5


def _remote_meta(remote_rel: str, timeout: int = 120) -> tuple[int | None, str]:
    """Hỏi ``meta`` một file. Lỗi/không parse được ⇒ ``(None, "")``, không raise."""
    try:
        cp = _baidu(["meta", _abs(remote_rel)], timeout=timeout)
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"[warn] meta {remote_rel} thất bại: {exc!r}")
        return None, ""
    if cp.returncode != 0:
        print(f"[warn] meta {remote_rel} exit {cp.returncode}")
        return None, ""
    return _parse_meta(cp.stdout or "")


def _second_opinion(remote_rel: str, path: Path, v: "Verdict") -> "Verdict":
    """Khi ``ls -l`` nói lệch: hỏi ``meta`` trước khi kết luận file hỏng.

    Vì sao cần: lần chạy 2026-10-25 ba file đã tải ĐỦ 100 % (Baidu in
    "下载结束 … 119.223720MB") rồi bị loại chỉ vì `ls -l` in size đã làm tròn.
    Sai số làm tròn đã được xử lý trong :func:`_parse_ls_estimates`, nhưng nếu
    vẫn lệch thì phải có bằng chứng độc lập mới được phép tuyên bố hỏng —
    `meta` cho size chính xác tới từng byte, và md5 để đối chiếu nội dung.
    """
    size, md5 = _remote_meta(remote_rel)
    local = path.stat().st_size if path.exists() else 0
    if size and size == local:
        return Verdict(True,
                       f"ok — `meta` xác nhận đúng {size} B (ls -l in size làm tròn)",
                       fmt=v.fmt)
    if md5:
        local_md5 = _md5(path) if path.exists() else ""
        if local_md5 and local_md5 == md5:
            return Verdict(True, f"ok — md5 khớp `meta` ({md5[:12]}…)", fmt=v.fmt)
        return Verdict(False,
                       f"{v.reason}; `meta` nói {size or '?'} B / md5 {md5[:12]}…, "
                       f"file là {local} B / md5 {local_md5[:12] or '?'}…",
                       retryable=v.retryable, fmt=v.fmt)
    if size:
        return Verdict(False,
                       f"{v.reason}; `meta` nói {size} B, file là {local} B "
                       f"(lệch {local - size:+d})",
                       retryable=v.retryable, fmt=v.fmt)
    return v


def _is_internal(rel: str) -> bool:
    """True nếu đường dẫn thuộc vùng nội bộ (staging / cache), không phải dữ liệu.

    ``/raw/.staging`` chứa file DỞ và ``/raw/.state`` chứa cache kiểm tra.
    Chúng nằm trong Volume để sống qua container, nhưng KHÔNG được xuất hiện
    trong báo cáo, trong `prune`, hay trong `verify` — file dở mà bị đem đi
    `gzip -t` thì lúc nào cũng "hỏng", và `prune` sẽ đề nghị xoá đúng thứ đang
    giữ tiến độ tải.
    """
    first = rel.replace("\\", "/").lstrip("/").split("/", 1)[0]
    # `.bench` là thư mục nháp của `bench_concurrency`: tải lại vài MB để đo
    # tốc độ, xong là xoá. Nếu không loại ở đây thì `prune` sẽ đề nghị xoá nó
    # (vô hại) và `verify` đem file dở đi `gzip -t` (báo hỏng sai).
    return first in {POLICY.dire_staging, ".staging", ".state", ".bench"}


def _staging_root() -> Path:
    """Nơi hạ file dở. MẶC ĐỊNH nằm TRONG Volume (``/raw/.staging``).

    Vì sao không dùng ``/tmp`` của container nữa:

    - ``/tmp`` là ephemeral. Container chết giữa chừng (hết thời gian, bị
      preempt, mất mạng) là **mất sạch tiến độ**, vì BaiduPCS-Go resume dựa
      trên file dở còn nằm trong thư mục đích. Ở 67 KB/s, một file 692 MB cần
      ~2.9 giờ — mất tiến độ ở phút 170 là mất gần 3 giờ.
    - ``/tmp`` khác filesystem với ``/raw`` ⇒ bước công bố phải COPY toàn bộ
      file qua FUSE (``shutil.move`` = copy + unlink). Với ``/raw/.staging``,
      bước đó là ``os.replace`` **cùng** filesystem: atomic, không copy byte nào.

    Đổi lại: mỗi ``write()`` của BaiduPCS-Go đi qua FUSE. Ở băng thông Baidu
    (67 KB/s đo được) thì đây là nhiễu, không phải nút cổ chai — mạng chậm hơn
    đĩa khoảng ba bậc độ lớn.

    Muốn quay lại ``/tmp`` (nhanh hơn một chút, nhưng mất resume xuyên
    container): đặt ``DIRE_STAGING_DIR=/tmp/baidu_staging``.
    """
    override = os.environ.get("DIRE_STAGING_DIR", "").strip()
    if override:
        return Path(override)
    return RAW / POLICY.dire_staging


def _state_dir() -> Path:
    """Cache kết quả kiểm tra. Cũng nằm trong Volume để dùng lại giữa các lần chạy."""
    return RAW / ".state"


def _publish(got: Path, local: Path) -> None:
    """Đưa file đã tải từ staging tới đường dẫn chính thức.

    Cùng filesystem (staging trong Volume) ⇒ ``os.replace``: atomic, O(1),
    không copy byte nào. Khác filesystem (ai đó đặt ``DIRE_STAGING_DIR=/tmp``)
    ⇒ ``rename(2)`` ném ``EXDEV`` (errno 18) và phải ``shutil.move`` = copy +
    unlink, đúng lỗi #1 của bản cũ.
    """
    try:
        os.replace(got, local)
        return
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        print(f"[warn] staging khác filesystem → copy thay vì rename: {exc.strerror}")
    if local.exists():
        local.unlink()
    shutil.move(str(got), str(local))


def _hold(path: Path) -> Path:
    """Chuyển một file KHÔNG qua kiểm sang ``<staging>/_held/<tmpdir>/``.

    Vì sao không ``unlink()``: lần chạy 2026-10-25 đã xoá ba file đã tải ĐỦ
    119 MB sau khi phép kiểm (sai) nói "tải dở dang" — 26 phút × 3 bay mất, và
    lần sau phải tải lại từ 0 B. Byte đã tải là tài sản; kết luận của code mới
    là thứ đáng ngờ. Chuyển sang ``_held`` vừa không mất dữ liệu, vừa để thư
    mục staging trống cho lần tải lại.

    ``_held`` nằm trong ``.staging`` nên `_is_internal()` coi là nội bộ: không
    hiện trong báo cáo, không bị ``prune`` đề nghị xoá, không bị ``verify`` đem
    đi ``gzip -t``. Xoá hẳn khi không cần nữa:
    ``modal volume rm dire-raw /raw/.staging/_held -r``

    **Vì sao ``os.replace`` mà không COPY** (cùng filesystem ⇒ rename, O(1),
    0 byte đọc/ghi):

    * Copy chậm hơn theo định nghĩa: nó phải làm đúng việc rename làm, cộng
      thêm 2N byte I/O qua FUSE và N byte dung lượng.
    * Copy để lại file trong thư mục tải ⇒ BaiduPCS-Go luôn thấy
      ``文件已经存在`` và ``跳过`` ⇒ mỗi lần chạy thêm một attempt 0 byte mà
      **không bao giờ tiến bộ**. Đó chính là "attempt vô nghĩa" cần tránh, và
      cái gây ra nó là *giữ file ở thư mục đích*, không phải việc xoá.
    * Yêu cầu "đừng xoá dữ liệu trong ``.staging``" vẫn được thoả: ``_held``
      NẰM TRONG ``.staging``.
    """
    nbytes = path.stat().st_size if path.exists() else 0
    try:
        dest = hold_into(path, _staging_root() / "_held" / path.parent.name)
        print(f"[HOLD] giữ {_human(nbytes)} → {dest} (KHÔNG xoá)")
        return dest
    except OSError as exc:
        print(f"[warn] không chuyển được {path} sang _held: {exc}")
        return path


def _tmpdir_for(remote_rel: str) -> Path:
    """Thư mục staging RIÊNG cho mỗi đường dẫn remote (sửa lỗi #4).

    Hash theo đường dẫn nên (a) không bao giờ trùng giữa ``train/adm.tar.gz`` và
    ``val/adm.tar.gz``, và (b) ỔN ĐỊNH giữa các attempt ⇒ BaiduPCS-Go resume
    được mảnh dở thay vì tải lại từ đầu. Với staging trong Volume, tính ổn định
    này còn sống **qua các container** — đó là toàn bộ lý do đổi chỗ.
    """
    return staging_dir_for(_staging_root(), remote_rel)


# ---------------------------------------------------------------------------
# Cache kiểm tra: đừng đọc lại 3.39 GB mỗi lần chạy
# ---------------------------------------------------------------------------
# `_validate_ex` chạy `gzip -t` trên từng file, tức là ĐỌC TOÀN BỘ file. Với
# Tier A mặc định (3.39 GB) thì mỗi lần chạy lại đều đọc hết qua FUSE chỉ để
# trả lời "file này vẫn tốt chứ". Cache theo (size, mtime_ns) bỏ được việc đó
# mà vẫn an toàn: file bị ghi đè hay sửa thì mtime đổi.
#
# CHỈ cache kết quả TỐT. Không bao giờ cache "hỏng" — nếu cache âm bị sai thì
# một file lành sẽ bị tải lại mãi mãi.
def _load_state() -> dict[str, dict]:
    path = _state_dir() / "validated.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_state(state: dict[str, dict]) -> None:
    try:
        path = _state_dir() / "validated.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, separators=(",", ":")),
                       encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:  # cache hỏng không được làm chết việc tải
        print(f"[warn] không ghi được cache kiểm tra: {exc}")


def _record_valid(rel: str, path: Path, digest: str, fmt: str) -> None:
    st = path.stat()
    state = _load_state()
    state[rel] = {
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "sha256": digest,
        "fmt": fmt,
        "checked": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    _save_state(state)


def _cached_valid(rel: str | None, path: Path, expected: int = 0) -> Verdict | None:
    """Trả ``Verdict`` từ cache nếu (size, mtime_ns) khớp, ngược lại ``None``.

    ``expected`` chỉ giữ cho tương thích chữ ký và **không** được dùng để từ
    chối cache: nó là size từ ``ls -l``, đã làm tròn (xem :class:`SizeEst`).
    Dùng nó ở đây thì một file đã kiểm tốt sẽ bị coi là "phải tải lại" mỗi lần
    chạy — và ở đường ``fetch_one`` là bị **xoá** rồi tải lại từ 0 B. Cache chỉ
    được ghi sau khi kiểm THÀNH CÔNG, nên (size, mtime_ns) khớp là đủ.
    """
    if not rel:
        return None
    rec = _load_state().get(rel)
    if not rec:
        return None
    try:
        st = path.stat()
    except OSError:
        return None
    if st.st_size != rec.get("size") or st.st_mtime_ns != rec.get("mtime_ns"):
        return None  # file đã bị ghi đè/sửa ⇒ không tin cache
    return Verdict(True, f"ok (cache, kiểm lúc {rec.get('checked', '?')})",
                   fmt=str(rec.get("fmt") or ""))


def _validate_cached(rel: str | None, path: Path, expected: int = 0,
                     expect_archive: bool = True, slack: int = 0) -> Verdict:
    """``_validate_ex`` + dùng cache khi có. Ghi cache khi kiểm thành công."""
    hit = _cached_valid(rel, path, expected)
    if hit is not None:
        return hit
    v = _validate_ex(path, expected, expect_archive=expect_archive, slack=slack)
    if v.ok and rel:
        try:
            _record_valid(rel, path, _sha256(path), v.fmt)
        except OSError:
            pass
    return v


def _find_downloaded(tmpdir: Path, want: str) -> Path | None:
    exact = tmpdir / want
    if exact.exists() and exact.stat().st_size > 0:
        return exact
    # BaiduPCS-Go có thể đổi tên khi trùng (adm(1).tar.gz) → lấy file duy nhất
    cands = [p for p in tmpdir.rglob("*") if p.is_file() and p.stat().st_size > 0]
    return cands[0] if len(cands) == 1 else None


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
    sizes: dict[str, "SizeEst | int | None"],
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
        est = sizes.get(f.remote)
        if est is None:
            unknown += 1
            cell = "? (ls -l không thấy)"
        else:
            s = _bytes_of(est)
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


def _clamp_concurrency(raw: int) -> int:
    """Số container tải song song: [1, 8], không hơn.

    Trước đây tham số ``--concurrency`` bị bỏ qua hoàn toàn (lỗi #3) nên người
    dùng tưởng chỉnh được mà không chỉnh được gì. Nay nó có tác dụng THẬT,
    nhưng vẫn bị chặn trần: Baidu throttle theo số phiên, và log thật cho thấy
    26 phiên cùng lúc ⇒ 0B/s hàng loạt. 8 là mức còn có thể có lợi.
    """
    return max(1, min(int(raw or 0), 8))


def _files_from_paths(spec: str) -> list[File]:
    """Dựng danh sách ``File`` từ ``--paths a.tar.gz,b.pth``.

    Vì sao có: tier là đơn vị của *bộ dữ liệu*, nhưng để có **con số đầu tiên**
    chỉ cần 3 file (~400 MB, 1.5 giờ ở 76.6 KiB/s) thay vì cả `e2` (1.94 GB,
    7.4 giờ). Bắt người dùng sửa manifest cho một lần chạy là sai: ``--paths``
    là công tắc, không phải cấu hình.

    Vẫn đi qua kiểm tra ``NEVER`` — không có đường tắt nào cho `recons/` hay
    `*ddim*`, kể cả khi gõ tay.
    """
    out: list[File] = []
    for raw in spec.replace("\n", ",").split(","):
        p = raw.strip().lstrip("/")
        if not p:
            continue
        if not p.endswith(_KNOWN_SUFFIX):
            raise SystemExit(
                f"[!] --paths chỉ nhận {'/'.join(_KNOWN_SUFFIX)}: {p!r}"
            )
        why = is_never(p)
        if why:
            raise SystemExit(f"[!] {p} bị NEVER chặn: {why}")
        out.append(File(remote=p, local=p, note="chọn tay qua --paths"))
    if not out:
        raise SystemExit("[!] --paths rỗng")
    return out


def _split_present(
    files: list[File], sizes: dict[str, "SizeEst | int | None"], force: bool
) -> tuple[list[File], list[File]]:
    """``(cần tải, đã có & hợp lệ)`` — dùng CHUNG cho lọc đầu và cho mỗi vòng.

    Chỉ tin cache (``_cached_valid``), không gọi ``_validate_cached``: bản đầy
    đủ đọc cả file để ``gzip -t`` + ``sha256``, tức 2 lần đọc toàn bộ dữ liệu
    ngay trong container orchestrator. Cache trượt thì giao cho worker — nó có
    đúng phép kiểm đó và sẽ tự ``[SKIP]``.
    """
    todo: list[File] = []
    have: list[File] = []
    for f in files:
        p_local = RAW / f.local
        exp = _bytes_of(sizes.get(f.remote))
        if force:
            todo.append(f)
            continue
        if p_local.exists() and p_local.stat().st_size > 0 and (
            _cached_valid(f.local, p_local, exp) is not None
        ):
            have.append(f)
        else:
            todo.append(f)
    return todo, have


def _execute(tiers: str, max_gb: float, do_fetch: bool, concurrency: int = 0,
             force: bool = False, paths: str = "") -> None:
    """Ruột của cả ``plan`` và ``main`` — tránh hai bản logic lệch nhau.

    ``concurrency`` > 0 ghi đè ``max_containers`` trong policy (clamp [1, 8]).
    ``force`` tải lại cả file đã có trong cache — chỉ dùng khi nghi cache sai.
    """
    paths = _demangle_list(paths)   # Git Bash: "/raw/x" → "C:/.../Git/raw/x"
    unresolved: list[DirSpec] = []
    blocked: list[tuple[File, str]] = []

    if paths.strip():
        # Danh sách tay: bỏ qua tier, nhưng vẫn qua guard dung lượng + NEVER.
        files = _files_from_paths(paths)
        print(f"\n[paths] {len(files)} file chọn tay (bỏ qua tier)")
        _baidu_login()
        sizes = _size_map(files)
    else:
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
        # DirSpec cần hỏi remote để biết tên file thật — không đoán.
        files, unresolved, blocked = expand(ents, lister=lambda d: list(_ls_sizes(d)))
        sizes = _size_map(files)

        # Đếm chính xác số file bỏ qua: chỉ File (lister=None để không gọi Baidu
        # thêm lần nào). Bản đầu đếm cả DirSpec nên báo sai 1 đơn vị.
        all_files, _, _ = expand(resolve_many(",".join(TIERS)))
        chosen = {f.remote for f in files}
        skipped_n = len({f.remote for f in all_files} - chosen)
        print(f"\n  Bỏ qua {skipped_n} file kê khai trong tier khác → KHÔNG tải.")

    known, unknown, largest = _report(files, sizes, blocked, unresolved)

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

    # --- đừng bật container cho việc đã xong ------------------------------
    # Trước đây payload dựng cho MỌI file rồi mới `.map()`: file đã có vẫn được
    # giao cho một container, container đó bật lên, mount Volume, chạy
    # `_validate_cached` rồi in [SKIP]. Đúng nhưng tốn cold start cho mỗi file
    # và làm `plan` không trả lời được câu "có cần chạy không".
    #
    # Ở đây chỉ dùng CACHE (`_cached_valid`), không gọi `_validate_cached`: bản
    # đầy đủ đọc cả file để `gzip -t` + `sha256`, tức 2 lần đọc 3.39 GB ngay
    # trong container orchestrator. Cache trượt thì giao cho worker — nó có
    # đúng phép kiểm đó và sẽ tự [SKIP].
    todo, have = _split_present(files, sizes, force)

    if have:
        print(f"\n[skip] {len(have)}/{len(files)} file đã có & hợp lệ (theo cache)"
              f" → không bật container cho chúng")
    if not todo:
        print("\n[done] Không còn file nào cần tải. Không bật container tải nào.")
        print("       (Kiểm lại toàn bộ nếu cần: --force, hoặc chạy ::verify)")
        return
    if force:
        print(f"[force] tải lại cả {len(files)} file, bỏ qua cache kiểm tra")

    workers = _clamp_concurrency(concurrency or POLICY.max_containers)
    print(f"\n[fetch] {len(todo)} file · tối đa {workers} container song song · "
          f"{POLICY.fetch_attempts} attempt/file")
    print(f"[fetch] staging: {_staging_root()} "
          f"(commit mỗi {_commit_interval():g}s — resume sống qua container chết)")
    if POLICY.baidu_parallel:
        print(f"[fetch] -p {POLICY.baidu_parallel} kết nối/file")
    if (POLICY.baidu_mode or "").strip():
        print(f"[fetch] --mode {(POLICY.baidu_mode or '').strip()}")
    print(f"[fetch] tự chạy nhiều vòng tới khi xong hoặc hết ngân sách "
          f"{ORCHESTRATOR_BUDGET_S / 3600:g}h")
    t0 = time.time()
    fn = fetch_one.with_options(max_containers=workers) if workers != POLICY.max_containers else fetch_one

    # Duyệt `.map()` theo TỪNG kết quả thay vì `list(...)`.
    #
    # `list(fn.map(...))` chỉ trả về khi MỌI task xong. Một task chạm trần thời
    # gian của Modal (`FunctionTimeoutError`) làm cả lời gọi raise ⇒ toàn bộ kết
    # quả của các file đã tải xong bị vứt, dù dữ liệu vẫn nằm trong Volume.
    # Duyệt từng cái + ghi manifest ngay (kèm commit) thì phần đã xong luôn được
    # ghi lại, kể cả khi container orchestrator bị giết ở giây sau đó.
    #
    # VÒNG LẶP: ở 76.6 KiB/s, một lần `.map()` chỉ tải được ~400 MB (3 attempt ×
    # 1800s). `e2` cần 1.94 GB ⇒ 5 vòng, `e2,e4` ⇒ 9 vòng. Không lặp thì người
    # dùng phải gõ lại lệnh 9 lần — và mỗi lần là một cold start.
    results: list[dict] = []
    aborted = ""
    wave = 0
    while wave < _MAX_WAVES:
        wave += 1
        payloads = [
            {"remote": f.remote, "local": f.local,
             "expected_size": _bytes_of(sizes.get(f.remote)),
             # Sai số của `ls -l`: phép kiểm PHẢI biết, nếu không mọi file tải
             # đúng đều bị báo "tải dở dang" (sự cố 2026-10-25).
             "expect_slack": _slack_of(sizes.get(f.remote))}
            for f in todo
        ]
        left = ORCHESTRATOR_BUDGET_S - (time.time() - t0)
        print(f"\n[vòng {wave}] {len(payloads)} file · đã dùng "
              f"{(time.time() - t0) / 3600:.2f}h · còn {left / 3600:.2f}h ngân sách")
        try:
            for r in fn.map(payloads, order_outputs=False):
                results.append(r)
                _write_manifest([r], quiet=True)
        except Exception as exc:  # noqa: BLE001 — phải in tổng kết dù đợt tải gãy
            aborted = f"{type(exc).__name__}: {exc}"
            print(f"\n[warn] vòng {wave} dừng giữa chừng — {aborted}")
            print("       Kết quả của các file đã xong VẪN được ghi (xem _manifest.tsv).")
            break

        # Tính lại từ ĐĨA, bằng đúng phép lọc của đầu lần chạy: file vừa xong
        # rời khỏi danh sách, file dở ở lại và resume từ byte đã có.
        # `force` thì không lặp: nó nghĩa là "tải lại một lượt", không phải
        # "tải lại mãi".
        if force:
            todo = []
            break
        before = len(todo)
        todo, _have_now = _split_present(files, sizes, force=False)
        if not todo:
            break
        if len(todo) >= before:
            # Một vòng không mang thêm file nào về đích ⇒ đừng đốt tiếp ngân
            # sách vào cùng chỗ. Không có chốt này thì BDUSS hết hạn (hoặc
            # throttle cứng) sẽ quay vòng tới hết 20 giờ mà không ai biết.
            print(f"\n[stop] vòng {wave} không xong thêm file nào — dừng để không"
                  f" đốt ngân sách. Còn {len(todo)} file.")
            print("       Kiểm: BDUSS/STOKEN còn hạn? thử --tiers e2min, hoặc"
                  " đặt baidu_mode: pcs trong tiers.yaml")
            break
        if time.time() - t0 + _MIN_WAVE_S >= ORCHESTRATOR_BUDGET_S:
            print(f"\n[budget] hết ngân sách {ORCHESTRATOR_BUDGET_S / 3600:g}h — "
                  f"còn {len(todo)} file. Chạy lại để resume.")
            break
    dt = time.time() - t0

    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    partial = [r for r in results if r["status"] == "partial"]
    errs = [r for r in results if r["status"] == "error"]

    # Đếm "còn thiếu" từ ĐĨA, không từ số payload: qua nhiều vòng thì
    # `len(payloads)` không còn nghĩa gì.
    still_todo, have_now = _split_present(files, sizes, force=False)

    _write_manifest([])  # in đường dẫn; không thêm dòng nào

    print(f"\n{'='*78}\nKẾT QUẢ  ({dt/60:.1f} phút · {wave} vòng)\n{'='*78}")
    print(f"  ok      : {len(ok)}")
    print(f"  skipped : {len(skipped)}  (đã có sẵn & hợp lệ)")
    print(f"  partial : {len(partial)}  (hết ngân sách thời gian — chạy lại để resume)")
    print(f"  error   : {len(errs)}")
    print(f"  còn thiếu: {len(still_todo)} / {len(files)} file"
          + ("  ✅ ĐỦ" if not still_todo else ""))
    n_stopped = len([r for r in partial if r.get("stopped")])
    if n_stopped:
        print(f"\n  ⏸  {n_stopped} file dừng giữa chừng theo yêu cầu (SIGTERM/SIGINT)."
              "\n     Phần đã tải nằm trong /raw/.staging và đã được commit —"
              " chạy lại đúng lệnh là resume.")
    for r in partial:
        print(f"      {r['remote']}\n          → {r['error']}")
    for r in errs:
        print(f"      {r['remote']}\n          → {r['error']}")
    if still_todo:
        for f in still_todo:
            print(f"      [thiếu] {f.remote}")

    if still_todo:
        print("\nChạy lại đúng lệnh trên: file đã xong sẽ skip, file dở resume từ"
              " đúng byte đã tải (staging nằm trong Volume).")

    if errs:
        # KHÔNG raise: để người dùng chạy lại và fetch_one sẽ skip file đã ok.
        print("\nNếu vẫn 0 byte: BDUSS hết hạn, hoặc giảm max_containers xuống 1.")


def _write_manifest(results: list[dict], quiet: bool = False) -> None:
    """Ghi thêm dòng vào ``/raw/_manifest.tsv`` và commit NGAY.

    ``quiet=True`` để gọi được sau TỪNG file trong lúc `.map()` đang chảy: khi
    container bị Modal giết, không có `finally` nào chạy, nên đây là chỗ duy nhất
    giữ được kết quả của những file đã xong.

    Không commit qua `vol_raw.commit()` trực tiếp mà qua `_commit_volume()`:
    cùng khoá với thread commit định kỳ, và commit lỗi không được giết việc tải.
    """
    out = RAW / "_manifest.tsv"
    write_header = not out.exists()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if results or write_header:
        with out.open("a", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            if write_header:
                w.writerow(
                    ["timestamp", "remote", "local", "size_bytes", "sha256", "status"]
                )
            for r in results:
                w.writerow(
                    [now, r.get("remote", ""), r["local"], r["size"], r.get("sha256", ""),
                     r["status"]]
                )
    _commit_volume("ghi _manifest.tsv")
    if not quiet:
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
    path = _demangle(path)
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
    dirs = _demangle_list(dirs)
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
    # Gọi `_fetch_one_impl` HAI lần, mỗi lần có ngân sách riêng FETCH_CALL_TIMEOUT_S.
    timeout=2 * FETCH_CALL_TIMEOUT_S + 600,
    cpu=1.0,
)
def probe_pair(
    a: str = "dire/test/lsun_bedroom/lsun_bedroom/real.tar.gz",
    b: str = "dire/test/lsun_bedroom/imagenet/real.tar.gz",
) -> None:
    a, b = _demangle(a), _demangle(b)
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
    # `_ls_estimates` chứ KHÔNG phải `_ls_sizes`: `probe_pair` so sha256 giữa
    # hai file, mà `expected_size` lấy từ `ls -l` là số ĐÃ LÀM TRÒN. Dùng bản
    # mất sai số thì probe tự loại đúng file nó vừa tải xong.
    est = {p: _ls_estimates(str(Path(p).parent)).get(Path(p).name) for p in (a, b)}
    for p, e in est.items():
        sz = _bytes_of(e)
        print(f"[probe] remote {p} = {_human(sz) if sz else '? (ls không thấy)'}"
              + (f" ±{e.slack} B (ls -l làm tròn)" if e and e.slack else ""))

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
                "expected_size": _bytes_of(est[p]),
                "expect_slack": _slack_of(est[p]),
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
def plan(tiers: str = POLICY.default_tiers, max_gb: float = POLICY.max_gb,
         paths: str = "") -> None:
    """In bảng file + size thật từ Baidu. **Không tải gì.**

    ``modal run -m dire_modal.download::plan --tiers e2,e4``
    ``modal run -m dire_modal.download::plan --paths a.tar.gz,b.pth``
    """
    _execute(tiers, max_gb, do_fetch=False, paths=paths)


@app.function(
    image=cpu_image,
    secrets=[baidu_secret],
    volumes={"/raw": vol_raw},
    # Orchestrator chỉ ngồi chờ `.map()` nên rất rẻ, nhưng nếu NÓ bị giết thì
    # không ai kịp in tổng kết. Trước đây nó dùng cùng công thức 18 000s như
    # `fetch_one` ⇒ hai bên chết cùng lúc, và vì `_execute` gọi
    # `list(fn.map(...))` nên kết quả của các file đã xong bị vứt sạch.
    timeout=ORCHESTRATOR_TIMEOUT_S,
)
def main(
    tiers: str = POLICY.default_tiers,
    max_gb: float = POLICY.max_gb,
    mode: str = "plan",
    concurrency: int = 0,
    force: bool = False,
    paths: str = "",
) -> None:
    """Lên kế hoạch rồi (tuỳ chọn) tải — tự chạy nhiều vòng tới khi xong.

    ``mode`` nhận ``plan`` (mặc định — KHÔNG tải) hoặc ``fetch`` / ``commit``
    (tải thật). Dùng chuỗi thay vì bool vì CLI của Modal parse ``--flag`` kiểu
    bool không nhất quán giữa các phiên bản; ``--mode fetch`` không mơ hồ.

    Ví dụ::

        # 1.5 giờ: chỉ 3 file đủ để chạy eval đầu tiên (xem tier `e2min`)
        modal run -m dire_modal.download::main --tiers e2min --mode fetch

        # 7.4 giờ ở 76.6 KiB/s: toàn bộ bảng eval DIRE (16 generator)
        modal run -m dire_modal.download::main --tiers e2 --mode fetch

        # file lẻ, bỏ qua tier
        modal run -m dire_modal.download::main \
            --paths checkpoints/lsun_adm.pth,dire/test/lsun_bedroom/lsun_bedroom/adm.tar.gz \
            --mode fetch

        # soát trước khi tiêu tiền (không tải gì)
        modal run -m dire_modal.download::main --tiers e2
    """
    _execute(
        tiers, max_gb,
        do_fetch=mode.strip().lower() in {"fetch", "commit", "yes"},
        concurrency=concurrency,
        force=force,
        paths=paths,
    )


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
    _install_stop_handlers()
    remote_rel = f["remote"]
    remote = _abs(remote_rel)
    local = RAW / f["local"]
    expected = int(f.get("expected_size") or 0)
    # Sai số của `expected`: `ls -l` in size đã làm tròn 2 chữ số thập phân MB
    # (xem `SizeEst`), nên đòi khớp từng byte là SAI. Không có tham số này thì
    # mọi file tải đúng đều bị loại — đúng sự cố 2026-10-25.
    slack = int(f.get("expect_slack") or 0)

    # --- đã có & hợp lệ → skip (idempotent, chạy lại bao nhiêu lần cũng được)
    if local.exists() and local.stat().st_size > 0:
        # `_validate_cached` dùng cache (size, mtime_ns) khi có: không phải đọc
        # lại 3.39 GB qua FUSE mỗi lần chạy chỉ để hỏi "file này vẫn tốt chứ".
        v = _validate_cached(f["local"], local, expected, slack=slack)
        if v.ok:
            rec = _load_state().get(f["local"]) or {}
            cached_digest = rec.get("sha256")
            print(f"[SKIP] {remote_rel} đã có & hợp lệ"
                  + (" (cache)" if cached_digest else ""))
            return {
                "remote": remote_rel, "local": str(local), "status": "skipped",
                "size": local.stat().st_size,
                # sha256 có sẵn trong cache thì không đọc lại file lần nữa
                "sha256": cached_digest or _sha256(local),
                "format": v.fmt or "gzip",
            }
        # Không xoá: đây là dữ liệu ĐÃ CÔNG BỐ. Nếu kết luận của code sai (đã
        # xảy ra) thì xoá là mất trắng; `_hold` giữ lại để còn đường kiểm tay.
        print(f"[REDO] {remote_rel}: {v.reason} → chuyển bản cũ sang _held")
        _hold(local)

    local.parent.mkdir(parents=True, exist_ok=True)
    tmpdir = _tmpdir_for(remote_rel)  # riêng + ổn định ⇒ resume được
    tmpdir.mkdir(parents=True, exist_ok=True)
    last = "chưa thử"
    prev_sha = ""  # sha256 của lần tải trước, để phát hiện lỗi TẤT ĐỊNH
    held_path: Path | None = None  # bản không qua kiểm đã được giữ lại (nếu có)
    bytes_before = 0               # byte có sẵn trước attempt hiện tại
    expect_archive = bool(f.get("expect_archive", True))
    started = time.monotonic()   # mốc tính ngân sách của CẢ lần gọi này
    budget_out = False           # dừng vì hết ngân sách (≠ lỗi tải)
    stopped = False              # dừng vì SIGTERM/SIGINT — cũng ≠ lỗi tải
    kept_bytes = 0
    attempt = 0                  # số attempt TẢI THẬT đã dùng
    holds = 0                    # số lần "Baidu bỏ qua" đã gặp (tối đa 1)

    try:
        # `while` chứ không `for`: lần "Baidu thấy file đã có nên bỏ qua" không
        # phải một attempt TẢI (nó luôn 0 byte, log thật: "下载结束, 时间: 0s").
        # Để nó tiêu một suất trong `fetch_attempts` thì một file cần tải lại
        # chỉ còn 1 suất cho lần thật — và mất luôn suất đó nếu mạng chập.
        while attempt < POLICY.fetch_attempts:
            attempt += 1
            # Không bao giờ để attempt vượt ngân sách: Modal giết container ở
            # FETCH_CALL_TIMEOUT_S và `FunctionTimeoutError` KHÔNG bắt được.
            tmo = _attempt_timeout(started)
            if tmo <= 0:
                budget_out = True
                last = (
                    f"hết ngân sách {FETCH_CALL_TIMEOUT_S}s của lần gọi này sau "
                    f"{attempt - 1} attempt — chạy lại để resume"
                )
                print(f"[BUDGET] {remote_rel}: {last}")
                break
            try:
                _baidu_login()
                # KHÔNG capture output: để progress bar chảy ra log của Modal.
                # Chính nhờ log đó mà chẩn đoán ra throttle 0B/s.
                # `-p N` = số kết nối song song cho MỘT file. Chỉ truyền khi
                # policy > 0: mặc định 0 nghĩa là "để BaiduPCS-Go tự quyết",
                # tránh phụ thuộc vào đúng tên cờ của từng phiên bản.
                par = int(POLICY.baidu_parallel or 0)
                args = ["d", "--saveto", str(tmpdir)]
                if par > 0:
                    args += ["-p", str(min(par, 32))]
                # `--mode` chỉ truyền khi có cấu hình: mỗi bản BaiduPCS-Go có
                # tập giá trị hơi khác nhau, và mặc định của client là an toàn.
                bmode = (POLICY.baidu_mode or "").strip()
                if bmode:
                    args += ["--mode", bmode]
                args.append(remote)

                # Bao nhiêu byte đã có TRƯỚC attempt này. Dùng để phân biệt
                # "Baidu tải tiếp" với "Baidu thấy file đã có nên bỏ qua"
                # (`文件已经存在 … 跳过`): hai ca này cần cách xử lý khác nhau.
                bytes_before = _dir_bytes(tmpdir)
                # Commit định kỳ TRONG lúc tải: đây là thứ khiến staging trong
                # Volume thực sự cho resume xuyên container. Không có nó thì
                # container chết là mất sạch tiến độ, y như khi dùng /tmp.
                with _PeriodicCommit(_commit_interval()):
                    rc = _baidu_download(
                        args,
                        timeout=tmo,
                        stall_s=float(POLICY.stall_timeout_s or 0),
                        watch=tmpdir,
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
                            f"BaiduPCS-Go không tạo file nào (exit {rc}) — "
                            "đường dẫn remote sai, hoặc BDUSS/STOKEN hết hạn"
                        )
                else:
                    # KIỂM TRA TRONG STAGING, chỉ công bố khi hợp lệ.
                    #
                    # Thứ tự này quan trọng: nếu move sang /raw rồi mới kiểm thì
                    # bước kiểm hỏng phải xoá file khỏi /raw — và mảnh dở (đúng
                    # thứ dùng để resume) biến mất. Giữ nó trong staging thì lần
                    # attempt sau, hoặc lần chạy sau ở CONTAINER KHÁC, resume
                    # được từ đúng chỗ đã tải.
                    v = _validate_ex(got, expected, expect_archive=expect_archive,
                                     slack=slack)
                    if not v.ok and v.retryable:
                        # Trước khi kết luận "file hỏng": hỏi `meta`. `ls -l`
                        # chỉ in size làm tròn nên nó không đủ tư cách một
                        # mình kết tội một file 119 MB. Chỉ tốn 1 phiên, và chỉ
                        # trên đường đang nghi ngờ.
                        v = _second_opinion(remote_rel, got, v)
                    if v.ok:
                        digest = _sha256(got)
                        size = got.stat().st_size
                        # Cùng filesystem ⇒ os.replace: atomic, không copy byte.
                        if local.exists():
                            local.unlink()
                        _publish(got, local)
                        _record_valid(f["local"], local, digest, v.fmt or "gzip")
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

                    # --- thất bại ---
                    last = v.reason
                    size_now = got.stat().st_size

                    if not v.retryable:
                        # Lỗi TẤT ĐỊNH (nguồn sai định dạng / Baidu trả trang
                        # lỗi). Tải lại ra đúng chuỗi byte đó nên dừng, NHƯNG
                        # vẫn không xoá: `_hold` giữ lại để người dùng kiểm tay
                        # nếu kết luận của code sai.
                        held = _hold(got)
                        print(f"[STOP] {remote_rel}: {v.reason}")
                        return {
                            "remote": remote_rel, "local": str(local),
                            "status": "error", "error": v.reason,
                            "size": 0, "attempts": attempt, "held": str(held),
                            "deterministic": True, "format": v.fmt,
                        }

                    # Baidu thấy file đã có trong thư mục đích nên BỎ QUA
                    # ("文件已经存在 … 跳过", 数据总量: 0B) ⇒ thử lại y nguyên là
                    # vô nghĩa. Chuyển file sang `_held` để attempt sau tải lại
                    # từ 0 B, và KHÔNG xoá byte nào.
                    #
                    # Đây là ca đã đốt 26 phút × 3 trong lần chạy 2026-10-25:
                    # lần 1 tải đủ, lần 2 Baidu bỏ qua, code cũ gọi đó là "hai
                    # lần cho ra byte GIỐNG HỆT ⇒ lỗi tất định" rồi xoá sạch.
                    if size_now <= bytes_before:
                        if holds == 0:
                            holds += 1
                            held_path = _hold(got)
                            # Hoàn lại suất: lần vừa rồi Baidu không tải gì cả.
                            attempt -= 1
                            last = (
                                f"Baidu không ghi thêm byte nào (coi file đã có là "
                                f"xong) mà vẫn không qua kiểm: {v.reason} → giữ ở "
                                f"{held_path}, tải lại từ 0 B (không tính là attempt)"
                            )
                            print(f"[HOLD] {remote_rel}: {last}")
                            _commit_volume("giữ file không qua kiểm")
                            continue
                        last = (
                            f"tải lại từ 0 B vẫn không qua kiểm: {v.reason} "
                            f"(bản cũ ở {held_path})"
                        )
                        print(f"[STOP] {remote_rel}: {last}")
                        return {
                            "remote": remote_rel, "local": str(local),
                            "status": "error", "error": last,
                            "size": 0, "attempts": attempt, "held": str(held_path),
                            "deterministic": False, "format": v.fmt,
                        }

                    digest = _sha256(got)
                    if digest == prev_sha:
                        # Byte GIỐNG HỆT qua hai lần tải ⇒ tải lại cũng thế.
                        held_path = _hold(got)
                        msg = (
                            f"hai lần tải cho ra byte GIỐNG HỆT (sha256={digest[:12]}…, "
                            f"{_human(size_now)}) ⇒ lỗi tất định, dừng thử lại. {v.reason}"
                            f" — giữ ở {held_path}"
                        )
                        print(f"[STOP] {remote_rel}: {msg}")
                        return {
                            "remote": remote_rel, "local": str(local),
                            "status": "error", "error": msg,
                            "size": 0, "attempts": attempt, "held": str(held_path),
                            "deterministic": True, "format": v.fmt,
                        }
                    prev_sha = digest
                    # Mảnh dở Ở LẠI staging: đây là tiến độ. Commit ngay để nó
                    # sống qua container chết (thay vì đợi chu kỳ định kỳ).
                    _commit_volume("giữ mảnh dở giữa các attempt")
            except _Stalled as exc:
                last = f"Baidu cắt băng thông: {exc} → thử lại (resume từ chỗ dở)"
            except subprocess.TimeoutExpired:
                last = f"BaiduPCS-Go treo quá {tmo:.0f}s (0B/s)"
            except subprocess.CalledProcessError as exc:
                last = f"BaiduPCS-Go exit {exc.returncode}"
            except Exception as exc:  # kể cả OSError/EXDEV nếu còn sót đường nào
                last = f"{type(exc).__name__}: {exc}"

            if _STOP.is_set():
                stopped = True
                last = (
                    f"dừng theo yêu cầu ({last}) — giữ "
                    f"{_human(_dir_bytes(tmpdir))} trong staging để resume"
                )
                print(f"[stop] {remote_rel}: {last}")
                break

            print(f"[RETRY {attempt}/{POLICY.fetch_attempts}] {remote_rel}: {last}")
            if attempt < POLICY.fetch_attempts:
                _sleep(30 * attempt)
    finally:
        # CỐ Ý không xoá staging: mảnh dở nằm trong Volume là thứ cho phép lần
        # chạy SAU (container khác) resume thay vì tải lại từ 0. Bản dùng /tmp
        # không thể làm điều này vì container chết là mất luôn thư mục.
        if tmpdir.exists():
            kept_bytes = _dir_bytes(tmpdir)
            if kept_bytes:
                print(f"[staging] giữ {_human(kept_bytes)} ở {tmpdir} để lần sau resume")
                _commit_volume("giữ mảnh dở")

    # Lỗi #5: KHÔNG raise. Exception lọt ra .map() sẽ cancel mọi task anh em
    # ("Runner terminated" ×15 trong log) và vứt hết file đang tải dở.
    #
    # `partial` KHÁC `error`: file chỉ dở vì hết ngân sách thời gian, không phải
    # vì nguồn hỏng. Gộp hai thứ này làm báo cáo sai — người đọc tưởng phải sửa
    # gì đó, trong khi việc cần làm chỉ là chạy lại.
    if stopped:
        # `partial` (không phải `error`): việc cần làm chỉ là chạy lại.
        return {"remote": remote_rel, "local": str(local), "status": "partial",
                "error": last, "size": 0, "kept_bytes": kept_bytes,
                "stopped": True}
    if budget_out:
        return {"remote": remote_rel, "local": str(local), "status": "partial",
                "error": last, "size": 0, "kept_bytes": kept_bytes}
    return {"remote": remote_rel, "local": str(local), "status": "error",
            "error": last, "size": 0}


@app.function(
    image=cpu_image,
    secrets=[baidu_secret],
    volumes={"/raw": vol_raw},
    # max_containers là fix của lỗi #3: bản cũ không đặt nên .map() bung ~26
    # container, tự làm Baidu throttle xuống 0B/s. retries=0 vì retry nằm trong
    # vòng lặp thủ công — exception lọt ra .map() sẽ huỷ cả đợt (lỗi #5).
    # Trần an toàn của NỀN TẢNG. Việc chặn thật nằm ở `_attempt_timeout()`
    # bên trong (`_FETCH_RESERVE_S` chừa chỗ cho cold start + validate + commit),
    # vì `FunctionTimeoutError` không bắt được và sẽ giết cả đợt `.map()`.
    timeout=FETCH_CALL_TIMEOUT_S,
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


# ============================================================================
# Nguồn thay thế: HF Hub (dùng SAU khi đã mirror — nhanh hơn Baidu 1-2 bậc)
# ============================================================================
# Đo 2026-10-25: Baidu cho **76.6 KiB/s cho cả tài khoản**, bất kể số container.
# `e2` = 1.94 GB ⇒ 7.4 giờ; `e2,e4` ⇒ 12.9 giờ. HF Hub thường cho vài MB/s.
#
# Đường đi: tải MỘT lần (bằng Baidu hoặc bằng máy nhà, nhanh hơn nhiều) →
# `archive::push_hf` → từ đó mọi lần chạy đều `download::from_hf`.
#
# Đây là lối thoát DUY NHẤT khỏi 77 KiB/s: mọi mẹo trong Baidu (thêm container,
# `-p N`) đều đã đo và đều không tăng tổng băng thông, chỉ tăng rủi ro throttle.
@app.function(
    image=cpu_image,
    volumes={"/raw": vol_raw},
    timeout=ORCHESTRATOR_TIMEOUT_S,
)
def from_hf(repo_id: str = "", tiers: str = POLICY.default_tiers,
            prefix: str = "", token: str = "") -> None:
    """Kéo dữ liệu của tier từ HF **dataset** repo đã mirror.

    ``modal run -m dire_modal.download::from_hf --repo-id <user>/<repo> --tiers e2``

    Không cần secret nếu repo công khai; repo private thì truyền ``--token``
    hoặc dùng secret ``hf-cred``.
    """
    import os as _os

    from huggingface_hub import hf_hub_download

    if not repo_id or "/" not in repo_id:
        raise SystemExit("[!] cần --repo-id dạng <user>/<repo>")

    ents = resolve_many(tiers)
    files, unresolved, _ = expand(ents, lister=None)
    if unresolved:
        print(f"[warn] {len(unresolved)} DirSpec chưa expand (cần --tiers tĩnh)")
    if not files:
        raise SystemExit(f"[!] tier {tiers!r} không có file tĩnh nào")

    tok = token or _os.environ.get("HF_TOKEN") or None
    print(f"[hf] {repo_id} · {len(files)} file · prefix={prefix!r}")
    ok = skipped = failed = 0
    for f in files:
        dest = RAW / f.local
        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            print(f"[SKIP] {f.local}")
            continue
        remote_name = f"{prefix.rstrip('/')}/{f.local}" if prefix else f.local
        try:
            got = hf_hub_download(
                repo_id=repo_id, filename=remote_name, repo_type="dataset",
                token=tok, local_dir=str(RAW),
            )
        except Exception as exc:  # noqa: BLE001 — một file hỏng không được giết cả đợt
            failed += 1
            print(f"[FAIL] {remote_name}: {type(exc).__name__}: {exc}")
            continue
        # `hf_hub_download` với local_dir giữ nguyên cây; nếu nó trả đường dẫn
        # khác `dest` (do prefix) thì chuyển về đúng chỗ manifest mong đợi.
        got_p = Path(got)
        if got_p.exists() and got_p.resolve() != dest.resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                dest.unlink()
            got_p.replace(dest)
        ok += 1
        size = _human(dest.stat().st_size) if dest.exists() else "?"
        print(f"[OK  ] {f.local}  {size}")
    _commit_volume("from_hf")
    print(f"\n[hf] ok={ok} skipped={skipped} failed={failed}")
    if failed:
        print("     File lỗi là file chưa có trên HF — mirror bổ sung rồi chạy lại:")
        print(f"     modal run -m dire_modal.archive::push_hf --repo-id {repo_id}"
              f" --src /raw")


def _bytes_per_sec(total_bytes: int, seconds: float) -> float:
    """Băng thông thực (B/s). Trả 0 khi không đo được — không bịa số."""
    if seconds <= 0 or total_bytes <= 0:
        return 0.0
    return total_bytes / seconds


def _bench_payloads(files: list[File], k: int) -> list[dict]:
    """Payload cho một lượt đo: mỗi lượt có thư mục riêng để KHÔNG dính cache.

    Phải tải lại thật thì mới đo được băng thông; trỏ vào ``/raw/.bench/k<k>/``
    nên lần đo sau không [SKIP] vì file đã có.
    """
    return [
        {"remote": f.remote, "local": f".bench/k{k}/{f.remote}", "expected_size": 0}
        for f in files
    ]


def _bench_report(rows: list[dict]) -> list[str]:
    """Bảng so sánh. Tách khỏi I/O để test được mà không cần GPU/Baidu."""
    out = [
        "",
        "=" * 78,
        "ĐO SONG SONG — bytes/s thực",
        "=" * 78,
        f"  {'k':>3} {'file':>5} {'giây':>8} {'MB':>8} {'KB/s':>10} {'KB/s/file':>10}",
    ]
    for r in sorted(rows, key=lambda x: x["concurrency"]):
        out.append(
            f"  {r['concurrency']:>3} {r['ok']:>5} {r['seconds']:>8.1f} "
            f"{r['bytes'] / 1024**2:>8.1f} {r['bytes_per_sec'] / 1024:>10.1f} "
            f"{r['bytes_per_sec'] / 1024 / max(r['concurrency'], 1):>10.1f}"
        )
    ok_rows = [r for r in rows if r["bytes_per_sec"] > 0]
    if len(ok_rows) >= 2:
        best = max(ok_rows, key=lambda r: r["bytes_per_sec"])
        slow = min(ok_rows, key=lambda r: r["bytes_per_sec"])
        out.append("")
        out.append(
            f"  → nhanh nhất: k={best['concurrency']} "
            f"({best['bytes_per_sec'] / 1024:.1f} KB/s)"
        )
        if best["concurrency"] != slow["concurrency"]:
            ratio = best["bytes_per_sec"] / slow["bytes_per_sec"]
            out.append(f"    so với k={slow['concurrency']}: {ratio:.2f}×")
            # Đo thật 2026-10-25: k=1 → 76.6 KB/s, k=2 → 75.0 KB/s, tỉ lệ 1.02×.
            # Chênh lệch dưới 10 % là nhiễu, KHÔNG phải "k=1 nhanh hơn" —
            # và càng không phải "nhiều container chậm hơn". Nói sai ở đây sẽ
            # đẩy người đọc đi chỉnh một tham số không liên quan.
            if ratio < 1.1:
                out.append(
                    "    ≈ như nhau (chênh < 10 %, trong nhiễu) ⇒ trần tốc độ là của"
                    " TÀI KHOẢN,"
                )
                out.append(
                    "      không phải của số container. Đừng chỉnh max_containers;"
                    " nhanh hơn thì đổi nguồn."
                )
            elif slow["concurrency"] > best["concurrency"]:
                out.append(
                    f"    ⚠ nhiều container CHẬM HƠN thật ({ratio:.2f}×): Baidu throttle"
                    " theo số phiên"
                )
                out.append("      của tài khoản ⇒ hạ max_containers.")
    out.append("")
    out.append("  Cách đọc: k=2 KHÔNG nhanh hơn k=1 (đo được 75.0 vs 76.6 KB/s) thì")
    out.append("  `max_containers: 1` là đúng. Muốn nhanh hơn thật thì đổi NGUỒN")
    out.append("  (`archive::push_hf` + `download::from_hf`), không phải đổi tham số.")
    return out


@app.function(
    image=cpu_image,
    secrets=[baidu_secret],
    volumes={"/raw": vol_raw},
    # Mỗi lượt đo có ngân sách riêng; số lượt = len(ks)
    timeout=ORCHESTRATOR_TIMEOUT_S,
)
def bench_concurrency(tiers: str = "e2", limit: int = 2, ks: str = "1,2",
                      paths: str = "") -> None:
    """Đo băng thông thật ở các mức song song khác nhau.

    ``modal run -m dire_modal.download::bench_concurrency --tiers e2 --limit 2 --ks 1,2``
    ``modal run -m dire_modal.download::bench_concurrency --paths <file> --limit 1 --ks 1``

    Kết quả đã đo 2026-10-25 (387.7 MB/lượt): ``k=1`` 5183.8 s → **76.6 KiB/s**;
    ``k=2`` 5293.3 s → **75.0 KiB/s** ⇒ trần là của TÀI KHOẢN, nên ``tiers.yaml``
    đặt ``max_containers: 1``. Chạy lại hàm này chỉ để kiểm khi đổi tài khoản
    (SVIP) hoặc đổi ``baidu_mode``.

    ``--paths`` để đo trên **một file cụ thể** (dùng cho ``tools/bench_baidu_mode.sh``):
    file nhỏ nhất trong cây là 26.57 MB, tức ~6 phút/mức k ở 76.6 KiB/s.

    Mỗi lượt tải lại THẬT vào ``/raw/.bench/k<k>/`` (không dính cache), đo tổng
    byte nhận được chia cho thời gian, rồi xoá. Chọn ``--limit`` nhỏ: đây là
    phép đo, không phải việc tải dữ liệu.
    """
    _baidu_login()
    if paths.strip():
        files = _files_from_paths(paths)
    else:
        ents = resolve_many(tiers)
        files, _, _ = expand(ents, lister=lambda d: list(_ls_sizes(d)))
    picked = files[: max(1, int(limit))]
    if not picked:
        raise SystemExit("[bench] không có file nào để đo")

    print(f"[bench] {len(picked)} file/lượt · các mức k: {ks}")
    for f in picked:
        print(f"        {f.remote}")

    rows: list[dict] = []
    for k in [max(1, min(int(x), 8)) for x in ks.split(",") if x.strip()]:
        root = RAW / ".bench" / f"k{k}"
        shutil.rmtree(root, ignore_errors=True)
        payloads = _bench_payloads(picked, k)
        t0 = time.time()
        res = list(
            fetch_one.with_options(max_containers=k).map(payloads, order_outputs=False)
        )
        dt = time.time() - t0
        got = [r for r in res if r["status"] in {"ok", "skipped"}]
        total = sum(int(r.get("size") or 0) for r in got)
        rows.append(
            {
                "concurrency": k,
                "ok": len(got),
                "attempted": len(res),
                "seconds": dt,
                "bytes": total,
                "bytes_per_sec": _bytes_per_sec(total, dt),
            }
        )
        shutil.rmtree(root, ignore_errors=True)
        vol_raw.commit()

    for line in _bench_report(rows):
        print(line)
    out = RAW / "_bench_concurrency.json"
    out.write_text(
        json.dumps(
            {
                "tiers": tiers,
                "paths": paths,
                "files": [f.remote for f in picked],
                "n": len(picked),
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    vol_raw.commit()
    print(f"\n[bench] → {out}")


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
        if rel in keep or _is_internal(rel):
            continue
        size = p.stat().st_size
        total += size
        orphans.append((p, size))

    # --- vùng nội bộ: không phải dữ liệu, cũng KHÔNG phải "file thừa" ---------
    # `_is_internal()` cố ý loại chúng khỏi danh sách orphan (đem `.staging` đi
    # `gzip -t` thì lúc nào cũng "hỏng"), nhưng im lặng về chúng thì có hại:
    # `.bench` từng chiếm 775 MB mà không lệnh nào nhắc tới. Ba vùng, ba kết
    # luận khác nhau — nên in riêng, kèm nhãn "xoá được / ĐỪNG xoá".
    def _dir_stat(d: Path) -> tuple[int, int]:
        files = [p for p in d.rglob("*") if p.is_file()]
        return len(files), sum(p.stat().st_size for p in files)

    staging = _staging_root()
    n_st, b_st = _dir_stat(staging) if staging.is_dir() else (0, 0)
    held = staging / "_held"
    n_hd, b_hd = _dir_stat(held) if held.is_dir() else (0, 0)

    if n_st - n_hd:
        print(f"\n[internal] .staging: {_human(b_st - b_hd)} · {n_st - n_hd} mảnh dở"
              "  ← TIẾN ĐỘ, ĐỪNG xoá (lần chạy sau resume từ đây)")
    if n_hd:
        print(f"[internal] .staging/_held: {_human(b_hd)} · {n_hd} file"
              "  ← file không qua kiểm, KHÔNG bị xoá tự động")
        print("            kiểm tay (`gzip -t`) rồi xoá nếu chắc là rác")
    bench = RAW / ".bench"
    if bench.is_dir():
        n_b, b_b = _dir_stat(bench)
        if n_b:
            print(f"[internal] .bench: {_human(b_b)} · {n_b} file"
                  "  ← RÁC của bench_concurrency, xoá được")
            print(f"            MSYS_NO_PATHCONV=1 modal volume rm {RAW_VOLUME} "
                  "/raw/.bench -r")
    state = _state_dir()
    if state.is_dir():
        n_s, b_s = _dir_stat(state)
        if n_s:
            print(f"[internal] .state: {_human(b_s)} · {n_s} file"
                  "  ← cache kiểm tra, xoá được (lần sau kiểm lại từ đầu)")
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
        rel = str(p.relative_to(RAW))
        if _is_internal(rel):
            continue  # file dở trong staging không phải dữ liệu để kiểm
        n += 1
        v = _validate_cached(rel, p)
        tag = "OK  " if v.ok else "FAIL"
        print(f"[{tag}] {_human(p.stat().st_size):>10}  {p.relative_to(RAW)}")
        if v.ok:
            continue
        # Chẩn đoán định dạng: "gzip FAIL" không nói được gì; "là rar/HTML"
        # mới cho biết phải làm gì.
        print(f"        ↳ {v.reason}")
        bad.append((rel, v.reason, v.retryable))

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


# Ba volume của repo, mount cùng lúc để `tree` soi được bất kỳ volume nào bằng
# đường dẫn tuyệt đối (/raw/..., /ckpt/..., /shards/...). Mount thêm volume
# KHÔNG tốn credit — chỉ là bind mount vào container đang chạy.
ALL_VOLUMES = {"/raw": vol_raw, "/ckpt": vol_ckpt, "/shards": vol_shards}


@app.function(image=cpu_image, volumes=ALL_VOLUMES, timeout=1800)
def tree(
    path: str = "/raw",
    depth: int = 0,
    flat: bool = False,
    hidden: bool = True,
    out: str = "",
) -> dict:
    """Liệt kê **toàn bộ** file dưới ``path`` dưới dạng cây (hoặc danh sách phẳng).

    ``modal run -m dire_modal.download::tree``
    ``modal run -m dire_modal.download::tree --path /raw/checkpoints``
    ``modal run -m dire_modal.download::tree --flat --out /raw/_tree_dire-raw.txt``

    Vì sao cần: ``modal volume ls`` **chỉ đọc một tầng** (không có ``--recursive``),
    nên không thể ra cây bằng CLI. Và khi stdout không phải TTY nó chỉ in đường
    dẫn — mất hết size. Hàm này đi một lần bằng ``os.scandir`` trên mount FUSE,
    rẻ hơn nhiều so với gọi CLI trong vòng lặp (mỗi lần một round-trip gRPC).

    ``depth=0`` = vô hạn · ``flat=True`` = ``<size>\t<đường_dẫn>`` sort theo path
    (dùng để grep/đối chiếu manifest) · ``out`` = ghi thêm ra file trong Volume rồi
    commit, phòng khi log bị cắt.
    """
    path = _demangle(path)
    out = _demangle(out)
    root = Path(path)
    if not root.exists():
        raise SystemExit(
            f"[!] {path} không tồn tại. Volume mount: {sorted(ALL_VOLUMES)}"
        )
    if not root.is_dir():
        raise SystemExit(f"[!] {path} là file, không phải thư mục")

    if flat:
        rows = _flat_lines(root, show_hidden=hidden)
        body = [f"{_human(sz)}\t{pp}" for sz, pp in rows]
        n_dir = sum(1 for _ in os.walk(root))
        n_file, total = len(rows), sum(sz for sz, _ in rows)
        text = "\n".join(body)
    else:
        body, n_dir, n_file, total = _tree_lines(root, depth, hidden)
        text = "\n".join(body)

    # Ẩn file của chính lần chạy này khỏi phần đếm (nếu `out` nằm trong `path`)
    print("=" * 78)
    print(f"{'DANH SÁCH' if flat else 'CÂY THƯ MỤC'} — {path}")
    print("=" * 78)
    print(text)

    # Tổng theo thư mục con trực tiếp: trả lời ngay "cái gì đang chiếm chỗ"
    subs: list[tuple[str, int, int]] = []
    for child in sorted(root.iterdir(), key=lambda c: c.name):
        if not child.is_dir() or (not hidden and child.name.startswith(".")):
            continue
        rows = _flat_lines(child, show_hidden=hidden)
        subs.append((child.name + "/", len(rows), sum(sz for sz, _ in rows)))
    subs.sort(key=lambda t: t[2], reverse=True)

    print("\n" + "-" * 78)
    print(f"  tổng: {n_dir} thư mục · {n_file} file · {_human(total)}")
    if subs:
        print("  theo thư mục con:")
        for name, cnt, sz in subs:
            print(f"    {_human(sz):>9}  {cnt:>5} file  {name}")
    print("-" * 78)

    if out:
        op = Path(out)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(text + "\n", encoding="utf-8")
        for vol in ALL_VOLUMES.values():
            try:
                vol.commit()
            except Exception:  # noqa: BLE001 — commit lỗi ≠ mất file
                pass
        print(f"[tree] đã ghi {out} ({_human(op.stat().st_size)})")
        for mnt, vol_name in (("/raw", "dire-raw"), ("/ckpt", "dire-ckpt"),
                              ("/shards", "dire-shards")):
            if out.startswith(mnt + "/"):
                print(f"[tree] tải về: modal volume get {vol_name} "
                      f"{out[len(mnt):]} ./")
                break

    return {
        "path": path,
        "dirs": n_dir,
        "files": n_file,
        "bytes": total,
        "children": [{"name": n, "files": c, "bytes": b} for n, c, b in subs],
        "out": out,
    }


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
    print(f"staging          = {_staging_root()}  (env DIRE_STAGING_DIR ghi đè)")
    print(f"commit định kỳ   = {_commit_interval():g}s khi đang tải")
    print(f"baidu_parallel   = {POLICY.baidu_parallel}  (1 = BẮT BUỘC với user free)")
    print(f"baidu_mode       = {POLICY.baidu_mode or '(mặc định của client)'}")
    print(f"ngân sách 1 file = {FETCH_CALL_TIMEOUT_S}s/attempt-budget "
          f"(chừa {_FETCH_RESERVE_S}s, tối thiểu {_MIN_ATTEMPT_S}s/attempt)")
    print(f"watchdog treo    = {POLICY.stall_timeout_s}s không lớn thêm → giết (0 = tắt)")
    print(f"timeout main     = {ORCHESTRATOR_TIMEOUT_S}s")
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
