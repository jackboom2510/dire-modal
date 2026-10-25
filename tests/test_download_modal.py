"""Test chống tái phát cho `download.py`, chạy OFFLINE, không cần Modal.

Vì sao bộ test này tồn tại
--------------------------
Bản đầu của `probe_pair` gọi thẳng ``fetch_one(...)``. Dưới ``@app.function``,
tên đó là ``modal.Function`` chứ không phải hàm Python, nên lệnh chết ngay:

    TypeError: 'Function' object is not callable

Bộ test cũ KHÔNG THỂ bắt được lỗi này vì nó stub ``app.function`` thành decorator
*identity* — tức là nó xoá đúng cái hành vi đang gây lỗi. Bài học: stub phải mô
phỏng đúng ngữ nghĩa (trả object có ``.map``/``.remote``, KHÔNG trả lại hàm).

Chạy::

    python3 -m pytest tests/ -q
    # hoặc không có pytest:
    python3 tests/test_download_modal.py
"""

from __future__ import annotations

import ast
import os
import sys
import tempfile
import time
import types
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
MODULE = SRC / "dire_modal" / "download.py"


# ---------------------------------------------------------------------------
# Stub Modal: mô phỏng ĐÚNG hành vi — decorator trả object, không trả hàm.
# ---------------------------------------------------------------------------
class FakeFunction:
    """Giống ``modal.Function``: có ``.map/.remote/.local`` nhưng KHÔNG gọi được."""

    def __init__(self, fn, kw):
        self._fn = fn
        self._kw = kw
        self.__name__ = getattr(fn, "__name__", "fn")
        self.__doc__ = getattr(fn, "__doc__", None)

    def __call__(self, *a, **k):
        raise TypeError("'Function' object is not callable")

    def map(self, items, **kw):
        return [self._fn(x) for x in items]

    def remote(self, *a, **k):
        return self._fn(*a, **k)

    def local(self, *a, **k):
        return self._fn(*a, **k)

    def with_options(self, **kw):
        merged = {**self._kw, **kw}
        return FakeFunction(self._fn, merged)


class _Img:
    def apt_install(self, *a, **k):
        return self

    def pip_install(self, *a, **k):
        return self

    def run_commands(self, *a, **k):
        return self


class _Vol:
    def __init__(self):
        self.commits = 0

    def commit(self, *a, **k):
        self.commits += 1


class _App:
    def __init__(self, *a, **k):
        pass

    def function(self, *a, **k):
        def deco(fn):
            return FakeFunction(fn, k)

        return deco

    def local_entrypoint(self, *a, **k):
        def deco(fn):
            return FakeFunction(fn, {"local": True})

        return deco


def install_stub():
    """Cài stub vào ``sys.modules`` rồi import module thật. Trả (module, vol)."""
    vol = _Vol()
    mod = types.ModuleType("modal")
    mod.App = _App
    mod.Image = types.SimpleNamespace(debian_slim=lambda **k: _Img())
    mod.Volume = types.SimpleNamespace(from_name=lambda *a, **k: vol)
    mod.Secret = types.SimpleNamespace(from_name=lambda *a, **k: object())
    mod.Retries = lambda **k: types.SimpleNamespace(**k)
    sys.modules["modal"] = mod

    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from dire_modal import download  # noqa: PLC0415

    # Patch MÓC riêng, KHÔNG patch time.sleep (module time là của cả process —
    # patch nó làm mọi sleep khác, kể cả trong pytest, thành no-op).
    download._sleep = lambda _s: None
    return download, vol


# ---------------------------------------------------------------------------
# BaiduPCS-Go giả — tự chứa, không phụ thuộc file ngoài repo
# ---------------------------------------------------------------------------
_FAKE_BAIDU = """#!/usr/bin/env bash
cmd="$1"; shift
case "$cmd" in
  login) exit 0 ;;
  ls)
    d="${*: -1}"
    case "$d" in
      *images/test/lsun_bedroom)
        for n in adm dalle2 ddpm diff-projectedgan diff-stylegan iddpm if ldm midjourney \
                 pndm projectedgan real sdv1_ddim5s_reversed sdv1_ddim5s_reversed_res \
                 sdv1_new sdv1_new2 sdv1_new_ddim10s_reversed sdv1_new_ddim10s_reversed_res \
                 sdv1_new_ddim50s_reversed sdv1_new_ddim50s_reversed_res \
                 sdv1_newddim5s_reversed sdv1_newddim5s_reversed_res sdv2 \
                 stylegan_official vqdiffusion; do
          echo " 1 - 2023-08-01 12:00:00 100.00MB \u5df2\u4fdd\u5b58 /apps/AirExplorer/$d/$n.tar.gz"; done ;;
      *dire/test/lsun_bedroom/imagenet)
        for n in adm ddpm iddpm ldm pndm real sdv1 sdv2 stylegan vqdiffusion; do
          echo " 1 - 2023-08-01 12:00:00 120.00MB \u5df2\u4fdd\u5b58 /apps/AirExplorer/$d/$n.tar.gz"; done ;;
      *)
        echo " 1 - 2023-08-01 12:00:00 269.45MB \u5df2\u4fdd\u5b58 /apps/AirExplorer/$d/adm.tar.gz"
        echo " 2 - 2023-08-01 12:00:00 106.15MB \u5df2\u4fdd\u5b58 /apps/AirExplorer/$d/real.tar.gz" ;;
    esac
    exit 0 ;;
  d)
    saveto=""; remote=""
    while [ $# -gt 0 ]; do case "$1" in --saveto) saveto="$2"; shift 2 ;; *) remote="$1"; shift ;; esac; done
    base=$(basename "$remote"); mkdir -p "$saveto"
    if [ -n "${COUNTER:-}" ]; then echo "x" >> "$COUNTER"; fi
    if [ "${MODE:-ok}" = "zero" ]; then
      # Tái hiện đúng lỗi thật: BaiduPCS-Go exit 0 nhưng chỉ tạo file 0 byte
      : > "$saveto/$base"
    elif [ "${MODE:-ok}" = "rar" ]; then
      # Tái hiện ca THẬT 2026-10-21: đúng số byte nhưng KHÔNG phải gzip.
      printf 'Rar!\x1a\x07\x00' > "$saveto/$base"
      head -c 2000 /dev/zero >> "$saveto/$base"
    elif [ "${MODE:-ok}" = "partial" ]; then
      # Tải dở: có file, đúng vài trăm byte, exit 0 — BaiduPCS-Go thật làm vậy
      head -c 300 /dev/urandom > "$saveto/$base"
    elif [ "${MODE:-ok}" = "stall" ]; then
      # Treo đúng như log thật 2026-10-25: in progress bar 0B/s, KHÔNG ghi byte
      # nào, và không tự bỏ cuộc — chỉ watchdog mới giết được.
      sleep "${STALL_S:-30}"
    elif [ "${MODE:-ok}" = "html" ]; then
      printf '<!DOCTYPE html><html><body>BDUSS expired</body></html>' > "$saveto/$base"
    elif [ "${MODE:-ok}" = "copy" ]; then
      # Test tự dựng nội dung (tar không nén, tar cụt, ...) rồi cho binary
      # chép ra. SRC_FILE_A / SRC_FILE_B cho phép hai nhánh khác nội dung.
      case "$remote" in
        *imagenet*) src="${SRC_FILE_B:-$SRC_FILE}" ;;
        *)          src="${SRC_FILE_A:-$SRC_FILE}" ;;
      esac
      cp "$src" "$saveto/$base"
    else
      t=$(mktemp -d); mkdir -p "$t/0_real" "$t/1_fake"
      echo "$remote/r" > "$t/0_real/a.jpg"; echo "$remote/f" > "$t/1_fake/b.jpg"
      # Byte PHẢI tất định: `tar -czf` ghi mtime hiện tại vào gzip header, nên
      # hai attempt khác giây cho sha256 khác → test "hai lần tải giống hệt"
      # fail NGẪU NHIÊN (đã xảy ra). --mtime cố định + `gzip -n` (bỏ tên/mtime).
      tar --mtime='@0' --owner=0 --group=0 --numeric-owner \
          -cf "$t/x.tar" -C "$t" 0_real 1_fake
      gzip -n -f "$t/x.tar"
      mv "$t/x.tar.gz" "$saveto/$base"; rm -rf "$t"
    fi
    echo "[1] 0B/269.45MB 0B/s in 10m3s, left -"
    echo "\u4e0b\u8f7d\u7ed3\u675f, \u65f6\u95f4: 0s, \u6570\u636e\u603b\u91cf: 0B"; exit 0 ;;
esac
"""


def _fake_baidu_path() -> str:
    """Tạo BaiduPCS-Go giả trong tempdir, trả thư mục để prepend vào PATH."""
    import stat

    d = Path(tempfile.mkdtemp(prefix="fakebaidu_"))
    exe = d / "BaiduPCS-Go"
    exe.write_text(_FAKE_BAIDU, encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(d)


# ---------------------------------------------------------------------------
# 1. Kiểm tra TĨNH qua AST — bắt lỗi mà stub không thấy được
# ---------------------------------------------------------------------------
def _called_names(fn_node: ast.FunctionDef) -> set[str]:
    out = set()
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
    return out


def test_probe_pair_does_not_call_modal_function():
    """`probe_pair` KHÔNG được gọi thẳng `fetch_one` (nó là modal.Function)."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    probe = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "probe_pair"
    )
    called = _called_names(probe)
    assert "fetch_one" not in called, (
        "probe_pair gọi thẳng fetch_one() → sẽ chết với "
        "TypeError: 'Function' object is not callable. Dùng _fetch_one_impl()."
    )
    assert "_fetch_one_impl" in called, "probe_pair phải dùng _fetch_one_impl()"


def test_only_plain_functions_are_called_directly():
    """Không Function nào bị gọi trực tiếp trong thân module (bắt lỗi tương lai)."""
    src = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    # tên các hàm CÓ decorator @app.function
    decorated = {
        n.name for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and any(
            isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
            and d.func.attr == "function"
            for d in n.decorator_list
        )
    }
    assert "fetch_one" in decorated, "fetch_one phải còn được trang trí @app.function"
    assert "probe_pair" in decorated

    plain = {
        n.name for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and not any(
            isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
            and d.func.attr in {"function", "local_entrypoint"}
            for d in n.decorator_list
        )
    }
    # chỉ xét lời gọi dạng `name(...)` thuần, bỏ qua `x.name(...)`
    offenders = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call) or not isinstance(n.func, ast.Name):
            continue
        if n.func.id in decorated:
            offenders.append((n.lineno, n.func.id))
    assert not offenders, (
        f"Gọi thẳng Function của Modal tại dòng {offenders} → "
        "TypeError: 'Function' object is not callable. "
        "Tách thân ra hàm thường (như _fetch_one_impl) hoặc dùng .map/.remote."
    )
    assert "_fetch_one_impl" in plain


# ---------------------------------------------------------------------------
# 2. Kiểm tra ĐỘNG: hành vi thật dưới stub mô phỏng đúng
# ---------------------------------------------------------------------------
def test_fetch_one_is_function_object_impl_is_plain():
    """`fetch_one` là Function (không gọi được), `_fetch_one_impl` là hàm thường."""
    dl, _ = install_stub()
    assert isinstance(dl.fetch_one, FakeFunction), "fetch_one phải là modal.Function"
    with_raises = False
    try:
        dl.fetch_one({"remote": "x", "local": "x", "expected_size": 0})
    except TypeError:
        with_raises = True
    assert with_raises, "fetch_one KHÔNG được gọi được trực tiếp"
    assert callable(dl._fetch_one_impl) and not isinstance(
        dl._fetch_one_impl, FakeFunction
    ), "_fetch_one_impl phải là hàm Python thường"


def test_probe_pair_runs_without_type_error():
    """Lỗi gốc: probe_pair phải chạy được thay vì TypeError."""
    dl, _ = install_stub()
    os.environ["BDUSS"] = "fake"
    os.environ["MODE"] = "ok"
    os.environ["PATH"] = _fake_baidu_path() + ":" + os.environ.get("PATH", "")
    dl.RAW = Path(tempfile.mkdtemp())
    dl.probe_pair.local()  # không được raise


def test_fetch_one_map_still_parallel():
    """Đường song song của `_execute` vẫn phải dùng `.map()` và còn hoạt động."""
    dl, _ = install_stub()
    os.environ["BDUSS"] = "fake"
    os.environ["MODE"] = "ok"
    os.environ["PATH"] = _fake_baidu_path() + ":" + os.environ.get("PATH", "")
    dl.RAW = Path(tempfile.mkdtemp())
    # `dire_modal.common` đã cache `vol_raw` từ lần import đầu, nên commit rơi
    # vào object ĐÓ chứ không phải `vol` vừa tạo trong install_stub().
    before = dl.vol_raw.commits
    res = list(
        dl.fetch_one.map(
            [{"remote": f"a/{i}.tar.gz", "local": f"a/{i}.tar.gz", "expected_size": 0}
             for i in range(3)]
        )
    )
    assert [r["status"] for r in res] == ["ok"] * 3
    assert dl.vol_raw.commits - before >= 3, "mỗi file phải commit Volume"


def test_zero_byte_is_error_not_silent_success():
    """BaiduPCS-Go exit 0 mà tạo file 0 byte (fix #2 của PR #2) phải bị bắt."""
    dl, _ = install_stub()
    os.environ["BDUSS"] = "fake"
    os.environ["MODE"] = "zero"
    os.environ["PATH"] = _fake_baidu_path() + ":" + os.environ.get("PATH", "")
    dl.RAW = Path(tempfile.mkdtemp())
    try:
        r = dl._fetch_one_impl(
            {"remote": "z/a.tar.gz", "local": "z/a.tar.gz", "expected_size": 0}
        )
    finally:
        os.environ["MODE"] = "ok"
    assert r["status"] == "error", "0 byte phải là error, không được coi là ok"
    assert "0 byte" in r["error"], r["error"]
    assert not (dl.RAW / "z/a.tar.gz").exists(), "không được để file hỏng trong /raw"


# ---------------------------------------------------------------------------
# 3. Timeout: probe_pair tải 2 file ⇒ gấp đôi timeout của một file
# ---------------------------------------------------------------------------
def test_probe_pair_timeout_covers_two_files():
    """`probe_pair` gọi `_fetch_one_impl` HAI lần ⇒ cần 2 ngân sách, không phải 1.

    Công thức cũ (`2×attempts×subprocess + 1800`) chỉ chừa 1800s cho *cả hai*
    lần gọi, trong khi mỗi lần đã có 1800s riêng cho cold start + validate +
    commit. Đó chính là kiểu tính thiếu đã làm task thật chết ở 18 000s
    (2026-10-25): 3×5400 = 16 200 dành cho attempt, chỉ còn 1 800 cho mọi thứ
    khác. Nay mỗi lần gọi tự cắt attempt theo ngân sách của nó
    (`download._attempt_timeout`), nên trần của `probe_pair` phải phủ 2 lần.
    """
    dl, _ = install_stub()
    one = dl.FETCH_CALL_TIMEOUT_S
    got = dl.probe_pair._kw["timeout"]
    assert got == 2 * one + 600, f"timeout probe_pair = {got}, cần {2 * one + 600}"
    assert got > 2 * dl.POLICY.fetch_attempts * dl.POLICY.subprocess_timeout, (
        "phải lớn hơn 2×attempts×subprocess, nếu không hai lần gọi không đủ chỗ"
    )
    assert dl.fetch_one._kw["timeout"] == one, "fetch_one chỉ tải 1 file nên giữ 1×"


# ---------------------------------------------------------------------------
# 3b. Ngân sách thời gian — lần chạy thật 2026-10-25 chết ở 18 000s
#     Log: "Task's current input ... hit its timeout of 18000s", sau khi 3 attempt
#     đốt 3×5400 = 16 200s, chỉ còn 1 800s cho cold start + login + validate +
#     commit ⇒ `FunctionTimeoutError`, mà lỗi này KHÔNG bắt được trong hàm.
# ---------------------------------------------------------------------------
def test_attempt_timeout_shrinks_to_fit_remaining_budget():
    dl, _ = install_stub()
    total = dl.FETCH_CALL_TIMEOUT_S
    # Đầu lần gọi: đủ một attempt đầy đủ.
    first = dl._attempt_timeout(0.0, now=0.0)
    assert first == min(dl.POLICY.subprocess_timeout, total - dl._FETCH_RESERVE_S)
    # Gần hết ngân sách: attempt phải NGẮN LẠI, không được vượt.
    short = dl._attempt_timeout(0.0, now=total - dl._FETCH_RESERVE_S - 1000)
    assert short == 1000, short
    # Hết ngân sách: trả 0 để caller dừng sạch.
    assert dl._attempt_timeout(0.0, now=total) == 0.0
    assert dl._attempt_timeout(0.0, now=total - dl._FETCH_RESERVE_S - 10) == 0.0


def test_sum_of_attempt_timeouts_fits_inside_call_budget():
    """Bất biến của bản vá: tổng các attempt + phần chừa ≤ ngân sách một lần gọi.

    Đây đúng là thứ đã thiếu: 3 × 5400 = 16 200s dành cho attempt trong khi
    ngân sách 18 000s chỉ còn 1 800s cho mọi thứ khác.
    """
    dl, _ = install_stub()
    budget = dl.FETCH_CALL_TIMEOUT_S
    spent, n = 0.0, 0
    while True:
        t = dl._attempt_timeout(0.0, now=spent)
        if t <= 0:
            break
        spent += t
        n += 1
        assert n < 50, "vòng lặp không hội tụ"
    assert spent <= budget - dl._FETCH_RESERVE_S, (spent, budget)
    assert n >= 1, "phải cho ít nhất một attempt"
    # Và ngân sách phải cho ĐỦ số attempt đã cấu hình — nếu không thì bản vá
    # biến "3 lần thử" thành "1 lần thử" một cách âm thầm.
    assert n >= dl.POLICY.fetch_attempts, n


def test_fetch_reports_partial_when_budget_runs_out():
    """Hết ngân sách ⇒ `partial`, KHÔNG phải `error`.

    `error` nghĩa là phải sửa gì đó; `partial` nghĩa là chỉ cần chạy lại. Gộp
    hai thứ này làm báo cáo sai.
    """
    dl, _ = install_stub()
    os.environ["BDUSS"] = "fake"
    os.environ["MODE"] = "partial"   # tải dở, sha khác nhau mỗi lần → retryable
    os.environ["PATH"] = _fake_baidu_path() + ":" + os.environ.get("PATH", "")
    dl.RAW = Path(tempfile.mkdtemp())

    calls = {"n": 0}

    def fake(started, now=None):
        # Attempt 1: một timeout DƯƠNG cố định (900 s) — đừng gọi
        # `_attempt_timeout` thật ở đây. Bản cũ gọi hàm thật nên kết quả phụ
        # thuộc đồng hồ; khi máy chậm/ngân sách tính ra <= 0 thì nhánh "ngân
        # sách" chạy ngay từ attempt 1 và test đỏ ngẫu nhiên (~1/16 lần chạy
        # cả bộ). Thứ test này canh là: attempt 2 thấy 0 ⇒ dừng, KHÔNG chạy
        # attempt 3.
        calls["n"] += 1
        return 900.0 if calls["n"] == 1 else 0.0

    real_timeout = dl._attempt_timeout
    dl._attempt_timeout = fake
    try:
        r = dl._fetch_one_impl(
            {"remote": "p/a.tar.gz", "local": "p/a.tar.gz", "expected_size": 0}
        )
    finally:
        dl._attempt_timeout = real_timeout   # KHÔNG để rò sang test khác
        os.environ["MODE"] = "ok"
    assert r["status"] == "partial", r
    assert "ngân sách" in r["error"], r["error"]
    assert calls["n"] == 2, "phải dừng ngay attempt kế tiếp, không chạy hết 3"


def test_stall_watchdog_kills_hung_download():
    """`0B/s in 42m48s`: BaiduPCS-Go không tự bỏ cuộc ⇒ phải có watchdog giết.

    Không có nó, một phiên treo ngồi hết `subprocess_timeout` (90 phút) và đốt
    sạch ngân sách của lần chạy — đúng thứ đã xảy ra.
    """
    import dataclasses

    dl, _ = install_stub()
    os.environ["BDUSS"] = "fake"
    os.environ["MODE"] = "stall"
    os.environ["STALL_S"] = "30"
    os.environ["PATH"] = _fake_baidu_path() + ":" + os.environ.get("PATH", "")
    dl.RAW = Path(tempfile.mkdtemp())
    old_policy = dl.POLICY
    dl.POLICY = dataclasses.replace(dl.POLICY, stall_timeout_s=0.15)
    old_poll = dl._POLL_S
    dl._POLL_S = 0.02
    try:
        t0 = time.monotonic()
        r = dl._fetch_one_impl(
            {"remote": "s/a.tar.gz", "local": "s/a.tar.gz", "expected_size": 0}
        )
        dt = time.monotonic() - t0
    finally:
        dl._POLL_S = old_poll
        dl.POLICY = old_policy
        os.environ["MODE"] = "ok"
    assert r["status"] == "error", r
    assert "cắt băng thông" in r["error"], r["error"]
    assert dt < 10, f"watchdog không giết kịp ({dt:.1f}s)"


def test_map_failure_keeps_results_already_written():
    """Container orchestrator bị giết ⇒ kết quả của file ĐÃ XONG vẫn phải còn.

    `list(fn.map(...))` chỉ trả về khi mọi task xong; một task chạm trần thời
    gian làm cả lời gọi raise và vứt sạch kết quả, dù dữ liệu đã nằm trong
    Volume. Đây là nửa còn lại của lần chạy hỏng 2026-10-25.
    """
    dl, _ = install_stub()
    os.environ["BDUSS"] = "fake"
    os.environ["PATH"] = _fake_baidu_path() + ":" + os.environ.get("PATH", "")
    dl.RAW = Path(tempfile.mkdtemp())

    made = {"remote": "a/x.tar.gz", "local": "a/x.tar.gz", "status": "ok",
            "size": 10, "sha256": "abc"}

    def boom(items, **kw):
        yield made
        raise RuntimeError("Task hit its timeout of 18000s")

    dl.fetch_one.map = boom
    dl._execute("e2", 25.0, do_fetch=True)  # KHÔNG được raise

    text = (dl.RAW / "_manifest.tsv").read_text(encoding="utf-8")
    assert "a/x.tar.gz" in text, "file đã xong bị mất khỏi manifest"
    assert "ok" in text


def test_fetch_one_keeps_safety_kwargs():
    """Các fix của PR #2 không bị bản vá này làm hỏng."""
    dl, _ = install_stub()
    kw = dl.fetch_one._kw
    # 1, không phải 2: đo 2026-10-25 cho thấy trần ~77 KiB/s là của TÀI KHOẢN.
    assert kw["max_containers"] == dl.POLICY.max_containers == 1
    assert kw["retries"].max_retries == 0
    assert kw["cpu"] == 0.25


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 4. Chẩn đoán định dạng: "gzip FAIL" không đủ để biết phải làm gì
#    (ra đời từ lần chạy thật 2026-10-21: tải đủ 27 863 040 B nhưng gzip fail)
# ---------------------------------------------------------------------------
def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    f = tmp_path / name
    f.write_bytes(data)
    return f


def test_diagnose_recognizes_formats(tmp_path=None):
    import tempfile as _tf

    dl, _ = install_stub()
    d = Path(_tf.mkdtemp())
    cases = {
        "gzip": b"\x1f\x8b\x08\x00" + b"\x00" * 40,
        "zip": b"PK\x03\x04" + b"\x00" * 40,
        "rar": b"Rar!\x1a\x07\x00" + b"\x00" * 40,
        "7z": b"7z\xbc\xaf\x27\x1c" + b"\x00" * 40,
        "bzip2": b"BZh9" + b"\x00" * 40,
        "xz": b"\xfd7zXZ\x00" + b"\x00" * 40,
        "zstd": b"\x28\xb5\x2f\xfd" + b"\x00" * 40,
    }
    got = {}
    for name, magic in cases.items():
        got[name] = dl._diagnose(_write(d, f"x_{name}.bin", magic))
    for name, magic in cases.items():
        assert got[name] == name, f"{name}: nhận thành {got[name]!r}"

    # tar: 'ustar' ở offset 257
    tar_bytes = bytearray(b"\x00" * 512)
    tar_bytes[257:262] = b"ustar"
    assert dl._diagnose(_write(d, "x.tar", bytes(tar_bytes))) == "tar (không nén)"

    # HTML / JSON: Baidu trả trang lỗi
    assert "HTML" in dl._diagnose(_write(d, "h.bin", b"<!DOCTYPE html><html>x</html>"))
    assert "JSON" in dl._diagnose(_write(d, "j.bin", b'{"error_code":2}'))

    # rỗng + rác
    assert dl._diagnose(_write(d, "e.bin", b"")) == "rỗng"
    assert "không rõ" in dl._diagnose(_write(d, "r.bin", bytes(range(256)) * 4))


def test_validate_flags_non_gzip_as_deterministic():
    """File .tar.gz nhưng nội dung là rar ⇒ lỗi TẤT ĐỊNH, không được retry."""
    import tempfile as _tf

    dl, _ = install_stub()
    d = Path(_tf.mkdtemp())
    f = _write(d, "real.tar.gz", b"Rar!\x1a\x07\x00" + b"\x00" * 4000)
    v = dl._validate_ex(f, expected=0)
    assert not v.ok
    assert v.retryable is False, "archive sai tên là lỗi tất định"
    assert v.fmt == "rar", v.fmt
    assert "rar" in v.reason


def test_validate_html_is_deterministic():
    import tempfile as _tf

    dl, _ = install_stub()
    d = Path(_tf.mkdtemp())
    f = _write(d, "x.tar.gz", b"<!DOCTYPE html><html>BDUSS expired</html>")
    v = dl._validate_ex(f, expected=0)
    assert not v.ok and v.retryable is False
    assert "HTML" in v.reason


def test_validate_size_mismatch_is_retryable():
    import tempfile as _tf

    dl, _ = install_stub()
    d = Path(_tf.mkdtemp())
    f = _write(d, "x.tar.gz", b"\x1f\x8b" + b"a" * 100)
    v = dl._validate_ex(f, expected=10_000)
    assert not v.ok and v.retryable is True, "lệch size là lỗi mạng ⇒ đáng retry"


# ---------------------------------------------------------------------------
# Sự cố THẬT 2026-10-25: size từ `ls -l` đã làm tròn 2 chữ số thập phân MB
# ---------------------------------------------------------------------------
# Log:
#   [RETRY 1/3] .../iddpm.tar.gz: size 125015131 ≠ remote 125011230 → tải dở dang
#   [STOP] ... hai lần tải cho ra byte GIỐNG HỆT ... ⇒ lỗi tất định
# File đã tải ĐỦ (Baidu in "下载结束 … 119.223720MB"), nhưng 125011230 chính là
# int(119.22MB × 2^20) — tức số `ls -l` in ra với 2 chữ số thập phân, quy đổi
# ngược ra byte. Đòi khớp từng byte với nó là bắt mọi file đúng phải hỏng.
def test_ls_listing_size_is_rounded_so_slack_must_be_kept():
    dl, _ = install_stub()
    text = (" 1 - 2026-09-24 09:17:25 119.22MB \u5df2\u4fdd\u5b58 "
            "/apps/AirExplorer/dire/test/lsun_bedroom/lsun_bedroom/iddpm.tar.gz\n"
            " 1 - 2026-09-24 09:17:25 124.64MB \u5df2\u4fdd\u5b58 "
            "/apps/AirExplorer/dire/test/lsun_bedroom/lsun_bedroom/pndm.tar.gz\n")
    est = dl._parse_ls_estimates(text)

    iddpm = est["iddpm.tar.gz"]
    assert iddpm.nbytes == 125_011_230, iddpm            # đúng số trong log
    assert iddpm.slack >= 3_901, "phải phủ được sai số làm tròn của 119.22MB"
    pndm = est["pndm.tar.gz"]
    assert pndm.nbytes == 130_694_512, pndm              # đúng số trong log
    assert pndm.slack >= 1_617

    # Size nguyên byte (không phần thập phân) thì KHÔNG có sai số.
    exact = dl._parse_ls_estimates(
        " 1 - 2026-09-24 09:17:25 125015131B /apps/x/iddpm.tar.gz")
    assert exact["iddpm.tar.gz"].nbytes == 125_015_131
    assert exact["iddpm.tar.gz"].slack == 0


def test_validate_accepts_complete_file_when_ls_size_is_rounded():
    """File gzip hợp lệ, lệch `ls -l` trong phạm vi làm tròn ⇒ PHẢI nhận."""
    import gzip as _gz
    import tempfile as _tf

    dl, _ = install_stub()
    d = Path(_tf.mkdtemp())
    body = _gz.compress(b"x" * 5000)
    f = _write(d, "iddpm.tar.gz", body)

    # mô phỏng đúng log: ls -l làm tròn xuống 3901 B so với file thật
    listed = len(body) - 3901
    v = dl._validate_ex(f, expected=listed, slack=5244)
    assert v.ok, v.reason
    assert "gzip" in v.reason or v.fmt == "gzip", v.reason


def test_validate_still_rejects_truncation_beyond_rounding_slack():
    """Sai số làm tròn KHÔNG được nới thành "tha hết": cụt thật thì vẫn loại."""
    import gzip as _gz
    import os as _os
    import tempfile as _tf

    dl, _ = install_stub()
    d = Path(_tf.mkdtemp())
    full = _gz.compress(_os.urandom(200_000))     # dữ liệu nén được ⇒ gzip dài
    f = _write(d, "iddpm.tar.gz", full[: len(full) // 2])   # file CỤT

    v = dl._validate_ex(f, expected=len(full), slack=5244)
    assert not v.ok and v.retryable is True, v.reason
    assert "gzip -t FAIL" in v.reason or "cụt" in v.reason or "FAIL" in v.reason


def test_validate_non_archive_within_slack_is_accepted():
    """``.pth`` không kiểm được bằng gzip ⇒ size (kèm sai số) là bằng chứng."""
    import tempfile as _tf

    dl, _ = install_stub()
    d = Path(_tf.mkdtemp())
    f = _write(d, "lsun_adm.pth", b"\x00" * 20_000)
    assert dl._validate_ex(f, expected=20_000 - 3901, slack=5244).ok
    bad = dl._validate_ex(f, expected=20_000 + 40_000, slack=5244)
    assert not bad.ok and bad.retryable is True


def test_payload_wires_slack_from_ls_into_validation():
    """Sai số phải đi CẢ đường: `_size_map` → payload → `_validate_ex`.

    Đây chính là chỗ hỏng của lần chạy 2026-10-25: phép kiểm biết size đã làm
    tròn nhưng payload chỉ mang con số, không mang sai số.
    """
    import tempfile as _tf

    dl, _ = install_stub()
    os.environ["BDUSS"] = "fake"
    os.environ["MODE"] = "ok"
    os.environ["PATH"] = _fake_baidu_path() + ":" + os.environ.get("PATH", "")
    dl.RAW = Path(_tf.mkdtemp())

    seen: list[dict] = []
    orig = dl._validate_ex

    def capture(path, expected=0, expect_archive=True, slack=0):
        seen.append({"expected": expected, "slack": slack})
        return orig(path, expected, expect_archive=expect_archive, slack=slack)

    dl._validate_ex = capture
    try:
        r = dl._fetch_one_impl({
            "remote": "dire/test/a/iddpm.tar.gz",
            "local": "dire/test/a/iddpm.tar.gz",
            "expected_size": 125_011_230,
            "expect_slack": 5244,
        })
    finally:
        dl._validate_ex = orig
    assert seen and seen[0]["slack"] == 5244, seen
    assert seen[0]["expected"] == 125_011_230, seen
    # File nhỏ hơn `expected` rất nhiều nhưng gzip -t qua ⇒ vẫn phải nhận
    assert r["status"] == "ok", r


def test_meta_is_the_second_opinion_when_ls_size_disagrees():
    """`meta` cho size chính xác + md5 ⇒ cứu được file mà `ls -l` tố sai."""
    import tempfile as _tf

    dl, _ = install_stub()
    d = Path(_tf.mkdtemp())
    f = _write(d, "adm.pth", b"\x00" * 3000)
    bad = dl.Verdict(False, "size 3000 ≠ ls -l 9999", retryable=True)

    orig = dl._remote_meta
    try:
        dl._remote_meta = lambda rel, timeout=120: (3000, "0" * 32)
        v = dl._second_opinion("x/adm.pth", f, bad)
        assert v.ok and "meta" in v.reason, v.reason

        # md5 khớp cũng đủ, dù size `meta` nói khác
        import hashlib
        dl._remote_meta = lambda rel, timeout=120: (None, hashlib.md5(b"\x00" * 3000).hexdigest())
        v2 = dl._second_opinion("x/adm.pth", f, bad)
        assert v2.ok and "md5" in v2.reason, v2.reason

        # Không khớp gì ⇒ giữ nguyên kết luận cũ, và nêu cả hai con số
        dl._remote_meta = lambda rel, timeout=120: (9999, "f" * 32)
        v3 = dl._second_opinion("x/adm.pth", f, bad)
        assert not v3.ok and "9999" in v3.reason and "3000" in v3.reason, v3.reason
    finally:
        dl._remote_meta = orig


def test_parse_meta_reads_size_and_md5_from_real_header():
    dl, _ = install_stub()
    text = """[1] ----
  类型              文件
  文件路径          /apps/AirExplorer/dire/test/lsun_bedroom/lsun_bedroom/iddpm.tar.gz
  文件名称          iddpm.tar.gz
  文件大小          125015131, 119.223720MB
  md5 (截图请打码)  5a6fb52a2bf9a4c1abf0d50bf30955d4
"""
    size, md5 = dl._parse_meta(text)
    assert size == 125_015_131, size
    assert md5 == "5a6fb52a2bf9a4c1abf0d50bf30955d4"


def test_hold_never_deletes_bytes():
    """`_hold` phải CHUYỂN file, không xoá — bài học từ 3 × 119 MB bị xoá."""
    import tempfile as _tf

    dl, _ = install_stub()
    dl.RAW = Path(_tf.mkdtemp())
    src = dl._tmpdir_for("dire/test/a/real.tar.gz")
    src.mkdir(parents=True, exist_ok=True)
    f = src / "real.tar.gz"
    f.write_bytes(b"z" * 1234)

    dest = dl._hold(f)
    assert not f.exists(), "đã chuyển đi thì không còn ở staging"
    assert dest.exists() and dest.stat().st_size == 1234
    assert "_held" in str(dest)
    # Chuyển lần hai cùng tên ⇒ không ghi đè bản cũ
    f.write_bytes(b"y" * 99)
    dest2 = dl._hold(f)
    assert dest2 != dest and dest.exists() and dest2.exists()


def test_cached_valid_not_invalidated_by_rounded_remote_size():
    """Cache ghi sau khi kiểm THÀNH CÔNG ⇒ size làm tròn không được đánh trượt.

    Nếu đánh trượt, đường `fetch_one` sẽ `_hold` bản đã công bố rồi tải lại
    đúng file đó từ 0 B — mỗi lần chạy.
    """
    import tempfile as _tf

    dl, _ = install_stub()
    dl.RAW = Path(_tf.mkdtemp())
    (dl.RAW / "z").mkdir(parents=True, exist_ok=True)
    f = _write(dl.RAW, "z/a.tar.gz", b"q" * 777)
    dl._record_valid("z/a.tar.gz", f, "0" * 64, "gzip")
    assert dl._cached_valid("z/a.tar.gz", f, 777 - 5) is not None


def test_validate_expect_archive_false_skips_gzip():
    """probe_pair cần so sha256 — đòi gzip sẽ chặn đúng câu trả lời cần."""
    import tempfile as _tf

    dl, _ = install_stub()
    d = Path(_tf.mkdtemp())
    f = _write(d, "real.tar.gz", b"Rar!\x1a\x07\x00" + b"\x00" * 100)
    assert dl._validate_ex(f, 0, expect_archive=False).ok, \
        "expect_archive=False phải chấp nhận file không phải gzip"


# ---------------------------------------------------------------------------
# 5. Không đốt thời gian vào lỗi tất định (3 attempt × 6 phút = 19 phút vô ích)
# ---------------------------------------------------------------------------
def test_deterministic_error_stops_after_one_attempt():
    import tempfile as _tf

    dl, _ = install_stub()
    os.environ["BDUSS"] = "fake"
    os.environ["MODE"] = "rar"
    os.environ["PATH"] = _fake_baidu_path() + ":" + os.environ.get("PATH", "")
    counter = Path(_tf.mkdtemp()) / "n.txt"
    os.environ["COUNTER"] = str(counter)
    dl.RAW = Path(_tf.mkdtemp())
    try:
        r = dl._fetch_one_impl(
            {"remote": "z/a.tar.gz", "local": "z/a.tar.gz", "expected_size": 0}
        )
    finally:
        os.environ["MODE"] = "ok"
        os.environ.pop("COUNTER", None)
    assert r["status"] == "error"
    assert r.get("deterministic") is True, r
    assert r["attempts"] == 1, f"phải dừng sau 1 attempt, đã chạy {r['attempts']}"
    assert counter.exists() and len(counter.read_text().splitlines()) == 1, \
        "chỉ được gọi BaiduPCS-Go MỘT lần"


def test_identical_bytes_stop_retries():
    """Byte giống hệt sau khi tải lại TỪ 0 B ⇒ tải lại cũng thế, dừng.

    Số attempt đổi 2 → 3 khi thêm luật "Baidu bỏ qua vì file đã có": lần 2
    Baidu không ghi thêm byte nào nên file được `_hold` rồi lần 3 tải lại từ
    đầu. Chỉ sau khi tải lại từ 0 B mà vẫn ra đúng chuỗi byte đó mới được kết
    luận "tất định" — và kể cả lúc đó cũng KHÔNG xoá file.
    """
    import tempfile as _tf

    dl, _ = install_stub()
    os.environ["BDUSS"] = "fake"
    os.environ["MODE"] = "ok"
    os.environ["PATH"] = _fake_baidu_path() + ":" + os.environ.get("PATH", "")
    dl.RAW = Path(_tf.mkdtemp())
    # giả lập nội dung không đổi + validate luôn fail
    orig = dl._validate_ex
    dl._validate_ex = lambda p, e=0, expect_archive=True, slack=0: dl.Verdict(
        False, "hỏng giả lập", retryable=True
    )
    counter = Path(_tf.mkdtemp()) / "n.txt"
    os.environ["COUNTER"] = str(counter)
    try:
        r = dl._fetch_one_impl(
            {"remote": "z/a.tar.gz", "local": "z/a.tar.gz", "expected_size": 0}
        )
    finally:
        dl._validate_ex = orig
        os.environ.pop("COUNTER", None)
    assert r["status"] == "error"
    assert r.get("deterministic") is True, r
    assert r["attempts"] == 3, f"phải tải lại từ 0 B một lần, đã chạy {r['attempts']}"
    # Byte bị loại KHÔNG được xoá: `_hold` giữ lại (sự cố 2026-10-25 đã xoá
    # 3 × 119 MB của những file thực ra đúng).
    held = Path(r.get("held") or "")
    assert held.exists(), f"file bị loại phải còn trong _held, đang là {held}"


# ---------------------------------------------------------------------------
# 6. probe_pair phải trả lời được cả khi file không giải nén được
# ---------------------------------------------------------------------------
def test_probe_pair_answers_even_when_not_gzip():
    """Regression của lần chạy thật: gzip fail không được chặn sha256."""
    import contextlib
    import io

    dl, _ = install_stub()
    # Hai blob KHÁC nhau (nếu giống hệt, probe rơi vào nhánh "sha256 TRÙNG" và
    # kết luận luôn — đúng, nhưng không kiểm được nhánh cần kiểm ở đây).
    _probe_env(
        dl,
        b"Rar!\x1a\x07\x00" + b"\x01" * 500,
        b"Rar!\x1a\x07\x00" + b"\x02" * 500,
    )
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            dl.probe_pair.local()
    finally:
        os.environ["MODE"] = "ok"
        for k in ("SRC_FILE_A", "SRC_FILE_B"):
            os.environ.pop(k, None)
    out = buf.getvalue()
    assert "sha256" in out, "phải in sha256 dù file không phải gzip"
    assert "KẾT LUẬN" in out, "phải đi tới kết luận, không bỏ cuộc giữa đường"
    assert "rar" in out.lower(), "phải báo định dạng thật"
    assert "KHÔNG so được" in out, "phải nói rõ không so được ảnh thay vì crash"
    assert "CHƯA KẾT LUẬN ĐƯỢC" in out, "không được kết luận bừa khi không đọc được"



# ---------------------------------------------------------------------------
# 7. Dữ liệu thật là tar KHÔNG NÉN, không phải gzip
#    (log 2026-10-21: cả hai file đều "định dạng: tar (không nén)")
# ---------------------------------------------------------------------------
def _make_tar(members: dict[str, bytes], mtime: int = 1_700_000_000,
              compress: bool = False) -> bytes:
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz" if compress else "w") as t:
        for name, data in members.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            ti.mtime = mtime
            t.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


_LSUN = {"0_real/a.jpg": b"AAA" * 100, "0_real/b.jpg": b"BBB" * 100}
_IMNET = {"0_real/x.jpg": b"XXX" * 100, "0_real/y.jpg": b"YYY" * 100}


def test_tar_profile_reads_uncompressed_tar():
    import tempfile as _tf

    dl, _ = install_stub()
    f = Path(_tf.mkdtemp()) / "real.tar.gz"
    f.write_bytes(_make_tar(_LSUN))  # tar KHÔNG nén, tên .tar.gz
    prof = dl._tar_profile(f)
    assert prof["files"] == 2, prof
    assert prof["classes"] == {"0_real": 2}, prof


def test_validate_accepts_uncompressed_tar():
    """Tar hợp lệ dù không nén ⇒ PHẢI chấp nhận. Coi là lỗi sẽ xoá dữ liệu tốt."""
    import tempfile as _tf

    dl, _ = install_stub()
    f = Path(_tf.mkdtemp()) / "real.tar.gz"
    f.write_bytes(_make_tar(_LSUN))
    v = dl._validate_ex(f, expected=f.stat().st_size)
    assert v.ok, v.reason
    assert v.fmt == "tar (không nén)", v.fmt
    assert "không nén" in v.reason.lower()


def test_validate_rejects_truncated_tar():
    """Cắt mất block kết thúc ⇒ tar cụt ⇒ đáng tải lại (retryable)."""
    import tempfile as _tf

    dl, _ = install_stub()
    blob = _make_tar({"0_real/a.jpg": b"A" * 5000})
    f = Path(_tf.mkdtemp()) / "real.tar.gz"
    f.write_bytes(blob[: len(blob) // 2])
    v = dl._validate_ex(f, expected=0)
    assert not v.ok, "tar cụt không được coi là hợp lệ"
    assert v.retryable is True, "tar cụt là lỗi tải dở ⇒ phải retry"


def test_fetch_keeps_uncompressed_tar_and_notes_format():
    """Tải tar không nén ⇒ status ok, file ĐƯỢC GIỮ, và báo định dạng thật."""
    import tempfile as _tf

    dl, _ = install_stub()
    src = Path(_tf.mkdtemp()) / "src.tar"
    src.write_bytes(_make_tar(_LSUN))
    os.environ["BDUSS"] = "fake"
    os.environ["MODE"] = "copy"
    os.environ["SRC_FILE"] = str(src)
    os.environ["PATH"] = _fake_baidu_path() + ":" + os.environ.get("PATH", "")
    dl.RAW = Path(_tf.mkdtemp())
    try:
        r = dl._fetch_one_impl(
            {"remote": "z/a.tar.gz", "local": "z/a.tar.gz", "expected_size": 0}
        )
    finally:
        os.environ["MODE"] = "ok"
        os.environ.pop("SRC_FILE", None)
    assert r["status"] == "ok", r
    assert r["format"] == "tar (không nén)", r
    assert (dl.RAW / "z/a.tar.gz").exists(), "dữ liệu hợp lệ KHÔNG được bị xoá"


def test_pack_opens_any_tar_not_only_gzip():
    """pack.py hardcode "r:gz" sẽ giết cả stage pack với dữ liệu thật.

    Kiểm bằng AST chứ không grep chuỗi: comment giải thích trong file cũng có
    chữ "r:gz", grep thô sẽ báo sai.
    """
    tree = ast.parse((SRC / "dire_modal" / "pack.py").read_text())
    modes = [
        node.args[1].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "open"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tarfile"
        and len(node.args) > 1
        and isinstance(node.args[1], ast.Constant)
    ]
    assert modes, "không tìm thấy tarfile.open(..., mode) trong pack.py"
    assert modes == ["r:*"], f"pack.py phải mở bằng 'r:*', đang là {modes}"


# ---------------------------------------------------------------------------
# 8. So sánh phải theo NỘI DUNG ẢNH, không theo sha256 container
#    (cùng ảnh + khác mtime ⇒ sha256 container khác, size y hệt)
# ---------------------------------------------------------------------------
def test_same_images_different_mtime_compare_identical():
    import hashlib
    import tempfile as _tf

    dl, _ = install_stub()
    d = Path(_tf.mkdtemp())
    a = dl._tar_content(_write(d, "a.tar", _make_tar(_LSUN, mtime=1_700_000_000)))
    b = dl._tar_content(_write(d, "b.tar", _make_tar(_LSUN, mtime=1_700_009_999)))
    blob_a = (d / "a.tar").read_bytes()
    blob_b = (d / "b.tar").read_bytes()
    # đúng cái bẫy: container khác byte, size y hệt
    assert hashlib.sha256(blob_a).hexdigest() != hashlib.sha256(blob_b).hexdigest()
    assert len(blob_a) == len(blob_b)

    cmp = dl._content_compare(a, b)
    assert cmp["status"] == "identical", cmp
    assert cmp["na"] == cmp["nb"] == 2


def test_different_images_compare_disjoint():
    import tempfile as _tf

    dl, _ = install_stub()
    d = Path(_tf.mkdtemp())
    a = dl._tar_content(_write(d, "a.tar", _make_tar(_LSUN)))
    b = dl._tar_content(_write(d, "b.tar", _make_tar(_IMNET)))
    cmp = dl._content_compare(a, b)
    assert cmp["status"] == "disjoint", cmp
    assert cmp["na"] == 2 and cmp["nb"] == 2, "phải đếm được ảnh, không phải -1"


def test_same_images_renamed_compare_same_images():
    import tempfile as _tf

    dl, _ = install_stub()
    d = Path(_tf.mkdtemp())
    a = dl._tar_content(_write(d, "a.tar", _make_tar({"0_real/a.jpg": b"Z" * 400})))
    b = dl._tar_content(
        _write(d, "b.tar", _make_tar({"0_real/khac_ten.jpg": b"Z" * 400}))
    )
    cmp = dl._content_compare(a, b)
    assert cmp["status"] == "same_images", cmp


def test_unreadable_compare_does_not_claim_equality():
    """-1 == -1 KHÔNG được coi là 'số ảnh bằng nhau' (đúng lỗi bản trước)."""
    import tempfile as _tf

    dl, _ = install_stub()
    d = Path(_tf.mkdtemp())
    bad = dl._tar_content(_write(d, "bad.tar.gz", b"Rar!\x1a\x07\x00" + b"\x00" * 500))
    good = dl._tar_content(_write(d, "good.tar", _make_tar(_LSUN)))
    cmp = dl._content_compare(bad, good)
    assert cmp["status"] == "unreadable", cmp
    assert "reason" in cmp


# ---------------------------------------------------------------------------
# 9. probe_pair: hai lần chạy thật, hai kết luận đúng
# ---------------------------------------------------------------------------
def _probe_env(dl, src_a: bytes, src_b: bytes):
    import tempfile as _tf

    d = Path(_tf.mkdtemp())
    pa, pb = d / "a.tar", d / "b.tar"
    pa.write_bytes(src_a)
    pb.write_bytes(src_b)
    os.environ["BDUSS"] = "fake"
    os.environ["MODE"] = "copy"
    os.environ["SRC_FILE_A"] = str(pa)
    os.environ["SRC_FILE_B"] = str(pb)
    os.environ["PATH"] = _fake_baidu_path() + ":" + os.environ.get("PATH", "")
    dl.RAW = Path(_tf.mkdtemp())
    # Patch CẢ HAI: `probe_pair` hỏi `_ls_estimates` (bản giữ sai số làm tròn),
    # còn `_ls_sizes` chỉ là wrapper của nó cho chỗ chỉ cần in số.
    dl._ls_sizes = lambda *a, **k: {}
    dl._ls_estimates = lambda *a, **k: {}
    return d


def _probe_output(dl):
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dl.probe_pair.local()
    return buf.getvalue()


def _probe_cleanup(dl, orig_ls):
    os.environ["MODE"] = "ok"
    for k in ("SRC_FILE_A", "SRC_FILE_B"):
        os.environ.pop(k, None)
    dl._ls_sizes = orig_ls[0]
    dl._ls_estimates = orig_ls[1]


def test_probe_says_same_images_when_only_mtime_differs():
    dl, _ = install_stub()
    orig_ls = (dl._ls_sizes, dl._ls_estimates)
    _probe_env(dl, _make_tar(_LSUN, mtime=1), _make_tar(_LSUN, mtime=999_999))
    try:
        out = _probe_output(dl)
    finally:
        _probe_cleanup(dl, orig_ls)
    assert "CÙNG HOÀN TOÀN" in out, out
    assert "KHÁC HOÀN TOÀN" not in out, "sha256 container khác KHÔNG được gọi là ảnh khác"
    assert "sha256 container : KHÁC" in out
    assert "bằng chứng YẾU" in out


def test_probe_counts_images_and_says_disjoint():
    dl, _ = install_stub()
    orig_ls = (dl._ls_sizes, dl._ls_estimates)
    _probe_env(dl, _make_tar(_LSUN), _make_tar(_IMNET))
    try:
        out = _probe_output(dl)
    finally:
        _probe_cleanup(dl, orig_ls)
    assert "KHÁC HOÀN TOÀN" in out, out
    assert "KHÔNG đọc được" not in out
    assert "-1" not in out, "không được để -1 lọt vào kết luận" 
    assert "test domain: HỢP LÝ" in out



# ---------------------------------------------------------------------------
# 10. Staging nằm TRONG Volume, không phải /tmp ephemeral
# ---------------------------------------------------------------------------
def test_staging_lives_in_volume_by_default():
    dl, _ = install_stub()
    dl.RAW = Path(tempfile.mkdtemp())
    d = dl._tmpdir_for("dire/test/a/real.tar.gz")
    assert str(d).startswith(str(dl.RAW)), (
        f"staging phải nằm trong Volume để sống qua container, đang ở {d}"
    )
    assert d.parent.name == dl.POLICY.dire_staging, d.parent
    assert d.parent.parent == dl.RAW, f"staging phải là con trực tiếp của Volume: {d}"


def test_staging_can_be_overridden_to_tmp():
    """Chốt an toàn: muốn đổi lại /tmp thì phải qua env, và phải nói rõ hậu quả."""
    dl, _ = install_stub()
    dl.RAW = Path(tempfile.mkdtemp())
    old = os.environ.get("DIRE_STAGING_DIR")
    os.environ["DIRE_STAGING_DIR"] = "/tmp/dire_staging_test"
    try:
        d = dl._tmpdir_for("x/y.tar.gz")
    finally:
        if old is None:
            os.environ.pop("DIRE_STAGING_DIR", None)
        else:
            os.environ["DIRE_STAGING_DIR"] = old
    assert str(d).startswith("/tmp/dire_staging_test"), d


def test_staging_dir_is_stable_across_calls():
    """Cùng remote ⇒ cùng thư mục, nếu không thì không resume được."""
    dl, _ = install_stub()
    dl.RAW = Path(tempfile.mkdtemp())
    a = dl._tmpdir_for("train/adm.tar.gz")
    b = dl._tmpdir_for("train/adm.tar.gz")
    c = dl._tmpdir_for("val/adm.tar.gz")
    assert a == b, "phải ổn định giữa các attempt/lần chạy"
    assert a != c, "hai file cùng tên khác thư mục KHÔNG được chung staging"


# ---------------------------------------------------------------------------
# 11. Công bố file: rename cùng filesystem, không copy
# ---------------------------------------------------------------------------
def test_publish_renames_instead_of_copying():
    """Staging trong Volume ⇒ os.replace (O(1)). Copy là lỗi hiệu năng thật."""
    import unittest.mock as mock

    dl, _ = install_stub()
    d = Path(tempfile.mkdtemp())
    got, local = d / "staging.bin", d / "final.bin"
    got.write_bytes(b"x" * 1000)
    with mock.patch.object(dl.os, "replace", wraps=dl.os.replace) as rep, \
         mock.patch.object(dl.shutil, "move", wraps=dl.shutil.move) as mv:
        dl._publish(got, local)
    assert rep.called, "phải dùng os.replace khi cùng filesystem"
    assert not mv.called, "không được copy khi cùng filesystem"
    assert local.read_bytes() == b"x" * 1000 and not got.exists()


def test_publish_falls_back_to_copy_across_filesystems():
    """DIRE_STAGING_DIR=/tmp ⇒ khác filesystem ⇒ EXDEV ⇒ phải copy được."""
    dl, _ = install_stub()
    import shutil as _sh

    if not Path("/dev/shm").is_dir():
        return
    src_dir = Path(tempfile.mkdtemp(dir="/dev/shm"))
    try:
        got = src_dir / "staging.bin"
        got.write_bytes(b"y" * 500)
        local = Path(tempfile.mkdtemp()) / "final.bin"
        dl._publish(got, local)  # không được raise EXDEV
        assert local.read_bytes() == b"y" * 500
    finally:
        _sh.rmtree(src_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 12. Commit định kỳ — thứ làm resume xuyên container có tác dụng
# ---------------------------------------------------------------------------
def test_periodic_commit_calls_volume_commit():
    dl, _ = install_stub()
    before = dl.vol_raw.commits
    with dl._PeriodicCommit(0.01) as pc:
        time.sleep(0.15)
    assert dl.vol_raw.commits - before >= 2, "phải commit trong lúc đang tải"
    assert pc.commits >= 2


def test_periodic_commit_disabled_when_zero():
    dl, _ = install_stub()
    before = dl.vol_raw.commits
    with dl._PeriodicCommit(0) as pc:
        time.sleep(0.05)
    assert dl.vol_raw.commits == before
    assert pc.commits == 0


def test_commit_interval_is_clamped():
    dl, _ = install_stub()
    import dataclasses as _dc

    orig = dl.POLICY
    try:
        dl.POLICY = _dc.replace(orig, staging_commit_s=99_999)
        assert dl._commit_interval() == 600, "commit 27 giờ một lần là vô nghĩa"
        dl.POLICY = _dc.replace(orig, staging_commit_s=-5)
        assert dl._commit_interval() == 0
    finally:
        dl.POLICY = orig


# ---------------------------------------------------------------------------
# 13. Tải hỏng KHÔNG được xoá mảnh dở (đó là tiến độ)
# ---------------------------------------------------------------------------
def test_failed_download_keeps_partial_for_resume():
    dl, _ = install_stub()
    os.environ["BDUSS"] = "fake"
    os.environ["MODE"] = "partial"
    os.environ["PATH"] = _fake_baidu_path() + ":" + os.environ.get("PATH", "")
    dl.RAW = Path(tempfile.mkdtemp())
    try:
        r = dl._fetch_one_impl(
            {"remote": "z/a.tar.gz", "local": "z/a.tar.gz", "expected_size": 10_000}
        )
    finally:
        os.environ["MODE"] = "ok"
    assert r["status"] == "error", r
    staging = dl._tmpdir_for("z/a.tar.gz")
    left = [p for p in staging.rglob("*") if p.is_file()]
    assert left, (
        "mảnh dở phải được GIỮ để lần chạy sau resume; bản /tmp xoá nó đi "
        "nên container chết là mất tiến độ"
    )
    assert not (dl.RAW / "z/a.tar.gz").exists(), "file dở không được lọt vào /raw"


# ---------------------------------------------------------------------------
# 14. Cache kiểm tra: đừng đọc lại 3.39 GB mỗi lần chạy
# ---------------------------------------------------------------------------
def test_validation_cache_skips_second_validation():
    dl, _ = install_stub()
    os.environ["BDUSS"] = "fake"
    os.environ["MODE"] = "ok"
    os.environ["PATH"] = _fake_baidu_path() + ":" + os.environ.get("PATH", "")
    dl.RAW = Path(tempfile.mkdtemp())
    payload = {"remote": "c/a.tar.gz", "local": "c/a.tar.gz", "expected_size": 0}
    first = dl._fetch_one_impl(payload)
    assert first["status"] == "ok", first

    # Nếu cache hỏng, lần hai sẽ gọi _validate_ex (đọc cả file).
    def _boom(*a, **k):
        raise AssertionError("đã kiểm lại từ đầu thay vì dùng cache")

    orig = dl._validate_ex
    dl._validate_ex = _boom
    try:
        second = dl._fetch_one_impl(payload)
    finally:
        dl._validate_ex = orig
    assert second["status"] == "skipped", second
    assert second["sha256"] == first["sha256"], "sha256 phải lấy từ cache, không đọc lại"


def test_validation_cache_invalidated_when_file_changes():
    dl, _ = install_stub()
    d = Path(tempfile.mkdtemp())
    f = d / "x.bin"
    f.write_bytes(b"a" * 100)
    dl._record_valid("x.bin", f, "sha-a", "gzip")
    assert dl._cached_valid("x.bin", f) is not None
    time.sleep(0.01)
    f.write_bytes(b"b" * 200)  # size + mtime đổi
    assert dl._cached_valid("x.bin", f) is None, "file đổi thì KHÔNG được tin cache"


def test_validation_cache_never_caches_failures():
    dl, _ = install_stub()
    d = Path(tempfile.mkdtemp())
    f = d / "bad.tar.gz"
    f.write_bytes(b"Rar!\x1a\x07\x00" + b"\x00" * 200)
    v = dl._validate_cached("bad.tar.gz", f)
    assert not v.ok
    assert dl._load_state().get("bad.tar.gz") is None, (
        "cache âm mà sai thì file lành sẽ bị tải lại mãi mãi"
    )


# ---------------------------------------------------------------------------
# 15. Vùng nội bộ không được lọt vào verify/prune
# ---------------------------------------------------------------------------
def test_verify_ignores_staging_and_state():
    import contextlib
    import io

    dl, _ = install_stub()
    dl.RAW = Path(tempfile.mkdtemp())
    staging = dl._tmpdir_for("z/a.tar.gz")
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "a.tar.gz").write_bytes(b"d\x1f\x8b khong phai tar")
    (dl.RAW / ".state").mkdir(parents=True, exist_ok=True)
    (dl.RAW / ".state" / "validated.json").write_text("{}")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dl.verify.local()  # không được SystemExit vì file dở "hỏng"
    out = buf.getvalue()
    assert "đã kiểm 0 file" in out, out
    assert "All OK." in out, out


def test_internal_paths_detected():
    dl, _ = install_stub()
    for rel in (".staging/x/y.tar.gz", ".state/validated.json", ".staging"):
        assert dl._is_internal(rel), rel
    for rel in ("dire/test/a/real.tar.gz", "checkpoints/x.pth"):
        assert not dl._is_internal(rel), rel


# ---------------------------------------------------------------------------
# 16. Concurrency: tham số phải có tác dụng thật (trước đây bị bỏ qua)
# ---------------------------------------------------------------------------
def test_clamp_concurrency():
    dl, _ = install_stub()
    assert dl._clamp_concurrency(0) == 1
    assert dl._clamp_concurrency(4) == 4
    assert dl._clamp_concurrency(26) == 8, "26 phiên là đúng cấu hình làm Baidu throttle"
    assert dl._clamp_concurrency(-3) == 1


def test_fetch_one_exposes_with_options_for_concurrency_override():
    dl, _ = install_stub()
    kw = dl.fetch_one._kw
    over = dl.fetch_one.with_options(max_containers=4)
    assert over._kw["max_containers"] == 4
    assert over._kw["timeout"] == kw["timeout"], "override không được làm mất timeout"


if __name__ == "__main__":
    import traceback

    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  OK   {name}")
            ok += 1
        except Exception:
            print(f"  FAIL {name}")
            traceback.print_exc()
    print(f"\n  {ok}/{len(tests)} test đạt")
    sys.exit(0 if ok == len(tests) else 1)


# ---------------------------------------------------------------------------
# 6. Đừng bật container cho file đã có
#     Trước: payload dựng cho MỌI file rồi mới .map() ⇒ mỗi file đã xong vẫn tốn
#     một container (cold start + mount Volume) chỉ để in [SKIP].
# ---------------------------------------------------------------------------
def _cache_entry(path: Path) -> dict:
    st = path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": "x",
            "fmt": "tar (không nén)", "checked": "2026-10-25T00:00:00+00:00"}


def _prepare_execute(dl, present: int, monkeypatch_map=None):
    """Dựng /raw có `present` file đầu của tier e2 + cache hợp lệ cho chúng."""
    dl.RAW = Path(tempfile.mkdtemp())
    dl._ls_sizes = lambda _d: {}          # không gọi Baidu cho phần size
    from dire_modal.manifest import resolve_many, expand

    files, _, _ = expand(resolve_many("e2"))
    state = {}
    for f in files[:present]:
        p = dl.RAW / f.local
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * 128)
        state[f.local] = _cache_entry(p)
    (dl.RAW / ".state").mkdir(parents=True, exist_ok=True)
    (dl.RAW / ".state" / "validated.json").write_text(
        __import__("json").dumps(state), encoding="utf-8"
    )
    return files


def test_existing_files_are_filtered_before_map():
    dl, _ = install_stub()
    os.environ["BDUSS"] = "fake"
    os.environ["PATH"] = _fake_baidu_path() + ":" + os.environ.get("PATH", "")
    files = _prepare_execute(dl, present=3)

    seen = []
    dl.fetch_one.map = lambda items, **k: seen.extend(items) or []

    dl._execute("e2", 25.0, do_fetch=True)

    assert len(seen) == len(files) - 3, (
        f"{len(seen)} payload tới .map(), cần {len(files) - 3} "
        "(3 file đã có phải bị lọc, không được bật container)"
    )
    present_locals = {f.local for f in files[:3]}
    assert not (present_locals & {p["local"] for p in seen}), "file đã có vẫn bị giao cho container"


def test_all_present_means_no_container_at_all():
    dl, _ = install_stub()
    os.environ["BDUSS"] = "fake"
    os.environ["PATH"] = _fake_baidu_path() + ":" + os.environ.get("PATH", "")
    _prepare_execute(dl, present=len(_all_e2(dl)))   # TẤT CẢ file đều đã có

    def boom(*a, **k):
        raise AssertionError("không được .map() khi mọi file đã có")

    dl.fetch_one.map = boom
    dl._execute("e2", 25.0, do_fetch=True)   # không được raise


def test_force_bypasses_the_skip_filter():
    dl, _ = install_stub()
    os.environ["BDUSS"] = "fake"
    os.environ["PATH"] = _fake_baidu_path() + ":" + os.environ.get("PATH", "")
    files = _prepare_execute(dl, present=3)

    seen = []
    dl.fetch_one.map = lambda items, **k: seen.extend(items) or []

    dl._execute("e2", 25.0, do_fetch=True, force=True)
    assert len(seen) == len(files), f"--force phải tải lại cả {len(files)} file, thấy {len(seen)}"


def _all_e2(dl):
    from dire_modal.manifest import resolve_many, expand

    return expand(resolve_many("e2"))[0]


# ---------------------------------------------------------------------------
# 7. Đo `--concurrency` (item còn treo: max_containers=2 là PHỎNG ĐOÁN)
# ---------------------------------------------------------------------------
def test_bench_payloads_use_separate_dirs_per_k():
    dl, _ = install_stub()
    f = dl.File(remote="dire/test/a/adm.tar.gz", local="dire/test/a/adm.tar.gz")
    p1 = dl._bench_payloads([f], 1)[0]
    p2 = dl._bench_payloads([f], 2)[0]
    assert p1["local"].startswith(".bench/k1/"), p1
    assert p2["local"].startswith(".bench/k2/"), p2
    assert p1["local"] != p2["local"], "hai lượt đo không được ghi cùng chỗ"
    assert dl._is_internal(p1["local"]), ".bench phải là vùng nội bộ"


def test_bench_dir_is_excluded_from_prune_and_verify():
    dl, _ = install_stub()
    assert dl._is_internal(".bench/k2/dire/test/a/adm.tar.gz")
    assert not dl._is_internal("dire/test/a/adm.tar.gz")


def test_bench_report_flags_slower_multi_container():
    """k=2 chậm hơn k=1 ⇒ phải CẢNH BÁO hạ max_containers, không chỉ in số."""
    dl, _ = install_stub()
    rows = [
        {"concurrency": 1, "ok": 2, "seconds": 100.0, "bytes": 5_000_000,
         "bytes_per_sec": 50_000.0},
        {"concurrency": 2, "ok": 2, "seconds": 400.0, "bytes": 5_000_000,
         "bytes_per_sec": 12_500.0},
    ]
    text = "\n".join(dl._bench_report(rows))
    assert "CHẬM HƠN" in text, text
    assert "nhanh nhất: k=1" in text, text


def test_bench_report_never_invents_throughput():
    dl, _ = install_stub()
    assert dl._bytes_per_sec(0, 10) == 0.0
    assert dl._bytes_per_sec(1000, 0) == 0.0
    assert dl._bytes_per_sec(1000, 2) == 500.0


def test_subprocess_timeout_is_short_enough_to_give_resume_a_chance():
    """5400s ⇒ một phiên bị throttle phải chờ 90 phút mới được thử lại.

    Ở tốc độ đo được (24–48 KB/s) không attempt nào tải xong file 692 MB, nên
    attempt ngắn = nhiều lần lấy phiên mới hơn. Khoá lại để không ai nâng về 5400
    mà không có số đo mới.
    """
    import re

    dl, _ = install_stub()
    assert dl.POLICY.subprocess_timeout <= 1800, dl.POLICY.subprocess_timeout
    assert dl.FETCH_CALL_TIMEOUT_S == (
        dl.POLICY.fetch_attempts * dl.POLICY.subprocess_timeout + 1800
    )
    # `tiers.yaml` là nguồn sự thật, NHƯNG giá trị fallback trong `Policy` cũng
    # phải khớp — nếu lệch, máy nào thiếu yaml sẽ chạy bằng 5400 mà không ai thấy.
    yaml_text = (SRC / "dire_modal" / "tiers.yaml").read_text(encoding="utf-8")
    m = re.search(r"^subprocess_timeout:\s*(\d+)", yaml_text, re.M)
    assert m, "tiers.yaml phải khai báo subprocess_timeout"
    assert int(m.group(1)) <= 1800, f"tiers.yaml: subprocess_timeout={m.group(1)}"
    assert int(m.group(1)) == dl.POLICY.subprocess_timeout, "yaml và Policy phải khớp"

    # Giá trị FALLBACK trong dataclass cũng phải đúng: `load_policy()` dùng nó khi
    # image không đóng gói được tiers.yaml, và khi đó không ai thấy là đang chạy
    # bằng 5400.
    import dataclasses

    import dire_modal.manifest as manifest

    default = next(
        f.default for f in dataclasses.fields(manifest.Policy)
        if f.name == "subprocess_timeout"
    )
    assert default <= 1800, f"Policy.subprocess_timeout mặc định = {default}"


# ---------------------------------------------------------------------------
# 8. `pack` không được làm mất nhánh
#    Bản cũ: /shards/{split}/{tên tar} ⇒ ba nhánh có `real.tar.gz` ghi vào CÙNG
#    một thư mục, và `_DONE` của nhánh đầu làm hai nhánh sau bị [SKIP] vĩnh viễn.
# ---------------------------------------------------------------------------
def test_pack_groups_keep_branch():
    dl, _ = install_stub()
    import importlib

    pack = importlib.import_module("dire_modal.pack")
    groups = {
        pack.shard_group(p, "test")
        for p in (
            "dire/test/lsun_bedroom/lsun_bedroom/real.tar.gz",
            "dire/test/lsun_bedroom/imagenet/real.tar.gz",
            "dire/test/imagenet/lsun_bedroom/real.tar.gz",
        )
    }
    assert len(groups) == 3, f"ba nhánh khác nhau phải ra ba nhóm khác nhau: {groups}"
    assert groups == {
        "lsun_bedroom__lsun_bedroom__real",
        "lsun_bedroom__imagenet__real",
        "imagenet__lsun_bedroom__real",
    }, groups


def test_pack_one_actually_uses_shard_group():
    """Test gọi thẳng `shard_group` là CHƯA ĐỦ — `pack_one` phải thật sự dùng nó.

    (`pack_one` là modal.Function nên không gọi được; kiểm bằng AST.)
    """
    dl, _ = install_stub()
    import ast

    tree = ast.parse((SRC / "dire_modal" / "pack.py").read_text(encoding="utf-8"))
    fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "pack_one"
    )
    called = {
        n.func.id
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "shard_group" in called, (
        "pack_one phải đặt tên shard bằng shard_group(); nếu không, ba nhánh có "
        "real.tar.gz lại ghi vào cùng /shards/{split}/real/"
    )


def test_pack_groups_are_stable_for_single_level_branch():
    dl, _ = install_stub()
    import importlib

    pack = importlib.import_module("dire_modal.pack")
    assert pack.shard_group("dire/test/celebahq/stargan.tar.gz", "test") == "celebahq__stargan"
    assert pack.shard_group("dire/train/lsun_bedroom/adm.tar.gz", "train") == "lsun_bedroom__adm"
