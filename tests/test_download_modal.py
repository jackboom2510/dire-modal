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

    download.time.sleep = lambda _s: None
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
    if [ "${MODE:-ok}" = "zero" ]; then
      # Tái hiện đúng lỗi thật: BaiduPCS-Go exit 0 nhưng chỉ tạo file 0 byte
      : > "$saveto/$base"
    else
      t=$(mktemp -d); mkdir -p "$t/0_real" "$t/1_fake"
      echo "$remote/r" > "$t/0_real/a.jpg"; echo "$remote/f" > "$t/1_fake/b.jpg"
      tar -czf "$saveto/$base" -C "$t" 0_real 1_fake; rm -rf "$t"
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
    dl, _ = install_stub()
    p = dl.POLICY
    one = p.fetch_attempts * p.subprocess_timeout + 1800
    want = 2 * p.fetch_attempts * p.subprocess_timeout + 1800
    got = dl.probe_pair._kw["timeout"]
    assert got == want, f"timeout probe_pair = {got}, cần {want} (=2×{one})"
    assert dl.fetch_one._kw["timeout"] == one, "fetch_one chỉ tải 1 file nên giữ 1×"


def test_fetch_one_keeps_safety_kwargs():
    """Các fix của PR #2 không bị bản vá này làm hỏng."""
    dl, _ = install_stub()
    kw = dl.fetch_one._kw
    assert kw["max_containers"] == dl.POLICY.max_containers == 2
    assert kw["retries"].max_retries == 0
    assert kw["cpu"] == 0.25


# ---------------------------------------------------------------------------
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
