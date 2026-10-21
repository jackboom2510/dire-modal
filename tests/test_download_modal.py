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
    if [ -n "${COUNTER:-}" ]; then echo "x" >> "$COUNTER"; fi
    if [ "${MODE:-ok}" = "zero" ]; then
      # Tái hiện đúng lỗi thật: BaiduPCS-Go exit 0 nhưng chỉ tạo file 0 byte
      : > "$saveto/$base"
    elif [ "${MODE:-ok}" = "rar" ]; then
      # Tái hiện ca THẬT 2026-10-21: đúng số byte nhưng KHÔNG phải gzip.
      printf 'Rar!\x1a\x07\x00' > "$saveto/$base"
      head -c 2000 /dev/zero >> "$saveto/$base"
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
    """Byte giống hệt qua 2 lần tải ⇒ tải lại cũng thế, dừng."""
    import tempfile as _tf

    dl, _ = install_stub()
    os.environ["BDUSS"] = "fake"
    os.environ["MODE"] = "ok"
    os.environ["PATH"] = _fake_baidu_path() + ":" + os.environ.get("PATH", "")
    dl.RAW = Path(_tf.mkdtemp())
    # giả lập nội dung không đổi + validate luôn fail
    orig = dl._validate_ex
    dl._validate_ex = lambda p, e=0, expect_archive=True: dl.Verdict(
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
    assert r["attempts"] == 2, f"dừng ở attempt 2, đã chạy {r['attempts']}"


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
    dl._ls_sizes = lambda *a, **k: {}
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
    dl._ls_sizes = orig_ls


def test_probe_says_same_images_when_only_mtime_differs():
    dl, _ = install_stub()
    orig_ls = dl._ls_sizes
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
    orig_ls = dl._ls_sizes
    _probe_env(dl, _make_tar(_LSUN), _make_tar(_IMNET))
    try:
        out = _probe_output(dl)
    finally:
        _probe_cleanup(dl, orig_ls)
    assert "KHÁC HOÀN TOÀN" in out, out
    assert "KHÔNG đọc được" not in out
    assert "-1" not in out, "không được để -1 lọt vào kết luận" 
    assert "test domain: HỢP LÝ" in out


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
