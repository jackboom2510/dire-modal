"""Test cho guard chi phí GPU của `compute_dire.py`. Chạy OFFLINE, không cần Modal.

Vì sao bộ test này tồn tại
--------------------------
Bản trước để cứng ``max_containers=40`` + ``timeout=24*3600`` trên A100:
40 × 24 h × ~$2.50/h ≈ **$2 400** trần xấu nhất cho một lần chạy hỏng.

Đây là loại code mà "đọc bằng mắt thấy đúng" không đủ: điều cần chứng minh
không phải là *có* một con số trong config, mà là **entrypoint dừng TRƯỚC khi
một container GPU nào được bật**. Nên test đếm số lần ``.map()`` và số lần
``.remote()`` — thứ tự đó là toàn bộ giá trị của guard.

Chạy::

    python3 -m pytest tests/test_compute_dire_guard.py -q
    # hoặc không có pytest:
    python3 tests/test_compute_dire_guard.py
"""

from __future__ import annotations

import dataclasses
import sys
import tempfile
import types
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
COMPUTE = SRC / "dire_modal" / "compute_dire.py"

# Sổ ghi: guard đúng nghĩa là hai sổ này PHẢI rỗng khi vượt trần.
MAP_LOG: list[dict] = []
REMOTE_LOG: list[str] = []


# ---------------------------------------------------------------------------
# Stub Modal: mô phỏng ĐÚNG ngữ nghĩa (bài học từ PR #4 — decorator identity
# che mất lỗi "Function object is not callable").
# ---------------------------------------------------------------------------
class FakeFunction:
    """Giống ``modal.Function``: có ``.map/.remote/.local`` nhưng KHÔNG gọi được."""

    def __init__(self, fn, kw):
        self.fn = fn
        self.kw = kw
        self.__name__ = getattr(fn, "__name__", "fn")

    def __call__(self, *a, **k):
        raise TypeError("'Function' object is not callable")

    def remote(self, *a, **k):
        REMOTE_LOG.append(self.__name__)
        return self.fn(*a, **k)

    def local(self, *a, **k):
        return self.fn(*a, **k)

    def map(self, *arglists, **kw):
        MAP_LOG.append({"fn": self.__name__, "n": len(arglists[0]) if arglists else 0})
        n = max((len(x) for x in arglists), default=0)
        return [{"i": i} for i in range(n)]


class FakeMethod(FakeFunction):
    """Method của ``@app.cls`` — cũng không gọi trực tiếp được."""

    def remote(self, *a, **k):
        REMOTE_LOG.append(self.__name__)
        return {"ok": True}


def _function(**kw):
    def deco(fn):
        return FakeFunction(fn, kw)

    return deco


class _Parameter:
    """Marker của ``modal.parameter()`` — giữ default, KHÔNG trả default trực tiếp.

    Trả thẳng default (``lambda default=None, **kw: default``) là stub quá dễ
    dãi: nó không bao giờ kiểm annotation, nên nó che mất đúng lớp lỗi đã làm
    ``compute_dire`` không import được bằng Modal thật (xem ``_check_params``).
    """

    __slots__ = ("default",)

    def __init__(self, default=None):
        self.default = default


# Các kiểu thật sự được `modal.parameter()` nhận (modal 1.6: str/int/bool/float).
_PARAM_TYPES = (str, int, float, bool)


def _check_params(C) -> dict:
    """Mô phỏng `modal.cls._Cls._validate_construction_mechanism`.

    Modal thật đọc annotation **thô** (``inspect.get_annotations`` mặc định
    ``eval_str=False``). Nếu file có ``from __future__ import annotations``
    (PEP 563) thì annotation là CHUỖI ``'str'`` chứ không phải ``str``, và Modal
    chết ngay lúc import:

        KeyError: 'str'
        → InvalidError("No class parameter encoder implemented for type `...`")
        → AttributeError: 'str' object has no attribute '__name__'

    Stub cũ trả thẳng default nên KHÔNG BAO GIỜ chạm tới đoạn này. Đây đúng là
    bài học của ``Function object is not callable``: stub phải mô phỏng hành vi
    gây lỗi, không được bỏ qua nó.

    Trả ``{tên: default}`` để ``__init__`` áp default như Modal thật.
    """
    params = {k: v for k, v in vars(C).items() if isinstance(v, _Parameter)}
    if not params:
        return {}

    # KHÔNG dùng inspect.get_annotations(eval_str=True): Modal thật không resolve.
    annotations = dict(C.__annotations__)
    missing = params.keys() - annotations.keys()
    if missing:
        raise TypeError(
            "All modal.parameter() specifications need to be type-annotated; "
            f"thiếu: {sorted(missing)}"
        )
    for name in params:
        declared = annotations[name]
        if isinstance(declared, str):
            raise TypeError(
                f"Class parameter '{name}': annotation là CHUỖI ({declared!r}), "
                "không phải kiểu thật.\n"
                "  Nguyên nhân: file có `from __future__ import annotations` "
                "(PEP 563) → Modal thật đọc `__annotations__` thô nên thấy 'str'.\n"
                "  Triệu chứng thật: KeyError: 'str' → InvalidError → "
                "AttributeError: 'str' object has no attribute '__name__'\n"
                "  Sửa: bỏ future import khỏi file chứa @app.cls."
            )
        if declared not in _PARAM_TYPES:
            raise TypeError(
                f"Class parameter '{name}': {declared!r} is not a supported "
                f"modal.parameter() type. Use one of: "
                f"{', '.join(t.__name__ for t in _PARAM_TYPES)}"
            )
    return {k: params[k].default for k in params}


def _cls(**kw):
    def deco(C):
        C._modal_kw = kw
        C._modal_params = _check_params(C)  # ← raise ngay lúc DECORATE (như Modal thật)
        orig_init = C.__dict__.get("__init__")

        def __init__(self, **kwargs):
            if orig_init is not None:
                orig_init(self)
            merged = {**C._modal_params, **kwargs}
            for k, v in merged.items():
                setattr(self, k, v)

        C.__init__ = __init__
        for name, val in list(vars(C).items()):
            if getattr(val, "_is_modal_method", False):
                setattr(C, name, FakeMethod(val, kw))
        return C

    return deco


def _method():
    def deco(fn):
        fn._is_modal_method = True
        return fn

    return deco


class _Smooth:
    """Image: mọi method chain trả về chính nó."""

    def apt_install(self, *a, **k):
        return self

    def pip_install(self, *a, **k):
        return self

    def run_commands(self, *a, **k):
        return self


class _Image:
    @staticmethod
    def debian_slim(**kw):
        return _Smooth()


class _Named:
    @staticmethod
    def from_name(name, **kw):
        return types.SimpleNamespace(name=name, commit=lambda: None)


class _App:
    def __init__(self, name, image=None):
        self.name = name

    function = staticmethod(_function)
    cls = staticmethod(_cls)

    @staticmethod
    def local_entrypoint(**kw):
        def deco(fn):
            return fn  # entrypoint local gọi được thật — đúng ngữ nghĩa Modal

        return deco


_OWNS_MODULES = False


def _purge():
    """Xoá stub + module đã import để không rò sang test file khác.

    `dire_modal.common` cache `vol_raw`/`app` ngay lần import đầu; nếu để lại,
    test của module khác sẽ dùng object của stub NÀY và fail với lỗi trông như
    bug thật. Đã xảy ra đúng như vậy: 4 test download đỏ khi chạy chung.
    """
    global _OWNS_MODULES
    if not _OWNS_MODULES:
        return
    for k in [
        k
        for k in sys.modules
        if k == "modal" or k == "dire_modal" or k.startswith("dire_modal.")
    ]:
        del sys.modules[k]
    _OWNS_MODULES = False


def install_stub():
    global _OWNS_MODULES
    _purge()
    modal = types.ModuleType("modal")
    modal.App = _App
    modal.Image = _Image
    modal.Volume = _Named
    modal.Secret = _Named
    modal.function = _function
    modal.cls = _cls
    modal.method = _method
    modal.enter = lambda *a, **k: (lambda fn: fn)
    modal.parameter = lambda default=None, **kw: _Parameter(default)
    sys.modules["modal"] = modal
    _OWNS_MODULES = True
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import dire_modal.compute_dire as cd  # noqa: PLC0415
    import dire_modal.manifest as mf  # noqa: PLC0415

    return cd, mf


def _policy(**over):
    cd, mf = install_stub()
    return dataclasses.replace(mf.Policy(notes={}), **over)


def _raised(fn, *a, **k):
    """Chạy fn, trả message của SystemExit (None nếu không raise)."""
    try:
        fn(*a, **k)
    except SystemExit as exc:
        return str(exc)
    return None


# ---------------------------------------------------------------------------
# 0. Stub phải CẮN: PEP 563 (`from __future__ import annotations`) làm Modal
#    thật chết lúc import. Stub cũ trả thẳng default nên không bao giờ thấy.
# ---------------------------------------------------------------------------
_PEP563_MODULE = '''
from __future__ import annotations

import modal

app = modal.App("pep563")


@app.cls()
class WithFuture:
    steps: int = modal.parameter(default=20)
'''


def _import_source_as_module(name: str, source: str):
    """Import một module từ chuỗi, dưới stub đang cài."""
    import importlib.util

    path = Path(tempfile.mkdtemp()) / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(name, None)
    return mod


def test_stub_rejects_pep563_string_annotations():
    """Stub phải báo lỗi y như Modal thật khi annotation là chuỗi.

    Nếu test này FAIL vì "không raise", nghĩa là stub đã quay lại quá dễ dãi —
    và nó sẽ lại che mất lỗi `AttributeError: 'str' object has no attribute
    '__name__'` đã làm `modal run -m dire_modal.compute_dire::dire_batch` chết
    ngay khi import.
    """
    install_stub()
    try:
        _import_source_as_module("_pep563_demo", _PEP563_MODULE)
    except TypeError as exc:
        msg = str(exc)
        assert "CHUỖI" in msg and "PEP 563" in msg, msg
        assert "__name__" in msg, "phải nhắc triệu chứng thật của Modal"
    else:
        raise AssertionError(
            "stub KHÔNG bắt được annotation dạng chuỗi ⇒ nó lại che lỗi import"
        )


def test_compute_dire_has_no_future_annotations():
    """`compute_dire.py` không được có `from __future__ import annotations`."""
    import ast as _ast

    tree = _ast.parse(COMPUTE.read_text())
    bad = [
        n
        for n in tree.body
        if isinstance(n, _ast.ImportFrom)
        and n.module == "__future__"
        and any(a.name == "annotations" for a in n.names)
    ]
    assert not bad, (
        "compute_dire.py có `from __future__ import annotations` ⇒ Modal đọc "
        "`__annotations__` thô và chết lúc import với AttributeError: 'str' "
        "object has no attribute '__name__'"
    )


def test_no_module_mixes_future_annotations_with_modal_parameter():
    """Module nào dùng `modal.parameter` cũng không được có future import.

    Quét cả `src/` chứ không chỉ compute_dire.py: lỗi này sẽ tái xuất hiện y
    nguyên nếu ai thêm `modal.parameter` vào một file đang có future import.
    """
    import ast as _ast

    offenders = []
    for path in sorted((SRC / "dire_modal").rglob("*.py")):
        src = path.read_text(encoding="utf-8", errors="replace")
        if "modal.parameter" not in src:
            continue
        tree = _ast.parse(src)
        has_future = any(
            isinstance(n, _ast.ImportFrom)
            and n.module == "__future__"
            and any(a.name == "annotations" for a in n.names)
            for n in tree.body
        )
        if has_future:
            offenders.append(path.name)
    assert not offenders, (
        f"{offenders} vừa có `from __future__ import annotations` vừa dùng "
        "`modal.parameter` ⇒ Modal thật sẽ chết lúc import"
    )


def test_compute_dire_parameters_are_real_types():
    cd, _ = install_stub()
    ann = cd.DIRECompute.__annotations__
    assert ann["model_path"] is str, ann
    assert ann["steps"] is int, ann
    assert cd.DIRECompute._modal_params == {
        "model_path": "/ckpt/pretrained/lsun_adm.pth",
        "steps": 20,
    }, cd.DIRECompute._modal_params


def test_dire_compute_constructs_with_defaults_and_overrides():
    cd, _ = install_stub()
    assert cd.DIRECompute().steps == 20
    assert cd.DIRECompute(steps=7).steps == 7
    assert cd.DIRECompute().model_path == "/ckpt/pretrained/lsun_adm.pth"


# ---------------------------------------------------------------------------
# 1. Cấu hình: không còn 40 container / 24 giờ
# ---------------------------------------------------------------------------
def test_class_decorator_takes_policy_not_hardcoded():
    cd, mf = install_stub()
    kw = cd.DIRECompute._modal_kw
    assert kw["max_containers"] == mf.POLICY.dire_max_containers, kw
    assert kw["timeout"] == mf.POLICY.dire_timeout_s, kw
    assert kw["max_containers"] <= 16, "guard cứng phải chặn 40 container"
    assert kw["timeout"] <= 7200, "timeout 24 h là trần chi phí thật — phải nhỏ"


def test_source_has_no_24h_timeout_or_40_containers():
    src = COMPUTE.read_text()
    code = "\n".join(
        ln for ln in src.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "24 * 3600" not in code, "24 giờ quay lại — đó là trần chi phí $2400"
    assert "max_containers=40" not in code.replace(" ", "")


def test_clamp_never_exceeds_hard_cap():
    cd, _ = install_stub()
    assert cd._clamp_containers(40) == 16, "config sai không được bật 40 container"
    assert cd._clamp_containers(0) == 1
    assert cd._clamp_containers(4, 2) == 2, "không bật nhiều container hơn số ảnh"
    assert cd._clamp_containers(4, 10_000) == 4


def test_policy_defaults_are_small():
    cd, mf = install_stub()
    p = mf.POLICY
    assert p.dire_max_containers <= 8
    assert p.dire_timeout_s <= 3600
    assert p.dire_max_usd <= 20
    assert p.dire_max_images <= 2000


# ---------------------------------------------------------------------------
# 2. Toán tiền: hai con số, và con số thứ hai mới là trần thật
# ---------------------------------------------------------------------------
def test_estimate_matches_hand_math():
    cd, _ = install_stub()
    p = _policy(
        dire_max_containers=4,
        dire_timeout_s=1800,
        dire_sec_per_image=5.0,
        dire_usd_per_gpu_hour=2.5,
    )
    est = cd.estimate_cost(200, p)
    assert est["gpu_hours"] == 200 * 5 / 3600
    assert abs(est["usd"] - (200 * 5 / 3600 * 2.5)) < 1e-9
    # trần xấu nhất: 4 container × 30 phút
    assert est["worst_gpu_hours"] == 4 * 0.5
    assert est["worst_usd"] == 4 * 0.5 * 2.5 == 5.0


def test_worst_case_is_bounded_even_with_insane_config():
    """Config đặt 40 container × 24 h vẫn không được tạo trần $2400."""
    cd, _ = install_stub()
    p = _policy(dire_max_containers=40, dire_timeout_s=24 * 3600)
    est = cd.estimate_cost(1000, p)
    assert est["containers"] == 16, "clamp container"
    assert est["timeout_s"] == 3600, "clamp timeout 24 h → 1 h"
    assert est["worst_usd"] == 16 * 1.0 * 2.5 == 40.0
    assert est["worst_usd"] < 40 * 24 * 2.5 / 10, "phải nhỏ hơn trần cũ rất nhiều"


def test_single_image_uses_one_container():
    cd, _ = install_stub()
    est = cd.estimate_cost(1, _policy(dire_max_containers=4))
    assert est["containers"] == 1, "không bật 4 container cho 1 ảnh"


def test_clamp_timeout_bounds():
    cd, _ = install_stub()
    assert cd._clamp_timeout(24 * 3600) == 3600
    assert cd._clamp_timeout(1800) == 1800
    assert cd._clamp_timeout(1) == 60


# ---------------------------------------------------------------------------
# 3. Preflight: ba điều kiện, đều raise SystemExit
# ---------------------------------------------------------------------------
def test_preflight_refuses_over_image_cap():
    cd, _ = install_stub()
    p = _policy(dire_max_images=500, dire_max_usd=1e9)
    msg = _raised(cd.preflight_cost, 10_000, p)
    assert msg and "trần" in msg and "500" in msg, msg


def test_preflight_refuses_over_usd_estimate():
    cd, _ = install_stub()
    p = _policy(dire_max_images=10**6, dire_max_usd=0.01, dire_max_containers=1)
    msg = _raised(cd.preflight_cost, 200, p)
    assert msg and "ước tính" in msg, msg


def test_preflight_refuses_when_worst_case_breaks_budget():
    """Kể cả 1 ảnh: nếu 4 container treo 30 phút đã vượt trần thì phải chặn."""
    cd, _ = install_stub()
    p = _policy(dire_max_containers=4, dire_timeout_s=1800, dire_max_usd=1.0)
    msg = _raised(cd.preflight_cost, 1, p)
    assert msg and "xấu nhất" in msg, msg


def test_force_bypasses_but_still_prints():
    import contextlib
    import io

    cd, _ = install_stub()
    p = _policy(dire_max_usd=0.01)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        est = cd.preflight_cost(200, p, force=True)
    assert est["images"] == 200
    out = buf.getvalue()
    assert "--force" in out and "modal app list" in out


# ---------------------------------------------------------------------------
# 4. ĐIỂM CHÍNH: vượt trần ⇒ không `.map()`, không `.remote()`
# ---------------------------------------------------------------------------
def test_entrypoint_refuses_before_any_container(monkeypatch=None):
    cd, mf = install_stub()
    orig = mf.POLICY
    cd.POLICY = dataclasses.replace(orig, dire_max_usd=0.001, dire_max_images=10**6)
    MAP_LOG.clear()
    REMOTE_LOG.clear()
    try:
        msg = _raised(cd.dire_batch, 100)
    finally:
        cd.POLICY = orig
    assert msg, "vượt trần mà không dừng"
    assert MAP_LOG == [], f"đã gọi .map() trước khi kiểm tiền: {MAP_LOG}"
    assert REMOTE_LOG == [], f"đã bật container trước khi kiểm tiền: {REMOTE_LOG}"


def test_entrypoint_dry_run_never_maps():
    cd, _ = install_stub()
    MAP_LOG.clear()
    REMOTE_LOG.clear()
    cd.dire_batch(n=3, dry_run=True)
    assert MAP_LOG == [] and REMOTE_LOG == []


def test_entrypoint_maps_when_within_budget():
    cd, _ = install_stub()
    d = Path(tempfile.mkdtemp())
    for i in range(3):
        (d / f"img{i}.jpg").write_bytes(b"x")
    MAP_LOG.clear()
    REMOTE_LOG.clear()
    cd.dire_batch(n=3, src_glob=f"{d}/*.jpg", out_dir="/shards/dire_out")
    assert REMOTE_LOG == ["list_images"], REMOTE_LOG
    assert len(MAP_LOG) == 1, MAP_LOG
    assert MAP_LOG[0]["n"] == 3, "phải map đúng số ảnh thật"


def test_entrypoint_refuses_when_too_many_images():
    cd, _ = install_stub()
    mapper = cd.DIRECompute.process_one
    MAP_LOG.clear()
    msg = _raised(cd.dire_batch, 10**6)
    assert msg and "ảnh" in msg
    assert MAP_LOG == []
    assert mapper is not None


# ---------------------------------------------------------------------------
# 5. Ngữ nghĩa cross-generator (probe_pair 2026-10-21)
# ---------------------------------------------------------------------------
def test_manifest_calls_that_branch_cross_generator():
    cd, mf = install_stub()
    notes = [
        e.note
        for e in mf.E3
        if getattr(e, "remote", "").startswith("dire/test/lsun_bedroom/imagenet/")
    ]
    assert notes, "không thấy file của nhánh lsun_bedroom/imagenet/ trong E3"
    for n in notes:
        assert "cross-generator" in n, n
        assert "cross-domain" not in n.replace("cross-generator", ""), n
        assert "chờ probe_pair" not in n, n


try:  # pytest: dọn module sau MỖI test, không chỉ ở cuối file
    import pytest as _pytest

    @_pytest.fixture(autouse=True)
    def _isolate_modules():
        yield
        _purge()
except ImportError:  # chạy bằng __main__ thì _run_all tự dọn
    pass


def _run_all() -> int:
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            print(f"  FAIL {name}\n         {exc}")
        except SystemExit as exc:
            print(f"  FAIL {name}\n         SystemExit bất ngờ: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {name}\n         {type(exc).__name__}: {exc}")
        else:
            print(f"  OK   {name}")
            ok += 1
        finally:
            _purge()
    print(f"\n{ok}/{len(tests)} test đạt")
    return 0 if ok == len(tests) else 1


if __name__ == "__main__":
    sys.exit(_run_all())
