"""Test import `dire_modal.compute_dire` bằng **modal THẬT** (không stub).

Vì sao cần file riêng
---------------------
Mọi test khác trong `tests/` chạy với stub Modal, và stub — dù cố mô phỏng đúng
— vẫn là code của chúng ta tự viết. Có những lỗi chỉ Modal thật mới bắt được::

    from __future__ import annotations   # PEP 563
    ...
    @app.cls()
    class DIRECompute:
        model_path: str = modal.parameter(default="...")

Modal thật đọc ``__annotations__`` **thô**, thấy ``'str'`` (chuỗi) chứ không phải
``str``, rồi chết ngay lúc IMPORT module::

    KeyError: 'str'
    → InvalidError("No class parameter encoder implemented for type `...`")
    → AttributeError: 'str' object has no attribute '__name__'

AttributeError cuối che mất InvalidError thật, nên traceback không chỉ ra nguyên
nhân. Lỗi ở import chứ không ở class ⇒ MỌI lệnh của module chết, kể cả
``make_gallery``. Đã xảy ra thật với ``dire_batch --dry-run`` (2026-10-21).

Test này chạy trong **subprocess** để có ``sys.modules`` sạch: nó import đúng
package thật với đúng ``modal`` từ PyPI, y như ``modal run`` làm.

Cài modal thật rồi chạy::

    pip install modal pyyaml pytest
    python3 -m pytest tests/test_real_modal_import.py -q

Hoặc ``bash scripts/check.sh`` (tự tạo venv + cài + chạy cả bộ test).
Không có ``modal`` trong môi trường ⇒ test SKIP, không FAIL: phần còn lại của bộ
test vẫn chạy được ở nơi không cài Modal.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MODULES = [
    "dire_modal.compute_dire",
    "dire_modal.download",
    "dire_modal.manifest",
    "dire_modal.common",
    "dire_modal.pack",
    "dire_modal.upload",
]

# Chạy trong subprocess với sys.modules sạch. Tên module truyền qua argv để
# không phải nhét repr() vào chuỗi (và .format() khỏi đụng { } của code).
_PROBE = r"""
import json
import sys

import modal

out = {"modal_version": getattr(modal, "__version__", "?"), "modules": {}}
for name in json.loads(sys.argv[1]):
    __import__(name, fromlist=["_"])
    out["modules"][name] = "ok"

import dire_modal.compute_dire as cd

cls = cd.DIRECompute
out["cls_type"] = type(cls).__name__
# Đọc từ user class: đây là chỗ Modal thật đọc annotation. Nếu annotation còn
# là chuỗi (PEP 563) thì import đã chết TRƯỚC dòng này — nên tới được đây đã
# là bằng chứng, nhưng vẫn kiểm tường minh cho rõ ràng.
user_cls = cls._get_user_cls()
out["annotations"] = {
    k: getattr(v, "__name__", repr(v)) for k, v in user_cls.__annotations__.items()
}
out["params"] = {
    k: getattr(getattr(user_cls, k), "default", getattr(user_cls, k))
    for k in user_cls.__annotations__
}
out["has"] = {
    "dire_batch": hasattr(cd, "dire_batch"),
    "make_gallery": hasattr(cd, "make_gallery"),
    "list_images": hasattr(cd, "list_images"),
    "preflight_cost": hasattr(cd, "preflight_cost"),
}
print("JSON::" + json.dumps(out))
"""

# Env tối giản, cố ý KHÔNG truyền MODAL_TOKEN_ID/SECRET: test không được phụ
# thuộc tài khoản. Dựng App/Cls ở client không cần token, chỉ lúc gọi mới cần.
_ENV = {
    "PYTHONPATH": str(SRC),
    "PATH": "/usr/bin:/bin:/usr/local/bin",
    "HOME": str(Path.home()),
    "PYTHONIOENCODING": "utf-8",
}


def _real_modal_available() -> tuple[bool, str]:
    """``modal`` có import được và có phải package THẬT (không phải stub) không.

    Kiểm trong SUBPROCESS, không `import modal` tại chỗ: khi chạy cả bộ test,
    `test_compute_dire_guard.py` đã cài stub vào ``sys.modules``. Kiểm tại chỗ
    sẽ thấy stub, kết luận "không có modal thật" và **SKIP ÂM THẦM** — đúng
    kiểu hỏng mà cả file này sinh ra để chống. Đã xảy ra thật ở lần chạy đầu:
    51 passed, nhưng 2 test này chưa hề chạy.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import modal; print(getattr(modal, '__version__', '?'))"],
            capture_output=True,
            text=True,
            env=_ENV,
            cwd=str(ROOT),
            timeout=120,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"không chạy được subprocess: {exc}"
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-1:] or ["(không có stderr)"]
        return False, f"modal không import được: {tail[0]}"
    return True, proc.stdout.strip()


def _skip(info: str) -> None:
    """SKIP cho pytest, in ra cho ``__main__``. Không bao giờ im lặng."""
    try:
        import pytest  # noqa: PLC0415
    except ImportError:
        print(f"[SKIP] {info}")
        return
    pytest.skip(f"không có modal thật: {info}")


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        env=_ENV,
        cwd=str(ROOT),
        timeout=300,
    )


def test_modules_import_with_real_modal():
    ok, info = _real_modal_available()
    if not ok:
        _skip(info)
        return

    proc = _run(["-c", _PROBE, json.dumps(MODULES)])
    assert proc.returncode == 0, (
        f"import chết với modal {info} thật:\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("JSON::")]
    assert line, f"không thấy JSON trong output:\n{proc.stdout}\n{proc.stderr}"

    out = json.loads(line[0][len("JSON::") :])
    for name in MODULES:
        assert out["modules"][name] == "ok", out
    assert out["cls_type"] == "Cls", (
        f"DIRECompute phải là modal.cls.Cls, đang là {out['cls_type']}"
    )
    # Bằng chứng trực tiếp PEP 563 đã được bỏ: annotation là KIỂU THẬT, không
    # phải chuỗi. Modal thật gọi `_get_encoder(str)`; nếu nhận `'str'` thì nó
    # KeyError → AttributeError ngay lúc import.
    assert out["annotations"] == {"model_path": "str", "steps": "int"}, out[
        "annotations"
    ]
    assert out["params"] == {
        "model_path": "/ckpt/pretrained/lsun_adm.pth",
        "steps": 20,
    }, out["params"]
    assert all(out["has"].values()), out["has"]


def test_dry_run_entrypoint_reaches_preflight_with_real_modal():
    """Chạy ``dire_batch`` với real modal, chỉ tới preflight (không gọi cloud).

    Đây đúng là đường người dùng chạy: ``modal run ...::dire_batch --n 200
    --dry-run``. Nếu import chết, lệnh đó chết trước khi in ra bất kỳ con số
    tiền nào — nên guard chi phí không bao giờ có cơ hội chạy.
    """
    ok, info = _real_modal_available()
    if not ok:
        _skip(info)
        return

    src = "import dire_modal.compute_dire as cd\ncd.dire_batch(n=200, dry_run=True)\n"
    proc = _run(["-c", src])
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "[tiền]" in out, out
    assert "GPU-h" in out, out
    assert "dry-run" in out, out


def _run_all() -> int:
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            print(f"  FAIL {name}\n         {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {name}\n         {type(exc).__name__}: {exc}")
        else:
            print(f"  OK   {name}")
            ok += 1
    print(f"\n{ok}/{len(tests)} test đạt")
    return 0 if ok == len(tests) else 1


if __name__ == "__main__":
    sys.exit(_run_all())
