"""`dire_core` là nguồn sự thật duy nhất — test này khoá điều đó.

Vì sao có file này
------------------
Tái cấu trúc theo hướng "local chạy được, Modal là wrapper" chỉ có giá trị nếu
**hai đường cho cùng một con số**. Kiểu hỏng đáng sợ nhất ở đây không phải crash
mà là *trôi im lặng*: CLI local dùng một pipeline tiền xử lý, eval trên Modal
dùng pipeline khác, và không có gì báo. Repo đã gặp đúng chuyện đó với
``demo.py`` của tác giả (thiếu ``Normalize``).

Bốn nhóm kiểm:

1. **``dire_core`` không phụ thuộc hạ tầng.** Import được khi ``modal`` bị chặn.
2. **Không có bản sao thứ hai.** ``eval.py`` không chứa lại pipeline tiền xử lý,
   không chứa lại công thức ``sigmoid``/``softmax``.
3. **Hai đường khớp nhau.** Cùng input ⇒ cùng output cho ``label_of``,
   ``prob_fake``, ``fc_out_features``.
4. **Wrapper biết mount code.** Image phải mount ``dire_core``/``dire_local``,
   nếu không cloud chết trong khi máy local vẫn xanh.

Chạy::

    python3 -m pytest tests/test_core_parity.py -q
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CORE = SRC / "dire_core"


def _src(name: str) -> str:
    return (CORE / name).read_text(encoding="utf-8")


def _tree(name: str) -> ast.Module:
    return ast.parse(_src(name))


# ---------------------------------------------------------------------------
# 1. core không phụ thuộc infra
# ---------------------------------------------------------------------------
def test_core_imports_without_modal():
    """Chạy trong tiến trình con: chặn `modal`, `torch` rồi import `dire_core`."""
    code = """
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'modal', 'torch', 'torchvision', 'PIL', 'fastapi'}:
            raise ImportError(f'BỊ CHẶN: {name}')
        return None
sys.meta_path.insert(0, Block())
sys.path.insert(0, %r)
import dire_core
print(dire_core.PREPROC)
""" % str(SRC)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "resize256_centercrop224_normalize"


def test_core_modules_do_not_import_modal_or_torch_at_top():
    for path in sorted(CORE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Import):
                roots = {a.name.split(".")[0] for a in node.names}
                assert not ({"modal", "torch", "torchvision"} & roots), path.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                assert root not in {"modal", "torch", "torchvision"}, path.name


def test_core_has_no_volume_or_app_references():
    for path in sorted(CORE.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "vol_raw" not in text, path.name
        assert "modal.App(" not in text, path.name
        assert "/raw/" not in text, path.name


# ---------------------------------------------------------------------------
# 2. không có bản sao pipeline trong eval.py
# ---------------------------------------------------------------------------
def _eval_tree() -> ast.Module:
    return ast.parse((SRC / "dire_modal" / "eval.py").read_text(encoding="utf-8"))


def test_eval_has_no_second_preprocess_pipeline():
    """`eval.py` không được chứa lại Resize/CenterCrop/Normalize."""
    tree = _eval_tree()
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
        if fn.name != "preprocess":
            continue
        attrs = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
        assert not ({"Resize", "CenterCrop", "Normalize", "ToTensor"} & attrs), (
            "eval.preprocess có bản sao pipeline — phải uỷ quyền cho dire_core"
        )
        calls = {
            n.func.id for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        assert "_core_preprocess" in calls, "eval.preprocess không gọi dire_core"


def test_eval_has_no_hand_written_sigmoid():
    """Công thức sigmoid/softmax chỉ được tồn tại trong `dire_core/score.py`."""
    text = (SRC / "dire_modal" / "eval.py").read_text(encoding="utf-8")
    assert "_math.exp" not in text, "eval.py còn viết tay exp() cho sigmoid"
    assert "math.exp" not in text


# ---------------------------------------------------------------------------
# 3. hai đường khớp nhau
# ---------------------------------------------------------------------------
@pytest.fixture()
def core():
    sys.path.insert(0, str(SRC))
    import dire_core

    return dire_core


def test_constants_are_the_same_object_as_eval(core):
    ev = pytest.importorskip("dire_modal.eval")
    assert (ev.RESIZE, ev.CROP) == (core.RESIZE, core.CROP) == (256, 224)
    assert ev.MEAN == core.MEAN == (0.485, 0.456, 0.406)
    assert ev.STD == core.STD == (0.229, 0.224, 0.225)
    assert ev.PREPROC == core.PREPROC


def test_prob_fake_parity_including_extremes(core):
    ev = pytest.importorskip("dire_modal.eval")
    cases = [
        ([-800.0, -1.0, 0.0, 1.0, 800.0], 1),
        ([[-3.0, 3.0], [2.0, -2.0]], 2),
        ([0.0], 1),
    ]
    for rows, n_out in cases:
        assert ev.prob_fake(rows, n_out) == core.prob_fake(rows, n_out)


def test_prob_fake_refuses_to_guess_with_too_few_logits(core):
    with pytest.raises(ValueError):
        core.prob_fake([[0.5]], n_out=2)


def test_label_of_parity(core):
    ev = pytest.importorskip("dire_modal.eval")
    cases = [
        ("0_real/0001.png", "adm.tar.gz"),
        ("1_fake/0001.png", "real.tar.gz"),
        ("0001.jpg", "real.tar.gz"),
        ("0001.jpg", "midjouney.tar.gz"),
        ("real/0001.jpg", "x.tar.gz"),
        ("fake/0001.jpg", "x.tar.gz"),
        ("a\\0_real\\b.png", "adm.tar.gz"),
    ]
    for member, archive in cases:
        assert ev.label_of(member, archive) == core.label_of(member, archive), member


class _T:
    def __init__(self, *shape):
        self.shape = tuple(shape)


def test_fc_out_features_parity(core):
    ev = pytest.importorskip("dire_modal.eval")
    cases = [
        {"fc.weight": _T(1, 2048)},
        {"fc.weight": _T(2, 2048)},
        {"module.fc.weight": _T(1, 2048)},
        {"fc.weight": _T(2048)},      # không phải 2 chiều ⇒ 0
        {"conv.weight": _T(64, 3, 7, 7)},
    ]
    for state in cases:
        assert ev.fc_out_features(state) == core.fc_out_features(state), state


def test_sigmoid_is_stable_at_both_tails(core):
    assert core.sigmoid(0) == 0.5
    assert core.sigmoid(-800) == 0.0
    assert core.sigmoid(800) == 1.0


# ---------------------------------------------------------------------------
# 4. wrapper phải mount code — thiếu là chết chỉ trên cloud
# ---------------------------------------------------------------------------
def test_common_mounts_all_three_packages():
    sys.path.insert(0, str(SRC))
    from dire_modal import common

    seen = {}

    class Img:
        def add_local_python_source(self, *names):
            seen["names"] = names
            return self

    out = common.mount_python_sources(Img())
    assert out is not None
    assert set(seen["names"]) == {"dire_core", "dire_local", "dire_modal"}


def test_api_image_uses_gpu_image_and_bakes_weights():
    sys.path.insert(0, str(SRC))
    from dire_modal import api

    assert api.BAKED_WEIGHTS.endswith("lsun_adm.pth")
    assert api.VOLUME_WEIGHTS.startswith("/raw/")
    assert api._MAX_CONTAINERS <= 2, "endpoint public không được scale rộng mặc định"


# ---------------------------------------------------------------------------
# 5. CLI chạy được mà không cần torch
# ---------------------------------------------------------------------------
def test_cli_info_runs_without_torch():
    r = subprocess.run(
        [sys.executable, "-m", "dire_local.cli", "info", "--weights", "khong_co.pth"],
        capture_output=True, text=True, cwd=str(ROOT),
        env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"},
    )
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["preproc"] == "resize256_centercrop224_normalize"
    assert data["weights_exists"] is False
    assert data["torch"] is None


def test_cli_help_lists_four_commands():
    r = subprocess.run(
        [sys.executable, "-m", "dire_local.cli", "--help"],
        capture_output=True, text=True, cwd=str(ROOT),
        env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"},
    )
    assert r.returncode == 0, r.stderr
    for cmd in ("info", "detect", "batch", "serve"):
        assert cmd in r.stdout


def test_server_payload_and_auth_are_testable_without_torch():
    sys.path.insert(0, str(SRC))
    from dire_local import server

    class FakeDet:
        def score_pils(self, pils, batch_size=64):
            return [0.9 for _ in pils]

        def info(self):
            return {"preproc": "x", "n_out": 1, "prob": "sigmoid"}

    # Không có ảnh ⇒ ValueError nói rõ thiếu field, không phải KeyError.
    with pytest.raises(ValueError):
        server.score_payload(FakeDet(), {})

    # base64 rác ⇒ ValueError, không phải 500.
    with pytest.raises(ValueError):
        server.score_payload(FakeDet(), {"image_b64": "!!!not-base64!!!"})


def test_api_key_check_is_pure_and_fails_closed():
    sys.path.insert(0, str(SRC))
    from dire_modal import api

    assert api.check_key("k", "k") == (True, 200)
    assert api.check_key("x", "k") == (False, 401)
    assert api.check_key(None, "k") == (False, 401)
    assert api.check_key(None, None) == (True, 200)

    class Req:
        headers = {"x-api-key": "abc"}

    assert api._key_from(Req()) == "abc"
    assert api._key_from(object(), "fallback") == "fallback"


def test_modal_parameters_are_not_used_in_api():
    """Cảnh báo trong `compute_dire.py`: PEP 563 + tham số Modal = KeyError lúc import."""
    text = (SRC / "dire_modal" / "api.py").read_text(encoding="utf-8")
    assert "modal.parameter" not in text
    assert types is not None  # giữ import cho pyflakes
