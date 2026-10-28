"""Test cho `tools/manim/` — chạy được khi máy **KHÔNG có manim**.

Vì sao file này tồn tại
----------------------
Video giải thích là chỗ dễ sinh ra "nguồn số thứ hai" nhất: một con số gõ tay
vào scene, đúng lúc viết, rồi sai dần khi code đổi. Và đường loss bịa trông y hệt
đường loss thật. Ba nhóm test:

1. **Nguồn số.** `scenes.py` không được chứa số liệu (chỉ được có tham số bố
   cục); `data.py` phải đọc TSV/manifest/JSON, và số nó trả phải **khớp** với
   `eval.load_paper_table` + `manifest.resolve_many`.
2. **Scene dựng được.** Stub `manim` bằng một module giả có whitelist API — nhờ
   vậy mỗi scene được `construct()` thật, bắt được `NameError`/`AttributeError`
   mà không cần cài cairo/pango. Whitelist cũng là danh sách API manim mà repo
   phụ thuộc: nâng manim thì đây là chỗ cần xem lại.
3. **Thiếu dữ liệu thì phải TRÔNG như thiếu.** `TrainMechanism` không có log ⇒
   hiện khung "chưa có log", không vẽ đường cong.

Chạy::

    python3 -m pytest tests/test_manim_scenes.py -q
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MANIM = ROOT / "tools" / "manim"
SCENES_PY = MANIM / "scenes.py"
STYLE_PY = MANIM / "style.py"
DATA_PY = MANIM / "data.py"

EXPECTED_SCENES = {
    "DiffusionTheory", "DirePipeline", "PaperTable2", "DataPipeline", "TrainMechanism",
}


# ---------------------------------------------------------------------------
# Stub manim: whitelist API, để lỗi gõ tên vẫn nổ
# ---------------------------------------------------------------------------
class _V(tuple):
    """Vector giả: manim cho phép `LEFT * 2`, nên phép nhân phải có."""

    def __mul__(self, k):
        return _V(v * k for v in self)

    def __rmul__(self, k):
        return self.__mul__(k)

    def __add__(self, other):
        return _V(a + b for a, b in zip(self, other))


_DIRS = {
    "UP": _V((0.0, 1.0, 0.0)), "DOWN": _V((0.0, -1.0, 0.0)),
    "LEFT": _V((-1.0, 0.0, 0.0)), "RIGHT": _V((1.0, 0.0, 0.0)),
    "ORIGIN": _V((0.0, 0.0, 0.0)),
}

# API manim mà repo thực sự dùng. Muốn dùng thêm thì thêm vào đây — đó là chủ ý:
# mỗi lần mở rộng là một lần xem lại xem manim bản mới còn hỗ trợ không.
_ALLOWED = {
    "arrange", "next_to", "to_edge", "to_corner", "shift", "scale", "rotate",
    "move_to", "align_to", "set_color", "set_opacity", "set_fill", "set_stroke",
    "set_width", "set_height", "copy", "get_center", "get_left", "get_right",
    "get_top", "get_bottom", "get_width", "get_height", "add", "become",
    "animate",
}


class _M:
    """Mobject giả: giữ ``text`` (nếu là Text) và ghi lại mọi lời gọi."""

    def __init__(self, *args, **kwargs):
        if args and isinstance(args[0], str):
            self.text = args[0]
        self.kwargs = kwargs
        self.children = [a for a in args if isinstance(a, _M)]
        self.width = float(kwargs.get("width", 1.0) or 1.0)
        self.height = float(kwargs.get("height", 1.0) or 1.0)
        self.calls: list[str] = []

    # --- API hợp lệ ------------------------------------------------------
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in _ALLOWED:
            raise AttributeError(
                f"mobject giả không có {name!r} — nếu đây là API manim thật, "
                f"thêm vào _ALLOWED trong tests/test_manim_scenes.py"
            )
        if name == "animate":
            return self

        def _method(*a, **k):
            self.calls.append(name)
            return self

        return _method

    def __add__(self, other):
        return _M(*([self] + ([other] if isinstance(other, _M) else [])))

    def __iter__(self):
        return iter(self.children)

    def __len__(self):
        return len(self.children)

    def __getitem__(self, i):
        return self.children[i]


class _Group(_M):
    """VGroup giả: list-like, để `VGroup(*mobs).arrange(...)` chạy được."""

    def __init__(self, *mobs):
        super().__init__()
        self.children = list(mobs)
        self.text = "".join(getattr(m, "text", "") for m in mobs)


class _Anim:
    def __init__(self, *a, **k):
        self.args, self.kwargs = a, k


class _Scene:
    """`Scene` giả — đủ để `construct()` chạy và đếm ``play``."""

    def __init__(self, *a, **k):
        self.played: list = []
        self.waited = 0

    def play(self, *anims, **kw):
        self.played.extend(anims)

    def wait(self, *a, **k):
        self.waited += 1

    def add(self, *a, **k):
        pass

    def remove(self, *a, **k):
        pass

    def bring_to_front(self, *a, **k):
        pass


def _manim_names() -> set[str]:
    """Mọi tên được import từ `manim` trong style.py + scenes.py (đọc bằng AST)."""
    names: set[str] = set()
    for path in (STYLE_PY, SCENES_PY):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "manim":
                names.update(a.name for a in node.names)
    return names


def _fake_manim_module() -> types.ModuleType:
    mod = types.ModuleType("manim")
    for name in _manim_names():
        if name == "Scene":
            setattr(mod, name, _Scene)
        elif name in _DIRS:
            setattr(mod, name, _DIRS[name])
        elif name in {"VGroup", "Group"}:
            setattr(mod, name, _Group)
        elif name[0].isupper() and name.endswith(("In", "Out", "Write", "Create",
                                                  "Transform", "Indicate", "Group",
                                                  "Start")):
            setattr(mod, name, _Anim)
        else:
            setattr(mod, name, _M)
    return mod


@pytest.fixture(scope="module")
def scenes_module():
    """Nạp `scenes.py` dưới stub manim, ở chế độ CHỈ ĐỌC dữ liệu thật."""
    sys.path.insert(0, str(SRC))
    saved = sys.modules.get("manim")
    sys.modules["manim"] = _fake_manim_module()
    try:
        spec = importlib.util.spec_from_file_location("_dire_manim_scenes", SCENES_PY)
        module = importlib.util.module_from_spec(spec)
        sys.modules["_dire_manim_scenes"] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop("_dire_manim_scenes", None)
        if saved is not None:
            sys.modules["manim"] = saved
        else:
            sys.modules.pop("manim", None)


# ---------------------------------------------------------------------------
# 1. Nguồn số
# ---------------------------------------------------------------------------
def test_data_module_is_stdlib_only():
    """`data.py` chạy được trong venv manim không có torch/pyyaml."""
    tree = ast.parse(DATA_PY.read_text(encoding="utf-8"))
    top: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.add(node.module.split(".")[0])
    assert not ({"manim", "torch", "numpy", "modal", "PIL", "yaml"} & top), top


def test_scenes_do_not_import_torch_or_modal():
    tree = ast.parse(SCENES_PY.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert not ({"torch", "modal", "numpy", "PIL"} & names), names


#: Số liệu ĐẶC TRƯNG, không được xuất hiện như literal trong `scenes.py`.
_DATA_NUMBERS = {99.9, 100.0, 98.7, 76.6, 269.4, 5.74, 1.94, 0.99, 125015131}


def test_no_data_number_is_hardcoded_in_scenes():
    """Số phải đến từ `data.py`; scene chỉ được có tham số bố cục."""
    tree = ast.parse(SCENES_PY.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if float(node.value) in _DATA_NUMBERS:
                bad.append((node.value, getattr(node, "lineno", "?")))
    assert not bad, f"số liệu viết thẳng trong scenes.py: {bad}"


def test_paper_numbers_come_from_the_same_tsv_as_eval():
    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(MANIM))
    from dire_modal import eval as E  # noqa: PLC0415
    import data as D  # noqa: PLC0415

    paper = E.load_paper_table()
    for ckpt in D.detectors():
        got = D.paper_matrix(ckpt)
        for gen, vals in got.items():
            assert vals["acc"] == paper[ckpt][gen]["acc"], (ckpt, gen, "acc")
            assert vals["ap"] == paper[ckpt][gen]["ap"], (ckpt, gen, "ap")


def test_mean_is_computed_not_copied_from_the_avg_row():
    """Hàng `avg` của paper in sẵn; scene tự tính nên phải khớp trong 0.05."""
    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(MANIM))
    from dire_modal import eval as E  # noqa: PLC0415
    import data as D  # noqa: PLC0415

    paper = E.load_paper_table()
    for ckpt in D.detectors():
        assert abs(D.mean_acc(ckpt) - paper[ckpt]["avg"]["acc"]) <= 0.05, ckpt


def test_pair_count_is_derived_from_the_manifest():
    """`e2` = 1 ckpt + 16 tar (có `real`) ⇒ 15 cặp, và 4 detector ⇒ 60 container."""
    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(MANIM))
    from dire_modal.manifest import resolve_many  # noqa: PLC0415
    import data as D  # noqa: PLC0415

    ents = resolve_many("e2")
    tars = [e for e in ents if e.remote.endswith(".tar.gz")]
    assert len(tars) == 16, len(tars)
    assert D.pairs("e2") == 15
    assert D.containers("e2") == 15 * 4 == 60
    # Số cặp phải bằng số generator mà scene vẽ bảng + 7 generator mở rộng.
    assert D.pairs("e2") == len(D.GENERATORS) + 7


def test_pipeline_facts_all_carry_a_source():
    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(MANIM))
    import data as D  # noqa: PLC0415

    facts = D.pipeline_facts()
    keys = {f["key"] for f in facts}
    assert {"baidu_rate", "pairs", "detectors", "containers"} <= keys
    for f in facts:
        assert f["source"].strip(), f
        assert f["text"].strip(), f


# ---------------------------------------------------------------------------
# 2. Log training: thiếu thì KHÔNG được bịa
# ---------------------------------------------------------------------------
def test_loss_log_returns_none_when_missing(tmp_path, monkeypatch):
    sys.path.insert(0, str(MANIM))
    import data as D  # noqa: PLC0415

    monkeypatch.setenv("DIRE_LOSS_LOG", str(tmp_path / "khong_co.json"))
    assert D.loss_log() is None
    assert "chưa có log" in D.demo_label(None)


def test_loss_log_reads_real_demo_train_output(tmp_path, monkeypatch):
    """Log do `demo_train` sinh ra phải đọc được, và nhãn demo đi kèm dữ liệu."""
    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(MANIM))
    import data as D  # noqa: PLC0415
    from dire_modal import demo_train as DT  # noqa: PLC0415

    steps = [DT.log_entry(1, 1, 0.70, 0.50, 64),
             DT.log_entry(10, 1, 0.42, 0.75, 64)]
    payload = DT.log_payload("t", steps, [{"epoch": 1, "val_acc": 0.8}], {"read": 128})
    path = tmp_path / "loss_log.json"
    DT.write_log(path, payload)
    monkeypatch.setenv("DIRE_LOSS_LOG", str(path))

    log = D.loss_log()
    assert log is not None
    assert D.loss_points(log) == [(1, 0.70), (10, 0.42)]
    assert D.val_points(log) == [(1, 0.8)]
    assert D.demo_label(log) == DT.DEMO_LABEL
    assert "paper" in D.demo_label(log)


def test_val_points_drop_missing_values():
    sys.path.insert(0, str(MANIM))
    import data as D  # noqa: PLC0415

    log = {"steps": [{"step": 1, "loss": 0.5}],
           "epochs": [{"epoch": 1, "val_acc": None}, {"epoch": 2, "val_acc": 0.7}]}
    assert D.val_points(log) == [(2, 0.7)]


# ---------------------------------------------------------------------------
# 3. Scene dựng được dưới stub
# ---------------------------------------------------------------------------
def test_exactly_the_five_documented_scenes(scenes_module):
    got = {n for n, v in vars(scenes_module).items()
           if isinstance(v, type) and issubclass(v, _Scene) and v is not _Scene}
    assert got == EXPECTED_SCENES, got


@pytest.mark.parametrize("name", sorted(EXPECTED_SCENES))
def test_scene_constructs_without_error(scenes_module, name, monkeypatch, tmp_path):
    """`construct()` chạy hết: bắt NameError/AttributeError mà không cần cairo."""
    monkeypatch.setenv("DIRE_LOSS_LOG", str(tmp_path / "khong_co.json"))
    scene = getattr(scenes_module, name)()
    scene.construct()
    assert scene.played, f"{name} không play gì — scene rỗng?"
    assert scene.waited >= 1, f"{name} không giữ khung cuối (GIF sẽ cut cụt)"


def test_train_scene_without_log_shows_the_missing_state(scenes_module, monkeypatch,
                                                         tmp_path, capsys):
    monkeypatch.setenv("DIRE_LOSS_LOG", str(tmp_path / "khong_co.json"))
    scene = scenes_module.TrainMechanism()
    scene.construct()
    texts = _texts(scene.played)
    assert any("chưa có log" in t.lower() for t in texts), texts
    assert any("demo_train::main" in t for t in texts), texts


def test_train_scene_with_log_draws_curve_and_label(scenes_module, monkeypatch, tmp_path):
    sys.path.insert(0, str(SRC))
    from dire_modal import demo_train as DT  # noqa: PLC0415

    steps = [DT.log_entry(1, 1, 0.70, 0.50, 64), DT.log_entry(10, 1, 0.42, 0.75, 64),
             DT.log_entry(20, 2, 0.30, 0.85, 64)]
    payload = DT.log_payload("t", steps, [{"epoch": 1, "val_acc": 0.8}], {"read": 128})
    path = tmp_path / "loss_log.json"
    DT.write_log(path, payload)
    monkeypatch.setenv("DIRE_LOSS_LOG", str(path))

    scene = scenes_module.TrainMechanism()
    scene.construct()
    texts = _texts(scene.played)
    assert any("0.700" in t and "0.300" in t for t in texts), texts
    assert any(DT.DEMO_LABEL in t for t in texts), texts
    assert not any("chưa có log" in t.lower() for t in texts)


def _texts(anims) -> list[str]:
    out: list[str] = []

    def walk(o):
        if isinstance(o, str):
            out.append(o)
        elif isinstance(o, _M):
            if getattr(o, "text", ""):
                out.append(o.text)
            for ch in o.children:
                walk(ch)
        elif isinstance(o, _Anim):
            for a in o.args:
                walk(a)

    for a in anims:
        walk(a)
    return out


# ---------------------------------------------------------------------------
# 4. Hạ tầng render
# ---------------------------------------------------------------------------
def test_render_script_is_strict_and_msys_safe():
    text = (MANIM / "render.sh").read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert "MSYS_NO_PATHCONV=1" in text, "Git Bash sẽ đổi đường dẫn tuyệt đối"
    assert "PYTHONPATH" in text and "src" in text, "data.py cần import dire_modal"
    assert "manim" in text


def test_render_script_lists_scenes_from_source():
    """Danh sách scene phải đọc từ scenes.py, không chép tay."""
    text = (MANIM / "render.sh").read_text(encoding="utf-8")
    assert "scenes.py" in text
    assert "grep" in text or "--check" in text


@pytest.mark.skipif(not (Path("/bin/bash").exists() or Path("/usr/bin/bash").exists()),
                    reason="máy không có bash")
def test_render_script_check_mode_lists_all_scenes():
    r = subprocess.run(["bash", str(MANIM / "render.sh"), "--check"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert set(r.stdout.split()) == EXPECTED_SCENES, r.stdout


def test_readme_documents_every_scene():
    text = (MANIM / "README.md").read_text(encoding="utf-8")
    for name in EXPECTED_SCENES:
        assert name in text, f"README không nhắc scene {name}"


def test_style_keeps_the_vietnamese_font_configurable():
    """Font thiếu glyph ⇒ ô vuông, và chỉ thấy sau khi render. Phải đổi được."""
    text = STYLE_PY.read_text(encoding="utf-8")
    assert "DIRE_FONT" in text
    assert "DejaVu Sans" in text


def test_no_mathtex_anywhere():
    """`MathTex` cần TeX Live 1–2 GB — scene phải chạy với `Text` + Unicode."""
    for path in (STYLE_PY, SCENES_PY):
        assert "MathTex" not in path.read_text(encoding="utf-8"), path.name


def test_manim_package_is_not_imported_by_the_repo_package():
    """manim không được lọt vào `src/` — nó cần numpy>=2.1, torch cần 1.26.4."""
    hits = [p.name for p in (ROOT / "src" / "dire_modal").glob("*.py")
            if "import manim" in p.read_text(encoding="utf-8")]
    assert not hits, hits
