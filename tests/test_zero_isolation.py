"""Cách ly — điều kiện để gọi bản này là "from scratch".

Một bản viết lại mà lặng lẽ `import guided_diffusion` ở đâu đó thì mọi tuyên bố
độc lập đều vô nghĩa. Test này đọc **mã nguồn** của package chứ không chỉ chạy
nó, vì một import nằm trong `try/except` hay trong hàm ít dùng vẫn là chép.

Ngoại lệ duy nhất, có đánh dấu: khối `CLEAN-ROOM-CHECK` trong `modal_app.py`,
dùng `find_spec` để **khẳng định** gói của tác giả không tồn tại trong image.
Test đòi khối đó (a) chỉ có một, (b) nằm trong `selfcheck`, (c) không chứa
`import` thật nào ngoài `importlib`. Một ngoại lệ được kiểm chặt thì không phải
lỗ hổng.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "dire_zero"

FORBIDDEN_MODULES = ("guided_diffusion", "dire_core", "dire_modal", "dire_local")
FORBIDDEN_STRINGS = ("/opt/dire", "ZhendongWang6", "compute_dire.py", "third_party/DIRE")
ALLOWED_TOP = {
    "__future__", "abc", "argparse", "collections", "dataclasses", "datetime", "hashlib",
    "importlib", "io", "json", "math", "os", "pathlib", "random", "re", "shutil", "sys",
    "tarfile", "tempfile", "time", "typing", "zlib", "numpy", "PIL", "torch", "modal",
}

BLOCK_START = "# CLEAN-ROOM-CHECK"
BLOCK_END = "# END-CLEAN-ROOM-CHECK"


def _sources() -> list[Path]:
    files = sorted(PKG.rglob("*.py"))
    assert len(files) >= 8, f"chỉ thấy {len(files)} file — package thiếu?"
    return files


def _strip_docstrings(text: str) -> str:
    """Bỏ docstring — chúng *được phép* nhắc tới tên gói của tác giả để giải thích."""
    return re.sub(r'"""[\s\S]*?"""', '""', text)


def _strip_checked_block(text: str) -> tuple[str, list[str]]:
    """Bỏ khối được đánh dấu, trả (phần còn lại, nội dung các khối)."""
    blocks: list[str] = []
    out, inside, buf = [], False, []
    for line in text.splitlines():
        if BLOCK_START in line:
            inside, buf = True, []
            continue
        if BLOCK_END in line:
            inside = False
            blocks.append("\n".join(buf))
            continue
        (buf if inside else out).append(line)
    assert not inside, "khối CLEAN-ROOM-CHECK không có dòng kết thúc"
    return "\n".join(out), blocks


@pytest.mark.parametrize("path", _sources(), ids=lambda p: p.name)
def test_no_forbidden_import(path: Path):
    text, _ = _strip_checked_block(path.read_text(encoding="utf-8"))
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name.split(".")[0] not in FORBIDDEN_MODULES, (path.name, a.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            root = node.module.split(".")[0]
            assert root not in FORBIDDEN_MODULES, (path.name, node.module)
            assert root in ALLOWED_TOP or root == "dire_zero", (path.name, node.module)


@pytest.mark.parametrize("path", _sources(), ids=lambda p: p.name)
def test_no_reference_to_authors_code(path: Path):
    text, _ = _strip_checked_block(_strip_docstrings(path.read_text(encoding="utf-8")))
    for bad in FORBIDDEN_STRINGS:
        assert bad not in text, f"{path.name} nhắc tới {bad!r}"


def test_checked_block_is_unique_and_minimal():
    blocks = []
    for path in _sources():
        _, found = _strip_checked_block(path.read_text(encoding="utf-8"))
        blocks.extend((path.name, b) for b in found)
    assert len(blocks) == 1, f"có {len(blocks)} khối CLEAN-ROOM-CHECK, chỉ được 1"
    name, body = blocks[0]
    assert name == "modal_app.py"
    assert "find_spec" in body
    assert '"guided_diffusion"' in body or "'guided_diffusion'" in body
    # chỉ được import importlib, không được import gói của tác giả
    imports = [ln.strip() for ln in body.splitlines() if ln.strip().startswith("import ")]
    assert imports == ["import importlib.util  # noqa: PLC0415"], imports


def test_modal_app_image_has_no_git_clone():
    text = _strip_docstrings((PKG / "modal_app.py").read_text(encoding="utf-8"))
    assert "git clone" not in text
    assert "run_commands" not in text


def test_package_basics_import_without_torch(monkeypatch):
    """`info`/`keys`/`budget` phải chạy được trên máy chưa cài torch."""
    import builtins
    import importlib
    import sys

    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name.split(".")[0] == "torch":
            raise ImportError("chặn torch trong test")
        return real_import(name, *a, **k)

    for mod in [m for m in list(sys.modules) if m.startswith("dire_zero")]:
        del sys.modules[mod]
    monkeypatch.setattr(builtins, "__import__", fake)
    for mod in ("dire_zero.schedule", "dire_zero.budget", "dire_zero.metrics", "dire_zero.data"):
        importlib.import_module(mod)
    monkeypatch.undo()


def test_only_expected_third_party_deps():
    """Package tự chứa: ngoài stdlib chỉ numpy/torch/PIL/modal."""
    seen = set()
    for path in _sources():
        text, _ = _strip_checked_block(path.read_text(encoding="utf-8"))
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                seen.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                seen.add(node.module.split(".")[0])
    external = seen - set(ALLOWED_TOP) - {"dire_zero"}
    assert not external, f"phụ thuộc ngoài danh sách: {sorted(external)}"


def test_readme_of_package_states_the_claim():
    """Tài liệu phải nói rõ cái gì *không* phải của mình — nếu không là khoe quá."""
    doc = (ROOT / "docs" / "from-zero-clean-room.md").read_text(encoding="utf-8")
    assert "**Trọng số ADM**" in doc or "trọng số ADM" in doc
    assert "OpenAI" in doc
    assert re.search(r"không (phải|tái lập)", doc, re.I)
