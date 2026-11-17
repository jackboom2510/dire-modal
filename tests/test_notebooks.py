"""Test cho `notebooks/` — notebook phải chạy được và **không** được chứa logic vẽ.

Vì sao bộ test này tồn tại
--------------------------
Notebook là chỗ dễ trôi nhất trong một repo: nó không được import, không được
gọi, nên không ai phát hiện khi nó chết. Ba kiểu hỏng cụ thể:

1. **Notebook tự vẽ lại** thay vì gọi `tools/figures.py` ⇒ hình trong notebook
   và hình trong báo cáo lệch nhau, và không ai biết bản nào đúng.
2. **Output cũ được lưu trong file** ⇒ người đọc thấy một con số từ lần chạy
   trước, không biết nó từ đâu, không tái lập được.
3. **Đường dẫn cá nhân** (`F:/…`) nằm trong code ⇒ chạy được trên đúng một máy.

Test chạy notebook 01 **thật** (bằng stub matplotlib) trong thư mục tạm, nên nó
bắt được cả lỗi cú pháp lẫn lỗi gọi CLI.

Chạy::

    python3 -m pytest tests/test_notebooks.py -q
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
TOOL = ROOT / "tools" / "figures.py"
STUB = ROOT / "tests" / "fixtures" / "mpl_stub"

NOTEBOOKS = sorted(NB_DIR.glob("*.ipynb"))


def _figures_registry() -> set[str]:
    """Tên hình mà `tools/figures.py` hỗ trợ — đọc từ chính registry của nó."""
    src = TOOL.read_text(encoding="utf-8")
    block = src[src.index("FIGURES = {"):]
    block = block[: block.index("\n}")]
    return set(re.findall(r'^    "([a-z0-9-]+)":', block, flags=re.M))


def test_notebooks_exist():
    assert NOTEBOOKS, "không có notebook nào trong notebooks/"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_json_and_has_no_saved_output(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["nbformat"] == 4
    assert d["cells"], f"{path.name} rỗng"
    for i, cell in enumerate(d["cells"]):
        assert cell["cell_type"] in ("markdown", "code")
        assert "source" in cell
        assert "id" in cell, f"{path.name} cell {i} thiếu id (nbformat 4.5 bắt buộc)"
        if cell["cell_type"] == "code":
            # Output được lưu là số liệu cũ không tái lập được.
            assert cell.get("outputs") == [], f"{path.name} cell {i} còn output cũ"
            assert cell.get("execution_count") is None


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_only_calls_commands_that_exist(path: Path):
    """`fig_cmd('xyz')` mà `xyz` không có trong registry ⇒ notebook chết im lặng."""
    text = path.read_text(encoding="utf-8")
    called = set(re.findall(r"fig_cmd\(\s*'([a-z0-9-]+)'", text))
    unknown = called - _figures_registry()
    assert not unknown, f"{path.name} gọi hình không tồn tại: {sorted(unknown)}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_does_not_hardcode_a_personal_path(path: Path):
    """Quy ước repo: không khoá vào ổ đĩa nào; đường dẫn qua biến môi trường."""
    code = "\n".join(
        "".join(c["source"]) for c in json.loads(path.read_text(encoding="utf-8"))["cells"]
    )
    # Chỉ xét code, không xét comment ví dụ — comment được phép minh hoạ.
    for line in code.splitlines():
        stripped = line.split("#")[0]
        assert not re.search(r"[A-Za-z]:[/\\]", stripped), f"{path.name}: {line.strip()}"


def test_notebooks_do_not_reimplement_plotting():
    """Có `plt.`/`imshow(` trong notebook nghĩa là đã có bản vẽ thứ hai."""
    for path in NOTEBOOKS:
        code = "\n".join(
            "".join(c["source"])
            for c in json.loads(path.read_text(encoding="utf-8"))["cells"]
            if c["cell_type"] == "code"
        )
        for bad in ("plt.", "imshow(", "savefig("):
            assert bad not in code, f"{path.name} tự vẽ ({bad}) thay vì gọi tools/figures.py"


def test_readme_lists_every_notebook():
    readme = (NB_DIR / "README.md").read_text(encoding="utf-8")
    for path in NOTEBOOKS:
        assert path.name in readme, f"{path.name} chưa có trong notebooks/README.md"
    extra = set(re.findall(r"`(\d\d_[a-z_0-9]+\.ipynb)`", readme))
    assert extra <= {p.name for p in NOTEBOOKS}, f"README nhắc notebook không tồn tại: {extra}"


def test_schematic_notebook_runs_end_to_end(tmp_path: Path):
    """Chạy **thật** các ô code của notebook 01, trong thư mục tạm.

    Không dùng jupyter (nặng, chậm): chỉ ghép source các ô code rồi `exec` bằng
    subprocess. Điều cần chứng minh là các lệnh trong notebook chạy được, không
    phải notebook render đẹp.
    """
    nb = json.loads((NB_DIR / "01_hinh_schematic.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(STUB), str(ROOT / "src"), env.get("PYTHONPATH", "")]
    )
    env["DIRE_FIG_OUT"] = str(tmp_path / "figs")
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=str(ROOT), env=env,
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]

    out = tmp_path / "figs"
    for name in ("f3-pipeline", "f2-concept", "a1-provenance"):
        assert (out / f"{name}.pdf").exists(), f"thiếu {name}.pdf"
        side = json.loads((out / "data" / f"{name}.json").read_text(encoding="utf-8"))
        assert side["claim_label"] is None       # schematic: không mang nhãn R
    # notebook có ô kiểm mốc thời gian: 0 … 950
    assert "950" in proc.stdout
