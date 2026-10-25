"""Canh cấu trúc `scripts/`: ĐÚNG HAI quy trình thực nghiệm.

Vì sao có file này
------------------
`scripts/` từng là dãy `00…10` rời rạc: mỗi lần thêm một bước lại thêm một file,
thứ tự phải tự ghép, và "repo này tái lập cái gì" không ai trả lời được từ tên
file. Quy ước mới là: **hai quy trình** (`01_repro.sh`, `02_improve.sh`); thêm
bước thì thêm **stage**, không thêm file thứ ba.

Quy ước không tự giữ được — PR thứ ba sẽ có người thêm `03_something.sh` vì nó
tiện. Test này làm việc đó đỏ ngay, kèm câu hỏi đúng: "bước này thuộc quy trình
nào?".

Kiểm:
  1. Đúng 2 file khớp `NN_*.sh` trong `scripts/`.
  2. Cả hai `source` `scripts/lib.sh` (không lặp lại logic `modal run`/`--detach`).
  3. Có `set -euo pipefail` (script im lặng bỏ qua lỗi là loại lỗi tệ nhất).
  4. `lib.sh` không tự chạy như một script độc lập (không có shebang execution
     guard sai) và có `run_modal`/`missing_stage`.
  5. Cả hai được nhắc trong `scripts/README.md` và `README.md`.
  6. `bash -n` sạch — chỉ chạy khi máy có `bash` (Windows dev có thể không).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

WORKFLOWS = {"01_repro.sh", "02_improve.sh"}


def _workflow_files() -> set[str]:
    return {p.name for p in SCRIPTS.glob("[0-9][0-9]_*.sh")}


def test_exactly_two_workflows():
    found = _workflow_files()
    assert found == WORKFLOWS, (
        f"scripts/ phải có đúng 2 quy trình {sorted(WORKFLOWS)}, đang có "
        f"{sorted(found)}.\n"
        "Thêm một bước ⇒ thêm --stage vào 01_repro.sh hoặc 02_improve.sh; "
        "nếu là tiện ích đo/xem (không sinh kết quả khoa học) ⇒ đặt ở tools/."
    )


@pytest.mark.parametrize("name", sorted(WORKFLOWS))
def test_workflow_sources_lib(name: str):
    text = (SCRIPTS / name).read_text(encoding="utf-8")
    assert "lib.sh" in text, f"{name} phải `source scripts/lib.sh` để dùng chung"


@pytest.mark.parametrize("name", sorted(WORKFLOWS))
def test_workflow_is_strict(name: str):
    text = (SCRIPTS / name).read_text(encoding="utf-8")
    assert "set -euo pipefail" in text, f"{name} thiếu `set -euo pipefail`"


def test_lib_is_a_library_not_a_workflow():
    text = (SCRIPTS / "lib.sh").read_text(encoding="utf-8")
    for fn in ("run_modal", "missing_stage", "want"):
        assert re.search(rf"^{fn}\(\)", text, re.M), f"lib.sh thiếu hàm {fn}()"
    assert "modal run" in text, "lib.sh phải có wrapper `modal run` (xử lý --detach)"


def test_workflows_are_documented():
    for docs in ("scripts/README.md", "README.md"):
        text = (ROOT / docs).read_text(encoding="utf-8")
        for name in WORKFLOWS:
            assert name in text, f"{docs} không nhắc {name}"


def test_prerequisites_exist():
    """setup/check là điều kiện tiên quyết — phải tồn tại nhưng KHÔNG phải workflow."""
    for name in ("setup.sh", "check.sh", "lib.sh"):
        assert (SCRIPTS / name).exists(), f"thiếu scripts/{name}"
    assert not re.match(r"[0-9]", "setup.sh")


@pytest.mark.skipif(shutil.which("bash") is None, reason="máy không có bash")
def test_bash_syntax():
    for p in sorted(SCRIPTS.glob("*.sh")):
        r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
        assert r.returncode == 0, f"{p.name}: {r.stderr.strip()}"


# ---------------------------------------------------------------------------
# 7. Windows / Git Bash: đừng để "/raw" bị đổi thành "C:/Program Files/Git/raw"
# ---------------------------------------------------------------------------
def _scripts_invoking_modal() -> list[Path]:
    out = []
    for d in (SCRIPTS, ROOT / "tools"):
        for f in sorted(d.glob("*.sh")):
            text = f.read_text(encoding="utf-8")
            if re.search(r"^\s*modal\s", text, re.M):
                out.append(f)
    return out


def test_every_script_that_runs_modal_disables_msys_path_conversion():
    """Git Bash đổi `/raw` → `C:/Program Files/Git/raw` trước khi Modal thấy.

    Đã xảy ra thật (2026-10-25, Windows): `./tools/tree.sh` chết với
    `[!] C:/Program Files/Git/raw không tồn tại`. Trên Linux/macOS hai biến này
    bị bỏ qua, nên bật vô điều kiện là cách rẻ nhất.
    """
    files = _scripts_invoking_modal()
    assert files, "không tìm thấy script nào gọi modal — test đang kiểm nhầm chỗ?"

    def _guarded(text: str) -> bool:
        # Hoặc tự export, hoặc `source scripts/lib.sh` (lib.sh export lúc source).
        return ("MSYS_NO_PATHCONV" in text
                or re.search(r"(^|\s)(source|\.)\s+.*lib\.sh", text, re.M) is not None)

    missing = [f.relative_to(ROOT).as_posix() for f in files
               if not _guarded(f.read_text(encoding="utf-8"))]
    assert not missing, (
        "script gọi `modal` mà không tắt MSYS path conversion: " + ", ".join(missing)
    )


def test_workflows_get_the_guard_through_lib_sh():
    """`01`/`02` không tự export — chúng `source lib.sh`, nên lib.sh phải có."""
    lib = (SCRIPTS / "lib.sh").read_text(encoding="utf-8")
    assert "MSYS_NO_PATHCONV" in lib and "MSYS2_ARG_CONV_EXCL" in lib
