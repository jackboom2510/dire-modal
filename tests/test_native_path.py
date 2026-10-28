"""`tools/lib/native_path.sh` — đổi đường dẫn cho chương trình WINDOWS thật.

Vì sao có file test này
-----------------------
Sự cố 2026-10-28 (Windows, Git Bash):

    $ bash tools/manim/render.sh --quality l
    FileNotFoundError: D:\\d\\Documents\\Code\\dire-modal\\tools\\manim\\scenes.py not found
                              ^^^ một tầng thư mục KHÔNG tồn tại

Nguyên nhân: `render.sh` (và mọi script của repo) tắt path conversion của MSYS2
để gửi nguyên văn `/raw` cho `modal`. Nhưng `manim`/`ffmpeg`/`python.exe` là
chương trình Windows: nhận `/d/Documents/...` thì Windows hiểu là "ổ hiện tại +
thư mục `d`". Không có exception nào nói "đây là lỗi đường dẫn"; chỉ có
`FileNotFoundError` với một đường dẫn *trông gần đúng*.

Test ở đây kiểm cả hai chiều:
  1. có `cygpath` ⇒ `/d/x` → `D:/x` (dạng "mixed": Windows đọc được, bash vẫn đọc được)
  2. không có `cygpath` ⇒ **nguyên văn** (Linux/macOS/container: không được đổi bừa)
  3. `cygpath` lỗi ⇒ vẫn trả nguyên văn, không nuốt đường dẫn
  4. chuỗi rỗng ⇒ rỗng, KHÔNG phải thư mục hiện tại

Chạy::

    python3 -m pytest tests/test_native_path.py -q
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "tools" / "lib" / "native_path.sh"
BASH = "/bin/bash" if Path("/bin/bash").exists() else "/usr/bin/bash"

pytestmark = pytest.mark.skipif(
    not Path(BASH).exists(), reason="máy không có bash"
)


def _run(script: str, env: dict | None = None) -> subprocess.CompletedProcess:
    full_env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    full_env.update(env or {})
    return subprocess.run([BASH, "-c", script], capture_output=True, text=True,
                          env=full_env, cwd=ROOT)


def _fake_cygpath(tmp_path: Path, behaviour: str = "map") -> Path:
    """`cygpath` giả: mô phỏng Git Bash — `/d/x` → `D:/x` (ở đây dùng `Z:`)."""
    p = tmp_path / "cygpath"
    if behaviour == "map":
        body = """#!/usr/bin/env bash
# git-bash giả: -m -- /x/y  →  Z:/x/y
for a in "$@"; do case "$a" in -*) ;; *) path="$a" ;; esac; done
printf 'Z:%s\\n' "$path"
"""
    else:               # lỗi: mô phỏng đường dẫn mà cygpath không xử lý được
        body = "#!/usr/bin/env bash\nexit 1\n"
    p.write_text(body, encoding="utf-8")
    p.chmod(0o755)
    return p


def test_helper_exists_and_is_not_executable_as_a_script():
    """File thư viện: có `set` guard không? Không — chỉ định nghĩa hàm."""
    text = HELPER.read_text(encoding="utf-8")
    assert "native_path()" in text and "is_windows_shell()" in text
    assert "set -euo" not in text.split("native_path()")[0].replace("#!/usr/bin/env bash", ""), (
        "file được `source`; đặt `set -e` ở đây sẽ đổi hành vi script gọi nó"
    )


def test_converts_to_mixed_path_when_cygpath_exists(tmp_path):
    _fake_cygpath(tmp_path)
    r = _run(f'. "{HELPER}"; native_path /d/Documents/Code/x', {"PATH": f"{tmp_path}:/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "Z:/d/Documents/Code/x"


def test_passthrough_when_no_cygpath(tmp_path):
    """Linux/macOS/container: KHÔNG được đổi gì."""
    empty = tmp_path / "bin"
    empty.mkdir()
    r = _run(f'. "{HELPER}"; native_path /home/user/dire-modal/src',
             {"PATH": f"{empty}:/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "/home/user/dire-modal/src"


def test_falls_back_to_input_when_cygpath_fails(tmp_path):
    """cygpath lỗi thì trả nguyên văn — lỗi thật phải hiện ở chỗ dùng."""
    _fake_cygpath(tmp_path, behaviour="fail")
    r = _run(f'. "{HELPER}"; native_path /d/x', {"PATH": f"{tmp_path}:/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "/d/x"


def test_empty_stays_empty(tmp_path):
    """`native_path ""` phải là "", không phải `$PWD` (cygpath không tham số hay
    trả thư mục hiện tại, nên phải chặn trước khi gọi)."""
    _fake_cygpath(tmp_path)
    r = _run(f'. "{HELPER}"; v="$(native_path "")"; printf "[%s]" "$v"',
             {"PATH": f"{tmp_path}:/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    assert r.stdout == "[]"


def test_path_with_spaces_survives():
    """`C:/Program Files/...` là đường dẫn Windows có thật (Git cài ở đó)."""
    r = _run(f'. "{HELPER}"; native_path "/c/Program Files/Git/raw"')
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "/c/Program Files/Git/raw"      # không có cygpath


def test_is_windows_shell_is_false_on_linux():
    r = _run(f'. "{HELPER}"; is_windows_shell && echo windows || echo other')
    assert r.stdout.strip() == "other"


# ---------------------------------------------------------------------------
# Nơi dùng: những chỗ gọi chương trình NATIVE phải đi qua helper
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("script", ["tools/manim/render.sh", "tools/local_download.sh"])
def test_scripts_that_call_native_binaries_source_the_helper(script):
    text = (ROOT / script).read_text(encoding="utf-8")
    if "manim" not in text and "PYTHONPATH" not in text:
        pytest.skip(f"{script} không gọi binary native")
    assert "native_path.sh" in text, f"{script} phải `source tools/lib/native_path.sh`"
    assert "native_path " in text, f"{script} chưa dùng native_path()"


def test_lib_is_shared_not_copied():
    """Chỉ MỘT bản ĐỊNH NGHĨA `native_path()` trong repo.

    Nhắc tới tên hàm trong ghi chú thì không tính (script khác phải `source`
    file này, không chép 6 dòng vào 5 chỗ để rồi lệch nhau).
    """
    pattern = re.compile(r"^\s*native_path\(\)\s*\{", re.M)
    defs = []
    for p in ROOT.rglob("*.sh"):
        if ".git" in p.parts:
            continue
        if pattern.search(p.read_text(encoding="utf-8")):
            defs.append(p.relative_to(ROOT).as_posix())
    assert defs == ["tools/lib/native_path.sh"], defs
