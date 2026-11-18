"""Tài liệu không được chứa lệnh **không copy-paste được**.

Sự cố 2026-10 (Windows, Git Bash). Tài liệu ghi::

    python tools/make_gallery.py --maps <volume>/maps/test --orig <volume>/images/test ...

Người dùng copy nguyên văn. Bash hiểu `<volume>` là **chuyển hướng stdin** từ một
tệp tên `volume`, nên kết quả là::

    bash: volume: No such file or directory

`python` không bao giờ chạy, và thông báo không nhắc gì tới placeholder. Cùng lỗi
với `<repo>`, `<user>`, `<ws>` — mọi thứ trong ngoặc nhọn.

Quy tắc ở đây: trong một khối lệnh, **không** được có `<tên>`. Muốn nói "chỗ này
do bạn điền" thì dùng biến shell (`$VOL`) hoặc một giá trị mẫu cụ thể. Nếu vẫn
muốn nhắc placeholder để giải thích, viết nó trong **dòng chú thích** (`#`) —
test bỏ qua dòng bắt đầu bằng `#`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: Chỉ những tệp người dùng thật sự copy lệnh ra chạy.
DOCS = [
    REPO / "README.md",
    REPO / "skills.md",
    REPO / "platforms" / "hf-spaces" / "README.md",
    *sorted((REPO / "docs").glob("*.md")),
]

FENCE = re.compile(r"^```([A-Za-z0-9_+-]*)\s*$")
SHELL_LANGS = {"bash", "sh", "shell", "console", "zsh", "cmd", "bat", "powershell"}
PLACEHOLDER = re.compile(r"<[A-Za-z][A-Za-z0-9_-]*>")


def shell_blocks(text: str) -> list[tuple[int, str]]:
    """`[(số_dòng, nội_dung)]` cho từng dòng nằm trong khối lệnh."""
    out: list[tuple[int, str]] = []
    inside = False
    for i, line in enumerate(text.splitlines(), start=1):
        m = FENCE.match(line)
        if m:
            inside = (m.group(1).lower() in SHELL_LANGS) if not inside else False
            continue
        if inside:
            out.append((i, line))
    return out


def offenders(path: Path) -> list[str]:
    bad = []
    for lineno, line in shell_blocks(path.read_text(encoding="utf-8")):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue                     # chú thích được phép nhắc placeholder
        for token in PLACEHOLDER.findall(line):
            bad.append(f"{path.relative_to(REPO)}:{lineno}: {token} trong `{stripped[:70]}`")
    return bad


@pytest.mark.parametrize("path", DOCS, ids=lambda p: str(p.relative_to(REPO)))
def test_no_placeholder_inside_a_copyable_command(path: Path):
    assert path.exists(), path
    bad = offenders(path)
    assert not bad, (
        "placeholder `<...>` trong khối lệnh: bash hiểu là chuyển hướng stdin và\n"
        "lệnh chết với '<tên>: No such file or directory'.\n"
        "Dùng biến shell (`$VOL`, `$HF_USER`) hoặc một giá trị cụ thể:\n  "
        + "\n  ".join(bad)
    )


def test_the_checker_itself_catches_the_incident():
    """Test của test: nếu bộ dò sai, mọi thứ trên đều xanh một cách vô nghĩa."""
    sample = """```bash
# đây là chú thích, được phép nhắc <volume>
python tools/make_gallery.py --maps <volume>/maps/test
```

```text
<volume> trong khối không phải lệnh thì không sao
```
"""
    tmp = REPO / ".tmp_placeholder_check.md"
    try:
        tmp.write_text(sample, encoding="utf-8")
        bad = offenders(tmp)
    finally:
        tmp.unlink(missing_ok=True)
    assert len(bad) == 1 and "<volume>" in bad[0]
    assert "--maps" in bad[0]
