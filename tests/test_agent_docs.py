"""Test cho bộ tài liệu agent (`AGENT.md`, `instructions.md`, `plan.md`, `skills.md`, `CLAUDE.md`).

Vì sao có file này
------------------
Tài liệu cho agent chỉ có giá trị khi **đúng**. Một link chết hay một
`dire_modal.xxx::yyy` không còn tồn tại sẽ khiến agent tiếp theo chạy lệnh sai
hoặc đi tìm file không có — và không ai phát hiện, vì tài liệu không có test.

Đây đúng loại lỗi đã xảy ra trong repo: `docs/iaas-options.md` hướng dẫn sửa giá
ở `tools/iaas_cost.py` suốt nhiều tháng trong khi file đó không tồn tại.

Test này kiểm:
  1. Mọi link markdown `.md` giữa các tài liệu đều trỏ tới file có thật.
  2. Mọi đường dẫn trong backtick (`src/...`, `tests/...`, `tools/...`) đều tồn tại
     — trừ danh sách `KNOWN_MISSING` được khai báo tường minh.
  3. Mọi `dire_modal.<module>::<entrypoint>` được nhắc đều là hàm thật.
  4. `KNOWN_MISSING` không bị "mục" — nếu file xuất hiện thì phải xoá khỏi danh
     sách, nếu không danh sách sẽ che lỗi mới trong tương lai.

Chạy::

    python3 -m pytest tests/test_agent_docs.py -q
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["AGENT.md", "instructions.md", "plan.md", "skills.md", "CLAUDE.md"]

# Đường dẫn được nhắc trong tài liệu nhưng CỐ Ý chưa tồn tại.
# Mỗi mục phải kèm lý do, và `plan.md` phải đang theo dõi nó.
KNOWN_MISSING = {
    "tools/iaas_cost.py": "được docs/iaas-options.md nhắc; plan.md mục 1 đang theo dõi",
}

_LINK_RE = re.compile(r"\]\(([A-Za-z0-9._/-]+\.md)\)")
_PATH_RE = re.compile(r"`((?:src|tests|tools|scripts|docs|configs|platforms)/[A-Za-z0-9._/-]+)`")
_ENTRY_RE = re.compile(r"dire_modal\.(\w+)::(\w+)")


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _entrypoints() -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for path in (ROOT / "src" / "dire_modal").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found.add((path.stem, node.name))
    return found


def test_all_docs_exist():
    missing = [d for d in DOCS if not (ROOT / d).exists()]
    assert not missing, f"thiếu tài liệu agent: {missing}"


def test_internal_links_resolve():
    broken = []
    for name in DOCS:
        for target in _LINK_RE.findall(_read(name)):
            if not (ROOT / target).exists():
                broken.append(f"{name} → {target}")
    assert not broken, "link chết trong tài liệu agent:\n  " + "\n  ".join(broken)


def test_referenced_paths_exist():
    bad = []
    for name in DOCS:
        for target in _PATH_RE.findall(_read(name)):
            target = target.rstrip("/")
            if any(c in target for c in "*?{}"):
                continue  # là glob, không phải đường dẫn cụ thể
            if (ROOT / target).exists():
                continue
            if target in KNOWN_MISSING:
                continue
            bad.append(f"{name} → {target}")
    assert not bad, (
        "đường dẫn không tồn tại (thêm vào KNOWN_MISSING nếu cố ý):\n  "
        + "\n  ".join(bad)
    )


def test_known_missing_is_actually_missing():
    """Nếu file đã xuất hiện thì phải xoá khỏi danh sách — nếu không nó che lỗi mới."""
    appeared = [p for p in KNOWN_MISSING if (ROOT / p).exists()]
    assert not appeared, (
        f"{appeared} đã tồn tại ⇒ xoá khỏi KNOWN_MISSING trong tests/test_agent_docs.py"
    )


def test_known_missing_is_tracked_in_plan():
    """Thứ đang thiếu phải có người chịu trách nhiệm, không phải ghi chú rơi vãi."""
    plan = _read("plan.md")
    for path in KNOWN_MISSING:
        assert Path(path).name in plan, (
            f"{path} được nhắc trong tài liệu nhưng không có trong plan.md — "
            "hoặc sửa tài liệu cho khớp thực tế, hoặc thêm việc vào plan"
        )


def test_referenced_entrypoints_exist():
    entries = _entrypoints()
    bad = []
    for name in DOCS:
        for mod, fn in _ENTRY_RE.findall(_read(name)):
            if (mod, fn) not in entries:
                bad.append(f"{name} → dire_modal.{mod}::{fn}")
    assert not bad, (
        "entrypoint được nhắc nhưng không tồn tại:\n  " + "\n  ".join(bad)
    )


def test_agent_md_points_to_all_other_docs():
    """Agent đọc AGENT.md trước; nó phải dẫn tới mọi file còn lại."""
    text = _read("AGENT.md")
    missing = [d for d in DOCS if d != "AGENT.md" and d not in text]
    assert not missing, f"AGENT.md không nhắc: {missing}"


def test_policy_keys_mentioned_in_docs_exist():
    """Tên khoá policy trong tài liệu phải là field thật của `Policy`."""
    import dataclasses

    sys.path.insert(0, str(ROOT / "src"))
    from dire_modal.manifest import Policy  # noqa: PLC0415

    fields = {f.name for f in dataclasses.fields(Policy)}
    # chỉ kiểm những tên trông như khoá policy và thực sự xuất hiện trong yaml
    yaml_text = (ROOT / "src/dire_modal/tiers.yaml").read_text(encoding="utf-8")
    bad = []
    for name in DOCS:
        for key in re.findall(r"`([a-z][a-z0-9_]{4,})`", _read(name)):
            if key in fields:
                continue
            # chỉ báo lỗi khi tài liệu trình bày nó NHƯ một khoá policy
            if re.search(rf"^{key}:", yaml_text, re.M):
                bad.append(f"{name} → {key}")
    assert not bad, (
        "khoá policy trong tài liệu không có trong `Policy`:\n  " + "\n  ".join(bad)
    )


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
