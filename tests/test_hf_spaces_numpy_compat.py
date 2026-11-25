from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "platforms" / "hf-spaces"
if str(SPACE) not in sys.path:
    sys.path.insert(0, str(SPACE))

from demolib import live  # noqa: E402


def test_numpy_two_is_reported_before_importing_incompatible_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(__version__="2.5.2"))
    torch, modules, error = live._try_import()
    assert torch is None and modules is None
    assert "NumPy 2.5.2" in error and "numpy<2" in error
    status = live.status()
    assert not status.available
    assert any("numpy<2" in hint for hint in status.hints)
