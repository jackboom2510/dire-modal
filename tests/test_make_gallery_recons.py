from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def _load_tool():
    path = ROOT / "tools" / "make_gallery_recons.py"
    spec = importlib.util.spec_from_file_location("make_gallery_recons_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_all_dry_run_uses_gallery_originals_and_needs_no_checkpoint(capsys):
    tool = _load_tool()
    gallery = ROOT / "modal-runs" / "gallery"
    assert tool.main(["--all", "--dry-run", "--gallery", str(gallery)]) == 0
    output = capsys.readouterr().out
    assert "_orig.png →" in output
    assert "_recon.png" in output
    assert "recon cần tính" in output
    assert "giờ CPU" in output
