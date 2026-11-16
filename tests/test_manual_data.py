"""Tải tay: danh sách phải khớp manifest, kiểm chứng phải bắt được file hỏng.

Ba lỗi mà người tải tay hay gặp, và test ở đây canh từng cái:

1. **đổi tên** file (Baidu hay thêm `(1)`, hoặc sửa `midjouney` → `midjourney`);
2. **tải hụt** — file 200 B hoặc trang HTML đăng nhập được lưu thành `.tar.gz`;
3. **đặt sai chỗ** — giải nén phẳng hết vào một thư mục.

Test dùng thư mục tạm, không cần mạng, không cần Modal.
"""

from __future__ import annotations

import io
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "manual_data.py"


def _run(args: list[str], cwd: Path | None = None):
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True, text=True, cwd=str(cwd or ROOT),
    )


def test_list_shows_typo_names_and_local_paths():
    r = _run(["list", "--need", "gallery", "--dest", "data/raw", "--tsv", ""])
    assert r.returncode == 0, r.stderr
    out = r.stdout
    # đường dẫn remote nguyên bản + đích local dưới data/raw/
    assert "images/test/lsun_bedroom/pndm.tar.gz" in out
    assert "data/raw/images/test/lsun_bedroom/pndm.tar.gz" in out
    # ba nhánh của cùng một generator
    assert "recons/test/lsun_bedroom/lsun_bedroom/pndm.tar.gz" in out
    assert "dire/test/lsun_bedroom/lsun_bedroom/pndm.tar.gz" in out
    # cảnh báo tên dễ sai
    assert "midjouney" in out and "stylegan_official" in out


def test_list_includes_outside_sources_for_path_a():
    r = _run(["list", "--need", "a", "--dest", "data/raw", "--tsv", ""])
    assert r.returncode == 0, r.stderr
    assert "256x256_diffusion_uncond.pt" in r.stdout
    assert "bedroom_val_lmdb.zip" in r.stdout
    # A không lấy gì từ dataset (khối cảnh báo cuối có nhắc tên, nên chỉ kiểm
    # phần ĐÍCH của danh sách)
    assert "data/raw/dire/" not in r.stdout
    assert "data/raw/recons/" not in r.stdout


def test_list_never_touches_train_or_ddim_tars():
    r = _run(["list", "--need", "e2,ckpts", "--dest", "data/raw", "--tsv", ""])
    out = r.stdout
    assert "/train/" not in out, "train chiếm 77% cây và không cần cho eval"
    assert "ddim" not in out.lower(), "8 tar ablation *ddim* nằm trong NEVER"


def test_unknown_need_is_rejected():
    r = _run(["list", "--need", "khong-co", "--dest", "data/raw"])
    assert r.returncode != 0
    assert "nhóm lạ" in (r.stderr + r.stdout)


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #

def _good_tar(path: Path, member: str = "0_real/a.jpg") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # nội dung ngẫu nhiên để gzip không nén còn vài chục byte (bản thân phép
    # kiểm từ chối file < 256 B — tar thật luôn lớn hơn nhiều)
    data = b"\xff\xd8" + bytes(range(256)) * 64
    with tarfile.open(path, "w:gz") as tf:
        ti = tarfile.TarInfo(member)
        ti.size = len(data)
        tf.addfile(ti, io.BytesIO(data))


def _good_pth(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"PK\x03\x04" + b"0" * 8192)


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    d = tmp_path / "raw"
    _good_pth(d / "checkpoints" / "lsun_adm.pth")
    _good_tar(d / "dire" / "test" / "lsun_bedroom" / "lsun_bedroom" / "real.tar.gz")
    _good_tar(d / "dire" / "test" / "lsun_bedroom" / "lsun_bedroom" / "adm.tar.gz")
    return d


def test_verify_passes_on_complete_tree(tree: Path):
    r = _run(["verify", "--need", "e2min", "--dest", str(tree), "--strict"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "3 đạt" in r.stdout


def test_verify_detects_html_saved_as_tar(tree: Path):
    p = tree / "dire/test/lsun_bedroom/lsun_bedroom/adm.tar.gz"
    p.write_bytes(b"<!DOCTYPE html><html>please log in</html>" * 40)
    r = _run(["verify", "--need", "e2min", "--dest", str(tree), "--strict"])
    assert r.returncode == 3
    assert "TRANG HTML" in r.stdout


def test_verify_detects_truncated_tar(tree: Path):
    p = tree / "dire/test/lsun_bedroom/lsun_bedroom/real.tar.gz"
    data = p.read_bytes()
    p.write_bytes(data[: len(data) // 2])
    r = _run(["verify", "--need", "e2min", "--dest", str(tree), "--strict"])
    assert r.returncode == 3
    assert "hỏng" in r.stdout or "rỗng" in r.stdout


def test_verify_reports_missing_with_exact_target(tree: Path):
    (tree / "checkpoints/lsun_adm.pth").unlink()
    r = _run(["verify", "--need", "e2min", "--dest", str(tree), "--strict"])
    assert r.returncode == 3
    assert "checkpoints/lsun_adm.pth" in r.stdout
    assert "data/raw/checkpoints/lsun_adm.pth" in r.stdout.replace(str(tree), "data/raw")


def test_verify_finds_file_placed_wrongly_but_uniquely(tree: Path):
    """Người dùng giải nén phẳng: vẫn tìm theo TÊN, nhưng nói rõ đã tìm ở đâu."""
    src = tree / "checkpoints/lsun_adm.pth"
    flat = tree / "lsun_adm.pth"
    src.rename(flat)
    r = _run(["verify", "--need", "e2min", "--dest", str(tree)])
    assert "theo tên" in r.stdout


def test_verify_flags_partial_download_junk(tree: Path):
    (tree / "dire/test/lsun_bedroom/lsun_bedroom/adm.tar.gz.part").write_bytes(b"x" * 2048)
    r = _run(["verify", "--need", "e2min", "--dest", str(tree)])
    assert "RÁC" in r.stdout


# --------------------------------------------------------------------------- #
# push: mặc định chỉ IN lệnh, không chạy
# --------------------------------------------------------------------------- #

def test_push_is_dry_run_by_default(tree: Path):
    r = _run(["push", "--need", "e2min", "--dest", str(tree)])
    assert r.returncode == 0, r.stderr
    assert "modal volume put dire-raw" in r.stdout
    assert "--execute" in r.stdout  # nhắc cách chạy thật


# --------------------------------------------------------------------------- #
# preflight: kiểm máy
# --------------------------------------------------------------------------- #

PREFLIGHT = ROOT / "tools" / "preflight_local.sh"


def test_preflight_parses():
    r = subprocess.run(["bash", "-n", str(PREFLIGHT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_preflight_disables_msys_path_conversion():
    text = PREFLIGHT.read_text(encoding="utf-8")
    assert "MSYS_NO_PATHCONV=1" in text
    assert "MSYS2_ARG_CONV_EXCL" in text


def test_preflight_runs_and_reports_python(tmp_path: Path):
    r = subprocess.run(["bash", str(PREFLIGHT)], capture_output=True, text=True,
                       env={**__import__("os").environ,
                            "DIRE_LOCAL_ROOT": str(tmp_path / "root")})
    out = r.stdout + r.stderr
    assert "Python" in out
    assert "DIRE_LOCAL_ROOT" in out
