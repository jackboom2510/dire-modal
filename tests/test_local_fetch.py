"""Test cho đường tải **LOCAL** (``dire_modal.local_fetch``).

Bối cảnh: đường Modal có ba trần (attempt 1800 s, lần gọi 7200 s, app 20 h) nên
một file 269 MB ở 76.6 KiB/s *luôn* bị cắt giữa chừng. Vì vậy mới có bản chạy
trên máy user — và bản đó phải trả lời **đúng** những câu mà đường Modal đã trả
lời sai một lần rồi:

1. ``ls -l`` in size ĐÃ LÀM TRÒN ⇒ lệch vài KB là bình thường, không được tải lại.
2. BaiduPCS-Go exit 0 mà 0 byte ⇒ phải retry, không được publish file rỗng.
3. File cũ hợp lệ bị thay ⇒ phải **giữ** bản cũ ở ``_held``, không xoá.
4. Staging phải **ổn định** giữa các lần chạy, nếu không resume vô nghĩa.
5. Không được kéo ``modal`` vào: máy user không cài và không cần.

Test ở đây stub đúng chỗ nghẽn thật (tiến trình ``BaiduPCS-Go``), không stub
phép kiểm — nếu stub cả phép kiểm thì test chỉ đang kiểm chính nó.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dire_modal import fetchcore as fc  # noqa: E402
from dire_modal import local_fetch as lf  # noqa: E402
from dire_modal.manifest import File  # noqa: E402


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Test không được chờ 30/60/120 s thật giữa các attempt."""
    monkeypatch.setattr(lf, "_sleep", lambda _s: None)


# ---------------------------------------------------------------------------
# 1. Không phụ thuộc Modal
# ---------------------------------------------------------------------------
def test_local_fetch_never_imports_modal():
    """Máy user không cài ``modal``; import nó ở đây là tính năng chết ngay."""
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import dire_modal.local_fetch\n"
        "print('modal' in sys.modules)\n" % str(Path(lf.__file__).parents[1])
    )
    cp = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert cp.returncode == 0, cp.stderr
    assert cp.stdout.strip() == "False", "local_fetch kéo `modal` vào ⇒ không chạy được ở local"


# ---------------------------------------------------------------------------
# 2. Staging ổn định ⇒ resume thật
# ---------------------------------------------------------------------------
def test_staging_dir_is_stable_across_processes(tmp_path):
    """``hash()`` đổi theo ``PYTHONHASHSEED`` ⇒ mỗi lần chạy staging một nơi,
    BaiduPCS-Go không thấy ``.BaiduPCS-Go-downloading`` cũ và tải lại từ 0 B."""
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from pathlib import Path\n"
        "from dire_modal.fetchcore import staging_dir_for\n"
        "print(staging_dir_for(Path('/x'), 'train/adm.tar.gz').name)\n" % str(Path(lf.__file__).parents[1])
    )
    outs = set()
    for seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        cp = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, env=env)
        assert cp.returncode == 0, cp.stderr
        outs.add(cp.stdout.strip())
    assert len(outs) == 1, f"staging đổi theo tiến trình: {outs} ⇒ resume mất tác dụng"


def test_staging_dir_separates_same_filename_in_different_folders(tmp_path):
    a = fc.staging_dir_for(tmp_path, "train/adm.tar.gz")
    b = fc.staging_dir_for(tmp_path, "val/adm.tar.gz")
    assert a != b, "hai file cùng tên khác thư mục mà chung staging ⇒ resume trộn mảnh dở"


# ---------------------------------------------------------------------------
# 3. Size làm tròn không được gây tải lại
# ---------------------------------------------------------------------------
def test_rounded_ls_size_does_not_trigger_a_redownload(tmp_path):
    """Ca thật: thư mục iddpm 125,015,131 B, ``ls -l`` in ``119.22MB``."""
    real = 125_015_131
    p = tmp_path / "iddpm.pth"
    p.write_bytes(b"\0" * 4096)
    with p.open("r+b") as fh:      # dựng file đúng size thật, không tốn 125 MB
        fh.truncate(real)
    est = fc.SizeEst(125_011_230, 5_244)   # `ls -l` in "119.22MB"
    v = fc._validate_ex(p, fc._bytes_of(est), slack=fc._slack_of(est))
    assert v.ok, f"lệch trong sai số làm tròn mà vẫn bị loại: {v.reason}"


def test_local_fetch_passes_the_slack_through(tmp_path, monkeypatch):
    """``fetch_one`` phải chuyển ``slack`` xuống ``_validate_ex`` — truyền thiếu
    là tái tạo đúng lỗi cũ (mọi file đúng bị coi là dở dang)."""
    seen = {}

    def fake_validate(path, expected=0, expect_archive=True, slack=0):
        seen["slack"] = slack
        return fc.Verdict(True, "ok", fmt="gzip")

    monkeypatch.setattr(lf, "_validate_ex", fake_validate)

    f = File(remote="e2/iddpm/train.tar.gz", local="e2/iddpm/train.tar.gz")
    dest = tmp_path / "raw"
    dest.mkdir()
    local = dest / f.local
    local.parent.mkdir(parents=True)
    local.write_bytes(b"x" * 16)

    est = fc.SizeEst(100, 7)
    lf.fetch_one("baidu", f, dest, est, stall_s=0, attempts=1)
    assert seen.get("slack") == 7, "slack bị nuốt ⇒ file đúng vẫn bị tải lại"


# ---------------------------------------------------------------------------
# 4. Baidu exit 0 nhưng 0 byte
# ---------------------------------------------------------------------------
def _mkfile(p: Path, n: int) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\0" * n)
    return p


def test_zero_byte_output_is_retried_and_never_published(tmp_path, monkeypatch):
    calls = []

    def fake_dl(bin_, remote, saveto, stall_s, max_s=0):
        calls.append(remote)
        (saveto / Path(remote).name).write_bytes(b"")   # Baidu "thành công" nhưng 0 B

    monkeypatch.setattr(lf, "_download_with_watchdog", fake_dl)
    f = File(remote="ckpts/lsun_adm.pth", local="ckpts/lsun_adm.pth")
    dest = tmp_path / "raw"
    r = lf.fetch_one("baidu", f, dest, None, stall_s=0, attempts=3)

    assert r["status"] == "error"
    assert "0 byte" in r["error"]
    assert not (dest / f.local).exists(), "file 0 byte đã được publish — đúng lỗi cần tránh"
    assert len(calls) == 3, f"phải thử lại, đang thử {len(calls)} lần"


# ---------------------------------------------------------------------------
# 5. Thành công: chuyển vào đích, ghi sha256 + manifest + cache
# ---------------------------------------------------------------------------
def test_success_moves_file_records_sha_and_manifest(tmp_path, monkeypatch):
    payload = b"DIRE" * 1024

    def fake_dl(bin_, remote, saveto, stall_s, max_s=0):
        (saveto / Path(remote).name).write_bytes(payload)

    monkeypatch.setattr(lf, "_download_with_watchdog", fake_dl)
    f = File(remote="ckpts/small.pth", local="ckpts/small.pth")
    dest = tmp_path / "raw"
    r = lf.fetch_one("baidu", f, dest, None, stall_s=0, attempts=1)

    assert r["status"] == "ok", r
    assert (dest / f.local).read_bytes() == payload
    assert r["sha256"] == fc._sha256(dest / f.local)

    lf._append_manifest(dest, r)
    row = (dest / "_manifest.tsv").read_text(encoding="utf-8").splitlines()[1]
    assert row.split("\t")[1] == "ckpts/small.pth"
    assert row.split("\t")[3] == str(len(payload))

    state = json.loads((dest / ".state" / "validated.json").read_text(encoding="utf-8"))
    assert state["ckpts/small.pth"]["size"] == len(payload)


# ---------------------------------------------------------------------------
# 6. Không xoá byte nào: bản cũ đi vào _held
# ---------------------------------------------------------------------------
def test_existing_valid_file_is_skipped_without_touching_the_disk(tmp_path, monkeypatch):
    payload = b"ok" * 100
    f = File(remote="ckpts/a.pth", local="ckpts/a.pth")
    dest = tmp_path / "raw"
    _mkfile(dest / f.local, len(payload))

    def boom(*a, **k):
        raise AssertionError("file đã hợp lệ mà vẫn gọi BaiduPCS-Go")

    monkeypatch.setattr(lf, "_download_with_watchdog", boom)
    r = lf.fetch_one("baidu", f, dest, fc.SizeEst(len(payload), 0), stall_s=0, attempts=1)
    assert r["status"] == "skipped"


def test_stale_file_is_held_not_deleted(tmp_path, monkeypatch):
    """Đích đã có file HỎNG (ngắn hơn) ⇒ chuyển sang ``_held``, tải bản mới.

    Điều cấm là ``unlink()``: 3.8 h/GB đã trả giá cho những byte đó."""
    new = b"N" * 4096

    def fake_dl(bin_, remote, saveto, stall_s, max_s=0):
        (saveto / Path(remote).name).write_bytes(new)

    monkeypatch.setattr(lf, "_download_with_watchdog", fake_dl)
    f = File(remote="ckpts/b.pth", local="ckpts/b.pth")
    dest = tmp_path / "raw"
    stale = _mkfile(dest / f.local, 10)
    r = lf.fetch_one("baidu", f, dest, fc.SizeEst(len(new), 0), stall_s=0, attempts=1)

    assert r["status"] == "ok", r
    assert (dest / f.local).read_bytes() == new
    held = list((dest / ".staging" / "_held").rglob("b.pth"))
    assert held and held[0].read_bytes() == b"\0" * 10, "bản cũ bị xoá thay vì giữ ở _held"
    assert stale.read_bytes() == new, "đích phải là bản MỚI (bản cũ đã os.replace sang _held)"


def test_held_name_collision_does_not_overwrite(tmp_path):
    d = tmp_path / "h"
    a = _mkfile(tmp_path / "a.pth", 3)
    b = _mkfile(tmp_path / "b" / "a.pth", 5)
    d1 = fc.hold_into(a, d)
    d2 = fc.hold_into(b, d)
    assert d1 != d2
    assert d1.read_bytes() == b"\0" * 3 and d2.read_bytes() == b"\0" * 5


# ---------------------------------------------------------------------------
# 7. Resume: mảnh dở nằm lại staging, KHÔNG bị xoá
# ---------------------------------------------------------------------------
def test_partial_download_keeps_staging_for_resume(tmp_path, monkeypatch):
    def fake_dl(bin_, remote, saveto, stall_s, max_s=0):
        (saveto / Path(remote).name).write_bytes(b"P" * 500)   # mới 1/8

    monkeypatch.setattr(lf, "_download_with_watchdog", fake_dl)
    f = File(remote="e2/big.tar.gz", local="e2/big.tar.gz")
    dest = tmp_path / "raw"
    r = lf.fetch_one("baidu", f, dest, fc.SizeEst(4000, 0), stall_s=0, attempts=2)

    assert r["status"] == "error"
    stage = fc.staging_dir_for(dest / ".staging", f.remote)
    assert (stage / "big.tar.gz").stat().st_size == 500, "mảnh dở bị xoá ⇒ mất tiến độ"
    assert not (dest / f.local).exists()


def test_growth_between_attempts_is_resumed_in_place(tmp_path, monkeypatch):
    """Lần 2 ghi thêm byte ⇒ cùng thư mục staging, không có thư mục mới."""
    sizes = iter([500, 4000])

    def fake_dl(bin_, remote, saveto, stall_s, max_s=0):
        (saveto / Path(remote).name).write_bytes(b"P" * next(sizes))

    monkeypatch.setattr(lf, "_download_with_watchdog", fake_dl)
    f = File(remote="e2/big.pth", local="e2/big.pth")
    dest = tmp_path / "raw"
    r = lf.fetch_one("baidu", f, dest, fc.SizeEst(4000, 0), stall_s=0, attempts=3)

    assert r["status"] == "ok", r
    assert r["attempts"] == 2
    stages = [p for p in (dest / ".staging").iterdir() if p.is_dir() and p.name.startswith("baidu_")]
    assert len(stages) == 1, f"mỗi attempt một thư mục ⇒ resume vô nghĩa: {stages}"


def test_baidu_skipping_a_broken_file_gets_one_fresh_restart(tmp_path, monkeypatch):
    """Baidu ``跳过`` ⇒ 0 byte mới. Phải chuyển bản hỏng đi (không xoá) và tải
    lại từ 0 B đúng MỘT lần; không được đốt hết 5 attempt vào cùng bế tắc."""
    calls = {"n": 0}

    def fake_dl(bin_, remote, saveto, stall_s, max_s=0):
        calls["n"] += 1
        if calls["n"] == 1:                              # lần 1: ghi 1 byte hỏng
            (saveto / Path(remote).name).write_bytes(b"B")
        # từ lần 2: "文件已经存在 … 跳过" — KHÔNG đụng tới file

    monkeypatch.setattr(lf, "_download_with_watchdog", fake_dl)
    f = File(remote="e2/c.pth", local="e2/c.pth")
    dest = tmp_path / "raw"
    r = lf.fetch_one("baidu", f, dest, fc.SizeEst(1000, 0), stall_s=0, attempts=5)

    assert r["status"] == "error"
    held = list((dest / ".staging" / "_held").rglob("c.pth*"))
    assert held, "bản Baidu bỏ qua không được giữ lại ⇒ byte đã tải bị xoá"


# ---------------------------------------------------------------------------
# 8. Watchdog: không ghi byte nào > stall_s ⇒ bỏ, giữ staging
# ---------------------------------------------------------------------------
@pytest.mark.skipif(os.name != "posix", reason="stub BaiduPCS-Go là sh script")
def test_watchdog_gives_up_when_nothing_is_written(tmp_path, monkeypatch):
    fake = tmp_path / "fake_baidu.sh"
    fake.write_text("#!/bin/sh\nsleep 30\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setattr(lf, "POLL_S", 0.05)

    with pytest.raises(TimeoutError) as ei:
        lf._download_with_watchdog(str(fake), "x", tmp_path / "s", stall_s=0.3)
    assert "không ghi thêm byte nào" in str(ei.value)


@pytest.mark.skipif(os.name != "posix", reason="stub BaiduPCS-Go là sh script")
def test_watchdog_returns_when_the_process_finishes(tmp_path, monkeypatch):
    fake = tmp_path / "fake_baidu.sh"
    fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setattr(lf, "POLL_S", 0.05)
    lf._download_with_watchdog(str(fake), "x", tmp_path / "s", stall_s=5)


# ---------------------------------------------------------------------------
# 9. CLI: plan không tải gì; guard chặn khi vượt trần
# ---------------------------------------------------------------------------
def _stub_baidu(monkeypatch, est=None):
    monkeypatch.setattr(lf, "_find_baidu", lambda *_a, **_k: "baidu")
    monkeypatch.setattr(lf, "_preflight", lambda *a, **k: None)
    monkeypatch.setattr(lf, "_ls_estimates", lambda *a, **k: est or {})
    monkeypatch.setattr(lf, "expand", lambda ents, lister=None: (ents, [], []))


def test_plan_mode_downloads_nothing(tmp_path, monkeypatch, capsys):
    _stub_baidu(monkeypatch, {"a.pth": fc.SizeEst(1000, 0)})
    monkeypatch.setattr(lf, "resolve_many",
                        lambda tiers: [File(remote="ckpts/a.pth", local="ckpts/a.pth")])
    monkeypatch.setattr(lf, "_download_with_watchdog",
                        lambda *a, **k: pytest.fail("plan không được tải"))

    rc = lf.main(["--tiers", "ckpts", "--dest", str(tmp_path / "raw")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY-RUN" in out
    assert not (tmp_path / "raw" / "ckpts").exists()


def test_guard_refuses_over_the_byte_budget(tmp_path, monkeypatch):
    _stub_baidu(monkeypatch, {"a.pth": fc.SizeEst(5 * 1024 ** 3, 0)})
    monkeypatch.setattr(lf, "resolve_many",
                        lambda tiers: [File(remote="ckpts/a.pth", local="ckpts/a.pth")])
    with pytest.raises(SystemExit) as ei:
        lf.main(["--tiers", "ckpts", "--dest", str(tmp_path), "--max-gb", "1"])
    assert "GUARD" in str(ei.value)


def test_never_paths_are_blocked_before_any_download(tmp_path, monkeypatch):
    _stub_baidu(monkeypatch)
    monkeypatch.setattr(lf, "_download_with_watchdog",
                        lambda *a, **k: pytest.fail("NEVER mà vẫn tải"))
    blocked = [("recons/whatever.tar.gz", "không cần cho Tier A")]
    monkeypatch.setattr(lf, "resolve_many", lambda tiers: [File("recons/x.tar.gz", "recons/x.tar.gz")])
    monkeypatch.setattr(lf, "expand", lambda ents, lister=None: ([], [], blocked))

    rc = lf.main(["--tiers", "e2", "--dest", str(tmp_path / "raw")])
    assert rc == 0


def test_ctrl_c_keeps_staging_and_exits_130(tmp_path, monkeypatch, capsys):
    _stub_baidu(monkeypatch)
    monkeypatch.setattr(lf, "resolve_many",
                        lambda tiers: [File(remote="ckpts/a.pth", local="ckpts/a.pth")])
    monkeypatch.setattr(lf, "expand", lambda ents, lister=None: (ents[:1], [], []))

    def interrupt(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(lf, "_download_with_watchdog", interrupt)
    rc = lf.main(["--tiers", "ckpts", "--dest", str(tmp_path / "raw"), "--mode", "fetch"])
    assert rc == 130
    assert "Ctrl+C" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 10. Git Bash: "/raw" bị đổi thành "C:/Program Files/Git/raw"
# ---------------------------------------------------------------------------
def test_msys_mangled_dest_is_demangled():
    """Windows + Git Bash: ``--dest /d/dire-raw`` → ``D:/Git/dire-raw``."""
    mangled = "C:/Program Files/Git/d/dire-raw"
    assert fc._demangle(mangled) == "/d/dire-raw"


def test_plain_windows_paths_are_left_alone():
    for p in ("C:/data/raw", "D:/GitLab/raw", "data/raw"):
        assert fc._demangle(p) == p


# ---------------------------------------------------------------------------
# 11. `--mode verify` — đã tải ĐÚNG chưa, không cần tải lại, không cần Baidu
# ---------------------------------------------------------------------------
import gzip as _gzip  # noqa: E402


def _real_gz(path: Path, payload: bytes = b"x" * 50_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_gzip.compress(payload))
    return path


def test_verify_passes_when_everything_is_there_and_valid(tmp_path, monkeypatch, capsys):
    dest = tmp_path / "raw"
    _real_gz(dest / "e2/a/iddpm.tar.gz")
    _stub_baidu(monkeypatch)
    monkeypatch.setattr(lf, "_download_with_watchdog",
                        lambda *a, **k: pytest.fail("verify không được tải gì"))

    rc = lf.main(["--dest", str(dest), "--mode", "verify", "--no-remote", "--paths",
                  "e2/a/iddpm.tar.gz"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ĐỦ và ĐÚNG" in out


def test_verify_reports_missing_and_broken_separately(tmp_path, capsys):
    dest = tmp_path / "raw"
    _real_gz(dest / "e2/a/good.tar.gz")
    body = _gzip.compress(b"y" * 80_000)
    (dest / "e2/b").mkdir(parents=True)
    (dest / "e2/b/bad.tar.gz").write_bytes(body[: len(body) // 2])   # cụt

    rc = lf.main(["--dest", str(dest), "--mode", "verify", "--no-remote", "--paths",
                  "e2/a/good.tar.gz,e2/b/bad.tar.gz,e2/c/absent.tar.gz"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "THIẾU" in out and "absent.tar.gz" in out
    assert "HỎNG" in out and "bad.tar.gz" in out
    assert "✅ đúng : 1/3" in out


def test_verify_works_without_baidu_installed(tmp_path, monkeypatch, capsys):
    """Máy đã có dữ liệu, không có BaiduPCS-Go: vẫn phải kiểm chứng được."""
    dest = tmp_path / "raw"
    _real_gz(dest / "e2/a/iddpm.tar.gz")
    monkeypatch.setattr(lf, "_find_baidu", lambda *_a, **_k: (_ for _ in ()).throw(
        SystemExit("không có BaiduPCS-Go")))

    rc = lf.main(["--dest", str(dest), "--mode", "verify", "--paths", "e2/a/iddpm.tar.gz"])
    assert rc == 0, capsys.readouterr().out


def test_verify_finds_files_downloaded_by_hand_under_another_tree(tmp_path, capsys):
    """Người dùng tải bằng GUI vào cây hơi khác — tìm theo TÊN nếu khớp duy nhất."""
    dest = tmp_path / "raw"
    _real_gz(dest / "tai-bang-gui" / "iddpm.tar.gz")

    rc = lf.main(["--dest", str(dest), "--mode", "verify", "--no-remote", "--paths",
                  "e2/lsun/iddpm.tar.gz"])
    assert rc == 0, capsys.readouterr().out


def test_verify_does_not_guess_when_two_files_share_a_name(tmp_path, capsys):
    """Hai ứng viên cùng tên ⇒ KHÔNG đoán: đoán sai là báo "đủ" cho file sai."""
    dest = tmp_path / "raw"
    _real_gz(dest / "a/iddpm.tar.gz")
    _real_gz(dest / "b/iddpm.tar.gz")

    rc = lf.main(["--dest", str(dest), "--mode", "verify", "--no-remote", "--paths",
                  "e2/lsun/iddpm.tar.gz"])
    assert rc == 1
    assert "THIẾU" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 12. `--mode manifest` — sha256 TSV CÙNG schema với archive::make_manifest
# ---------------------------------------------------------------------------
def test_manifest_matches_archive_schema_and_skips_internal_areas(tmp_path, capsys):
    dest = tmp_path / "raw"
    _real_gz(dest / "e2/a/iddpm.tar.gz")
    (dest / "ckpts").mkdir(parents=True)
    (dest / "ckpts/lsun_adm.pth").write_bytes(b"\0" * 500)          # .pth, có trong manifest
    (dest / ".staging/baidu_x").mkdir(parents=True)
    _real_gz(dest / ".staging/baidu_x/half.tar.gz")                  # nội bộ ⇒ bỏ
    (dest / ".state").mkdir(parents=True)
    (dest / ".state/validated.json").write_text("{}", encoding="utf-8")

    assert lf.main(["--dest", str(dest), "--mode", "manifest"]) == 0
    lines = (dest / "_manifest_sha256.tsv").read_text(encoding="utf-8").splitlines()

    assert lines[0].split("\t") == ["source", "path", "size_bytes", "sha256"]
    paths = [ln.split("\t")[1] for ln in lines[1:]]
    assert paths == ["ckpts/lsun_adm.pth", "e2/a/iddpm.tar.gz"], paths
    row = [ln for ln in lines if "iddpm" in ln][0].split("\t")
    assert row[2] == str((dest / "e2/a/iddpm.tar.gz").stat().st_size)
    assert row[3] == fc._sha256(dest / "e2/a/iddpm.tar.gz")


def test_manifest_reuses_the_hash_cache_instead_of_rereading_gigabytes(tmp_path, monkeypatch):
    """Băm lại 3.39 GB mỗi lần chạy là lý do người ta thôi không kiểm tra nữa."""
    dest = tmp_path / "raw"
    f = _real_gz(dest / "e2/a/iddpm.tar.gz")
    calls = {"n": 0}
    real = fc._sha256

    def counting(p):
        calls["n"] += 1
        return real(p)

    monkeypatch.setattr(lf, "_sha256", counting)
    lf.make_manifest(dest)
    assert calls["n"] == 1
    lf.make_manifest(dest)          # size + mtime_ns không đổi ⇒ dùng cache
    assert calls["n"] == 1, "băm lại file dù (size, mtime_ns) không đổi"

    f.write_bytes(_gzip.compress(b"z" * 1000))   # đổi nội dung ⇒ phải băm lại
    lf.make_manifest(dest)
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# 13. Tài liệu phải dùng ĐÚNG đường dẫn local mà manifest sinh ra
# ---------------------------------------------------------------------------
def test_runbook_paths_under_data_raw_are_real():
    """Mình đã viết sai chỗ này một lần: local KHÔNG có `data/raw/e2/`. Cây local
    giữ đúng đường dẫn manifest (`dire/test/…`, `checkpoints/…`), nên một ví dụ
    `./data/raw/e2` là đường dẫn không tồn tại — copy-paste từ runbook sẽ đẩy
    nhầm thư mục rồi `eval::main` báo "không tìm thấy cặp real/generator nào",
    lỗi trông như thiếu dữ liệu chứ không như sai đường dẫn."""
    from dire_modal.manifest import TIERS
    from dire_modal.manifest import File as _F

    prefixes = {f.local.split("/", 1)[0] for tier in TIERS.values()
                if isinstance(tier, list) for f in tier if isinstance(f, _F)}
    prefixes |= {"dire", "checkpoints"}      # DirSpec mở ra dưới hai gốc này

    text = (Path(__file__).resolve().parents[1] / "docs/runbook-free.md").read_text(encoding="utf-8")
    refs = re.findall(r"\./data/raw(?:/([^\s`'\")]+))?", text)
    assert refs, "runbook không nhắc đường dẫn local nào — test đang kiểm nhầm chỗ?"
    bad = [r for r in refs if r and r.split("/")[0] not in prefixes]
    assert not bad, f"runbook nhắc `data/raw/{bad[0]}` — không phải cây local thật"


# ---------------------------------------------------------------------------
# 14. `verify` tại máy cũng phải đối chiếu sha256 cho `.pth`
# ---------------------------------------------------------------------------
def test_local_verify_uses_manifest_sha_for_checkpoints(tmp_path, capsys):
    """Đường local không có `gzip -t` cho `.pth`; manifest là bằng chứng duy nhất.

    Cùng `fetchcore.check_artifact` với `download::verify` — hai phía không được
    trả lời khác nhau cho cùng một câu hỏi (bài học 2026-10-25).
    """
    import hashlib as _hashlib

    dest = tmp_path / "raw"
    ck = dest / "checkpoints/lsun_adm.pth"
    ck.parent.mkdir(parents=True)
    body = b"\0" * 4096
    ck.write_bytes(body)
    (dest / "_manifest_sha256.tsv").write_text(
        "source\tpath\tsize_bytes\tsha256\n"
        f"raw\tcheckpoints/lsun_adm.pth\t{len(body)}\t{_hashlib.sha256(body).hexdigest()}\n",
        encoding="utf-8")

    rc = lf.main(["--dest", str(dest), "--mode", "verify", "--no-remote",
                  "--paths", "checkpoints/lsun_adm.pth"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "đối chiếu sha256" in out, out

    # Cùng size, khác byte ⇒ chỉ sha bắt được.
    ck.write_bytes(b"\1" * len(body))
    import os as _os
    st = ck.stat()
    _os.utime(ck, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    rc = lf.main(["--dest", str(dest), "--mode", "verify", "--no-remote",
                  "--paths", "checkpoints/lsun_adm.pth"])
    out = capsys.readouterr().out
    assert rc == 1, out
    assert "sha256" in out and "nội dung khác" in out, out
