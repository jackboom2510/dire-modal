"""Test cho đường **máy local → Volume Modal** (`dire_modal.upload`).

Ba lỗi mà bộ test này canh, đều là lỗi "trông như thành công":

1. `modal volume put` nhận đường dẫn **tính từ gốc Volume** (`dire/test/…`), còn
   container thấy `/raw/dire/test/…`. Đưa `/raw/...` vào CLI ⇒ dữ liệu nằm ở
   `/raw/raw/...` ⇒ `eval::main` báo "không tìm thấy cặp real/generator nào".
2. `modal volume put <dir> <dest>` đặt thư mục **vào trong** đích ⇒
   `/raw/dire/dire/...`. Vì vậy module này đẩy từng file.
3. Xoá `.staging` trên Volume **trước khi** dữ liệu nằm trên Volume ⇒ xoá bản
   duy nhất còn lại trên cloud (máy có thể đã hỏng / đã dọn).

Cách ly module: `dire_modal.upload` chỉ được import DƯỚI stub `modal`, và mọi
`dire_modal.*` được khôi phục nguyên trạng sau khi module test này chạy xong.
Không làm vậy thì `dire_modal.common` (cache `app`/`vol_raw` ngay lần import đầu)
sẽ rò stub sang test khác — đúng lỗi đã xảy ra một lần trong repo này.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Stub Modal — cùng hợp đồng với tests/test_download_modal.py
# ---------------------------------------------------------------------------
class FakeFunction:
    """Giống ``modal.Function``: ``.local/.remote/.map`` nhưng KHÔNG gọi được."""

    def __init__(self, fn, kw):
        self._fn = fn
        self._kw = kw
        self.__name__ = getattr(fn, "__name__", "fn")

    def __call__(self, *a, **k):
        raise TypeError("'Function' object is not callable")

    def remote(self, *a, **k):
        return self._fn(*a, **k)

    def local(self, *a, **k):
        return self._fn(*a, **k)


class _Img:
    def apt_install(self, *a, **k):
        return self

    def pip_install(self, *a, **k):
        return self

    def run_commands(self, *a, **k):
        return self


class _Vol:
    def __init__(self):
        self.commits = 0

    def commit(self, *a, **k):
        self.commits += 1


class _App:
    def __init__(self, *a, **k):
        pass

    def function(self, *a, **k):
        return lambda fn: FakeFunction(fn, k)

    def local_entrypoint(self, *a, **k):
        return lambda fn: FakeFunction(fn, {"local": True})


def _stub_modal() -> types.ModuleType:
    mod = types.ModuleType("modal")
    mod.App = _App
    mod.Image = types.SimpleNamespace(debian_slim=lambda **k: _Img())
    mod.Volume = types.SimpleNamespace(from_name=lambda *a, **k: _Vol())
    mod.Secret = types.SimpleNamespace(from_name=lambda *a, **k: object())
    mod.Retries = lambda **k: types.SimpleNamespace(**k)
    return mod


def _owned() -> list[str]:
    return [k for k in list(sys.modules)
            if k == "modal" or k == "dire_modal" or k.startswith("dire_modal.")]


#: Module dưới stub, do fixture module-scope gán vào.
up = None


@pytest.fixture(scope="module", autouse=True)
def _upload_under_stub():
    """Import `dire_modal.upload` với `modal` giả, rồi TRẢ sys.modules về cũ.

    Phải làm ở đây (không phải lúc import file test) vì pytest collect mọi file
    trước khi chạy: cài stub lúc import sẽ để `dire_modal.common` mang `app` của
    stub này cho tất cả module khác (chúng cần `app.cls`, `modal.parameter`…).
    """
    global up
    saved = {k: sys.modules[k] for k in _owned()}
    try:
        for k in _owned():
            del sys.modules[k]
        sys.modules["modal"] = _stub_modal()
        import dire_modal.upload as _u        # noqa: PLC0415

        up = _u
        yield
    finally:
        for k in _owned():
            del sys.modules[k]
        sys.modules.update(saved)
        up = None


@pytest.fixture()
def up_mod(monkeypatch):
    """Module + Volume giả lập được điều khiển từ test.

    Tier được stub bằng ``state["tier"]`` để test nói về HÀNH VI, không phải về
    việc manifest hôm nay có mấy file; `test_tier_contents_come_from_the_real_manifest`
    canh chỗ nối đó.
    """
    u = up
    state = {"files": {}, "internal": {}, "cleaned": None,
             "tier": ["checkpoints/lsun_adm.pth"]}
    monkeypatch.setattr(u, "resolve_many",
                        lambda _t: [u.File(remote=r, local=r) for r in state["tier"]])
    monkeypatch.setattr(u, "_remote_index",
                        lambda: {"files": dict(state["files"]),
                                 "internal": dict(state["internal"])})

    def fake_clean(areas, yes):
        state["cleaned"] = (areas, yes)
        return {"dry": not yes, "files": 3, "bytes": 999,
                "areas": [{"area": ".staging", "bytes": 999, "files": 3}]}

    monkeypatch.setattr(u, "_clean", fake_clean)
    return u, state


def _mk(root: Path, rel: str, n: int) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\0" * n)
    return p


# ---------------------------------------------------------------------------
# 1. Đường dẫn: tính từ GỐC VOLUME, không phải "/raw/..."
# ---------------------------------------------------------------------------
def test_put_target_is_relative_to_volume_root(up_mod):
    """Volume mount ở /raw; CLI thì không biết `/raw`. `/raw/...` ⇒ `/raw/raw/...`."""
    u, _ = up_mod
    assert u.remote_path("dire/test/x.tar.gz") == "/dire/test/x.tar.gz"
    assert u.remote_path("/dire/test/x.tar.gz") == "/dire/test/x.tar.gz"
    assert not u.remote_path("dire/test/x.tar.gz").startswith("/raw")


def test_put_command_shape(up_mod, tmp_path):
    u, _ = up_mod
    cmd = u.put_command(tmp_path / "x.tar.gz", "dire/test/x.tar.gz", volume="dire-raw")
    assert cmd[:5] == ["modal", "volume", "put", "dire-raw", str(tmp_path / "x.tar.gz")]
    assert cmd[5] == "/dire/test/x.tar.gz"
    assert "--force" not in cmd
    assert u.put_command(tmp_path / "x", "a/b", force=True)[-1] == "--force"


def test_force_flag_is_only_used_when_the_cli_supports_it(up_mod, monkeypatch):
    """Bản `modal` cũ không có `--force`; truyền bừa ⇒ CLI exit 2 và người dùng
    phải tự đoán lỗi. Hỏi `--help` rồi nhớ."""
    u, _ = up_mod

    class R:
        def __init__(self, text):
            self.stdout, self.stderr, self.returncode = text, "", 0

    monkeypatch.setattr(u.subprocess, "run", lambda cmd, **k: R("--force  ghi đè"))
    assert u.put_supports_force() is True

    monkeypatch.setattr(u.subprocess, "run", lambda cmd, **k: R("usage: modal volume put"))
    assert u.put_supports_force() is False

    def boom(*a, **k):
        raise OSError("không có modal")

    monkeypatch.setattr(u.subprocess, "run", boom)
    assert u.put_supports_force() is False, "không hỏi được thì phải coi như KHÔNG có"


# ---------------------------------------------------------------------------
# 2. index_tree / plan_upload: so size tuyệt đối, tách vùng nội bộ
# ---------------------------------------------------------------------------
def test_index_tree_separates_internal_areas(up_mod, tmp_path):
    u, _ = up_mod
    _mk(tmp_path, "dire/test/a/real.tar.gz", 100)
    _mk(tmp_path, ".staging/baidu_abc/real.tar.gz", 40)       # mảnh dở
    _mk(tmp_path, ".staging/_held/baidu_abc/real.tar.gz", 7)  # giữ lại
    _mk(tmp_path, ".bench/k1/x.tar.gz", 5)

    idx = u.index_tree(tmp_path)
    assert idx["files"] == {"dire/test/a/real.tar.gz": 100}
    assert idx["internal"][".staging"] == 47
    assert idx["internal"][".staging/_held"] == 7
    assert idx["internal"][".bench"] == 5


def test_plan_upload_classifies_missing_equal_and_size_mismatch(up_mod, tmp_path):
    u, _ = up_mod
    files = [u.File(r, r) for r in ("dire/a.tar.gz", "dire/b.tar.gz",
                                    "dire/c.tar.gz", "dire/d.tar.gz")]
    _mk(tmp_path, "dire/a.tar.gz", 10)      # chưa có trên Volume
    _mk(tmp_path, "dire/b.tar.gz", 20)      # khớp
    _mk(tmp_path, "dire/c.tar.gz", 30)      # Volume có size khác
    # d: thiếu ở local

    todo, ok, absent = u.plan_upload(
        files, tmp_path, {"dire/b.tar.gz": 20, "dire/c.tar.gz": 15})
    assert [f.local for f, *_ in todo] == ["dire/a.tar.gz", "dire/c.tar.gz"]
    assert todo[1][3].startswith("Volume có")
    assert [f.local for f, *_ in ok] == ["dire/b.tar.gz"]
    assert [f.local for f in absent] == ["dire/d.tar.gz"]


def test_plan_finds_files_downloaded_by_hand(up_mod, tmp_path):
    """Tải bằng GUI vào cây khác ⇒ vẫn đẩy được (khớp tên DUY NHẤT)."""
    u, _ = up_mod
    _mk(tmp_path, "tai-bang-gui/iddpm.tar.gz", 5)
    todo, _ok, absent = u.plan_upload(
        [u.File("dire/test/x/iddpm.tar.gz", "dire/test/x/iddpm.tar.gz")], tmp_path, {})
    assert [f.local for f, *_ in todo] == ["dire/test/x/iddpm.tar.gz"]
    assert not absent


# ---------------------------------------------------------------------------
# 3. Guard trước khi xoá `.staging`
# ---------------------------------------------------------------------------
def test_check_ready_lists_files_that_must_exist_on_volume_first(up_mod):
    u, _ = up_mod
    files = [u.File("dire/a.tar.gz", "dire/a.tar.gz"),
             u.File("dire/b.tar.gz", "dire/b.tar.gz")]
    assert u.check_ready(files, {"dire/a.tar.gz": 1}) == ["thiếu: dire/b.tar.gz"]


def test_clean_refuses_to_delete_staging_before_data_is_on_volume(up_mod, tmp_path, capsys):
    """Guard quan trọng nhất: xoá sai là mất bản duy nhất còn lại trên cloud."""
    u, state = up_mod
    state["files"] = {}          # Volume TRỐNG

    with pytest.raises(SystemExit) as ei:
        u.main.local(mode="clean", tiers="t", dest=str(tmp_path), yes=True)
    assert state["cleaned"] is None, "đã xoá dù dữ liệu chưa nằm trên Volume"
    assert "xoá bản duy nhất" in str(ei.value)


def test_clean_proceeds_when_every_file_is_already_on_volume(up_mod, tmp_path, capsys):
    u, state = up_mod
    state["files"] = {"checkpoints/lsun_adm.pth": 40}
    rc = u.main.local(mode="clean", tiers="t", dest=str(tmp_path), yes=True)
    assert rc == 0
    assert state["cleaned"] == (".staging", True)
    assert "DỌN VÙNG NỘI BỘ" in capsys.readouterr().out


def test_clean_is_dry_run_without_yes(up_mod, tmp_path):
    u, state = up_mod
    state["files"] = {"checkpoints/lsun_adm.pth": 40}
    u.main.local(mode="clean", tiers="t", dest=str(tmp_path))
    assert state["cleaned"] == (".staging", False), "mặc định phải là dry-run"


def test_clean_needs_to_know_what_must_survive(up_mod, tmp_path):
    u, _ = up_mod
    with pytest.raises(SystemExit) as ei:
        u.main.local(mode="clean", dest=str(tmp_path), yes=True)
    assert "--tiers" in str(ei.value)


def test_skip_check_is_explicit(up_mod, tmp_path):
    u, state = up_mod
    rc = u.main.local(mode="clean", dest=str(tmp_path), yes=True, skip_check=True)
    assert rc == 0 and state["cleaned"] == (".staging", True)


# ---------------------------------------------------------------------------
# 4. clean_tree: chỉ đụng đúng tên vùng nội bộ
# ---------------------------------------------------------------------------
def test_clean_tree_removes_only_internal_areas(up_mod, tmp_path):
    u, _ = up_mod
    _mk(tmp_path, ".staging/baidu_a/x.tar.gz", 30)
    _mk(tmp_path, ".bench/k/x", 7)
    _mk(tmp_path, "dire/test/a/real.tar.gz", 100)

    rep = u.clean_tree(tmp_path, (".staging", ".bench"), dry=False)
    assert not (tmp_path / ".staging").exists()
    assert not (tmp_path / ".bench").exists()
    assert (tmp_path / "dire/test/a/real.tar.gz").exists(), "xoá nhầm dữ liệu thật"
    assert rep["files"] >= 1


def test_clean_tree_refuses_anything_that_is_not_internal(up_mod, tmp_path):
    u, _ = up_mod
    _mk(tmp_path, "dire/test/a/real.tar.gz", 100)
    with pytest.raises(ValueError) as ei:
        u.clean_tree(tmp_path, ("dire",), dry=False)
    assert "không phải vùng nội bộ" in str(ei.value)
    assert (tmp_path / "dire/test/a/real.tar.gz").exists()


def test_clean_tree_dry_run_touches_nothing(up_mod, tmp_path):
    u, _ = up_mod
    _mk(tmp_path, ".staging/x", 3)
    rep = u.clean_tree(tmp_path, (".staging",), dry=True)
    assert (tmp_path / ".staging/x").exists()
    assert rep["dry"] is True


# ---------------------------------------------------------------------------
# 5. push: đẩy đúng file, đúng đích, không dọn khi Volume còn lệch
# ---------------------------------------------------------------------------
def test_push_uploads_only_what_is_missing_and_verifies_after(up_mod, tmp_path,
                                                              monkeypatch, capsys):
    u, state = up_mod
    _mk(tmp_path, "checkpoints/lsun_adm.pth", 40)
    index = {"files": {}}
    seen = []

    monkeypatch.setattr(u, "_remote_index",
                        lambda: {"files": dict(index["files"]), "internal": {".staging": 500}})

    def fake_run(cmd, **k):
        seen.append(cmd)
        if cmd[:3] == ["modal", "volume", "put"]:
            index["files"]["checkpoints/lsun_adm.pth"] = Path(cmd[4]).stat().st_size

        class R:
            returncode = 0
            stdout = stderr = ""

        return R()

    monkeypatch.setattr(u.subprocess, "run", fake_run)
    monkeypatch.setattr(u, "put_supports_force", lambda *a, **k: False)

    rc = u.main.local(mode="push", tiers="t", dest=str(tmp_path))
    assert rc == 0, capsys.readouterr().out
    puts = [c for c in seen if c[:3] == ["modal", "volume", "put"]]
    assert len(puts) == 1
    assert puts[0][5] == "/checkpoints/lsun_adm.pth", puts[0]
    assert state["cleaned"] is None, "không được dọn staging khi chưa được yêu cầu"


def test_push_never_cleans_when_volume_still_disagrees(up_mod, tmp_path,
                                                       monkeypatch, capsys):
    u, state = up_mod
    _mk(tmp_path, "checkpoints/lsun_adm.pth", 40)

    class R:
        returncode = 0
        stdout = stderr = ""

    monkeypatch.setattr(u.subprocess, "run", lambda cmd, **k: R())   # Volume KHÔNG nhận
    monkeypatch.setattr(u, "put_supports_force", lambda *a, **k: False)

    rc = u.main.local(mode="push", tiers="t", dest=str(tmp_path), clean_after=True, yes=True)
    assert rc == 1
    assert state["cleaned"] is None, "đã xoá staging dù Volume vẫn lệch local"
    assert "KHÔNG dọn staging" in capsys.readouterr().out


def test_push_refuses_when_local_is_incomplete(up_mod, tmp_path, capsys):
    u, state = up_mod
    state["tier"] = ["checkpoints/lsun_adm.pth", "checkpoints/imagenet_adm.pth"]
    _mk(tmp_path, "checkpoints/lsun_adm.pth", 40)      # thiếu file thứ hai
    rc = u.main.local(mode="push", tiers="t", dest=str(tmp_path))
    assert rc == 1
    assert "chưa có ở local" in capsys.readouterr().out


def test_plan_mode_changes_nothing(up_mod, tmp_path, monkeypatch, capsys):
    u, _ = up_mod
    _mk(tmp_path, "checkpoints/lsun_adm.pth", 40)
    monkeypatch.setattr(u.subprocess, "run",
                        lambda *a, **k: pytest.fail("plan không được gọi CLI"))
    rc = u.main.local(mode="plan", tiers="t", dest=str(tmp_path))
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY-RUN" in out and "KẾ HOẠCH ĐẨY LÊN VOLUME" in out


def test_clean_after_push_happens_only_after_verification(up_mod, tmp_path, monkeypatch):
    u, state = up_mod
    _mk(tmp_path, "checkpoints/lsun_adm.pth", 40)
    index = {"files": {}}
    monkeypatch.setattr(u, "_remote_index",
                        lambda: {"files": dict(index["files"]), "internal": {".staging": 500}})

    class R:
        returncode = 0
        stdout = stderr = ""

    def fake_run(cmd, **k):
        if cmd[:3] == ["modal", "volume", "put"]:
            index["files"]["checkpoints/lsun_adm.pth"] = Path(cmd[4]).stat().st_size
        return R()

    monkeypatch.setattr(u.subprocess, "run", fake_run)
    monkeypatch.setattr(u, "put_supports_force", lambda *a, **k: False)

    rc = u.main.local(mode="push", tiers="t", dest=str(tmp_path), clean_after=True, yes=True)
    assert rc == 0
    assert state["cleaned"] == (".staging", True)


# ---------------------------------------------------------------------------
# 6. Không đụng tới đường tải cũ
# ---------------------------------------------------------------------------
def test_tier_contents_come_from_the_real_manifest():
    """Chỗ nối với manifest thật: đường dẫn phải ở gốc ``dire/`` hoặc
    ``checkpoints/`` — nếu không, `remote_path()` đẩy vào sai gốc Volume."""
    from dire_modal.manifest import expand, resolve_many

    files, _unresolved, _blocked = expand(resolve_many("e2min"))
    assert files, "tier e2min rỗng?"
    for f in files:
        assert f.local.split("/")[0] in {"dire", "checkpoints"}, f.local


def test_upload_does_not_reimplement_download_validation():
    """Phép kiểm "file đã đúng chưa" phải là của `fetchcore`, không bản sao."""
    src = (ROOT / "src/dire_modal/upload.py").read_text(encoding="utf-8")
    for banned in ("gzip -t", "def _validate_ex", "def _sha256"):
        assert banned not in src, f"upload.py tự viết lại phép kiểm ({banned})"
    assert "from .fetchcore import" in src
