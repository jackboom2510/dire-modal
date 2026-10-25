"""`run.json` + CSV — "một lần chạy phải tự giải thích được" (FR-7 của SPEC-001).

Vì sao có file này
------------------
Một bảng AP không kèm gì là **vô dụng sau ba tuần**: không biết nó đo trên nhánh
nào, tiền xử lý nào, checkpoint nào. Không ai tái lập được, và tệ hơn: hai lần
chạy khác nhau sẽ bị so với nhau như thể cùng một thứ.

Test này khoá ba thứ:
  1. ``run.json`` có ĐỦ khoá mà spec §10.2 đòi — thiếu là báo lỗi ngay.
  2. ``git_sha`` không bao giờ bị bịa: env thiếu thì ghi ``"unknown"``.
  3. ``n_rows`` trong ``run.json`` khớp SỐ DÒNG của CSV — nếu lệch nghĩa là
     một bên ghi hỏng, và mọi con số trích từ đó đều đáng ngờ.

Chạy::

    python3 -m pytest tests/test_run_json.py -q
"""

import csv
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _ev():
    mod = types.ModuleType("modal")

    class _Vol:
        def commit(self, *a, **k):
            pass

    class _Fn:
        def __init__(self, fn, kw):
            self._fn, self._kw, self.__name__ = fn, kw, fn.__name__

        def __call__(self, *a, **k):
            raise TypeError("'Function' object is not callable")

        def map(self, items, **k):
            return [self._fn(x) for x in items]

        def starmap(self, items, **k):
            return [self._fn(*x) for x in items]

        def remote(self, *a, **k):
            return self._fn(*a, **k)

        def local(self, *a, **k):
            return self._fn(*a, **k)

        def with_options(self, **k):
            return _Fn(self._fn, {**self._kw, **k})

    class _App:
        def __init__(self, *a, **k):
            pass

        def function(self, *a, **k):
            return lambda fn: _Fn(fn, k)

        def local_entrypoint(self, *a, **k):
            return lambda fn: _Fn(fn, {"local": True})

    class _Img:
        def apt_install(self, *a, **k):
            return self

        def pip_install(self, *a, **k):
            return self

        def run_commands(self, *a, **k):
            return self

    mod.App = _App
    mod.Image = types.SimpleNamespace(debian_slim=lambda **k: _Img())
    mod.Volume = types.SimpleNamespace(from_name=lambda *a, **k: _Vol())
    mod.Secret = types.SimpleNamespace(from_name=lambda *a, **k: object())
    mod.Retries = lambda **k: types.SimpleNamespace(**k)
    sys.modules["modal"] = mod

    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from dire_modal import eval as ev  # noqa: PLC0415

    return ev


# Khoá bắt buộc, lấy từ spec §10.2. Thiếu một cái là không tái lập được.
REQUIRED = {
    "spec", "experiment_id", "detector", "git_sha", "created_at",
    "preproc", "tiers", "n_rows", "n_images", "versions",
}

ROW = {
    "experiment_id": "exp1", "detector": "dire", "branch": "lsun_bedroom__lsun_bedroom",
    "generator": "adm", "split": "test", "transform": "resize256_centercrop224_normalize",
    "n_real": 1000, "n_fake": 1000, "acc": 0.995, "ap": 0.998,
    "tpr_at_5fpr": 0.98, "threshold": 0.5, "git_sha": "abc1234", "created_at": "2026-10-25T00:00:00+00:00",
}


def test_run_json_has_all_spec_keys():
    import tempfile

    ev = _ev()
    path = Path(tempfile.mkdtemp()) / "run.json"
    doc = ev.run_json(
        path, experiment_id="exp1", tiers="e2,e4", n_images={"real_total": 1000},
        rows=1, sysinfo={"torch": "2.4.0"},
    )
    missing = REQUIRED - set(doc)
    assert not missing, f"run.json thiếu khoá: {sorted(missing)}"
    assert doc["spec"] == "SPEC-001"
    assert doc["detector"] == "dire"
    # Ghi ra đĩa phải đọc lại được và y hệt.
    assert json.loads(path.read_text(encoding="utf-8")) == doc


def test_git_sha_is_never_invented():
    import tempfile

    ev = _ev()
    old = ev.os.environ.pop("GIT_SHA", None)
    try:
        path = Path(tempfile.mkdtemp()) / "run.json"
        doc = ev.run_json(path, experiment_id="x", tiers="e2", n_images={}, rows=0)
        assert doc["git_sha"] == "unknown", "thiếu env thì phải là 'unknown', không bịa sha"
    finally:
        if old is not None:
            ev.os.environ["GIT_SHA"] = old

    ev.os.environ["GIT_SHA"] = "deadbeef"
    try:
        doc = ev.run_json(Path(tempfile.mkdtemp()) / "run.json",
                          experiment_id="x", tiers="e2", n_images={}, rows=0)
        assert doc["git_sha"] == "deadbeef"
    finally:
        ev.os.environ.pop("GIT_SHA", None)


def test_tiers_are_split_into_a_list():
    import tempfile

    ev = _ev()
    doc = ev.run_json(Path(tempfile.mkdtemp()) / "run.json", experiment_id="x",
                      tiers=" e2 , e4 ", n_images={}, rows=0)
    assert doc["tiers"] == ["e2", "e4"], doc["tiers"]


def test_csv_header_matches_spec_schema():
    import tempfile

    ev = _ev()
    path = Path(tempfile.mkdtemp()) / "dire.csv"
    ev.write_csv([ROW], path)
    with path.open(encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert header == ev.CSV_FIELDS, header
    # Các cột spec §10.1 đòi, phải có mặt.
    for col in ("experiment_id", "detector", "generator", "split", "transform",
                "n_real", "n_fake", "acc", "ap", "tpr_at_5fpr", "threshold"):
        assert col in header, f"CSV thiếu cột {col}"


def test_csv_does_not_invent_missing_metrics():
    """Thiếu metric thì để TRỐNG, không điền 0 (NFR-6)."""
    import tempfile

    ev = _ev()
    path = Path(tempfile.mkdtemp()) / "dire.csv"
    ev.write_csv([{"experiment_id": "e", "generator": "adm", "n_real": 10}], path)
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["ap"] == "", "ap chưa đo phải để trống, không được là 0"
    assert rows[0]["acc"] == ""


def test_csv_row_count_matches_run_json():
    """`n_rows` phải khớp số dòng CSV thật — lệch nghĩa là một bên ghi hỏng."""
    import tempfile

    ev = _ev()
    d = Path(tempfile.mkdtemp())
    rows = [dict(ROW, generator=g) for g in ("adm", "ddpm", "sdv1")]
    ev.write_csv(rows, d / "dire.csv")
    ev.run_json(d / "run.json", experiment_id="exp1", tiers="e2", n_images={},
                rows=len(rows))
    with (d / "dire.csv").open(encoding="utf-8") as fh:
        n_csv = sum(1 for _ in csv.DictReader(fh))
    n_json = json.loads((d / "run.json").read_text(encoding="utf-8"))["n_rows"]
    assert n_csv == n_json == 3, (n_csv, n_json)


def test_every_csv_row_carries_git_sha():
    """Mỗi dòng tự mang `git_sha` — CSV bị cắt rời khỏi run.json vẫn truy được."""
    import tempfile

    ev = _ev()
    path = Path(tempfile.mkdtemp()) / "dire.csv"
    ev.write_csv([ROW], path)
    with path.open(encoding="utf-8") as fh:
        assert next(csv.DictReader(fh))["git_sha"] == "abc1234"


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
