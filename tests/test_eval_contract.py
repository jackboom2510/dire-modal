"""Hợp đồng của `eval.py` — phần kiểm được mà KHÔNG cần GPU, torch hay ảnh thật.

Vì sao có file này
------------------
`eval.py` là chỗ dễ sai nhất trong repo: nó biến ảnh thành con số, và một con số
sai vẫn "trông hợp lý". Ba lớp lỗi từng xảy ra thật, mỗi lớp cần một kiểu test:

1. **Sai metric.** AP của `sklearn` gộp score TRÙNG nhau vào cùng một ngưỡng;
   tự viết lại mà không khớp chỗ đó thì AP lệch âm thầm. Test so với giá trị
   `average_precision_score` đã biết.
2. **Gộp nhánh.** `dire/test/lsun_bedroom/imagenet/` dùng CHUNG 1000 ảnh real với
   `.../lsun_bedroom/` (chốt bằng `probe_pair` 2026-10-21). Gộp hai nhánh là
   đếm trùng và kéo mean AP về phía LSUN — không có exception nào báo.
3. **Một lớp rỗng.** Nếu `n_fake == 0` (đường dẫn/nhãn sai) thì AP vẫn ra một số.
   Phải đếm và phơi ra.

Chạy::

    python3 -m pytest tests/test_eval_contract.py -q
"""

import sys
import tarfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _install_stub():
    """Stub Modal: decorator trả OBJECT, không trả hàm — giống Modal thật.

    Trả hàm thì mọi lỗi "gọi Function như hàm thường" biến mất khỏi test suite.
    """
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


# ---------------------------------------------------------------------------
# 1. Metric phải khớp sklearn trên những ca đã biết đáp án
# ---------------------------------------------------------------------------
def test_metrics_perfect_separation():
    ev = _install_stub()
    m = ev.metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["acc"] == 1.0
    assert m["ap"] == 1.0
    assert m["tpr_at_5fpr"] == 1.0, "tách hoàn hảo ⇒ TPR@5%FPR phải bằng 1"
    assert m["n_real"] == 2 and m["n_fake"] == 2


def test_metrics_inverted_scores_are_the_worst_case():
    """Đảo nhãn ⇒ AP thấp. Đây là test bắt lỗi ĐẢO DẤU điểm (AC-3.2)."""
    ev = _install_stub()
    m = ev.metrics([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
    assert m["ap"] < 0.6, m["ap"]
    assert m["acc"] == 0.0


def test_average_precision_matches_sklearn_value():
    """Giá trị đã đối chiếu `sklearn.metrics.average_precision_score`.

    y_true = [0,0,1,1], score = [0.1,0.9,0.8,0.4]
    sklearn: các ngưỡng 0.9/0.8/0.4/0.1 → AP = 0.5 * (1/2) + … = 0.583333…
    """
    ev = _install_stub()
    got = ev._average_precision([0, 0, 1, 1], [0.1, 0.9, 0.8, 0.4])
    assert abs(got - 0.5833333) < 1e-6, got


def test_average_precision_groups_tied_scores():
    """Score bằng nhau phải gộp thành MỘT ngưỡng, như sklearn.

    Nếu cộng dồn từng phần tử, AP của ca này ra 1.0 (sai) thay vì 0.5.
    """
    ev = _install_stub()
    assert ev._average_precision([0, 1], [0.5, 0.5]) == 0.5
    assert ev._average_precision([0, 0, 1, 1], [0.5] * 4) == 0.5


def test_tpr_at_fpr_takes_max_not_last_point():
    """Phải lấy TPR LỚN NHẤT trong vùng FPR ≤ 5%, không phải điểm cuối.

    20 real + 20 fake. Đúng 1 real bị chấm cao nhất ⇒ FPR = 1/20 = 5% (đúng biên,
    phải được TÍNH). Sau đó 10 fake lọt vào ⇒ TPR = 10/20 = 0.5. Điểm cuối của
    ROC là (1.0, 1.0) — nếu code trả 1.0 thì nó đang đọc sai điểm.
    """
    ev = _install_stub()
    y = [0] * 20 + [1] * 20
    # score theo ĐÚNG nhãn: 1 real cao nhất, rồi 10 fake, rồi 19 real, rồi 10 fake
    s = [0.99] + [0.1] * 19 + [0.9] * 10 + [0.05] * 10
    tpr = ev._tpr_at_fpr(y, s)
    assert tpr == 0.5, f"TPR@5%FPR = {tpr}, cần 0.5 (biên 1/20 = 5% phải tính)"
    assert ev._tpr_at_fpr(y, s) != 1.0, "không được lấy điểm cuối của ROC"


def test_tpr_at_fpr_excludes_points_beyond_budget():
    """1 real bị chấm cao nhất trong 10 real ⇒ FPR = 10% > 5% ⇒ TPR phải là 0."""
    ev = _install_stub()
    y = [0] * 10 + [1] * 10
    s = [0.99] + [0.1] * 9 + [0.9] * 10
    assert ev._tpr_at_fpr(y, s) == 0.0


# ---------------------------------------------------------------------------
# 2. Một lớp rỗng phải NHÌN THẤY ĐƯỢC (AC-1.2)
# ---------------------------------------------------------------------------
def test_empty_fake_class_is_reported_not_silent():
    ev = _install_stub()
    m = ev.metrics([0, 0, 0], [0.1, 0.2, 0.3])
    assert m["n_fake"] == 0, "phải đếm được 0 ảnh giả"
    assert m["ap"] == 0.0
    assert m["tpr_at_5fpr"] == 0.0


def test_metrics_on_empty_input_does_not_divide_by_zero():
    ev = _install_stub()
    m = ev.metrics([], [])
    assert m["n_real"] == 0 and m["n_fake"] == 0
    assert m["acc"] == 0.0


# ---------------------------------------------------------------------------
# 3. Nhãn
# ---------------------------------------------------------------------------
def test_label_of_prefers_member_structure():
    ev = _install_stub()
    assert ev.label_of("0_real/0001.png", "adm.tar.gz") == 0
    assert ev.label_of("1_fake/0001.png", "real.tar.gz") == 1
    # Không có cấu trúc bên trong ⇒ đoán theo tên archive.
    assert ev.label_of("0001.jpg", "real.tar.gz") == 0
    assert ev.label_of("0001.jpg", "adm.tar.gz") == 1


# ---------------------------------------------------------------------------
# 4. Nhóm cặp: KHÔNG được gộp nhánh (lỗi đếm trùng 1000 ảnh real)
# ---------------------------------------------------------------------------
def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tf:
        data = b"x"
        info = tarfile.TarInfo("0_real/a.jpg")
        info.size = len(data)
        import io

        tf.addfile(info, io.BytesIO(data))


def test_pairs_keep_branches_separate():
    import tempfile

    ev = _install_stub()
    root = Path(tempfile.mkdtemp())
    # lsun: hai tầng; celebahq: một tầng — đúng cấu trúc thật của dataset.
    for name in ("real", "adm", "ddpm"):
        _touch(root / "dire/test/lsun_bedroom/lsun_bedroom" / f"{name}.tar.gz")
        _touch(root / "dire/test/lsun_bedroom/imagenet" / f"{name}.tar.gz")
    for name in ("real", "stargan"):
        _touch(root / "dire/test/celebahq" / f"{name}.tar.gz")

    pairs = ev.pairs_from_raw(root, "test")
    branches = {p["branch"] for p in pairs}
    assert branches == {"lsun_bedroom__lsun_bedroom", "lsun_bedroom__imagenet", "celebahq"}, branches
    # 2 fake ở nhánh lsun ×2 nhánh + 1 fake ở celebahq.
    assert len(pairs) == 5, [(p["branch"], p["generator"]) for p in pairs]
    assert all(p["real"].name == "real.tar.gz" for p in pairs)
    assert all(p["fake"].name != "real.tar.gz" for p in pairs)


def test_pairs_skip_group_without_real():
    """Nhóm không có `real.tar.gz` thì bỏ — không tự bịa nhãn real."""
    import tempfile

    ev = _install_stub()
    root = Path(tempfile.mkdtemp())
    _touch(root / "dire/test/x" / "adm.tar.gz")
    assert ev.pairs_from_raw(root, "test") == []


def test_pairs_are_sorted_deterministically():
    import tempfile

    ev = _install_stub()
    root = Path(tempfile.mkdtemp())
    for name in ("real", "sdv2", "adm"):
        _touch(root / "dire/test/b" / f"{name}.tar.gz")
    a = [(p["branch"], p["generator"]) for p in ev.pairs_from_raw(root, "test")]
    b = [(p["branch"], p["generator"]) for p in ev.pairs_from_raw(root, "test")]
    assert a == b == [("b", "adm"), ("b", "sdv2")], a


# ---------------------------------------------------------------------------
# 5. Tiền xử lý phải khớp tác giả — sai một bước là lệch số
# ---------------------------------------------------------------------------
def test_preprocess_constants_match_author():
    ev = _install_stub()
    assert ev.RESIZE == 256 and ev.CROP == 224
    assert ev.MEAN == (0.485, 0.456, 0.406), "mean ImageNet của utils/datasets.py"
    assert ev.STD == (0.229, 0.224, 0.225)
    assert ev.ACC_THRESHOLD == 0.5, "tác giả dùng `y_pred > 0.5`"
    assert ev.FPR_TARGET == 0.05


def test_preprocess_includes_normalize():
    """`demo.py` của tác giả THIẾU Normalize ⇒ điểm demo lệch điểm test.

    Test đọc AST của hàm `preprocess` để chắc nó gọi `Normalize` và `CenterCrop`
    — không chỉ tin vào hằng số.
    """
    import ast

    src = (SRC / "dire_modal" / "eval.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "preprocess")
    names = {
        n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)
    }
    for need in ("Resize", "CenterCrop", "ToTensor", "Normalize"):
        assert need in names, f"preprocess thiếu {need}"


def test_eval_reads_raw_tars_not_shards():
    """Eval đọc `/raw` — `/shards` bị `pack` làm mất nhánh (xem docstring `pack`).

    Kiểm bằng AST và BỎ QUA docstring: docstring có quyền giải thích *vì sao*
    không dùng `/shards`, nhưng không được có chuỗi `/shards` nào trong code.
    """
    import ast

    src = (SRC / "dire_modal" / "eval.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))

    bad = [
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
        and id(n) not in docstrings and "/shards" in n.value
    ]
    assert not bad, (
        "eval.py dùng /shards trong code: " + repr(bad) + ". `pack_one` đặt shard "
        "vào /shards/{split}/{stem} nên nhánh lsun_bedroom/ và imagenet/ trùng "
        "thư mục ⇒ eval phải đọc /raw."
    )


# ---------------------------------------------------------------------------
# 6. Stub ngữ nghĩa: eval_one là modal.Function, không gọi trực tiếp được
# ---------------------------------------------------------------------------
def test_eval_one_is_a_modal_function():
    ev = _install_stub()
    try:
        ev.eval_one("a", "b", "c")
    except TypeError as exc:
        assert "not callable" in str(exc)
    else:
        raise AssertionError("eval_one phải là modal.Function, không phải hàm thường")


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


# ---------------------------------------------------------------------------
# 7. Chạy `score_pair` thật với torch/PIL GIẢ — kiểm nhãn, batch và chiều điểm
#    (Không cần GPU: mọi thứ torch đụng tới đều được stub.)
# ---------------------------------------------------------------------------
class _Probs:
    """Giả lập `model(x).softmax(dim=1)[:, 1]`: trả đúng danh sách điểm đã đặt."""

    def __init__(self, values):
        self.values = values

    def softmax(self, dim):
        return self

    def __getitem__(self, item):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)


class _Stacked:
    def __init__(self, items):
        self.items = items

    def to(self, device):
        return self


def _fake_torch(_unused=None):
    """torch giả: `stack(buf)` giữ nguyên buf, `to()` không làm gì.

    `preprocess` trả về một con số (điểm của ảnh đó); `model(x)` trả `_Probs`
    đúng bằng các con số đó — nhờ vậy test điều khiển được điểm từng ảnh.
    """

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _T:
        @staticmethod
        def inference_mode():
            return _Ctx()

        @staticmethod
        def stack(buf):
            return _Stacked(buf)

    class _Model:
        def __call__(self, stacked):
            return _Probs(stacked.items)

    return _T, _Model()


def _fake_pil():
    import types as _types

    class _Img:
        def __init__(self, data):
            self.data = data

        def convert(self, mode):
            return self

    class _Image:
        @staticmethod
        def open(buf):
            data = buf.getvalue()
            if data == b"BROKEN":
                raise OSError("ảnh hỏng")
            return _Img(data)

    return _types.SimpleNamespace(Image=_Image)


def test_score_pair_assigns_labels_from_each_archive():
    """real.tar.gz ⇒ nhãn 0; tar generator ⇒ nhãn 1. Không lẫn, không đoán."""
    import io
    import tarfile
    import tempfile

    ev = _install_stub()
    ev.preprocess = lambda img, torch, transforms: img.data  # "ảnh" = id của nó

    def make(path, members):
        with tarfile.open(path, "w") as tf:
            for name in members:
                data = b"img-" + name.encode()
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))

    d = Path(tempfile.mkdtemp())
    real = d / "real.tar.gz"
    fake = d / "adm.tar.gz"
    make(real, ["a.png", "b.png"])
    make(fake, ["a.png", "b.png"])

    # Ảnh real được điểm thấp, ảnh fake điểm cao ⇒ tách hoàn hảo.
    calls = {"n": 0}
    orig = ev.preprocess

    def preprocess(img, torch, transforms):
        # nửa đầu là real, nửa sau là fake ⇒ gán điểm theo archive
        calls["n"] += 1
        base = 0.1 if calls["n"] <= 2 else 0.9
        return base

    ev.preprocess = preprocess
    torch, model = _fake_torch()
    try:
        sys.modules.setdefault("PIL", _fake_pil())
        r = ev.score_pair(model, real, fake, torch, None, "cpu", batch_size=1)
    finally:
        ev.preprocess = orig
        sys.modules.pop("PIL", None)

    assert r["n_real"] == 2 and r["n_fake"] == 2, r
    assert r["acc"] == 1.0, r
    assert r["ap"] == 1.0, r


def test_score_pair_counts_unreadable_images_instead_of_crashing():
    """Một ảnh hỏng KHÔNG được giết cả eval — phải đếm vào `skipped`."""
    import io
    import tarfile
    import tempfile

    ev = _install_stub()
    orig = ev.preprocess
    ev.preprocess = lambda img, torch, transforms: 0.0

    def make(path, blobs):
        with tarfile.open(path, "w") as tf:
            for name, data in blobs:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))

    d = Path(tempfile.mkdtemp())
    real, fake = d / "real.tar.gz", d / "adm.tar.gz"
    make(real, [("0_real/ok.png", b"IMG"), ("0_real/bad.png", b"BROKEN")])
    make(fake, [("1_fake/ok.png", b"IMG")])

    torch, model = _fake_torch({0.0: 0.9})
    try:
        sys.modules.setdefault("PIL", _fake_pil())
        r = ev.score_pair(model, real, fake, torch, None, "cpu", batch_size=8)
    finally:
        ev.preprocess = orig
        sys.modules.pop("PIL", None)

    assert r["skipped"] == 1, r
    assert r["n_real"] == 1, "ảnh hỏng không được tính là mẫu real"
    assert r["n_fake"] == 1


def test_score_pair_honours_limit_budget():
    """`limit` phải cắt đúng số ảnh MỖI archive — đây là công tắc chi phí."""
    import io
    import tarfile
    import tempfile

    ev = _install_stub()
    orig = ev.preprocess
    ev.preprocess = lambda img, torch, transforms: 0.0

    def make(path, n):
        with tarfile.open(path, "w") as tf:
            for i in range(n):
                data = b"x"
                info = tarfile.TarInfo(f"img{i}.png")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))

    d = Path(tempfile.mkdtemp())
    real, fake = d / "real.tar.gz", d / "adm.tar.gz"
    make(real, 5)
    make(fake, 5)

    torch, model = _fake_torch({0.0: 0.5})
    try:
        sys.modules.setdefault("PIL", _fake_pil())
        r = ev.score_pair(model, real, fake, torch, None, "cpu", batch_size=4, limit=2)
    finally:
        ev.preprocess = orig
        sys.modules.pop("PIL", None)

    assert r["n_real"] == 2 and r["n_fake"] == 2, r
