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

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _restore_pil(prev):
    """Trả ``sys.modules['PIL']`` về trạng thái trước test.

    Vì sao không dùng ``setdefault``: nếu một test khác (hoặc plugin) đã import
    PIL thật thì ``setdefault`` là no-op ⇒ ``score_pair`` mở ảnh giả bằng PIL
    thật, mọi ảnh thành ``skipped``, test đỏ với ``n_real == 0``. Stub phải ép,
    không phải "thử xem".
    """
    if prev is None:
        sys.modules.pop("PIL", None)
    else:
        sys.modules["PIL"] = prev


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

    Từ khi tách `dire_core`, pipeline nằm ở **một** chỗ duy nhất:
    `src/dire_core/preprocess.py`. Test đọc AST của hàm `preprocess` ở đó để chắc
    nó gọi đủ 4 bước — không chỉ tin vào hằng số. `tests/test_core_parity.py`
    kiểm thêm rằng `eval.py` KHÔNG có bản sao thứ hai.
    """
    import ast

    src = (SRC / "dire_core" / "preprocess.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "preprocess")
    # 4 bước nằm trong `build_transform`; `preprocess` phải gọi nó.
    build = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_transform")
    names = {n.attr for n in ast.walk(build) if isinstance(n, ast.Attribute)}
    for need in ("Resize", "CenterCrop", "ToTensor", "Normalize"):
        assert need in names, f"build_transform thiếu {need}"
    # và `preprocess` phải thật sự dùng `build_transform`
    called = {n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "build_transform" in called, "preprocess không gọi build_transform"


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
    """Đầu ra THÔ của model: logit, chưa qua sigmoid/softmax.

    `score_pair` đổi logit → P(fake) bằng `prob_fake` (n_out=1 ⇒ sigmoid), đúng
    quy ước checkpoint DIRE gốc. Trước đây test này giả lập `softmax(...)[:, 1]`
    nên không phát hiện được lỗi "model 2 lớp vs checkpoint 1 logit".
    """

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

    `preprocess` trả về một con số (LOGIT của ảnh đó); `model(x)` trả `_Probs`
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
        # nửa đầu là real, nửa sau là fake ⇒ gán LOGIT theo archive.
        # -2.2 → sigmoid = 0.10 (real); +2.2 → 0.90 (fake).
        calls["n"] += 1
        return -2.2 if calls["n"] <= 2 else 2.2

    ev.preprocess = preprocess
    torch, model = _fake_torch()
    try:
        _prev_pil = sys.modules.get("PIL")
        sys.modules["PIL"] = _fake_pil()
        r = ev.score_pair(model, real, fake, torch, None, "cpu", batch_size=1)
    finally:
        ev.preprocess = orig
        _restore_pil(_prev_pil)

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

    torch, model = _fake_torch({0.0: 2.2})
    try:
        _prev_pil = sys.modules.get("PIL")
        sys.modules["PIL"] = _fake_pil()
        r = ev.score_pair(model, real, fake, torch, None, "cpu", batch_size=8)
    finally:
        ev.preprocess = orig
        _restore_pil(_prev_pil)

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

    torch, model = _fake_torch({0.0: 0.0})
    try:
        _prev_pil = sys.modules.get("PIL")
        sys.modules["PIL"] = _fake_pil()
        r = ev.score_pair(model, real, fake, torch, None, "cpu", batch_size=4, limit=2)
    finally:
        ev.preprocess = orig
        _restore_pil(_prev_pil)

    assert r["n_real"] == 2 and r["n_fake"] == 2, r


# ---------------------------------------------------------------------------
# 8. Lớp cuối của checkpoint: 1 logit (DIRE gốc, BCE) vs 2 lớp (CrossEntropy)
# ---------------------------------------------------------------------------
# Sự cố THẬT 2026-10-28, log:
#   RuntimeError: Error(s) in loading state_dict for ResNet:
#     size mismatch for fc.weight: copying a param with shape torch.Size([1, 2048])
#     from checkpoint, the shape in current model is torch.Size([2, 2048]).
# Toàn bộ eval chết ở dòng `load_state_dict` — sau khi đã trả tiền bật container
# GPU — vì code tạo `resnet50(num_classes=2)` trong khi checkpoint của tác giả
# (`vendor/utils/utils.py`: `resnet(num_classes=1)`, loss `BCEWithLogitsLoss`)
# có đúng MỘT logit.
class _Shape:
    """Giả lập `torch.Size` — chỉ cần `len()` và truy cập theo chỉ số."""

    def __init__(self, *dims):
        self._d = tuple(dims)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, i):
        return self._d[i]


class _Tensor:
    """Giả lập tensor: thứ `load_detector` cần duy nhất là `.shape`."""

    def __init__(self, *dims):
        self.shape = _Shape(*dims)


def test_fc_out_features_reads_the_author_one_logit_checkpoint():
    ev = _install_stub()
    assert ev.fc_out_features({"fc.weight": _Tensor(1, 2048), "fc.bias": _Tensor(1)}) == 1


def test_fc_out_features_reads_a_two_class_checkpoint():
    ev = _install_stub()
    assert ev.fc_out_features({"fc.weight": _Tensor(2, 2048)}) == 2


def test_fc_out_features_handles_dataparallel_prefix():
    """Checkpoint train bằng DataParallel có tiền tố `module.`."""
    ev = _install_stub()
    assert ev.fc_out_features({"module.fc.weight": _Tensor(1, 2048)}) == 1


def test_fc_out_features_returns_zero_when_it_cannot_tell():
    """Không thấy `fc.weight` ⇒ 0, để caller BÁO RÕ thay vì đoán 2 cho chắc."""
    ev = _install_stub()
    assert ev.fc_out_features({"conv1.weight": _Shape(64, 3, 7, 7)}) == 0


def test_prob_fake_uses_sigmoid_for_one_logit():
    """1 logit ⇒ sigmoid; đây là quy ước của checkpoint tác giả."""
    ev = _install_stub()
    p = ev.prob_fake([0.0, 2.2, -2.2], n_out=1)
    assert abs(p[0] - 0.5) < 1e-9
    assert abs(p[1] - 0.9010) < 1e-3      # sigmoid(2.2)
    assert abs(p[2] - 0.0998) < 1e-3      # sigmoid(-2.2)
    # Điểm phải ĐƠN ĐIỆU theo logit, nếu không ngưỡng 0.5 vô nghĩa.
    assert p[2] < p[0] < p[1]


def test_prob_fake_uses_softmax_for_two_logits():
    ev = _install_stub()
    p = ev.prob_fake([[0.0, 0.0], [0.0, 5.0], [5.0, 0.0]], n_out=2)
    assert abs(p[0] - 0.5) < 1e-9
    assert p[1] > 0.99 and p[2] < 0.01


def test_prob_fake_accepts_nested_rows_for_one_logit():
    """`model(x)` thật trả tensor [B, 1] ⇒ `.tolist()` ra [[z], [z], …]."""
    ev = _install_stub()
    flat = ev.prob_fake([2.2, -2.2], n_out=1)
    nested = ev.prob_fake([[2.2], [-2.2]], n_out=1)
    assert flat == nested


def test_prob_fake_does_not_overflow_on_extreme_logits():
    """`sigmoid(-800)` phải là 0.0, không phải OverflowError giết cả eval."""
    ev = _install_stub()
    p = ev.prob_fake([-800.0, 800.0], n_out=1)
    assert p[0] == 0.0 and p[1] == 1.0


def test_prob_fake_refuses_to_invent_a_score_when_columns_are_missing():
    ev = _install_stub()
    try:
        ev.prob_fake([[0.3]], n_out=2)
    except ValueError as exc:
        assert "logit" in str(exc)
    else:
        raise AssertionError("thiếu cột lớp fake mà vẫn trả điểm ⇒ bịa số")


class _FakeTorch:
    """Chỉ đủ cho `load_detector`: `torch.load`."""

    def __init__(self, state):
        self.state = state

    def load(self, ckpt, map_location=None):
        return self.state


class _FakeModel:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.loaded = None

    def load_state_dict(self, state):
        # Mô phỏng đúng việc torch làm: so shape của `fc.weight`.
        shape = getattr(state.get("fc.weight"), "shape", None)
        got = shape[0] if shape is not None else None
        if got != self.n_classes:
            raise RuntimeError(
                "Error(s) in loading state_dict for ResNet:\n"
                f"\tsize mismatch for fc.weight: copying a param with shape "
                f"torch.Size([{got}, 2048]) from checkpoint, the shape in current "
                f"model is torch.Size([{self.n_classes}, 2048])."
            )
        self.loaded = state

    def to(self, device):
        return self

    def eval(self):
        return self


def _fake_resnet(record):
    def resnet50(num_classes=1000, **kw):
        record.append(num_classes)
        return _FakeModel(num_classes)

    return resnet50


def test_load_detector_builds_the_model_that_matches_the_checkpoint():
    """Đây chính là lỗi 2026-10-28: phải tạo 1 lớp, không phải 2."""
    ev = _install_stub()
    made: list[int] = []
    state = {"fc.weight": _Tensor(1, 2048), "fc.bias": _Tensor(1)}
    model, n_out = ev.load_detector("x.pth", _FakeTorch(state), _fake_resnet(made), "cpu")

    assert made == [1], f"tạo resnet50(num_classes={made}) cho checkpoint 1 logit"
    assert n_out == 1 and model.loaded is not None


def test_load_detector_still_accepts_a_two_class_checkpoint():
    ev = _install_stub()
    made: list[int] = []
    state = {"fc.weight": _Tensor(2, 2048)}
    _model, n_out = ev.load_detector("x.pth", _FakeTorch(state), _fake_resnet(made), "cpu")
    assert made == [2] and n_out == 2


def test_load_detector_unwraps_nested_state_dicts():
    ev = _install_stub()
    made: list[int] = []
    nested = {"epoch": 7, "model": {"module.fc.weight": _Tensor(1, 2048)}}
    _model, n_out = ev.load_detector("x.pth", _FakeTorch(nested), _fake_resnet(made), "cpu")
    assert made == [1] and n_out == 1


def test_load_detector_explains_itself_instead_of_a_bare_runtimeerror():
    """Nếu vẫn lệch, thông điệp phải nói quy ước 1 logit vs 2 lớp."""
    ev = _install_stub()

    class _Wrong(_FakeModel):
        def load_state_dict(self, state):
            raise RuntimeError("size mismatch for fc.weight")

    try:
        ev.load_detector("x.pth", _FakeTorch({"fc.weight": _Tensor(1, 2048)}),
                         lambda **k: _Wrong(2), "cpu")
    except SystemExit as exc:
        msg = str(exc)
        assert "n_out đọc từ checkpoint = 1" in msg, msg
        assert "[1,2048]" in msg and "[2,2048]" in msg, msg
    else:
        raise AssertionError("load lệch mà không báo gì")


def test_load_detector_says_so_when_it_cannot_find_fc():
    ev = _install_stub()
    try:
        ev.load_detector("x.pth", _FakeTorch({"conv1.weight": _Shape(64, 3, 7, 7)}),
                         lambda **k: _FakeModel(2), "cpu")
    except SystemExit as exc:
        assert "Không đọc được lớp cuối" in str(exc)
    else:
        raise AssertionError("không thấy fc.weight mà vẫn đoán bừa")


# ---------------------------------------------------------------------------
# 9. Một cặp lỗi không được vứt cả lượt eval
# ---------------------------------------------------------------------------
def test_failed_pair_row_keeps_metrics_empty_and_records_the_reason():
    """`acc = 0` đọc như "đoán sai hết"; ô trống + `error` đọc đúng là "không chấm"."""
    ev = _install_stub()
    meta = {"branch": "lsun_bedroom__lsun_bedroom", "generator": "adm"}
    row = ev.row_for_result("exp", {"fake": "/x/adm.tar.gz",
                                    "error": "size mismatch for fc.weight"},
                            meta, "test", "deadbeef", "2026-10-28T00:00:00Z")

    assert row["acc"] == "" and row["ap"] == "" and row["n_fake"] == ""
    assert row["error"] == "size mismatch for fc.weight"
    assert row["branch"] == "lsun_bedroom__lsun_bedroom"
    # Không được bịa 0
    assert row["acc"] != 0 and row["ap"] != 0


def test_ok_pair_row_has_no_error_and_keeps_numbers():
    ev = _install_stub()
    meta = {"branch": "b", "generator": "adm"}
    row = ev.row_for_result("exp", {"n_real": 100, "n_fake": 100, "acc": 0.995,
                                    "ap": 0.998, "tpr_at_5fpr": 0.99},
                            meta, "test", "sha", "t")
    assert row["error"] == "" and row["acc"] == 0.995 and row["n_real"] == 100


def test_eval_one_never_raises_out_of_the_map():
    """`.map()` của Modal huỷ mọi task anh em khi một task raise ⇒ phải bắt."""
    import ast as _ast

    src = (SRC / "dire_modal" / "eval.py").read_text(encoding="utf-8")
    tree = _ast.parse(src)
    fn = next(n for n in tree.body
              if isinstance(n, _ast.FunctionDef) and n.name == "eval_one")
    handlers = [h for node in _ast.walk(fn) if isinstance(node, _ast.Try)
                for h in node.handlers]
    assert handlers, "eval_one phải có try/except để không raise ra .map()"
    caught = " ".join(_ast.dump(h.type) if h.type else "Bare" for h in handlers)
    assert "Exception" in caught, "phải bắt cả Exception, không chỉ lỗi cụ thể"
    assert "SystemExit" in caught, "SystemExit mang thông điệp đã rõ, phải giữ"


# ---------------------------------------------------------------------------
# 10. Demo HF Spaces: đường "ảnh RGB → xác suất" đã bị bỏ, và phải ở yên như thế
# ---------------------------------------------------------------------------
# Hai test dưới đây từng kiểm app Gradio cũ: nó nạp `fc.weight` để biết số lớp,
# và chuẩn hoá ImageNet đúng như lúc train. Cả hai đều đúng — và cả hai đều **vô
# nghĩa**, vì app đó đưa ảnh RGB vào một model học trên DIRE map. Sửa phép chuẩn
# hoá cho khớp không làm con số có nghĩa hơn; nó chỉ làm con số khó nghi ngờ hơn.
#
# Nên app Streamlit hiện tại không có bộ phân loại nào để nạp sai. Điểm số chỉ đến
# từ `.json` của gallery, sau khi qua cổng `gate == "passed"`. Ba test dưới khoá
# đúng điều đó; phần số học và giao diện nằm ở `tests/test_hf_spaces_demo.py`.
def test_hf_space_has_no_classifier_to_misload():
    src = (ROOT / "platforms/hf-spaces/app.py").read_text(encoding="utf-8")
    for banned in ("num_classes", "fc.weight", "sigmoid", "softmax", "torchvision",
                   "import torch"):
        assert banned not in src, f"app.py không được dựng lại đường chấm ảnh ({banned!r})"
    # Có `torch` trong câu thông báo trạng thái là chuyện khác với *dùng* torch:
    # phần tính toán nằm ở `demolib/live.py`, nơi có test parity với `dire_map`.
    assert "live" in src


def test_hf_space_has_no_image_preprocessing():
    """Không còn `Resize(256) → CenterCrop(224) → Normalize` vì không còn ảnh đầu vào.

    Đây là điều *tốt*: mỗi bước tiền xử lý là một chỗ để lệch âm thầm so với lúc
    train (`Normalize` sai từng làm hỏng điểm của tác giả, README dòng 40).
    """
    src = (ROOT / "platforms/hf-spaces/app.py").read_text(encoding="utf-8")
    for banned in ("RESIZE", "CenterCrop", "Normalize", "file_uploader"):
        assert banned not in src, f"app.py không được nhận ảnh người dùng ({banned!r})"


def test_hf_space_score_only_after_the_gate():
    src = (ROOT / "platforms/hf-spaces/demolib/provenance.py").read_text(encoding="utf-8")
    assert 'gate != "passed"' in src, "phải chặn điểm khi detector chưa qua cổng"
    assert 'threshold_split") != "val"' in src, "ngưỡng phải chọn trên val, không trên test"


# ---------------------------------------------------------------------------
# 9. `--tiers` phải QUYẾT ĐỊNH tập cặp, không chỉ được ghi vào run.json
# ---------------------------------------------------------------------------
def _pairs_under(tmp_root, rels):
    """Cặp giả: (real, fake) cùng thư mục, để test bộ lọc mà không cần tar."""
    return [
        {"branch": "x", "generator": Path(rel).name[: -len(".tar.gz")],
         "real": Path(tmp_root) / Path(rel).parent / "real.tar.gz",
         "fake": Path(tmp_root) / rel}
        for rel in rels
    ]


def test_tier_e2_selects_exactly_fifteen_pairs():
    """`e2` = 17 file: 1 checkpoint + real + 15 generator ⇒ **15 cặp**."""
    ev = _install_stub()
    assert ev.expected_n_pairs("e2") == 15


def test_filter_pairs_keeps_only_the_requested_tier(tmp_path):
    """Xin `e2` mà ổ đĩa có cả `e3` ⇒ nhánh cross-generator KHÔNG được chấm.

    `lsun_bedroom__imagenet` dùng chung `real.tar.gz` với nhánh LSUN, nên nếu
    lọt vào thì mean AP bị tính hai lần trên cùng 1000 ảnh real.
    """
    ev = _install_stub()
    pairs = _pairs_under(tmp_path, [
        "dire/test/lsun_bedroom/lsun_bedroom/adm.tar.gz",
        "dire/test/lsun_bedroom/lsun_bedroom/sdv2.tar.gz",
        "dire/test/lsun_bedroom/imagenet/adm.tar.gz",
        "dire/test/celebahq/sdv2.tar.gz",
    ])
    kept, dropped = ev.filter_pairs(pairs, "e2", tmp_path)
    assert [p["generator"] for p in kept] == ["adm", "sdv2"]
    assert all("imagenet" not in str(p["fake"]) for p in kept)
    assert dropped == 2


def test_filter_pairs_does_not_guess_when_tier_has_dirspec(tmp_path):
    """Tier có `DirSpec` (e4x) ⇒ KHÔNG lọc. Thà chấm thừa còn hơn bỏ cặp."""
    ev = _install_stub()
    pairs = _pairs_under(tmp_path, ["images/test/lsun_bedroom/adm/0001.png"
                                    .replace(".png", ".tar.gz")])
    kept, dropped = ev.filter_pairs(pairs, "e4x", tmp_path)
    assert kept == pairs and dropped == 0


def test_unknown_tier_fails_loudly(tmp_path):
    ev = _install_stub()
    with pytest.raises(SystemExit, match="tier không có trong manifest"):
        ev.filter_pairs(_pairs_under(tmp_path, ["dire/test/a/b.tar.gz"]), "ez9", tmp_path)


def test_local_pairs_previews_tier_without_data(tmp_path, capsys):
    """Chưa tải byte nào vẫn trả lời được "e2 sẽ chấm bao nhiêu cặp"."""
    ev = _install_stub()
    ev.local_pairs.local(str(tmp_path), "test", "e2")
    out = capsys.readouterr().out
    assert "15 cặp" in out
    assert "chưa có dữ liệu" in out
