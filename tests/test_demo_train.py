"""Test cho `demo_train.py` — phần kiểm được mà KHÔNG cần GPU, torch hay ảnh thật.

Vì sao có file này
------------------
`demo_train.py` là chỗ dễ trôi nhất trong repo: nó *train thật* nên đầu ra trông
giống kết quả, nhưng mục đích chỉ là **minh hoạ cơ chế**. Ba kiểu sai cần chặn:

1. **Trôi thành "số của dự án".** Nếu checkpoint/log của demo lọt vào
   ``/ckpt/results/**`` thì ``eval::report`` sẽ trộn ACC của 2 000 ảnh × 5 epoch
   vào bảng Δ so với paper — và Δ sẽ "đúng" trên một con số vô nghĩa.
2. **Bịa siêu tham số.** Giá trị phải khớp ``vendor/utils/config.py`` của tác
   giả, nên test đọc thẳng file vendor thay vì tin vào một hằng số chép tay.
3. **Bịa đường cong.** Log chỉ được chứa số hữu hạn, có ``first/last`` thật;
   scene manim không được có nguồn số nào khác.

Test không cần PIL/torch: chỗ đọc ảnh nhận ``opener``/``transform`` tiêm từ
ngoài, nên kiểm được nhãn/đếm/limit bằng bytes giả.

Chạy::

    python3 -m pytest tests/test_demo_train.py -q
"""

from __future__ import annotations

import ast
import inspect
import io
import json
import re
import sys
import tarfile
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

VENDOR = SRC / "dire_modal" / "vendor" / "utils"


# ---------------------------------------------------------------------------
# Stub Modal — và PHẢI dọn sạch sau khi xong
# ---------------------------------------------------------------------------
# Bài học đã trả giá trong repo này: một module test import `dire_modal.common`
# với `modal` THẬT rồi để nó nằm lại trong `sys.modules` sẽ làm mọi module test
# sau đó (chúng cài stub) nhận `common.app` là App THẬT ⇒
# ``InvalidError: Retries parameter must be ... Found: SimpleNamespace``.
# Nên: cài stub TRƯỚC khi import, và teardown xoá mọi `dire_modal*` đã nhiễm.


class _Fn:
    """Giống ``modal.Function``: có ``.map/.remote/.local`` nhưng KHÔNG gọi được."""

    def __init__(self, fn, kw):
        self._fn, self._kw = fn, kw
        self.__name__ = getattr(fn, "__name__", "fn")
        self.__doc__ = getattr(fn, "__doc__", None)

    def __call__(self, *a, **k):
        raise TypeError("'Function' object is not callable")

    def map(self, items, **kw):
        return [self._fn(x) for x in items]

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
    def commit(self, *a, **k):
        pass


class _App:
    def __init__(self, *a, **k):
        pass

    def function(self, *a, **k):
        return lambda fn: _Fn(fn, k)

    def cls(self, *a, **k):
        return lambda c: c

    def local_entrypoint(self, *a, **k):
        return lambda fn: _Fn(fn, k)


D = None   # module `demo_train`, cài trong fixture (xem lý do bên dưới)
E = None   # module `eval`


def _purge() -> None:
    """Xoá stub + mọi module dire_modal đã import — KHÔNG khôi phục lại.

    Khôi phục bản cũ là tự chuốc lỗi: `dire_modal.common` cache `app`/`vol_*`
    ngay lần import đầu, nên trả lại một `common` do stub KHÁC sinh ra sẽ làm
    file test sau nhận `app`/`Function`/`Volume` sai kiểu (đã xảy ra: 5 test
    download đỏ). Cùng cách `test_compute_dire_guard._purge` đã sửa lỗi này.
    """
    for name in [n for n in sys.modules if n == "modal" or n == "dire_modal"
                 or n.startswith("dire_modal.")]:
        sys.modules.pop(name, None)


def _install_stub() -> None:
    """Cài stub rồi import. PHẢI purge trước: `common` cache `app`/`vol_*` ngay
    lần import đầu, nên một `dire_modal.common` do stub khác sinh ra sẽ làm
    `compute_dire` chết ở `@app.cls` (đã xảy ra: 1 test đỏ)."""
    global D, E
    _purge()
    mod = types.ModuleType("modal")
    mod.App = _App
    mod.Image = types.SimpleNamespace(debian_slim=lambda **k: _Img())
    mod.Volume = types.SimpleNamespace(from_name=lambda *a, **k: _Vol())
    mod.Secret = types.SimpleNamespace(from_name=lambda *a, **k: object())
    mod.Retries = lambda **k: types.SimpleNamespace(**k)
    mod.enter = lambda *a, **k: (lambda fn: fn)
    mod.parameter = lambda default=None, **k: default
    sys.modules["modal"] = mod
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from dire_modal import demo_train, eval as ev  # noqa: PLC0415

    D, E = demo_train, ev


@pytest.fixture(scope="module", autouse=True)
def _modules():
    """Cài stub + import Ở THỜI ĐIỂM CHẠY, không phải lúc collect.

    Import lúc collect nghĩa là `dire_modal.common` bị neo vào stub NÀY trước
    khi bất kỳ test nào chạy — và mọi file test khác (chúng có stub riêng, đầy
    đủ hơn) sẽ nhận `app` không có `.cls`/`.map` đúng nghĩa. Đây đúng lỗi
    "stub rò giữa module test" mà repo đã gặp.
    """
    global D, E
    _install_stub()
    yield
    _purge()
    D = E = None


# ---------------------------------------------------------------------------
# 1. Cấu hình phải khớp code của tác giả
# ---------------------------------------------------------------------------
def _vendor_value(pattern: str, text: str) -> str:
    m = re.search(pattern, text, re.M)
    assert m, f"không thấy {pattern!r} trong code tác giả — vendor đã đổi?"
    return m.group(1)


@pytest.fixture(scope="module")
def vendor_config() -> str:
    return (VENDOR / "config.py").read_text(encoding="utf-8")


def test_config_matches_the_authors_source(vendor_config):
    """Từng siêu tham số của demo phải bằng giá trị trong `DefaultConfigs`."""
    cases = [
        ("seed", 3407, r"^\s*seed = (\d+)"),
        ("batch_size", 64, r"^\s*batch_size = (\d+)"),
        ("lr", 1e-4, r"^\s*lr = ([\d.e-]+)"),
        ("beta1", 0.9, r"^\s*beta1 = ([\d.]+)"),
        ("load_size", 256, r"^\s*loadSize = (\d+)"),
        ("crop_size", 224, r"^\s*cropSize = (\d+)"),
    ]
    cfg = D.load_config()
    for key, expected, pattern in cases:
        got_vendor = _vendor_value(pattern, vendor_config)
        assert float(got_vendor) == float(expected), (
            f"{key}: vendor ghi {got_vendor}, hằng số trong test là {expected}"
        )
        assert float(cfg[key]) == float(expected), f"{key}: config demo lệch vendor"

    assert _vendor_value(r'^\s*arch = "(\w+)"', vendor_config) == cfg["arch"] == D.ARCH


def test_aug_flags_match_the_author(vendor_config):
    """Tác giả bật hình học, TẮT quang học — demo cũng phải vậy."""
    aug = D.load_config()["aug"]
    assert aug["resize"] is True
    assert aug["crop"] is True
    assert aug["flip"] is True
    for key in ("blur_prob", "jpg_prob", "gray_prob"):
        assert float(_vendor_value(rf"^\s*{key} = ([\d.]+)", vendor_config)) == 0.0
        assert float(aug[key]) == 0.0, f"{key}: demo bật một aug mà tác giả tắt"


def test_loss_and_width_are_the_ones_that_broke_eval():
    """1 logit + BCEWithLogitsLoss — đúng quy ước đã gây `size mismatch`."""
    cfg = D.load_config()
    assert cfg["n_out"] == D.N_OUT == 1
    assert cfg["loss"] == "bce_with_logits"
    trainer = (VENDOR / "trainer.py").read_text(encoding="utf-8")
    assert "BCEWithLogitsLoss" in trainer, "vendor đổi loss ⇒ phải xem lại n_out"


def test_config_lives_next_to_the_module():
    """Modal chỉ đóng gói `src/dire_modal/` ⇒ file ở gốc repo KHÔNG vào container."""
    assert D.CONFIG_PATH.exists(), f"thiếu {D.CONFIG_PATH}"
    assert D.CONFIG_PATH.parent.name == "dire_modal"


def test_load_config_never_raises(tmp_path):
    """File thiếu/hỏng ⇒ fallback, không giết job đã bật GPU."""
    assert D.load_config(tmp_path / "khong_ton_tai.yaml")["seed"] == D.SEED
    broken = tmp_path / "hong.yaml"
    broken.write_text("dòng này: [không đóng ngoặc", encoding="utf-8")
    assert D.load_config(broken)["batch_size"] == D.BATCH


# ---------------------------------------------------------------------------
# 2. Guard chi phí — phần duy nhất tiêu tiền
# ---------------------------------------------------------------------------
def test_estimate_is_hand_computable():
    """2000 ảnh / 64 = 32 step mỗi epoch × 5 = 160; 160 × 0.35 s = 56 s."""
    est = D.estimate(images=2000, epochs=5, batch=64, sec_per_step=0.35,
                     usd_per_gpu_hour=0.80, val_images=0)
    assert est["train_steps"] == 160
    assert est["gpu_seconds"] == pytest.approx(56.0)
    # Tiền làm tròn 5 chữ số thập phân (micro-USD là vô nghĩa ở đây).
    assert est["usd"] == pytest.approx(56.0 / 3600.0 * 0.80, abs=1e-5)


def test_estimate_counts_validation_as_forward_only():
    """Val chỉ forward: 4 step × 5 epoch × 0.35 s × 1/3."""
    est = D.estimate(images=0, epochs=5, batch=64, sec_per_step=0.35,
                     usd_per_gpu_hour=0.80, val_images=200)
    assert est["train_steps"] == 0
    assert est["val_steps"] == 20
    assert est["gpu_seconds"] == pytest.approx(20 * 0.35 / 3.0, abs=1e-3)


def test_estimate_is_monotonic():
    a = D.estimate(images=500, epochs=1)["usd"]
    b = D.estimate(images=1000, epochs=1)["usd"]
    c = D.estimate(images=1000, epochs=2)["usd"]
    assert a < b < c


def test_enforce_rejects_too_many_images():
    with pytest.raises(SystemExit) as exc:
        D.enforce(D.estimate(images=4000, epochs=1), max_images=2000)
    msg = str(exc.value)
    assert "4000" in msg and "2000" in msg


def test_enforce_rejects_too_many_minutes():
    est = D.estimate(images=2000, epochs=100)   # ≈ 19 phút
    assert est["gpu_minutes"] > 15
    with pytest.raises(SystemExit) as exc:
        D.enforce(est, max_minutes=15)
    assert "phút" in str(exc.value)


def test_default_plan_passes_the_guard():
    """Kế hoạch mặc định phải nằm trong trần — nếu không, mọi người gặp lỗi đầu tiên."""
    est = D.enforce(D.estimate())
    assert est["images"] == 2000
    assert est["gpu_minutes"] < D.MAX_MINUTES


def test_plan_mode_prints_label_and_trains_nothing(capsys):
    out = D.describe_plan(D.estimate())
    printed = capsys.readouterr().out
    assert out["verdict"] == "ok" and out["demo"] is True
    assert D.DEMO_LABEL in printed
    assert "chưa train gì" in printed
    assert "[tiền]" in printed


def test_main_defaults_to_plan():
    """`--mode` mặc định phải là plan: gõ thiếu cờ không được train.

    Đọc bằng AST vì `D.main` là `modal.Function` (decorator trả object, không
    trả hàm) — đúng ngữ nghĩa Modal thật, và cũng là lý do không dùng
    `inspect.signature` ở đây.
    """
    tree = ast.parse(Path(D.__file__).read_text(encoding="utf-8"))
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "main")
    names = [a.arg for a in fn.args.args]
    defaults = dict(zip(names[-len(fn.args.defaults):], fn.args.defaults))
    assert isinstance(defaults["mode"], ast.Constant)
    assert defaults["mode"].value == "plan"


# ---------------------------------------------------------------------------
# 3. Nhãn + đọc tar (không cần PIL, không cần torch)
# ---------------------------------------------------------------------------
PNG_1PX = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
    b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01"
    b"\x01\x00\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _tar(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def test_iter_labeled_counts_labels_and_limits(tmp_path):
    real = _tar(tmp_path / "real.tar.gz", {f"0_real/{i}.png": PNG_1PX for i in range(3)})
    fake = _tar(tmp_path / "adm.tar.gz", {f"1_fake/{i}.png": PNG_1PX for i in range(5)})
    stats: dict = {}
    got = list(D.iter_labeled([(real, E.REAL), (fake, E.FAKE)], per_class=2,
                              opener=lambda b: b, stats=stats))
    assert stats["read"] == 4 and stats["skipped"] == 0
    assert [lbl for _, lbl in got] == [E.REAL, E.REAL, E.FAKE, E.FAKE]
    assert stats["per_label"] == {E.REAL: 2, E.FAKE: 2}


def test_iter_labeled_skips_broken_images_without_dying(tmp_path):
    """Một ảnh hỏng không được giết cả lần train (bài học fetch_one/eval_one)."""
    tar = _tar(tmp_path / "real.tar.gz", {"0_real/a.png": b"ok", "0_real/b.png": b"bad"})
    stats: dict = {}
    got = list(D.iter_labeled([(tar, E.REAL)], opener=lambda b: None if b == b"bad" else b,
                              stats=stats))
    assert len(got) == 1 and stats["skipped"] == 1


def test_iter_labeled_ignores_missing_archive(tmp_path):
    got = list(D.iter_labeled([(tmp_path / "khong_co.tar.gz", E.REAL)], opener=lambda b: b))
    assert got == []


def test_batches_keeps_the_tail():
    assert [len(b) for b in D.batches(range(5), 2)] == [2, 2, 1]
    assert [len(b) for b in D.batches(range(0), 2)] == []


# ---------------------------------------------------------------------------
# 4. Log / checkpoint — không bịa số, không đụng `results`
# ---------------------------------------------------------------------------
class _Shape:
    def __init__(self, shape):
        self.shape = shape


def test_checkpoint_carries_one_logit_and_eval_can_read_it():
    """Khép vòng với bản vá 2026-10-28: checkpoint demo phải nạp được bằng eval."""
    state = D.state_for_save({"fc.weight": _Shape((1, 2048)), "fc.bias": _Shape((1,))},
                             epoch=5, images=2000)
    assert state["demo"] is True and state["label"] == D.DEMO_LABEL
    assert state["n_out"] == 1
    assert E.fc_out_features(state["model"]) == 1


def test_output_dir_is_never_the_results_tree():
    d = D.ckpt_dir("lan1")
    assert d.as_posix().endswith("/demo_train/lan1")
    assert "results" not in d.parts
    with pytest.raises(SystemExit):
        D.ckpt_dir("results")


def test_log_entry_refuses_nan():
    with pytest.raises(ValueError):
        D.log_entry(1, 1, float("nan"), 0.5, 64)
    with pytest.raises(ValueError):
        D.log_entry(1, 1, 0.5, float("inf"), 64)


def test_summarize_uses_real_first_and_last_step():
    steps = [D.log_entry(1, 1, 0.7, 0.5, 64), D.log_entry(2, 1, 0.4, 0.8, 64)]
    s = D.summarize(steps, [{"epoch": 1, "val_acc": 0.75}])
    assert (s["first_loss"], s["last_loss"]) == (0.7, 0.4)
    assert s["best_val_acc"] == 0.75 and s["n_steps"] == 2


def test_summarize_keeps_missing_val_empty():
    """Chưa có val ⇒ None, KHÔNG phải 0 (ô trống ≠ số 0)."""
    s = D.summarize([], [{"epoch": 1, "val_acc": None}])
    assert s["best_val_acc"] is None and s["first_loss"] is None


def test_log_payload_is_the_only_source_for_scenes(tmp_path):
    steps = [D.log_entry(1, 1, 0.7, 0.5, 64)]
    payload = D.log_payload("lan1", steps, [{"epoch": 1, "val_acc": 0.5}], {"read": 64})
    p = D.write_log(tmp_path / "loss_log.json", payload)
    back = json.loads(p.read_text(encoding="utf-8"))     # cp1252 sẽ nổ ở đây
    assert back["label"] == D.DEMO_LABEL and back["demo"] is True
    assert back["steps"] == steps
    assert "—" in back["label"]                           # em dash sống sót


# ---------------------------------------------------------------------------
# 5. Ranh giới với phần còn lại của repo
# ---------------------------------------------------------------------------
def test_module_does_not_import_torch_at_top():
    """Import module phải nhẹ: test/plan không được kéo torch vào."""
    tree = ast.parse(Path(D.__file__).read_text(encoding="utf-8"))
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.add(node.module.split(".")[0])
    assert not ({"torch", "torchvision", "PIL"} & top), top


def test_no_paper_number_is_hardcoded():
    """Không được có SỐ 99.9 trong code: con số paper không thuộc file này.

    Docstring được phép *nhắc* tới 99.9 (để cảnh báo đừng so sánh với nó); thứ
    bị cấm là một literal số có thể lọt vào biểu đồ.
    """
    tree = ast.parse(Path(D.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            assert abs(float(node.value) - 99.9) > 1e-9, "literal 99.9 trong code"
    assert "paper_table2" not in Path(D.__file__).read_text(encoding="utf-8")


def test_train_transform_uses_geometry_not_photometry():
    src = inspect.getsource(D.train_transforms)
    for need in ("Resize(LOAD_SIZE)", "RandomCrop(CROP)", "RandomHorizontalFlip",
                 "Normalize(E.MEAN, E.STD)"):
        assert need in src, need
    assert "Jpeg" not in src and "GaussianBlur" not in src and "Grayscale" not in src


def test_preproc_constants_match_eval():
    """256/224 phải TRÙNG eval, nếu không demo và eval nhìn hai loại ảnh khác nhau."""
    assert (D.LOAD_SIZE, D.CROP) == (E.RESIZE, E.CROP)
    assert D.N_OUT == 1 and E.FAKE == 1 and E.REAL == 0


def test_importing_the_module_does_not_pull_eval_in():
    """`demo_train` không được đăng ký `eval.main` vào cùng app từ tầng module.

    Modal báo `function name 'main' collision!` khi hai module cùng app có hàm
    `main`, và `::main` trở nên phụ thuộc thứ tự import. Import `eval` bên trong
    hàm giữ cho `modal run -m dire_modal.demo_train::main` không mơ hồ.
    """
    import subprocess

    code = (
        "import sys; import dire_modal.demo_train; "
        "assert 'dire_modal.eval' not in sys.modules, 'eval đã bị import ở tầng module'; "
        "print('ok')"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                       env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"},
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    assert "ok" in r.stdout


def test_data_paths_are_the_train_split_not_the_test_split():
    """Train trên test split là leakage — đường này phải trỏ vào split `train`."""
    for p in (D.TRAIN_REAL, D.TRAIN_FAKE):
        assert "/dire/train/" in p
    for p in (D.VAL_REAL, D.VAL_FAKE):
        assert "/dire/val/" in p
