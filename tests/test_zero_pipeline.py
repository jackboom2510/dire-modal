"""Chạy trọn bốn bước trên mô hình tí hon — bài kiểm "pipeline có nối được không".

Cái được kiểm không phải chất lượng (mô hình ngẫu nhiên 32×32, 8 ảnh), mà là
**hợp đồng giữa các bước**:

* `gen.json` có đủ trường để người khác dựng lại (seed, bước, sha256, split);
* `dire.json` khai đúng thang `[0, 2]` và NFE đếm được;
* `train.json` khai nhãn do **người gọi** truyền vào (không tự nhận R3 khi ảnh
  giả là của tác giả), chọn ngưỡng trên val, và test chỉ chấm một lần;
* split train/test **rời nhau tuyệt đối** (không một tên file chung).

Nếu bốn điều này sai, con số đẹp nhất cũng vô nghĩa. Và chúng sai được một cách
im lặng — không exception, không cảnh báo.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from dire_zero.detector import ResNet, ResNetDetector  # noqa: E402
from dire_zero.diffusion import ModelCounter  # noqa: E402
from dire_zero.pipeline import RunMeta, collect_maps, compute_maps, generate, load_image, train  # noqa: E402
from dire_zero.unet import UNetConfig, ZeroUNet  # noqa: E402

TINY = UNetConfig(image_size=32, model_channels=32, num_res_blocks=1,
                  attention_ds=(4,), channel_mult=(1, 2), num_heads=2, head_channels=-1)


@pytest.fixture()
def tmp_run(tmp_path: Path):
    torch.manual_seed(0)
    model = ZeroUNet(TINY).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    f = ModelCounter(model)
    meta = RunMeta("a" * 64, steps=3)
    gen = generate(f, n=10, out_dir=tmp_path / "gen", meta=meta, seed_base=100, size=32)
    items = [
        (str(tmp_path / "gen" / n), 1, "train" if n in gen["split"]["train"] else "test", "ours_adm")
        for n in gen["names"]
    ]
    maps = compute_maps(f, items, tmp_path / "maps", meta, size=32)

    # Ảnh **thật** cho tập train: thiếu chúng thì `train` chỉ thấy một lớp và
    # (đúng như phải thế) từ chối chạy — xem `pipeline.SplitError`. Fixture cũ chỉ
    # có ảnh sinh, nên test "train chạy được" thực chất đang train trên một lớp.
    import numpy as np
    from PIL import Image

    real_dir = tmp_path / "real"
    real_dir.mkdir(parents=True, exist_ok=True)
    n_real = len(gen["names"])
    n_train = len(gen["split"]["train"])
    real_items = []
    for i in range(n_real):
        png = real_dir / f"real_{i:02d}.png"
        rng = np.random.default_rng(i)
        Image.fromarray((rng.random((32, 32, 3)) * 255).astype("uint8")).save(png)
        real_items.append((str(png), 0, "train" if i < n_train else "test", "real"))
    # Ghi vào gốc riêng: `dire.json` là sổ ghi **một** lượt chạy cho mỗi lần gọi,
    # và có test khoá đúng điều đó (`test_dire_json_is_a_ledger_not_a_rewrite`).
    real_maps = compute_maps(f, real_items, tmp_path / "maps-real", meta, size=32)

    rows = [
        {"name": x["name"],
         "path": str((tmp_path / "maps" if x in maps["items"] else tmp_path / "maps-real") / x["path"]),
         "label": x["label"], "split": x["split"], "generator": x["generator"]}
        for x in maps["items"] + real_maps["items"]
    ]
    return tmp_path, gen, maps, rows, f


def test_gen_json_has_everything_needed_to_rebuild(tmp_run):
    _, gen, _, _, f = tmp_run
    for key in ("ckpt_sha256", "sampler", "eta", "steps", "t_max", "nfe_per_image", "seeds", "split", "created"):
        assert key in gen, key
    assert gen["sampler"] == "ddim" and gen["eta"] == 0.0
    from dire_zero.diffusion import Schedule

    assert gen["t_max"] == Schedule(steps=3).timesteps[-1]        # S=3 ⇒ 667
    assert Schedule(steps=20).timesteps[-1] == 950                # S=20 ⇒ 950
    assert gen["nfe_per_image"] == 6            # 2·S, kể cả khi S=3
    assert gen["nfe_total"] == 10 * 3           # sinh chỉ cần S lượt/ảnh
    assert len(gen["seeds"]) == 10 and gen["seeds"][0] == 100


def test_split_is_disjoint_and_covers_everything(tmp_run):
    _, gen, _, _, _ = tmp_run
    tr, te = set(gen["split"]["train"]), set(gen["split"]["test"])
    assert not (tr & te)
    assert tr | te == set(gen["names"])
    assert tr and te


def test_gen_writes_real_pngs_of_the_right_size(tmp_run):
    tmp, gen, _, _, _ = tmp_run
    first = load_image(tmp / "gen" / gen["names"][0])
    assert first.shape == (1, 3, 32, 32)
    assert float(first.min()) >= -1.0 and float(first.max()) <= 1.0


def test_dire_json_declares_scale_and_counts_nfe(tmp_run):
    _, _, maps, _, f = tmp_run
    assert maps["scale"].startswith("[0, 2]")
    assert maps["nfe_total"] == 10 * 2 * 3      # N ảnh × 2S
    assert f.nfe == 60
    assert maps["skipped"] == 0
    assert maps["steps"] == 3


def test_maps_are_float16_in_range(tmp_run):
    import numpy as np

    tmp, _, maps, _, _ = tmp_run
    arr = np.load(tmp / "maps" / maps["items"][0]["path"])
    assert arr.dtype == np.float16 and arr.shape == (3, 32, 32)
    assert float(arr.min()) >= 0.0 and float(arr.max()) <= 2.0


def test_skips_images_with_wrong_size_instead_of_crashing(tmp_run, tmp_path):
    """Ảnh sai cỡ bị **đếm** rồi bỏ, không làm sập cả lô và không im lặng."""
    from PIL import Image
    import numpy as np

    _, _, _, _, f = tmp_run
    bad = tmp_path / "bad"
    bad.mkdir()
    Image.fromarray((np.random.rand(48, 48, 3) * 255).astype("uint8")).save(bad / "x.png")
    out = compute_maps(f, [(str(bad / "x.png"), 1)], tmp_path / "maps_bad", RunMeta("a" * 64, steps=2), size=32)
    assert out["n"] == 0 and out["skipped"] == 1


def test_train_json_is_honest_about_what_it_is(tmp_run):
    tmp, _, _, rows, _ = tmp_run
    out = train(
        lambda: ResNetDetector(ResNet(blocks=(1, 1))),
        rows,
        tmp / "train",
        epochs=1,
        bs=4,
        crop=28,
        device="cpu",
        patience=99,
    )
    # Bản trước ghi cứng "R1+R2+R3" cho mọi lượt train, kể cả khi ảnh giả là của
    # tác giả (`fake_source="author"`) — tức tự nhận mức tái lập cao hơn sự thật.
    assert out["label"] == "R1+R2"
    assert out["threshold_space"] == "logit"
    assert out["init"] == "scratch"
    assert "Table 2" in out["honesty"]
    assert 0.0 <= out["threshold_from_val"] <= 1.0
    assert out["split"]["n_train"] + out["split"]["n_val"] + out["split"]["n_test"] == len(rows)
    assert (tmp / "train" / "detector.pth").exists()
    payload = json.loads((tmp / "train" / "train.json").read_text(encoding="utf-8"))
    assert payload["test"]["n"] == out["split"]["n_test"]


def test_collect_maps_reads_label_from_generator_dir(tmp_path):
    """Nhãn đến từ **tên thư mục**, không từ tham số — không thể truyền nhầm."""
    import numpy as np

    for split, gen in (("train", "real"), ("train", "adm"), ("test", "real"), ("test", "pndm")):
        d = tmp_path / split / gen
        d.mkdir(parents=True)
        np.save(d / "a.npy", np.zeros((3, 4, 4), dtype="float16"))
    rows = collect_maps(tmp_path)
    assert len(rows) == 4
    assert {(r["split"], r["generator"], r["label"]) for r in rows} == {
        ("train", "real", 0), ("train", "adm", 1), ("test", "real", 0), ("test", "pndm", 1)
    }


def test_split_is_stable_across_calls(tmp_run):
    _, gen, _, _, _ = tmp_run
    from dire_zero.data import split_indices

    a = split_indices(gen["names"], seed=3407)
    b = split_indices(list(reversed(gen["names"])), seed=3407)
    assert a == b      # thứ tự đầu vào không đổi tập chia


def test_compute_maps_moves_images_to_the_model_device(tmp_path, monkeypatch):
    """Ảnh phải được đưa sang **đúng thiết bị của mô hình**.

    Lỗi thật đã xảy ra trên Modal (A10G): `load_image` trả tensor CPU, `compute_maps`
    dùng nó trực tiếp, nên `timesteps` sinh trên CPU ⇒ `time_embedding` ra CPU ⇒
    `time_mlp` (CUDA) nổ ở `addmm` với "Expected all tensors to be on the same
    device". Test chạy trên CPU **không** bắt được, vì ở đó hai bên cùng CPU — nên
    ở đây mô hình được đặt trên `meta` (một thiết bị khác hẳn) và `dire_map` được
    thay bằng bản ghi lại thiết bị nhận được.
    """
    from PIL import Image
    import numpy as np

    import dire_zero.pipeline as P

    src = tmp_path / "src"
    src.mkdir()
    Image.fromarray((np.random.rand(32, 32, 3) * 255).astype("uint8")).save(src / "a.png")

    model = ZeroUNet(TINY).eval().to("meta")
    f = ModelCounter(model)
    assert next(f.model.parameters()).device.type == "meta"

    seen = {}

    def fake_dire_map(f_, img, sched=None, clip=True):
        seen["img_device"] = str(img.device)
        seen["sched_device"] = None if sched is None else str(sched.device)
        # trả về tensor CPU để phần lưu `.npy` vẫn chạy được
        z = torch.zeros(1, 3, 32, 32)
        return z, z

    monkeypatch.setattr(P, "dire_map", fake_dire_map)
    out = P.compute_maps(f, [(str(src / "a.png"), 1, "test", "adm")], tmp_path / "m",
                         RunMeta("a" * 64, steps=2), size=32)

    assert out["n"] == 1
    assert seen["img_device"] == "meta", "compute_maps quên `.to(device)` của mô hình"
    assert seen["sched_device"] == "meta"


def test_dire_map_normalises_device_of_the_input_image():
    """`dire_map` tự đưa ảnh về thiết bị của lịch, thay vì nổ ở giữa `time_mlp`."""
    from dire_zero.diffusion import Schedule, dire_map

    model = ZeroUNet(TINY).eval()
    f = ModelCounter(model)
    sched = Schedule(steps=2, device="cpu")
    img = torch.zeros(1, 3, 32, 32)
    dm, recon = dire_map(f, img, sched)
    assert dm.device == img.device and recon.shape == img.shape
    assert f.nfe == 4


def test_save_png_moves_the_tensor_to_cpu_before_numpy(tmp_path):
    """Lỗi thật trên Modal (A10G): `TypeError: can't convert cuda:0 device type tensor to numpy`.

    `to_uint8` giữ nguyên thiết bị của đầu vào, nên `save_png` gọi `.numpy()` trên
    tensor CUDA và chết ngay ảnh đầu tiên — sau khi đã trả tiền nạp checkpoint.
    CPU không tái hiện được lỗi, nên test này khoá **hai** thứ: (a) đường CPU vẫn
    ghi đúng, (b) mã nguồn có `.cpu()` **trước** `.numpy()` trong chuỗi gọi.

    Cách (b) hơi thô, nhưng đây là lỗi mà test chức năng không thể bắt nếu không
    có GPU: kiểm thứ tự trong chuỗi là cách duy nhất còn lại. Cùng lý do với
    `test_compute_maps_moves_images_to_the_model_device` dùng thiết bị `meta`.
    """
    from PIL import Image
    import numpy as np

    from dire_zero.pipeline import save_png, to_hwc_uint8

    img = torch.full((1, 3, 8, 8), -1.0)
    img[:, :, :4, :] = 1.0          # nửa trên trắng, nửa dưới đen
    out = tmp_path / "sub" / "a.png"
    save_png(img, out)

    with Image.open(out) as im:
        arr = np.asarray(im.convert("RGB"))
    assert arr.shape == (8, 8, 3)
    assert arr[0, 0].tolist() == [255, 255, 255]    # +1 → 255
    assert arr[7, 0].tolist() == [0, 0, 0]          # −1 → 0

    src = inspect.getsource(to_hwc_uint8)
    body = src.split('"""')[-1] if '"""' in src else src
    assert ".cpu()" in body and ".numpy()" in body
    assert body.index(".cpu()") < body.index(".numpy()"), \
        "phải `.cpu()` TRƯỚC `.numpy()`; đảo lại là crash trên CUDA"
    assert ".detach()" in body


def test_save_png_accepts_a_bare_channel_image_and_one_channel_maps(tmp_path):
    from PIL import Image

    from dire_zero.pipeline import save_png

    save_png(torch.zeros(3, 4, 4), tmp_path / "bare.png")
    save_png(torch.zeros(1, 1, 4, 4), tmp_path / "gray.png")
    with Image.open(tmp_path / "bare.png") as im:
        assert im.size == (4, 4) and im.mode == "RGB"
    with Image.open(tmp_path / "gray.png") as im:
        assert im.size == (4, 4) and im.mode == "L"


def test_save_png_refuses_a_batch_instead_of_writing_the_first_image(tmp_path):
    from dire_zero.pipeline import save_png

    with pytest.raises(ValueError, match="batch"):
        save_png(torch.zeros(2, 3, 4, 4), tmp_path / "x.png")
    with pytest.raises(ValueError, match="tensor"):
        save_png(torch.zeros(4, 4), tmp_path / "y.png")


def test_no_module_converts_a_device_tensor_without_cpu():
    """Quét cả `src/` theo **từng hàm**: `.numpy()` phải có `.cpu()` đứng trước.

    Đây là bài kiểm bắt **cả lớp lỗi**, không chỉ một chỗ: bất kỳ ai viết lại
    `to_uint8(x).permute(...).numpy()` ở một module khác cũng sẽ chết y như vậy
    trên GPU, và trên laptop thì không thấy gì.

    Quét theo hàm chứ theo dòng, vì cách viết đúng thường tách `.cpu()` ra một
    dòng riêng:

    ::

        u8 = to_uint8(arr).detach().cpu()   # dòng 1
        hwc = u8.permute(1, 2, 0).numpy()   # dòng 2 — vẫn đúng

    Điều kiện chỉ áp cho hàm *có* chuyển ảnh sang numpy sau `permute`/`to_uint8`;
    các chỗ khác (ví dụ `load_image` đọc ảnh đã ở CPU) không bị đụng.
    """
    import ast

    root = Path(__file__).resolve().parent.parent / "src"
    bad = []
    for py in root.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        lines = text.splitlines()
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body_start = node.lineno
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                # Bỏ docstring: nó nhắc tới `.numpy()`/`.cpu()` bằng lời, và lời
                # không phải là thứ tự thực thi.
                body_start = node.body[0].end_lineno + 1
            seg = "\n".join(lines[body_start - 1: node.end_lineno])
            if ".numpy()" not in seg:
                continue
            if ".permute(" not in seg and "to_uint8(" not in seg:
                continue
            if ".cpu()" not in seg or seg.index(".cpu()") > seg.index(".numpy()"):
                bad.append(f"{py.relative_to(root.parent)}:{node.lineno} {node.name}()")
    assert not bad, "hàm chuyển tensor sang numpy mà thiếu `.cpu()` trước đó:\n" + "\n".join(bad)


# ---------------------------------------------------------------------------
# Tính thêm map về sau: không trả tiền hai lần cho cùng một ảnh
# ---------------------------------------------------------------------------
def test_skip_existing_reuses_maps_instead_of_paying_twice(tmp_run, monkeypatch):
    """Chạy lại chỉ để thêm một generator mới **không** được tính lại ảnh `real`.

    Tình huống thật: đã có map cho `ours_adm` lô 1; sinh thêm ảnh (lô 2) rồi
    chạy `::dire --skip-existing`. Ảnh `real` nằm trong danh sách ở **mọi** lần
    chạy, nên không có cờ này thì mỗi lần thêm generator là một lần trả tiền cho
    toàn bộ `real` — rồi ghi đè đúng file vừa trả tiền để tạo.
    """
    tmp, gen, maps, _, f = tmp_run
    items = [(str(tmp / "gen" / n), 1, "train" if n in gen["split"]["train"] else "test", "ours_adm")
             for n in gen["names"]]

    import dire_zero.pipeline as P

    def boom(*a, **k):
        raise AssertionError("cờ skip_existing mà vẫn gọi dire_map")

    monkeypatch.setattr(P, "dire_map", boom)
    again = compute_maps(f, items, tmp / "maps", RunMeta("a" * 64, steps=3), size=32,
                         skip_existing=True)
    assert again["n_new"] == 0 and again["n_reused"] == len(items)
    assert again["nfe_total"] == 0               # không một lượt forward nào


def test_without_the_flag_it_recomputes_everything(tmp_run):
    tmp, gen, _, _, f = tmp_run
    items = [(str(tmp / "gen" / n), 1, "train", "ours_adm") for n in gen["names"]]
    out = compute_maps(f, items, tmp / "maps", RunMeta("a" * 64, steps=3), size=32)
    assert out["n_new"] == len(items) and out["n_reused"] == 0
    assert out["nfe_total"] == len(items) * 2 * 3


def test_dire_json_is_a_ledger_not_a_rewrite(tmp_run):
    """Lần chạy thứ hai ghi **thêm** vào sổ, không xoá dấu lần đầu.

    Lỗi thật: `compute_maps` ghi đè `maps/dire.json` mỗi lần, nên bản ghi của lần
    đầu (số ảnh, NFE, số giây) mất trong khi các `.npy` của lần đó vẫn còn — một
    cây map không còn tài liệu nào giải thích. Nếu hai lần chạy khác `steps` thì
    `steps` cấp ngoài phải là **null** kèm `steps_mixed`, không lấy bừa một giá trị.
    """
    tmp, gen, maps, _, f = tmp_run
    index = json.loads((tmp / "maps" / "dire.json").read_text(encoding="utf-8"))
    assert [r["steps"] for r in index["runs"]] == [3]
    first_n = index["n"]

    other = [(str(tmp / "gen" / gen["names"][0]), 1, "test", "adm")]
    second = compute_maps(f, other, tmp / "maps", RunMeta("a" * 64, steps=5), size=32)

    index = json.loads((tmp / "maps" / "dire.json").read_text(encoding="utf-8"))
    assert [r["steps"] for r in index["runs"]] == [3, 5]
    assert index["steps"] is None and index["steps_mixed"] == [3, 5]
    assert index["nfe_total"] == maps["nfe_total"] + second["nfe_total"], "sổ phải cộng dồn NFE"
    assert index["n"] >= first_n                       # không mất item của lần đầu
    assert (tmp / "maps" / "dire-runs").is_dir()
    assert len(list((tmp / "maps" / "dire-runs").glob("*.json"))) == 2
    assert index["generators"] == ["adm", "ours_adm"]


def test_reused_items_are_marked_so_they_cannot_masquerade_as_fresh(tmp_run):
    """Item dùng lại mang cờ `reused`; `n_new`/`n_reused` nói rõ cái nào là mới."""
    tmp, gen, _, _, f = tmp_run
    items = [(str(tmp / "gen" / n), 1, "train", "ours_adm") for n in gen["names"][:3]] + [
        (str(tmp / "gen" / gen["names"][3]), 1, "test", "adm")
    ]
    out = compute_maps(f, items, tmp / "maps", RunMeta("a" * 64, steps=3), size=32,
                       skip_existing=True)
    reused = [it for it in out["items"] if it.get("reused")]
    assert len(reused) == 3 and out["n_new"] == 1 and out["n_reused"] == 3
    assert out["n"] == 4


def test_train_label_comes_from_the_caller_not_a_constant(tmp_path, tmp_run):
    """Nhãn mức tái lập phải theo dữ liệu thật, không theo một hằng số trong code."""
    from dire_zero.pipeline import train

    from dire_zero.detector import ResNetDetector, ResNet

    tmp, _, _, rows, _ = tmp_run
    out = train(
        lambda: ResNetDetector(ResNet(blocks=(1, 1))),
        rows,
        tmp / "train-labelled",
        epochs=1,
        bs=4,
        crop=28,
        device="cpu",
        patience=99,
        label="R1+R2",
    )
    assert out["label"] == "R1+R2"


# ---------------------------------------------------------------------------
# Chặn "kết quả" giả: train trên tập một lớp
# ---------------------------------------------------------------------------
def test_train_refuses_a_single_class_train_split(tmp_path):
    """`acc = 0,500` tuyệt đối không phải kết quả — nó là hàm hằng số.

    Đây là lỗi đọc được từ log thật: 5 generator chỉ có map ở `test` (dalle2,
    ddpm, diff-projectedgan, diff-stylegan, if) cho ra `acc = 0,5` với `AUC
    0,21–0,45`, và bảng vẫn in chúng như những dòng Table 2. Nguyên nhân: tập
    `train` của chúng chỉ còn ảnh `real`, nên mô hình học "mọi thứ đều thật".
    """
    from dire_zero.detector import ResNet, ResNetDetector
    from dire_zero.pipeline import SplitError, train

    maps = tmp_path / "maps"
    rows = []
    for i in range(12):                       # chỉ có `real` ở train, fake ở test
        for split, label in (("train", 0), ("test", 0), ("test", 1)):
            d = maps / split / ("real" if label == 0 else "adm")
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"{i}_{split}_{label}.npy"
            p.write_bytes(b"x")
            rows.append({"name": p.name, "path": str(p), "label": label,
                         "split": split, "generator": "real" if label == 0 else "adm"})

    with pytest.raises(SplitError) as exc:
        train(lambda: ResNetDetector(ResNet(blocks=(1, 1))), rows, tmp_path / "out",
              epochs=1, bs=4, crop=28, device="cpu")
    msg = str(exc.value)
    assert "train" in msg and "0,5" in msg
    assert "split-mode hash" in msg             # có đường thoát, không chỉ từ chối


def test_hash_split_mode_accepts_test_only_generators_and_says_so(tmp_path):
    """`--split-mode hash` chia lại mọi map: dùng được, nhưng phải khai ra.

    Cây DiffusionForensics chỉ có `test` cho phần lớn generator, nên nếu không có
    chế độ này thì 10/15 dòng của Table 2 **không tồn tại** — không phải vì DIRE
    kém, mà vì không có ảnh để train. Chế độ này mua lại các dòng đó bằng một
    tuyên bố trung thực: `split_mode: hash`, không so trực tiếp với bài báo.
    """
    from dire_zero.detector import ResNet, ResNetDetector
    from dire_zero.pipeline import train

    import numpy as np

    maps = tmp_path / "maps"
    rows = []
    for label, gen in ((0, "real"), (1, "adm")):
        for i in range(40):
            d = maps / "test" / gen
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"{label}_{i}.npy"
            np.save(p, np.zeros((1, 32, 32), dtype="float16"))
            rows.append({"name": p.name, "path": str(p), "label": label,
                         "split": "test", "generator": gen})

    out = train(lambda: ResNetDetector(ResNet(blocks=(1, 1))), rows, tmp_path / "out",
                epochs=1, bs=4, crop=28, device="cpu", split_mode="hash", patience=99)
    assert out["split_mode"] == "hash"
    assert out["split"]["mode"] == "hash"
    assert out["split"]["n_train"] and out["split"]["n_test"]
    assert "HASH" in out["honesty"]
