"""Chạy trọn bốn bước trên mô hình tí hon — bài kiểm "pipeline có nối được không".

Cái được kiểm không phải chất lượng (mô hình ngẫu nhiên 32×32, 8 ảnh), mà là
**hợp đồng giữa các bước**:

* `gen.json` có đủ trường để người khác dựng lại (seed, bước, sha256, split);
* `dire.json` khai đúng thang `[0, 2]` và NFE đếm được;
* `train.json` tự khai nhãn `R1+R2+R3`, chọn ngưỡng trên val, và test chỉ chấm
  một lần;
* split train/test **rời nhau tuyệt đối** (không một tên file chung).

Nếu bốn điều này sai, con số đẹp nhất cũng vô nghĩa. Và chúng sai được một cách
im lặng — không exception, không cảnh báo.
"""

from __future__ import annotations

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
    items = [(str(tmp_path / "gen" / n), 1) for n in gen["names"]]
    maps = compute_maps(f, items, tmp_path / "maps", meta, size=32)
    rows = [{"name": x["name"], "path": str(tmp_path / "maps" / x["name"]), "label": x["label"]} for x in maps["items"]]
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
    arr = np.load(tmp / "maps" / maps["items"][0]["name"])
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
    assert out["label"] == "R1+R2+R3"
    assert out["init"] == "scratch"
    assert "Table 2" in out["honesty"]
    assert 0.0 <= out["threshold_from_val"] <= 1.0
    assert out["split"]["n_train"] + out["split"]["n_val"] + out["split"]["n_test"] == len(rows)
    assert (tmp / "train" / "detector.pth").exists()
    payload = json.loads((tmp / "train" / "train.json").read_text(encoding="utf-8"))
    assert payload["test"]["n"] == out["split"]["n_test"]


def test_collect_maps_labels_by_directory(tmp_path):
    import numpy as np

    for sub, label in (("real", 0), ("fake", 1)):
        d = tmp_path / sub
        d.mkdir()
        np.save(d / "a.npy", np.zeros((3, 4, 4), dtype="float16"))
    rows = collect_maps([tmp_path / "real", tmp_path / "fake"], [0, 1])
    assert {r["label"] for r in rows} == {0, 1}
    assert len(rows) == 2


def test_split_is_stable_across_calls(tmp_run):
    _, gen, _, _, _ = tmp_run
    from dire_zero.data import split_indices

    a = split_indices(gen["names"], seed=3407)
    b = split_indices(list(reversed(gen["names"])), seed=3407)
    assert a == b      # thứ tự đầu vào không đổi tập chia
