"""ResNet-50 tự viết — hình dạng, bảng tên ImageNet, và một lần học thật.

Test cuối (`test_it_can_learn`) là thứ chứng minh detector **không** hỏng ở mức
cơ bản: cho nó 20 map tổng hợp có tín hiệu rõ, 3 epoch, và đòi ACC > 0,8. Mô
hình hỏng (nhãn đảo, mất nhánh dư, chuẩn hoá sai) sẽ trượt bài này trong khi
mọi test hình dạng vẫn xanh.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dire_zero.detector import (  # noqa: E402
    AUTHOR_DETECTOR_PARAMS,
    Bottleneck,
    ResNet,
    ResNetDetector,
    load_dire_detector,
)


def test_param_count_is_resnet50_with_one_logit():
    model = ResNetDetector()
    # ResNet-50 ImageNet = 25,56 M; thay fc 1000 lớp bằng 1 logit ⇒ 23,51 M
    assert 23.4e6 < model.n_params < 23.6e6


def test_forward_shapes():
    model = ResNetDetector()
    x = torch.randn(2, 3, 224, 224)
    assert model.logits(x).shape == (2,)
    assert model(x).shape == (2,)
    assert ((model(x) >= 0) & (model(x) <= 1)).all()


def test_normalisation_is_imagenet():
    model = ResNetDetector()
    assert model.mean.flatten().tolist() == pytest.approx([0.485, 0.456, 0.406])
    assert model.std.flatten().tolist() == pytest.approx([0.229, 0.224, 0.225])


def test_rgb_dire_detector_has_six_channels_and_repeated_imagenet_normalisation():
    model = ResNetDetector(in_channels=6)
    assert model.backbone.stem[0].weight.shape[1] == 6
    assert model.mean.shape[1] == model.std.shape[1] == 6
    assert model.mean.flatten().tolist() == pytest.approx([0.485, 0.456, 0.406] * 2)
    assert model.logits(torch.randn(1, 6, 64, 64)).shape == (1,)


def test_imagenet_rgb_stem_initializes_both_rgb_dire_branches():
    model = ResNet(blocks=(1, 1), widths=(16, 32, 32, 32), in_channels=6)
    rgb_weight = torch.arange(64 * 3 * 7 * 7, dtype=torch.float32).reshape(64, 3, 7, 7)
    skipped = model.load_imagenet({"conv1.weight": rgb_weight})
    got = model.stem[0].weight
    assert "stem.0.weight" not in skipped
    assert torch.equal(got[:, :3], rgb_weight * 0.5)
    assert torch.equal(got[:, 3:], rgb_weight * 0.5)


def test_imagenet_alias_roundtrip():
    """Đổi tên sang kiểu torchvision rồi nạp lại ⇒ tham số không đổi.

    Bảng ánh xạ này là thứ dễ sai âm thầm nhất (16 khối phẳng ↔ 4 layer), và
    test vòng tròn là cách duy nhất kiểm nó mà không cần tải 100 MB ImageNet.
    """
    model = ResNet()
    ref = {}
    for my, theirs in model.imagenet_alias().items():
        for suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
            key = f"{my}.{suffix}"
            if key in model.state_dict():
                ref[f"{theirs}.{suffix}"] = model.state_dict()[key].clone()
    before = {k: v.clone() for k, v in model.state_dict().items()}

    fresh = ResNet()
    torch.nn.init.normal_(fresh.fc.weight)               # đổi để thấy nó bị thay
    skipped = fresh.load_imagenet(ref)
    assert len(ref) > 100, "bảng alias không sinh ra đủ leaf"
    assert "fc.weight" in skipped and "fc.bias" in skipped
    for key, val in before.items():
        if key.startswith("fc."):
            continue
        assert torch.equal(fresh.state_dict()[key], val), key


def test_downsample_only_where_needed():
    block = Bottleneck(64, 64, stride=2)
    assert isinstance(block.downsample, torch.nn.Sequential)
    block2 = Bottleneck(256, 64, stride=1)
    assert isinstance(block2.downsample, torch.nn.Identity)


def test_it_can_learn():
    """3 epoch trên 20 map tổng hợp: map của lớp 1 có vệt sáng, lớp 0 thì không."""
    torch.manual_seed(0)
    import numpy as np
    import tempfile
    from pathlib import Path

    from dire_zero.pipeline import train

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rows = []
        for i in range(20):
            label = i % 2
            arr = np.zeros((3, 64, 64), dtype="float32")
            if label:
                arr[:, 20:44, 20:44] = 0.8
            p = tmp / f"{i:03d}.npy"
            np.save(p, arr.astype("float16"))
            rows.append({"name": p.name, "path": str(p), "label": label})
        out = train(
            lambda: ResNetDetector(ResNet(blocks=(1, 1), widths=(16, 32, 32, 32))),
            rows, tmp / "out", epochs=8, bs=4, lr=1e-3, crop=48, device="cpu", patience=99,
        )
    # Bài học từ chính test này: với mô hình còn non, điểm sigmoid còn rất thấp
    # (≈0,02 cho cả hai lớp) nên ACC@0,5 = 0,5 trong khi AP/AUC = 1,0. Đó là vấn
    # đề **hiệu chỉnh**, không phải phân biệt — báo cả hai, đừng chọn một.
    assert out["test"]["ap"] >= 0.8
    assert out["test"]["auc"] >= 0.8
    assert "acc_at_0.5" in out["test"]
    assert out["best_epoch"] >= 1


# --------------------------------------------------------------------------- #
# R0: nạp detector **của tác giả**
# --------------------------------------------------------------------------- #

def _author_state(model: ResNetDetector) -> dict:
    """Đổi state của mình sang **torchvision naming** như file `.pth` của tác giả."""
    table = model.backbone.imagenet_alias()
    own, out = model.backbone.state_dict(), {}
    for my_key, ref in table.items():
        for suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
            if f"{my_key}.{suffix}" in own:
                out[f"{ref}.{suffix}"] = own[f"{my_key}.{suffix}"]
    out["fc.weight"] = own["fc.weight"]
    out["fc.bias"] = own["fc.bias"]
    return out


def test_author_param_count_matches_the_real_file():
    """File thật: 23.563.254 `numel`, trong đó 53.173 là buffer BN ⇒ 23.510.081 tham số."""
    assert AUTHOR_DETECTOR_PARAMS == 23_510_081
    assert 23_563_254 - AUTHOR_DETECTOR_PARAMS == 53_173     # 2×26.560 + 53 num_batches_tracked
    assert ResNetDetector().n_params == AUTHOR_DETECTOR_PARAMS


def test_load_dire_detector_roundtrip(tmp_path):
    """`fc` **phải** được nạp: bỏ nó thì 23,5 M tham số đúng mà đầu ra vẫn ngẫu nhiên."""
    src = ResNetDetector()
    with torch.no_grad():
        src.backbone.fc.weight.fill_(0.25)
        src.backbone.fc.bias.fill_(-0.75)
    path = tmp_path / "lsun_adm.pth"
    torch.save(_author_state(src), path)

    got = load_dire_detector(path)
    assert torch.equal(got.backbone.fc.weight, src.backbone.fc.weight)
    assert torch.equal(got.backbone.fc.bias, src.backbone.fc.bias)
    assert got.n_params == AUTHOR_DETECTOR_PARAMS


def test_load_dire_detector_rejects_imagenet_1000_classes(tmp_path):
    state = _author_state(ResNetDetector())
    state["fc.weight"] = torch.zeros(1000, 2048)
    state["fc.bias"] = torch.zeros(1000)
    path = tmp_path / "imagenet.pth"
    torch.save(state, path)
    with pytest.raises(ValueError, match="1000"):
        load_dire_detector(path)


def test_load_dire_detector_rejects_a_non_resnet_file(tmp_path):
    path = tmp_path / "adm.pth"
    torch.save({"input_blocks.0.0.weight": torch.zeros(3, 3, 3, 3)}, path)
    with pytest.raises(KeyError, match="conv1.weight"):
        load_dire_detector(path)
