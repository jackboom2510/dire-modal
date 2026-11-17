"""U-Net + bảng ánh xạ checkpoint — phần quyết định "nạp được trọng số hay không".

Ba lớp kiểm, từ rẻ tới đắt:

1. **Đếm tham số** — 552.814.086, đúng con số của ADM 256 vô điều kiện. Lệch
   một khối là lệch vài chục triệu, thấy ngay.
2. **Song ánh 566 key** với fixture sinh từ kiến trúc tham chiếu. Đây là test
   bắt được lỗi "đánh số block lệch một" — loại lỗi mà `strict=True` chỉ phát
   hiện khi đã có checkpoint 2,5 GB trên GPU.
3. **Parity số** với bản tham chiếu, nếu máy có nó (tự skip nếu không): nạp
   cùng trọng số, so đầu ra từng phần tử. Bắt lỗi *ngữ nghĩa* (thứ tự scale/shift,
   chia head) mà hai lớp trên không thấy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from dire_zero.checkpoint import check_bijection, iter_name_pairs, map_state_dict  # noqa: E402
from dire_zero.blocks import time_embedding  # noqa: E402
from dire_zero.unet import ADM_LSUN, UNetConfig, ZeroUNet  # noqa: E402

FIXTURE = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "adm_lsun_state_keys.json").read_text(encoding="utf-8")
)

TINY = UNetConfig(
    image_size=32, model_channels=32, num_res_blocks=1, attention_ds=(4,),
    channel_mult=(1, 2), num_heads=2, head_channels=-1,
)


def test_full_model_param_count_matches_adm():
    with torch.device("meta"):
        model = ZeroUNet()
    assert model.n_params == 552_814_086
    assert len(model.state_dict()) == 566
    assert FIXTURE["_n_params"] == model.n_params


def test_fixture_is_spec_not_weights():
    """Fixture chỉ có tên + shape. Không một byte trọng số nào của OpenAI trong repo."""
    assert set(FIXTURE) == {"_what", "_config", "_n_params", "_n_tensors", "keys"}
    assert all(isinstance(v, list) for v in FIXTURE["keys"].values())
    assert sum(1 for v in FIXTURE["keys"].values() if any(isinstance(x, int) for x in v)) > 0


def test_bijection_over_all_566_keys():
    with torch.device("meta"):
        model = ZeroUNet()
    mapping = check_bijection(model, FIXTURE["keys"])
    assert len(mapping) == 566
    assert len(set(mapping.values())) == 566


def test_bijection_catches_a_missing_block():
    """Bỏ một khối khỏi bảng ⇒ test phải đỏ, không được im lặng."""
    with torch.device("meta"):
        model = ZeroUNet()
    broken = dict(FIXTURE["keys"])
    broken.pop("middle_block.1.proj_out.weight")
    with pytest.raises(AssertionError):
        check_bijection(model, broken)


def test_bijection_catches_wrong_shape():
    with torch.device("meta"):
        model = ZeroUNet()
    broken = {k: list(v) for k, v in FIXTURE["keys"].items()}
    broken["out.2.weight"] = [3, 256, 3, 3]      # sai số kênh ra
    with pytest.raises(AssertionError):
        check_bijection(model, broken)


def test_mapping_rejects_unknown_checkpoint_key():
    with torch.device("meta"):
        model = ZeroUNet()
    bad = dict(model.state_dict())
    bad["enc.0.0.res.conv_in.weight"] = bad.pop("enc.0.0.res.conv_in.weight")
    with pytest.raises(AssertionError):
        map_state_dict({**{r: None for r in dict(iter_name_pairs(model)).values()}, "không.tồn.tại": None}, model)


def test_tiny_model_roundtrip_shape_and_no_nan():
    torch.manual_seed(0)
    model = ZeroUNet(TINY).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 32, 32), torch.tensor([0, 950]))
    assert out.shape == (2, 6, 32, 32)
    assert torch.isfinite(out).all()


def test_time_embedding_is_float32_and_time_mlp_takes_half():
    """Nhúng sin/cos **phải** ở float32: ở fp16, `f_i·t` với t tới 1000 mất ~3 chữ số.

    Nhưng đầu ra float32 đó không được đi thẳng vào `time_mlp` fp16 — đó chính là
    lỗi `mat1 and mat2 must have the same dtype` gặp trên Modal.
    """
    model = ZeroUNet(TINY).half()
    e = time_embedding(torch.tensor([0, 950]), TINY.model_channels)
    assert e.dtype == torch.float32
    assert model.time_mlp(e.to(torch.float16)).dtype == torch.float16


def test_forward_in_half_matches_dtype_of_input():
    """Đường fp16 — đường duy nhất chạy trên GPU (`dtype="auto"`), và là đường
    mà mọi test CPU float32 không chạm tới.

    Lỗi thật, nguyên văn từ Modal::

        RuntimeError: mat1 and mat2 must have the same dtype, but got Float and Half
    """
    model = ZeroUNet(TINY).eval().half()
    x = torch.randn(1, 3, 32, 32)              # cố tình float32, như `selfcheck`
    try:
        with torch.inference_mode():
            out = model(x, torch.tensor([10]))
    except RuntimeError as exc:
        # Chỉ bỏ qua khi *phép toán* fp16 không có trên CPU; lỗi dtype thì phải nổ.
        if "not implemented for 'Half'" in str(exc) or "Could not run" in str(exc):
            pytest.skip(f"CPU này không chạy conv fp16: {exc}")
        raise
    assert out.dtype == torch.float32           # trả về dtype của đầu vào, không phải của trọng số
    assert torch.isfinite(out.float()).all()


def test_resblock_downsample_is_parameterless():
    """ADM giảm mẫu bằng average-pool, không bằng conv stride 2 — đếm tham số là
    cách duy nhất thấy điều đó mà không cần đọc code người khác."""
    with torch.device("meta"):
        model = ZeroUNet(TINY)
    down_entry = model.enc[0][TINY.num_res_blocks]
    assert down_entry.solo
    names = [n for n, _ in down_entry.named_parameters()]
    assert not any("resample" in n for n in names)


def test_lsuns_config_is_the_released_one():
    assert ADM_LSUN.channel_mult == (1, 1, 2, 2, 4, 4)
    assert ADM_LSUN.attention_ds == (8, 16, 32)
    assert ADM_LSUN.out_channels == 6
    with pytest.raises(ValueError):
        UNetConfig(model_channels=48)          # không chia hết 32
    with pytest.raises(NotImplementedError):
        UNetConfig(resblock_updown=False)      # ngoài phạm vi bản này


def test_parity_with_reference_if_available():
    """Nạp **cùng trọng số** vào hai mô hình rồi so đầu ra.

    Chỉ chạy khi có bản tham chiếu (đặt `DIRE_REFERENCE_UNET=/đường/dẫn/tới/
    guided-diffusion-main`). Trên CI thì skip — nhưng đây là test đã được chạy
    thật khi phát triển, và nó bắt được lỗi thứ tự attention/upsample.
    """
    import os
    import sys

    root = os.environ.get("DIRE_REFERENCE_UNET")
    if not root or not Path(root).is_dir():
        pytest.skip("không có bản tham chiếu")
    sys.path.insert(0, root)
    from guided_diffusion.unet import UNetModel  # noqa: PLC0415

    torch.manual_seed(0)
    ref = UNetModel(
        image_size=32, in_channels=3, model_channels=32, out_channels=6,
        num_res_blocks=1, attention_resolutions=(4,), dropout=0, channel_mult=(1, 2),
        num_heads=2, num_head_channels=-1, use_scale_shift_norm=True,
        resblock_updown=True, conv_resample=True, dims=2, num_classes=None,
        use_checkpoint=False, use_fp16=False,
    )
    with torch.no_grad():
        for p in ref.parameters():          # init mặc định có conv ra = 0 ⇒ đổi hết
            p.normal_(0, 0.02)
    mine = ZeroUNet(TINY)
    mine.load_state_dict(map_state_dict(ref.state_dict(), mine), strict=True)
    mine.eval()
    ref.eval()
    x = torch.randn(2, 3, 32, 32)
    t = torch.tensor([3, 900])
    with torch.no_grad():
        a, b = ref(x, t), mine(x, t)
    assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()
