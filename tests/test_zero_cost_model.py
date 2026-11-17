"""Mô hình chi phí có kiểm: 829 GFLOP/forward, 9 TFLOPS hiệu dụng, sàn 14 phút.

Vì sao cần test này
-------------------
"`::dire` mất ~49 phút cho 800 ảnh" là một câu **kiểm được**, nhưng trước đây nó
chỉ dựa vào một con số `sec/step` không ai đối chiếu. Test này đối chiếu nó bằng
vật lý: đếm FLOP của đúng kiến trúc trong `dire_zero.unet`, rồi hỏi xem
`sec/step` đo được có nằm trong khoảng một GPU thật có thể làm hay không.

Ba thứ được khoá:

1. **Kiến trúc ↔ chi phí.** `ADM_LSUN` cho 552,8 M tham số và ~415 GMAC/forward.
   Nếu ai đổi `channel_mult`, `num_res_blocks` hay `attention_ds`, con số này đổi
   và test đỏ — đó là lúc phải **cố ý** cập nhật mọi bảng chi phí, không phải để
   chúng lặng lẽ sai.
2. **Số đo phải khả dĩ.** 829 GFLOP / 0,0917 s = 9,0 TFLOPS = 29 % đỉnh fp16
   A10G. Test đòi hiệu suất nằm trong [5 %, 60 %]: dưới 5 % là code chậm bất
   thường (hoặc số đo sai), trên 60 % là bất khả ở batch 1.
3. **Sàn thời gian.** 26,5 PFLOP ở 100 % đỉnh vẫn là ~14 phút. Nên mọi con số
   dưới 14 phút cho 800 ảnh trên A10G đều **sai** — kể cả khi ai đó muốn tin.

Đo FLOP trên thiết bị `meta`: không cần GPU, không cần checkpoint, không cấp phát
bộ nhớ — chỉ có shape đi qua các lớp.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from dire_zero.blocks import Attention  # noqa: E402
from dire_zero.budget import Estimate  # noqa: E402
from dire_zero.unet import ADM_LSUN, ZeroUNet  # noqa: E402

#: Đỉnh fp16 (tensor core, dense) của A10G, TFLOPS — thông số nhà sản xuất.
A10G_FP16_PEAK_TFLOPS = 31.2
#: Số đo của `::bench` (A10G, fp16, batch 1), `sec/step`.
SEC_PER_STEP = 0.0917
#: BNhững gì tài liệu công bố, để test canh chúng khỏi trôi.
DOC_GMAC = 415.0
DOC_PARAMS_M = 552.8


def macs_per_forward(cfg=ADM_LSUN) -> tuple[float, float, int, float]:
    """`(GMAC, GFLOP, params, GMAC của chú ý)` của một lượt forward."""
    with torch.device("meta"):
        model = ZeroUNet(cfg).eval()
    params = sum(p.numel() for p in model.parameters())

    macs = 0
    attn_heads: list[tuple[int, int]] = []

    def conv_hook(m: nn.Module, _inp, out) -> None:
        nonlocal macs
        k = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
        spatial = out.numel() // (out.shape[0] * out.shape[1])
        macs += k * (m.in_channels // m.groups) * m.out_channels * spatial

    def linear_hook(m: nn.Module, _inp, out) -> None:
        nonlocal macs
        macs += m.in_features * m.out_features * (out.numel() // m.out_features)

    def attn_hook(m: Attention, inp, _out) -> None:
        x = inp[0]
        t = math.prod(x.shape[2:])
        attn_heads.append((t, x.shape[1] // m.num_heads))

    for mod in model.modules():
        if isinstance(mod, (nn.Conv2d, nn.Conv1d)):
            mod.register_forward_hook(conv_hook)
        elif isinstance(mod, nn.Linear):
            mod.register_forward_hook(linear_hook)
        elif isinstance(mod, Attention):
            mod.register_forward_hook(attn_hook)

    s = cfg.image_size
    model(torch.zeros(1, 3, s, s, device="meta"), torch.zeros(1, dtype=torch.long, device="meta"))

    # q·k và w·v: mỗi cái `t·t·ch` MAC cho mỗi head
    attn_macs = sum(2 * t * t * ch for t, ch in attn_heads)
    macs += attn_macs
    return macs / 1e9, 2 * macs / 1e9, params, attn_macs / 1e9


def test_architecture_costs_what_the_docs_say() -> None:
    gmac, gflop, params, _ = macs_per_forward()
    assert params / 1e6 == pytest.approx(DOC_PARAMS_M, rel=0.01)
    assert gmac == pytest.approx(DOC_GMAC, rel=0.02), f"{gmac:.1f} GMAC"
    assert gflop == pytest.approx(2 * DOC_GMAC, rel=0.02)


def test_conv_dominates_attention_not_the_other_way_round() -> None:
    """Trực giác hay sai: 16 khối chú ý nghe "đắt", nhưng conv chiếm ~98 % FLOP.

    Đây là lý do đổi `attention_ds` **không** phải đòn tối ưu, còn batching thì
    có: phần đắt là conv, và conv ở batch 8 dùng tensor core tốt hơn hẳn batch 1.
    """
    gmac, _, _, attn_gmac = macs_per_forward()
    assert attn_gmac < 0.03 * gmac, f"chú ý {attn_gmac:.1f} GMAC / tổng {gmac:.1f} GMAC"


def test_measured_step_time_is_physically_plausible() -> None:
    _, gflop, _, _ = macs_per_forward()          # GFLOP / forward
    tflops = gflop / SEC_PER_STEP / 1000.0       # GFLOP/s → TFLOPS
    frac = tflops / A10G_FP16_PEAK_TFLOPS
    assert 0.05 < frac < 0.60, (
        f"{tflops:.1f} TFLOPS = {frac:.0%} đỉnh A10G. Dưới 5 % ⇒ code chậm bất thường "
        "hoặc số đo hỏng; trên 60 % ⇒ bất khả ở batch 1. Kiểm lại `::bench`."
    )


def test_fifty_minutes_is_arithmetic_not_pessimism() -> None:
    """800 ảnh × 40 NFE × 0,0917 s = 2 934 s, và sàn vật lý là ~14 phút."""
    est = Estimate(n_images=800, nfe_per_image=40, sec_per_step=SEC_PER_STEP)
    assert est.gpu_h * 3600 == pytest.approx(2934, abs=1.0)
    assert est.gpu_h * 60 == pytest.approx(48.9, abs=0.2)

    _, gflop, _, _ = macs_per_forward()          # GFLOP / forward
    total_flop = gflop * 1e9 * est.nfe_total     # FLOP
    floor_s = total_flop / (A10G_FP16_PEAK_TFLOPS * 1e12)
    assert floor_s == pytest.approx(851, rel=0.03)
    # Không một phép tối ưu nào trên A10G batch 1 xuống dưới sàn này.
    assert floor_s / 60 > 13


def test_a100_is_cheaper_per_image_than_a10g_at_the_same_efficiency() -> None:
    """Suy luận trong `docs/scratch-reproduction.md` §4.2: nhanh 10×, đắt 2,3× ⇒ rẻ 4×."""
    a10 = Estimate(n_images=800, nfe_per_image=40, sec_per_step=SEC_PER_STEP, gpu="A10G")
    # A100-40GB: đỉnh fp16 312 TFLOPS ≈ 10× A10G ⇒ cùng hiệu suất, sec/step giảm 10×
    a100 = Estimate(n_images=800, nfe_per_image=40, sec_per_step=SEC_PER_STEP / 10, gpu="A100-40GB")
    assert a100.usd < 0.30 and a10.usd == pytest.approx(0.90, abs=0.02)
    assert a100.usd < a10.usd / 3
