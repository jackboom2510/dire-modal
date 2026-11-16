"""Khối dựng của U-Net ADM — viết lại, không chép.

Ba khối, đúng ba khối bài báo cần: `ResBlock` (tích chập dư + điều kiện thời
điểm kiểu FiLM), `Attention` (tự chú ý trên vị trí không gian), và tích chập
xuống/lên mẫu.

Hai chi tiết "nhỏ" quyết định trọng số có nạp được hay không, nên chúng được
viết rõ và có test:

* **GroupNorm chạy ở float32 rồi trả về dtype cũ.** ADM chạy fp16; GroupNorm
  fp16 tích luỹ sai số đủ để map DIRE lệch. Bản gốc cũng làm vậy — đây là chi
  tiết *số học*, không phải phong cách.
* **Thứ tự chia head của attention.** Bản ADM dùng "chia head trước, chia qkv
  sau": reshape `(B, 3C, T) → (B·H, 3·c, T)` rồi mới cắt ba phần. Làm ngược lại
  (cắt qkv trước) vẫn chạy, shape vẫn đúng, nhưng **trộn sai kênh** ⇒ đầu ra
  vô nghĩa mà không có exception nào. Đây là loại lỗi mà chỉ parity số mới bắt.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = [
    "GroupNorm32",
    "ResBlock",
    "Attention",
    "time_embedding",
    "zero_",
]


class GroupNorm32(nn.GroupNorm):
    """GroupNorm 32 nhóm, tính ở float32, trả lại dtype đầu vào."""

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


def zero_(module: nn.Module) -> nn.Module:
    """Khởi tạo 0 cho tham số — chốt chặn cuối của mọi nhánh dư."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def time_embedding(timesteps: Tensor, dim: int, max_period: float = 10_000.0) -> Tensor:
    """Nhúng thời điểm bằng sin/cos nửa tần số (Vaswani et al., 2017).

    Công thức mà ADM dùng::

        f_i = exp(−ln(max_period) · i / (dim/2)),  i = 0 … dim/2−1
        e_t = [cos(f_i·t), sin(f_i·t)]              (nối theo chiều kênh)

    `dim` lẻ thì đệm một cột 0 — không dùng ở ADM nhưng giữ cho hàm tổng quát.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ResBlock(nn.Module):
    """Khối dư có điều kiện thời điểm, kiểu FiLM (scale & shift).

    ::

        h = conv(GN(x))                      # nhánh chính
        e = Linear(SiLU(emb))                # 2·C_out kênh khi scale_shift
        h = conv(SiLU(GN(h) · (1+scale) + shift))
        y = skip(x) + h

    `down`/`up` đặt phép giảm/tăng mẫu **giữa** GroupNorm và tích chập, đúng
    như ADM: cả nhánh chính lẫn đường tắt đều bị giảm mẫu, nên `skip` là
    identity khi số kênh không đổi.

    Điểm dễ sai và đã sai một lần khi viết file này: phép giảm mẫu của ADM là
    **average-pool 2×2**, phép tăng mẫu là **nearest 2×** — *không có tích chập
    nào trong cả hai*. Dùng conv stride 2 (như nhiều U-Net khác) vẫn chạy, vẫn
    ra ảnh, nhưng thêm ~vài chục triệu tham số không tồn tại trong checkpoint
    ⇒ `load_state_dict(strict=True)` nổ ngay. Đó là lý do cổng nạp trọng số đáng
    giá: nó bắt được sai kiến trúc mà mắt thường bỏ qua.
    """

    def __init__(
        self,
        channels: int,
        emb_channels: int,
        out_channels: int | None = None,
        dropout: float = 0.0,
        scale_shift: bool = True,
        up: bool = False,
        down: bool = False,
        use_conv_skip: bool = False,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels or channels
        self.scale_shift = scale_shift
        self.updown = up or down

        self.norm_in = GroupNorm32(32, channels)
        self.conv_in = nn.Conv2d(channels, self.out_channels, 3, padding=1)

        # cả hai phép đều KHÔNG có tham số, nên không xuất hiện trong state_dict
        if up:
            self.scale_hw = 2
            self.pool_hw = None
        elif down:
            self.scale_hw = None
            self.pool_hw = 2
        else:
            self.scale_hw = self.pool_hw = None

        self.emb_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels if scale_shift else self.out_channels),
        )

        self.norm_out = GroupNorm32(32, self.out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv_out = zero_(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1))

        if self.out_channels == channels:
            self.skip = nn.Identity()
        elif use_conv_skip:
            self.skip = nn.Conv2d(channels, self.out_channels, 3, padding=1)
        else:
            self.skip = nn.Conv2d(channels, self.out_channels, 1)

    def _resample(self, t: Tensor) -> Tensor:
        if self.scale_hw:
            return torch.nn.functional.interpolate(t, scale_factor=self.scale_hw, mode="nearest")
        return torch.nn.functional.avg_pool2d(t, self.pool_hw, self.pool_hw)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        h = self.norm_in(x)
        h = torch.nn.functional.silu(h)
        if self.scale_hw or self.pool_hw:
            h = self._resample(h)
            x = self._resample(x)
        h = self.conv_in(h)

        e = self.emb_mlp(emb).type(h.dtype)
        while e.dim() < h.dim():
            e = e[..., None]

        if self.scale_shift:
            scale, shift = torch.chunk(e, 2, dim=1)
            h = self.norm_out(h) * (1 + scale) + shift
            h = self.dropout(torch.nn.functional.silu(h))
        else:
            h = h + e
            h = self.dropout(torch.nn.functional.silu(self.norm_out(h)))
        h = self.conv_out(h)
        return self.skip(x) + h


class Attention(nn.Module):
    """Tự chú ý trên toàn bộ vị trí không gian, `num_heads` head.

    Điểm dễ sai: hệ số `1/√(√c)` chia **cả q lẫn k** (tổng 1/√c) chứ không phải
    `1/√c` cho riêng q. Với fp16, chia đôi như vậy ổn định hơn chia một lần.
    """

    def __init__(self, channels: int, num_heads: int = 1, head_channels: int = -1) -> None:
        super().__init__()
        if head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % head_channels == 0, (channels, head_channels)
            self.num_heads = channels // head_channels
        self.norm = GroupNorm32(32, channels)
        # Conv1d: sau khi trải không gian, chiều dài T là trục tích chập
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = zero_(nn.Conv1d(channels, channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        b, c, *spatial = x.shape
        flat = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(flat))

        # chia head TRƯỚC, rồi mới cắt q/k/v (thứ tự "legacy" của ADM)
        width = qkv.shape[1]
        assert width % (3 * self.num_heads) == 0
        ch = width // (3 * self.num_heads)
        q, k, v = qkv.reshape(b * self.num_heads, 3 * ch, -1).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        w = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        w = torch.softmax(w.float(), dim=-1).type(w.dtype)
        out = torch.einsum("bts,bcs->bct", w, v)
        out = out.reshape(b, -1, flat.shape[-1])
        return (flat + self.proj(out)).reshape(b, c, *spatial)
