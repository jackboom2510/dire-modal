"""U-Net của ADM — viết lại từ mô tả trong bài báo (ADM, arXiv:2105.05233, Phụ lục B).

Cấu trúc, với `channel_mult = (1,1,2,2,4,4)`, `num_res_blocks = 2`,
attention ở độ phân giải giảm 8/16/32::

    stem 3→256
    enc[0] 256      ×2 (+down)              ds=1
    enc[1] 256      ×2 (+down)              ds=2
    enc[2] 512      ×2 (+down)              ds=4
    enc[3] 512      ×2 + attn (+down)       ds=8      ← attention bắt đầu
    enc[4] 1024     ×2 + attn (+down)       ds=16
    enc[5] 1024     ×2 + attn               ds=32
    mid             res + attn + res
    dec[5] 1024     ×3 + attn (+up)
    dec[4] 1024     ×3 + attn (+up)
    dec[3] 512      ×3 + attn (+up)
    dec[2] 512      ×3 (+up)
    dec[1] 256      ×3 (+up)
    dec[0] 256      ×3
    head 256→6      (learn_sigma ⇒ 3 kênh ε + 3 kênh phương sai)

Tên tham số ở đây là **của mình** (`enc.3.1.res.conv_in`, …). Việc nối chúng
với tên trong checkpoint (`input_blocks.7.0.in_layers.2.weight`) nằm ở
:mod:`dire_zero.checkpoint`, và được kiểm bằng test song ánh 566 key.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .blocks import Attention, GroupNorm32, ResBlock, time_embedding

__all__ = ["UNetConfig", "ADM_LSUN", "EncEntry", "DecEntry", "ZeroUNet"]


@dataclass(frozen=True)
class UNetConfig:
    """Siêu tham số kiến trúc. Đổi một trường là đổi checkpoint phải nạp."""

    image_size: int = 256
    in_channels: int = 3
    model_channels: int = 256
    out_channels: int = 6          # learn_sigma ⇒ 2·3
    num_res_blocks: int = 2
    attention_ds: tuple[int, ...] = (8, 16, 32)
    channel_mult: tuple[int, ...] = (1, 1, 2, 2, 4, 4)
    num_heads: int = 4
    head_channels: int = 64
    dropout: float = 0.0
    scale_shift: bool = True
    resblock_updown: bool = True

    def __post_init__(self) -> None:
        # GroupNorm của ADM luôn 32 nhóm ⇒ số kênh phải chia hết cho 32.
        if self.model_channels % 32:
            raise ValueError(
                f"model_channels={self.model_channels} không chia hết cho 32 (GroupNorm 32 nhóm)"
            )
        if not self.resblock_updown:
            raise NotImplementedError(
                "bản này chỉ hỗ trợ resblock_updown=True (cấu hình của ADM 256)"
            )

    @property
    def time_channels(self) -> int:
        return self.model_channels * 4


#: Cấu hình của `256x256_diffusion_uncond.pt` (ADM 256 vô điều kiện).
ADM_LSUN = UNetConfig()


class EncEntry(nn.Module):
    """Một mục của nhánh xuống: `res` (+ `attn`). `solo` = mục chỉ có phép giảm mẫu."""

    def __init__(self, res: ResBlock, attn: Attention | None = None, solo: bool = False) -> None:
        super().__init__()
        self.res = res
        self.attn = attn
        self.solo = solo

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        x = self.res(x, emb)
        if self.attn is not None:
            x = self.attn(x)
        return x


class DecEntry(nn.Module):
    """Một mục của nhánh lên: `res` → `attn` → `up`.

    Phép tăng mẫu là một `ResBlock` **riêng** đứng sau attention, không phải cờ
    trên `res`. Thứ tự này không phải chi tiết vụn: chú ý chạy trước khi tăng
    mẫu thì bản đồ chú ý có 1/4 số vị trí so với chạy sau. Làm sai thứ tự vẫn
    ra ảnh, nhưng tốn gấp 4 lần compute và lệch hẳn kết quả.
    """

    def __init__(
        self,
        res: ResBlock,
        attn: Attention | None = None,
        up: ResBlock | None = None,
    ) -> None:
        super().__init__()
        self.res = res
        self.attn = attn
        self.up = up

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        x = self.res(x, emb)
        if self.attn is not None:
            x = self.attn(x)
        if self.up is not None:
            x = self.up(x, emb)
        return x


class ZeroUNet(nn.Module):
    """U-Net vô điều kiện, dự đoán ε (và log-phương sai nếu `out_channels = 2·C`)."""

    def __init__(self, cfg: UNetConfig = ADM_LSUN) -> None:
        super().__init__()
        self.cfg = cfg
        c = cfg.model_channels
        tdim = cfg.time_channels

        self.time_mlp = nn.Sequential(
            nn.Linear(c, tdim),
            nn.SiLU(),
            nn.Linear(tdim, tdim),
        )

        c0 = cfg.channel_mult[0] * c
        self.stem = nn.Conv2d(cfg.in_channels, c0, 3, padding=1)

        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()
        skips: list[int] = [c0]

        ch = c0
        ds = 1
        levels = len(cfg.channel_mult)
        for level, mult in enumerate(cfg.channel_mult):
            entries = nn.ModuleList()
            for _ in range(cfg.num_res_blocks):
                out_ch = int(mult * c)
                attn = (
                    Attention(out_ch, cfg.num_heads, cfg.head_channels)
                    if ds in cfg.attention_ds
                    else None
                )
                entries.append(EncEntry(ResBlock(ch, tdim, out_ch, cfg.dropout, cfg.scale_shift), attn))
                ch = out_ch
                skips.append(ch)
            if level != levels - 1:
                entries.append(EncEntry(ResBlock(ch, tdim, ch, cfg.dropout, cfg.scale_shift, down=True), solo=True))
                skips.append(ch)
                ds *= 2
            self.enc.append(entries)

        self.mid = nn.ModuleList([
            ResBlock(ch, tdim, ch, cfg.dropout, cfg.scale_shift),
            Attention(ch, cfg.num_heads, cfg.head_channels),
            ResBlock(ch, tdim, ch, cfg.dropout, cfg.scale_shift),
        ])

        for level, mult in reversed(list(enumerate(cfg.channel_mult))):
            entries = nn.ModuleList()
            for i in range(cfg.num_res_blocks + 1):
                out_ch = int(mult * c)
                skip_ch = skips.pop()
                attn = (
                    Attention(out_ch, cfg.num_heads, cfg.head_channels)
                    if ds in cfg.attention_ds
                    else None
                )
                res = ResBlock(ch + skip_ch, tdim, out_ch, cfg.dropout, cfg.scale_shift)
                up = None
                if level and i == cfg.num_res_blocks:
                    up = ResBlock(out_ch, tdim, out_ch, cfg.dropout, cfg.scale_shift, up=True)
                entries.append(DecEntry(res, attn, up))
                ch = out_ch
                if up:
                    ds //= 2
            self.dec.append(entries)

        self.head_norm = GroupNorm32(32, ch)
        self.head_conv = nn.Conv2d(ch, cfg.out_channels, 3, padding=1)
        for p in self.head_conv.parameters():
            p.detach().zero_()

    # ------------------------------------------------------------------ #
    def forward(self, x: Tensor, timesteps: Tensor) -> Tensor:
        # `time_embedding` **luôn** trả float32: sin/cos của `f_i·t` mất chính xác
        # nhanh ở fp16 (tới 1000 ⇒ mất ~3 chữ số), nên tính ở float32 rồi mới đổi
        # sang dtype của mô hình. Nhưng phải đổi **trước** `time_mlp`: trọng số fp16
        # × đầu vào fp32 ⇒ `mat1 and mat2 must have the same dtype`, và lỗi này chỉ
        # xuất hiện trên GPU (nơi `dtype="auto"` chọn fp16), không bao giờ ở test CPU
        # chạy float32.
        dtype = next(self.parameters()).dtype
        emb = self.time_mlp(time_embedding(timesteps, self.cfg.model_channels).to(dtype))
        h = self.stem(x.type(dtype))
        hs = [h]
        for entries in self.enc:
            for entry in entries:
                h = entry(h, emb)
                hs.append(h)
        for block in self.mid:
            h = block(h) if isinstance(block, Attention) else block(h, emb)
        for entries in self.dec:
            for entry in entries:
                h = torch.cat([h, hs.pop()], dim=1)
                h = entry(h, emb)
        h = torch.nn.functional.silu(self.head_norm(h))
        return self.head_conv(h).type(x.dtype)

    # ------------------------------------------------------------------ #
    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
