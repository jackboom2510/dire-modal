"""Giao thức DDIM của đường scratch — không cần checkpoint, không cần GPU.

Vì sao file này tồn tại
-----------------------
`dire_modal.ddim` là đường mà runbook bảo người dùng chạy (`--stage dire`), và
nó **chưa từng** được chạy với mô hình nào. Ba lỗi dưới đây chỉ lộ ra khi kiểm
bằng một mô hình giả:

1. mốc 0 lấy `ᾱ = 1,0` ⇒ `√(1−ᾱ) = 0` ⇒ `ε̂ = 0/0` ⇒ **toàn bộ map NaN**;
2. `invert` và `reconstruct` mỗi bên chỉ chạy `S−1` lượt ⇒ NFE = 38 chứ không
   phải 40, trong khi `check_nfe` đòi 40;
3. `reconstruct` không có lượt nào tại mốc 0, nên đầu ra còn ~0,5 % nhiễu.

Test dùng **mô hình tiên tri**: nó trả về đúng ε của một ảnh gốc đã biết, nên
phép đảo ngược + tái tạo phải trả lại đúng ảnh đó. Sai một chỉ số `ᾱ` là test
đỏ ngay, không cần 2,5 GB trọng số.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dire_modal.ddim import Schedule, dire_map, invert, reconstruct, sample  # noqa: E402


class Oracle(torch.nn.Module):
    """Trả đúng ε của ảnh gốc, theo đúng quy ước `ᾱ` mà `_predict` dùng."""

    def __init__(self, x0: torch.Tensor, sched: Schedule) -> None:
        super().__init__()
        self.x0 = x0
        self.sched = sched

    def forward(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        a = torch.tensor(self.sched.abar[int(ts[0]) + 1], dtype=x.dtype)
        eps = (x - a.sqrt() * self.x0) / (1 - a).sqrt()
        return torch.cat([eps, torch.zeros_like(eps)], dim=1)


class Blind(torch.nn.Module):
    """Trả 0 — đủ để kiểm tính hữu hạn và số lượt chạy."""

    def forward(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.zeros_like(x), torch.zeros_like(x)], dim=1)


@pytest.mark.parametrize("steps", [2, 3, 20])
def test_nfe_is_two_s(steps):
    sched = Schedule(steps=steps)
    x0 = torch.randn(1, 3, 8, 8).clamp(-1, 1)
    _, n1 = invert(Blind(), x0, sched, "cpu")
    top, _ = invert(Blind(), x0, sched, "cpu")
    _, n2 = reconstruct(Blind(), top, sched, "cpu")
    assert (n1, n2) == (steps, steps)


@pytest.mark.parametrize("steps", [2, 3, 20])
def test_no_nan_at_the_first_milestone(steps):
    """Cổng NaN: `ᾱ[0]` phải là 0,9999, không phải 1,0."""
    sched = Schedule(steps=steps)
    x0 = torch.randn(1, 3, 8, 8).clamp(-1, 1)
    dm, _recon, nfe = dire_map(Blind(), x0, sched, "cpu")
    assert torch.isfinite(dm).all()
    assert nfe == 2 * steps
    assert float(dm.min()) >= 0.0 and float(dm.max()) <= 2.0


@pytest.mark.parametrize("steps", [2, 5])
def test_oracle_roundtrip_returns_the_image(steps):
    """Đảo ngược rồi tái tạo bằng mô hình hoàn hảo ⇒ đúng ảnh gốc."""
    sched = Schedule(steps=steps)
    x0 = torch.rand(1, 3, 8, 8) * 2 - 1
    top, _ = invert(Oracle(x0, sched), x0, sched, "cpu")
    back, _ = reconstruct(Oracle(x0, sched), top, sched, "cpu")
    assert torch.allclose(back, x0, atol=5e-5), (back - x0).abs().max()


def test_dire_of_a_perfect_model_is_near_zero():
    sched = Schedule(steps=5)
    x0 = torch.rand(1, 3, 8, 8) * 2 - 1
    dm, _recon, _nfe = dire_map(Oracle(x0, sched), x0, sched, "cpu")
    assert dm.max() < 1e-3


def test_sampling_uses_s_evaluations():
    sched = Schedule(steps=4)
    _img, nfe = sample(Blind(), (1, 3, 8, 8), sched, "cpu", seed=0)
    assert nfe == 4


def test_abar_at_zero_is_just_below_one():
    """``ᾱ`` tại mốc 0 phải **< 1** (nếu = 1 thì chia cho 0), và rất gần 1.

    Đúng bằng ``1 − β`` của lịch, không phải một con số gõ tay: bản repo dùng
    chỉ số 1-based cho ``β`` nên ``ᾱ`` tại mốc 0 là 0,99988 chứ không phải
    0,9999 — lệch 1,2e-4, không ảnh hưởng gì nhưng phải nhất quán với lịch.
    """
    from dire_core.ddim_schedule import linear_betas

    sched = Schedule()
    a0 = float(sched._ab(0, "cpu", torch.float32))
    assert a0 == pytest.approx(1 - linear_betas()[0], rel=1e-6)   # float32
    assert 0.999 < a0 < 1.0
    top = float(sched._ab(950, "cpu", torch.float32))
    assert 1e-5 < top < 1e-3
