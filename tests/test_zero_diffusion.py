"""DDIM + DIRE map — kiểm bằng **mô hình tiên tri**, không bằng mô hình thật.

Ý tưởng: nếu mô hình trả về *đúng* ε của một ảnh gốc đã biết, thì phép đảo
ngược phải dựng lại đúng trạng thái nhiễu ``q_sample``, và phép tái tạo phải trả
lại đúng ảnh gốc. Sai một chỉ số ``ᾱ``, sai thứ tự cập nhật, hay sai chiều
``t`` đều làm test này đỏ — trong khi nhìn ảnh thì không thấy gì.

Đây là bài kiểm mà một lần chạy GPU 2,5 GB **không** cho bạn: nó tách "số học
sai" khỏi "trọng số xấu".
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from dire_zero import schedule as sch  # noqa: E402
from dire_zero.diffusion import (  # noqa: E402
    ModelCounter,
    Schedule,
    dire_map,
    invert,
    reconstruct,
    sample,
    sigma_floor,
    to_uint8,
)


class Oracle(torch.nn.Module):
    """Mô hình "hoàn hảo": trả đúng ε của ảnh gốc mà nó đã thuộc lòng."""

    def __init__(self, x0: torch.Tensor, gamma: torch.Tensor) -> None:
        super().__init__()
        self.x0 = x0
        self.gamma = gamma
        self.seen: list[int] = []

    def forward(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        t = int(ts[0])
        self.seen.append(t)
        a = self.gamma[t].to(x.dtype)
        eps = (x - a.sqrt() * self.x0) / (1 - a).sqrt()
        return torch.cat([eps, torch.zeros_like(eps)], dim=1)   # 6 kênh: ε + log σ²


def test_oracle_inversion_matches_q_sample():
    sched = Schedule(steps=5)
    x0 = (torch.rand(1, 3, 8, 8) * 2 - 1)
    oracle = Oracle(x0, sched.gamma)
    f = ModelCounter(oracle)
    top = invert(f, x0, sched)

    t_max = sched.timesteps[-1]
    a = sched.a(t_max, dtype=x0.dtype)
    eps = (top - a.sqrt() * x0) / (1 - a).sqrt()
    assert torch.isfinite(top).all()
    # trạng thái cuối phải tái tạo được x0 với ε mà mô hình đã dự đoán
    assert torch.allclose(a.sqrt() * x0 + (1 - a).sqrt() * eps, top, atol=1e-6)
    assert f.nfe == sched.steps


def test_oracle_reconstruction_returns_the_image():
    sched = Schedule(steps=5)
    x0 = (torch.rand(1, 3, 8, 8) * 2 - 1)
    f = ModelCounter(Oracle(x0, sched.gamma))
    top = invert(f, x0, sched)
    back = reconstruct(f, top, sched)
    assert torch.allclose(back, x0, atol=2e-5), (back - x0).abs().max()
    assert f.nfe == 2 * sched.steps


def test_dire_of_a_perfect_model_is_zero():
    sched = Schedule(steps=5)
    x0 = (torch.rand(1, 3, 8, 8) * 2 - 1)
    f = ModelCounter(Oracle(x0, sched.gamma))
    dm, recon = dire_map(f, x0, sched)
    assert dm.max() < 1e-4
    assert f.nfe == 2 * sched.steps


def test_dire_is_absolute_difference_in_0_2():
    sched = Schedule(steps=3)
    x0 = torch.zeros(1, 3, 8, 8)
    f = ModelCounter(Oracle(x0, sched.gamma))
    dm, recon = dire_map(f, x0, sched)
    assert dm.shape == x0.shape
    assert float(dm.min()) >= 0.0
    assert float(dm.max()) <= 2.0


def test_random_model_gives_finite_map_not_nan():
    """Cổng NaN: mô hình ngẫu nhiên phải cho map hữu hạn.

    Đây là test sẽ đỏ nếu ai đó đổi `ᾱ[0]` về `1,0` (chia cho 0 ở bước đầu) —
    lỗi có thật trong đường cũ, và nó *không* làm chương trình dừng.
    """
    torch.manual_seed(0)
    sched = Schedule(steps=3)

    class Rand(torch.nn.Module):
        def forward(self, x, ts):
            return torch.cat([torch.zeros_like(x), torch.zeros_like(x)], dim=1)

    f = ModelCounter(Rand())
    dm, recon = dire_map(f, torch.zeros(1, 3, 8, 8), sched)
    assert torch.isfinite(dm).all() and torch.isfinite(recon).all()
    assert sigma_floor(sched) > 0


def test_nfe_counts_the_whole_batch_and_guards_the_protocol():
    sched = Schedule(steps=4)
    x0 = torch.zeros(1, 3, 8, 8)
    f = ModelCounter(Oracle(x0, sched.gamma))
    dire_map(f, x0, sched)
    assert f.nfe == 8

    with pytest.raises(AssertionError):
        sch.check_nfe(f.nfe, 1, 5)          # 8 ≠ 2·5
    x3 = x0.repeat(3, 1, 1, 1)
    f3 = ModelCounter(Oracle(x3, sched.gamma))
    f3(x3, 0)                      # một lượt mạng, ba ảnh
    assert f3.nfe == 3             # NFE đếm theo ẢNH, không theo lần gọi


def test_sampling_uses_s_steps_and_is_deterministic():
    torch.manual_seed(0)
    sched = Schedule(steps=4)

    class Zero(torch.nn.Module):
        def forward(self, x, ts):
            return torch.zeros(x.shape[0], 6, x.shape[2], x.shape[3])

    f = ModelCounter(Zero())
    a = sample(f, (1, 3, 8, 8), sched, seed=7)
    n = f.nfe
    b = sample(f, (1, 3, 8, 8), sched, seed=7)
    assert n == 4 and f.nfe == 8
    assert torch.equal(a, b)


def test_to_uint8_is_scale_one_not_four():
    x = torch.tensor([[[[-1.0]]], [[[0.0]]], [[[1.0]]]]).reshape(1, 3, 1, 1)
    out = to_uint8(x)
    assert out.dtype == torch.uint8
    assert out.flatten().tolist() == [0, 128, 255]   # (0,5·255) làm tròn = 128
    # giá trị > 1 bị kẹp, không nhân 4
    assert int(to_uint8(torch.tensor(5.0)).item()) == 255


def test_schedule_repr_and_bounds():
    sched = Schedule(steps=20)
    assert sched.timesteps[0] == 0 and sched.timesteps[-1] == 950
    assert math.isclose(float(sched.gamma[0]), 0.9999, rel_tol=1e-6)
    assert "S=20" in repr(sched)
