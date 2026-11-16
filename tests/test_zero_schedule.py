"""Lịch nhiễu — toán thuần, chạy được ở mọi máy (không cần torch).

Ba con số ở đây không phải "kiểm cho có": chúng là chỗ đã có lỗi thật.

* ``ᾱ[0]`` phải là ``0,9999`` chứ không phải ``1,0`` — nếu là 1 thì
  ``√(1−ᾱ)=0`` và phép đảo ngược chia cho 0 ⇒ map toàn NaN.
* mốc cuối phải là **950**, không phải 999.
* ``NFE/ảnh = 2S`` — sai số này từng làm hoá đơn sai 25 lần.
"""

from __future__ import annotations

import math

import pytest

from dire_zero import schedule as sch


def test_abar_starts_at_one_minus_beta_not_one():
    betas = sch.linear_betas()
    gamma = sch.alphas_cumprod(betas)
    assert len(gamma) == 1000
    assert gamma[0] == pytest.approx(1 - betas[0])
    assert gamma[0] == pytest.approx(0.9999)
    # điều kiện sống còn: mẫu số của ε̂ tại mốc 0 phải khác 0
    assert math.sqrt(1 - gamma[0]) > 0


def test_abar_milestones_match_known_values():
    gamma = sch.alphas_cumprod(sch.linear_betas())
    # 0-based như ADM. Bảng cũ trong repo dùng chỉ số 1-based nên lệch ~2 % ở
    # t=950 (1,08e-4 so với 1,06e-4) — đúng loại lệch làm hai tài liệu cãi nhau.
    known = {250: 0.5214, 500: 0.0778, 750: 0.0033, 950: 1.06e-4}
    for t, want in known.items():
        assert gamma[t] == pytest.approx(want, rel=5e-3), t


def test_timesteps_are_20_points_ending_at_950():
    ts = sch.timesteps()
    assert len(ts) == 20
    assert ts[0] == 0
    assert ts[-1] == 950
    assert ts[:4] == [0, 50, 100, 150]
    assert ts == sorted(ts)


def test_every_timestep_has_nonzero_sigma():
    gamma = sch.alphas_cumprod(sch.linear_betas())
    floors = [math.sqrt(1 - gamma[t]) for t in sch.timesteps()]
    assert min(floors) > 0.009      # mốc 0: √(1−0,99999) ≈ 0,01


def test_nfe_is_2s():
    assert sch.nfe_per_image(20) == 40
    assert sch.nfe_per_image(50) == 100
    with pytest.raises(ValueError):
        sch.nfe_per_image(0)


def test_check_nfe_accepts_correct_and_rejects_the_25x_bug():
    sch.check_nfe(40, 1, 20)            # 2·20·1
    with pytest.raises(AssertionError):
        sch.check_nfe(1020, 1, 20)      # lỗi cũ: chạy trọn 1000 bước
    with pytest.raises(AssertionError):
        sch.check_nfe(38, 1, 20)        # 2·(S−1): thiếu hai lượt


@pytest.mark.parametrize("steps", [1, 2, 5, 10, 20, 50, 100])
def test_timesteps_always_include_zero_and_stay_in_range(steps):
    ts = sch.timesteps(1000, steps)
    assert ts[0] == 0
    assert max(ts) < 1000
    assert len(ts) == steps


def test_beta_schedule_matches_adm_linear():
    betas = sch.linear_betas()
    assert betas[0] == pytest.approx(1e-4)
    assert betas[-1] == pytest.approx(2e-2)
    with pytest.raises(ValueError):
        sch.linear_betas(beta_min=0.1, beta_max=0.05)
    with pytest.raises(ValueError):
        sch.linear_betas(t_train=1)


def test_timesteps_reject_impossible_requests():
    with pytest.raises(ValueError):
        sch.timesteps(1000, 0)
    with pytest.raises(ValueError):
        sch.timesteps(1000, 1001)
