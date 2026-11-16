"""Cổng chặn của đường from-scratch — test được trên máy không có torch/GPU.

Vì sao test riêng cho phần này
------------------------------
Ba thứ dưới đây quyết định **kết quả có nghĩa hay không**, và cả ba đều sai
được một cách im lặng:

1. đọc nhầm `dire/**` của tác giả ⇒ thành "chấm lại" mà tưởng là "tái lập";
2. seed train/test trùng ⇒ ACC 100% vì mô hình đã thấy ảnh test;
3. NFE sai ⇒ hoá đơn sai 25 lần, không có exception.

Không test nào ở đây cần GPU, nên chúng chạy trong CI bình thường.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from dire_core import ddim_schedule as sch
from dire_core import scratch

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# G0 — không đọc dire/**
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("path", [
    "/raw/dire/test/lsun_bedroom/lsun_bedroom/adm.tar.gz",
    "dire/train/x.tar.gz",
    "./dire/train/x.tar.gz",
    "D:\\data\\dire\\train\\x.tar.gz".replace("\\", "/"),
])
def test_published_paths_are_rejected(path):
    assert scratch.is_published_path(path)


@pytest.mark.parametrize("path", [
    "/raw/scratch/maps/fake/0001.npy",
    "/raw/ckpt/scratch_adm/detector.pth",
    "/raw/scratch/real/0001.png",
])
def test_scratch_paths_pass(path):
    assert not scratch.is_published_path(path)


def test_assert_no_published_names_the_culprit():
    with pytest.raises(PermissionError) as exc:
        scratch.assert_no_published(["/raw/scratch/ok.npy", "/raw/dire/x.tar.gz"])
    assert "dire/x.tar.gz" in str(exc.value)
    assert "chấm lại" in str(exc.value)


def test_assert_no_published_is_silent_when_clean():
    scratch.assert_no_published(["/raw/scratch/a.npy", "/raw/scratch/b.npy"])


# --------------------------------------------------------------------------- #
# G5 — NFE và mốc respacing
# --------------------------------------------------------------------------- #

def test_respacing_matches_paper_setup():
    assert sch.ddim_timesteps(1000, 20) == list(range(0, 1000, 50))
    assert sch.ddim_timesteps(1000, 20)[-1] == 950, "mốc cuối là 950, không phải 999"
    assert len(sch.ddim_timesteps(1000, 20)) == 20


def test_nfe_is_two_s():
    assert sch.nfe_per_image(20) == 40
    assert sch.nfe_total(1000, 20) == 40_000
    # Lỗi cũ: chạy trọn 1000 bước đảo ngược + 20 tái tạo = 1020.
    assert sch.nfe_per_image(20) != 1020


def test_check_nfe_blocks_the_old_bug():
    sch.check_nfe(40, 1, 20)
    with pytest.raises(AssertionError) as exc:
        sch.check_nfe(1020, 1, 20)
    assert "25" in str(exc.value) or "40" in str(exc.value)


def test_inversion_goes_up_and_reconstruction_goes_down():
    ts = sch.ddim_timesteps(1000, 5)
    assert sch.inversion_pairs(ts)[0] == (0, 200)
    assert sch.inversion_pairs(ts)[-1] == (600, 800)
    # xuôi: 800→600 … 200→0, nên cặp CUỐI là (200, 0)
    assert sch.reconstruction_pairs(ts)[-1] == (200, 0)
    assert sch.reconstruction_pairs(ts)[0] == (800, 600)


# --------------------------------------------------------------------------- #
# Số của lịch — đối chiếu bảng trong giáo trình (Phần 5 §5.7)
# --------------------------------------------------------------------------- #

def test_abar_matches_book_table():
    """So với bảng Phần 5 §5.7, sai số ≤ 5·10⁻⁴ (bảng in 4 chữ số)."""
    a = sch.alphas_cumprod(sch.linear_betas())
    assert a[0] == 1.0
    for t, want in ((50, 0.9700), (250, 0.5214), (500, 0.0778),
                    (750, 0.0033), (950, 0.0001)):
        assert abs(a[t] - want) < 5e-4, f"t={t}: {a[t]:.6f} vs {want}"


def test_sqrt_abar_at_last_step_gives_the_1_percent_figure():
    """`√ᾱ_950 ≈ 0,0103` ⇒ `x_950` còn ~1,03% tín hiệu — con số dùng trong sách."""
    a = sch.alphas_cumprod(sch.linear_betas())
    assert round(a[950] ** 0.5, 4) == 0.0103


def test_beta_schedule_rejects_nonsense():
    with pytest.raises(ValueError):
        sch.linear_betas(t_train=1)
    with pytest.raises(ValueError):
        sch.linear_betas(beta_min=0.02, beta_max=0.0001)
    with pytest.raises(ValueError):
        sch.ddim_timesteps(1000, 0)
    with pytest.raises(ValueError):
        sch.ddim_timesteps(1000, 2000)


# --------------------------------------------------------------------------- #
# G2 — seed
# --------------------------------------------------------------------------- #

def test_seed_split_is_disjoint_and_deterministic():
    a = scratch.split_seeds(400, 100, base=1000)
    b = scratch.split_seeds(400, 100, base=1000)
    assert a == b, "cùng base phải ra cùng seed — nếu không thì không tái lập được"
    assert set(a["train"]).isdisjoint(a["test"]), "trùng seed ⇒ ảnh test nằm trong train"


def test_seed_split_changes_with_base():
    a = scratch.split_seeds(50, 50, base=1)
    b = scratch.split_seeds(50, 50, base=2)
    assert a["train"] != b["train"]


def test_assert_disjoint_catches_leak():
    scratch.assert_disjoint([1, 2], [3, 4])
    with pytest.raises(ValueError, match="seed 3"):
        scratch.assert_disjoint([1, 2, 3], [3, 4])


# --------------------------------------------------------------------------- #
# G3 — metadata của ảnh sinh
# --------------------------------------------------------------------------- #

def _gen_meta(**over):
    m = {
        "generator": "adm", "ckpt_sha256": "a" * 64, "sampler": "ddim",
        "steps": 20, "seeds": [1, 2], "n_images": 2, "resolution": 256,
        "prompt": None, "created_utc": "2026-01-01T00:00:00+00:00",
    }
    m.update(over)
    return m


def test_gen_meta_complete_passes():
    assert scratch.check_gen_meta(_gen_meta()) == []


def test_gen_meta_missing_seed_is_reported():
    m = _gen_meta()
    del m["seeds"]
    assert "seeds" in scratch.check_gen_meta(m)


def test_gen_meta_length_mismatch_is_reported():
    assert scratch.check_gen_meta(_gen_meta(seeds=[1, 2, 3])) == ["seeds (độ dài khác n_images)"]


# --------------------------------------------------------------------------- #
# Nhãn được phép nói + run.json
# --------------------------------------------------------------------------- #

def test_scratch_claim_is_r1_r2_r3():
    c = scratch.scratch_claim()
    assert c.label == "R1+R2+R3"
    assert c.is_full
    assert "authors:" not in c.data_source


def test_run_meta_roundtrips_through_json():
    meta = scratch.run_meta(
        generator="adm", ckpt_sha256="b" * 64, dir_steps=20,
        n_train=400, n_test=100, seed_base=1000, sec_per_step=0.004,
        gates={"G0": True, "G8": True}, git_sha="deadbeef",
    )
    assert meta["nfe_per_image"] == 40
    assert meta["sec_per_step_is_measured"] is True
    assert json.loads(json.dumps(meta)) == meta


def test_run_meta_says_when_sec_per_step_is_a_guess():
    meta = scratch.run_meta(
        generator="adm", ckpt_sha256="c" * 64, dir_steps=20,
        n_train=1, n_test=1, seed_base=0, sec_per_step=None, gates={},
    )
    assert meta["sec_per_step_is_measured"] is False


def test_gate_report_lists_all_gates_and_finds_first_blocking():
    rep = scratch.gate_report({"G0": True, "G1": False})
    assert [g["id"] for g in rep] == [g[0] for g in scratch.GATES]
    assert scratch.first_blocking(rep)["id"] == "G1"
    assert scratch.first_blocking(scratch.gate_report({g: True for g in
                                                       [x[0] for x in scratch.GATES]})) is None


# --------------------------------------------------------------------------- #
# Kịch bản bash — cú pháp và luật "không đụng dire/"
# --------------------------------------------------------------------------- #

pytestmark_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="không có bash")


@pytestmark_bash
def test_scratch_script_parses():
    p = ROOT / "tools" / "scratch_repro.sh"
    r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytestmark_bash
def test_scratch_script_never_reads_published_dire():
    text = (ROOT / "tools" / "scratch_repro.sh").read_text(encoding="utf-8")
    # Dòng `grep` là chính cái guard, không phải một lần đọc.
    bad = [ln for ln in text.splitlines()
           if "/raw/dire" in ln and "grep" not in ln]
    assert not bad, f"script tham chiếu dire/**: {bad[:3]}"


def test_scripts_dir_still_has_exactly_two_workflows():
    """Đường scratch ở `tools/` để không phá quy ước 2 quy trình."""
    found = {p.name for p in (ROOT / "scripts").glob("[0-9][0-9]_*.sh")}
    assert found == {"01_repro.sh", "02_improve.sh"}
