"""Giai đoạn 2: tự tính DIRE từ ảnh gốc bằng ADM DDIM inversion.

Có 2 việc ở đây:
  1. `DIRECompute`  — class chạy song song cho cả dataset (max_containers=40).
  2. `make_gallery` — sinh vài mẫu `orig | recon | |x-recon|` cho HF Spaces demo.

Cả hai **cần ADM checkpoint** (`/ckpt/pretrained/lsun_adm.pth`) và repo guided-diffusion
(đã clone sẵn vào image tại `/opt/dire` — xem `common.py::gpu_image`).

⚠ TRẠNG THÁI: code dưới đây viết theo API của guided-diffusion (`create_model_and_diffusion`,
`ddim_reverse_sample_loop`, `ddim_sample_loop`) nhưng **CHƯA được chạy thử** vì sandbox soạn
code không có Modal/checkpoint. Lần chạy đầu tiên cần kiểm tra kỹ 3 điểm:
  (a) tên tham số model khớp checkpoint,
  (b) `ddim_reverse_sample_loop` trả (x, _) hay chỉ x,
  (c) kích thước bộ nhớ khi batch > 1 ở 256×256.
"""
from __future__ import annotations

import modal

from .common import app, gpu_image, vol_shards, vol_ckpt

DIRE_SRC = "/opt/dire/guided-diffusion"  # clone trong image, xem common.py


def _import_dire():
    """Import guided_diffusion từ bản clone của repo DIRE gốc."""
    import sys

    if DIRE_SRC not in sys.path:
        sys.path.insert(0, DIRE_SRC)
    from guided_diffusion import script_util  # noqa: PLC0415
    from guided_diffusion.dist_util import dev  # noqa: F401,PLC0415

    return script_util


def _load_adm(ckpt: str, image_size: int = 256):
    """Nạp ADM theo đúng cấu hình 256×256_diffusion_uncond của guided-diffusion."""
    import torch

    script_util = _import_dire()
    model, diffusion = script_util.create_model_and_diffusion(
        image_size=image_size,
        learn_sigma=True,
        sigma_small=False,
        class_cond=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        num_channels=256,
        num_res_blocks=2,
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        attention_resolutions="32,16,8",
        channel_mult="",
        dropout=0.0,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=True,
        use_new_attention_order=False,
    )
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, diffusion


def _to_neg1_1(img):
    import numpy as np

    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return arr


def _dire_for_image(model, diffusion, pil_img, steps: int = 20):
    """Trả (orig_arr, recon_arr, diff_arr) dạng uint8 HxWx3."""
    import numpy as np
    import torch

    from guided_diffusion import gaussian_diffusion as gd

    x0 = torch.from_numpy(_to_neg1_1(pil_img)).permute(2, 0, 1).unsqueeze(0)
    with torch.inference_mode():
        # 1) DDIM inversion: ảnh → nhiễu (eta=0 → tất định)
        spacing = gd.SpacedDiffusion.get_ddim_sample_times if hasattr(gd.SpacedDiffusion, "get_ddim_sample_times") else None
        ddim = gd.SpacedDiffusion(
            use_timesteps=gd.space_timesteps(diffusion.num_timesteps, str(steps)),
            betas=diffusion.betas,
            model_mean_type=diffusion.model_mean_type,
            model_var_type=diffusion.model_var_type,
            loss_type=diffusion.loss_type,
            rescale_timesteps=diffusion.rescale_timesteps,
        )
        noise = diffusion.ddim_reverse_sample_loop(
            model, x0, clip_denoised=True, model_kwargs={}, eta=0.0, progress=False
        )
        if isinstance(noise, tuple):
            noise = noise[0]
        # 2) Tái tạo: nhiễu → ảnh (chính là "x_recon" của DIRE)
        recon = ddim.ddim_sample_loop(
            model, noise.shape, noise=noise, clip_denoised=True, model_kwargs={}, progress=False
        )
        if isinstance(recon, tuple):
            recon = recon[0]

    to_uint8 = lambda t: np.clip((t.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) * 127.5, 0, 255).astype(np.uint8)
    recon_u8 = to_uint8(recon)
    orig_u8 = ((_to_neg1_1(pil_img) + 1) * 127.5).clip(0, 255).astype(np.uint8)
    # DIRE = |x - recon|, hiển thị khuếch đại 4× cho dễ thấy
    diff_u8 = np.clip(np.abs(orig_u8.astype(np.int16) - recon_u8.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
    return orig_u8, recon_u8, diff_u8


# ===========================================================================
# 1. Class chạy song song cho cả dataset
# ===========================================================================
@app.cls(
    image=gpu_image,
    volumes={"/shards": vol_shards, "/ckpt": vol_ckpt},
    gpu="A100",
    timeout=24 * 3600,
    max_containers=40,
)
class DIRECompute:
    model_path: str = modal.parameter(default="/ckpt/pretrained/lsun_adm.pth")
    steps: int = modal.parameter(default=20)

    @modal.enter()
    def load(self):
        self.model, self.diffusion = _load_adm(self.model_path)
        self.model = self.model.cuda()

    @modal.method()
    def process_one(self, src_path: str, dst_path: str) -> dict:
        """Đọc 1 ảnh → DIRE → ghi .npy kèm metadata."""
        import numpy as np
        from pathlib import Path
        from PIL import Image
        import torch

        with torch.inference_mode():
            orig, recon, diff = _dire_for_image(self.model, self.diffusion, Image.open(src_path), self.steps)
        Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(dst_path, diff)
        return {"src": src_path, "dst": dst_path, "mean_abs_diff": float(diff.mean())}


# ===========================================================================
# 2. Gallery cho HF Spaces demo
# ===========================================================================
@app.function(
    image=gpu_image,
    volumes={"/shards": vol_shards, "/ckpt": vol_ckpt},
    gpu="A100",
    timeout=3600,
)
def make_gallery(n: int = 12, steps: int = 20,
                 ckpt: str = "/ckpt/pretrained/lsun_adm.pth",
                 out: str = "/shards/_gallery",
                 src_glob: str = "/shards/test/*/real/*.jpg",
                 src_glob_fake: str = "/shards/test/*/adm/*.jpg") -> dict:
    """Sinh gallery cho tab "DIRE viz": mỗi mẫu = _orig.png + _recon.png + _diff.png (+ .txt).

    Chọn nửa mẫu real, nửa mẫu fake để demo thấy khác biệt (|diff| fake ≈ nhỏ hơn real).

    Chạy:
        modal run -m dire_modal.compute_dire::make_gallery --n 12 --steps 20
    Tải về:
        modal volume get dire-shards /_gallery ./platforms/hf-spaces/gallery
    """
    import glob
    from pathlib import Path

    from PIL import Image
    import numpy as np

    if not Path(ckpt).exists():
        raise SystemExit(
            f"[!] Không thấy ADM checkpoint tại {ckpt}.\n"
            "    Cần: modal volume cp /raw/checkpoints/lsun_adm.pth dire-ckpt:/pretrained/lsun_adm.pth\n"
            "    (xem scripts/04_eval_pretrained.sh)"
        )
    reals = sorted(glob.glob(src_glob))[: n // 2 + 1]
    fakes = sorted(glob.glob(src_glob_fake))[: n // 2 + 1]
    if not reals or not fakes:
        raise SystemExit(
            f"[!] Không tìm thấy ảnh nguồn.\n  real: {src_glob} → {len(reals)} file\n  fake: {src_glob_fake} → {len(fakes)} file\n"
            "    → cần chạy `pack` trước, hoặc trỏ --src-glob vào thư mục ảnh khác trong /shards."
        )

    model, diffusion = _load_adm(ckpt)
    import torch
    model = model.cuda()

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = [(p, "REAL (ảnh thật)") for p in reals] + [(p, "FAKE (ADM sinh)") for p in fakes]
    written = []
    for i, (src, label) in enumerate(pairs[:n]):
        stem = f"{i:02d}_{Path(src).parent.name}_{Path(src).stem[:12]}"
        try:
            orig, recon, diff = _dire_for_image(model, diffusion, Image.open(src), steps)
        except Exception as exc:
            print(f"[SKIP] {src}: {exc}")
            continue
        for tag, arr in (("orig", orig), ("recon", recon), ("diff", diff)):
            Image.fromarray(arr).save(out_dir / f"{stem}_{tag}.png")
        mean_abs = float(np.abs(orig.astype(np.int16) - recon.astype(np.int16)).mean())
        (out_dir / f"{stem}.txt").write_text(
            f"{label} — DIRE |x−recon| trung bình = {mean_abs:.2f}/255 (khuếch đại 4× khi hiển thị)\n"
        )
        written.append(stem)
        print(f"[ok] {stem}: mean|diff|={mean_abs:.2f}  ({label})")

    vol_shards.commit()
    pdf = __import__("statistics").mean(
        [float((out_dir / f"{s}.txt").read_text().split("=")[1].split("/")[0]) for s in written]
    ) if written else 0.0
    print(f"\n[done] {len(written)} mẫu → {out_dir} · DIRE trung bình {pdf:.2f}")
    return {"samples": written, "out": str(out_dir),
            "download": "modal volume get dire-shards /_gallery ./platforms/hf-spaces/gallery"}
