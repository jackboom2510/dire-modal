"""Giai đoạn 2: tự tính DIRE từ ảnh gốc bằng ADM DDIM inversion.

Có 2 việc ở đây:
  1. `DIRECompute`  — class chạy song song cho cả dataset (số container + timeout
     lấy từ `tiers.yaml`; xem khối "Guard chi phí" — đừng để cứng 40/24h).
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
# ⚠ TUYỆT ĐỐI KHÔNG thêm `from __future__ import annotations` vào file này.
#
# PEP 563 biến mọi annotation thành CHUỖI. Modal đọc `__annotations__` thô để
# validate `modal.parameter`, nên `model_path: str` tới nó dưới dạng `'str'`
# chứ không phải `str`, và nó chết ngay lúc IMPORT module:
#
#     KeyError: 'str'
#     → InvalidError("No class parameter encoder implemented for type `...`")
#     → AttributeError: 'str' object has no attribute '__name__'
#
# AttributeError cuối cùng che mất InvalidError thật sự, nên đọc traceback
# không thấy nguyên nhân. Lỗi nằm ở import chứ không ở class ⇒ MỌI lệnh của
# module này chết, kể cả `make_gallery`.
#
# Dockerfile dùng Python 3.11 nên mọi annotation trong file (`dict`, `list[str]`,
# `int | None`) đều evaluate được ở runtime — không cần future import.
import modal

from .common import app, cpu_image, gpu_image, vol_shards, vol_ckpt
from .manifest import POLICY

DIRE_SRC = "/opt/dire/guided-diffusion"  # clone trong image, xem common.py


# ===========================================================================
# 0. Guard chi phí — chạy TRƯỚC khi bật container nào
# ===========================================================================
# Bản trước để cứng `max_containers=40` + `timeout=24*3600` trên A100. Một
# container treo bị tính tiền GPU suốt 24 giờ, nhân 40 container:
#     40 × 24 h × ~$2.50/h ≈ $2 400 cho MỘT lần chạy hỏng.
# Điểm chết người là TIMEOUT chứ không phải số container: một ảnh 256² × 20
# step mất vài giây, nên timeout 24 h chỉ có tác dụng khi container đã treo —
# đúng lúc nó tốn tiền nhất.
_CONTAINER_HARD_CAP = 16
# Timeout tối đa cho MỘT ảnh. Một ảnh 256² × 20 step mất vài giây, nên 1 giờ
# đã là margin ~1000×. Đặt ở đây để một dòng config sai (24 h) không thể biến
# 16 container thành $960 trần xấu nhất.
_TIMEOUT_HARD_CAP = 3600


def _clamp_timeout(raw: int) -> int:
    """Chặn cứng timeout một ảnh vào [60, 3600] giây."""
    return max(60, min(int(raw), _TIMEOUT_HARD_CAP))


def _clamp_containers(raw: int, n_images: int | None = None) -> int:
    """Chặn cứng số container vào [1, 16].

    Một dòng config sai (``dire_max_containers: 40``) không được phép bật lại
    40 A100. Không dùng biến env: config đi theo package nên nó là nguồn duy
    nhất, và guard phải áp dụng cả khi ai đó sửa file.
    """
    c = max(1, min(int(raw), _CONTAINER_HARD_CAP))
    if n_images is not None:
        c = max(1, min(c, max(1, int(n_images))))
    return c


def estimate_cost(n_images: int, policy=None, containers: int | None = None) -> dict:
    """Ước tính chi phí GPU cho ``n_images`` ảnh.

    Trả về HAI con số, và con số thứ hai mới là thứ guard phải chặn:

    - ``usd``: kỳ vọng, theo ``sec_per_image`` (bảo thủ, chưa đo thật).
    - ``worst_usd``: mọi container đều treo tới ``timeout`` — đây là **hoá đơn
      thật** khi lần chạy hỏng, và là trần cứng của mọi lần chạy.
    """
    p = policy or POLICY
    n = max(0, int(n_images))
    c = containers if containers is not None else _clamp_containers(
        p.dire_max_containers, n
    )
    timeout_s = _clamp_timeout(p.dire_timeout_s)
    gpu_hours = n * p.dire_sec_per_image / 3600.0
    worst_gpu_hours = c * timeout_s / 3600.0
    return {
        "images": n,
        "containers": c,
        "gpu_hours": gpu_hours,
        "usd": gpu_hours * p.dire_usd_per_gpu_hour,
        "worst_gpu_hours": worst_gpu_hours,
        "worst_usd": worst_gpu_hours * p.dire_usd_per_gpu_hour,
        "cap_usd": float(p.dire_max_usd),
        "max_images": int(p.dire_max_images),
        "sec_per_image": float(p.dire_sec_per_image),
        "timeout_s": timeout_s,
    }


def _print_estimate(est: dict) -> None:
    print(
        f"[tiền] {est['images']} ảnh × {est['sec_per_image']:.1f} s/ảnh = "
        f"{est['gpu_hours']:.2f} GPU-h ≈ ${est['usd']:.2f}"
    )
    print(
        f"[tiền] trần xấu nhất: {est['containers']} container × "
        f"{est['timeout_s'] / 60:.0f} phút = {est['worst_gpu_hours']:.2f} GPU-h "
        f"≈ ${est['worst_usd']:.2f}   (trần cho phép ${est['cap_usd']:.2f})"
    )


def preflight_cost(
    n_images: int, policy=None, max_usd: float | None = None, force: bool = False
) -> dict:
    """Kiểm tiền TRƯỚC khi bật container. Vượt trần ⇒ ``SystemExit``.

    Gọi hàm này là việc đầu tiên trong entrypoint, trước cả bước liệt kê file.
    ``force=True`` bỏ qua cả ba điều kiện (người dùng tự chịu trách nhiệm), vẫn
    in đủ số.
    """
    p = policy or POLICY
    cap = float(max_usd) if max_usd else float(p.dire_max_usd)
    est = estimate_cost(n_images, p)
    est["cap_usd"] = cap
    _print_estimate(est)

    if force:
        if est["usd"] > cap or est["images"] > est["max_images"]:
            print("[tiền] ⚠ --force: đã bỏ qua trần chi phí. Theo dõi bằng")
            print("       modal app list  /  modal billing report --resolution h")
        return est

    if est["images"] > est["max_images"]:
        raise SystemExit(
            f"[DỪNG] {est['images']} ảnh > trần {est['max_images']} ảnh/lần "
            f"(dire_max_images trong src/dire_modal/tiers.yaml).\n"
            "  Chạy thành nhiều đợt nhỏ, hoặc nâng trần một cách có ý thức."
        )
    if est["usd"] > cap:
        raise SystemExit(
            f"[DỪNG] ước tính ${est['usd']:.2f} > trần ${cap:.2f}.\n"
            f"  Giảm --n (đang {est['images']}), hoặc nâng dire_max_usd, "
            "hoặc dùng --force nếu đã cân nhắc."
        )
    if est["worst_usd"] > cap:
        raise SystemExit(
            f"[DỪNG] trần xấu nhất ${est['worst_usd']:.2f} > ${cap:.2f}: "
            f"{est['containers']} container × {est['timeout_s'] / 60:.0f} phút "
            "treo vẫn vượt ngân sách.\n"
            "  Giảm dire_max_containers hoặc dire_timeout_s trong tiers.yaml, "
            "hoặc dùng --force nếu chấp nhận rủi ro đó."
        )
    return est


def _import_dire():
    """Import guided_diffusion từ bản clone của repo DIRE gốc."""
    import sys

    if DIRE_SRC not in sys.path:
        sys.path.insert(0, DIRE_SRC)
    from guided_diffusion import script_util  # noqa: PLC0415

    # Import vì TÁC DỤNG PHỤ: `dist_util` chọn device (cuda/cpu) lúc import.
    # Dùng `__import__` thay vì `from ... import dev  # noqa: F401` vì pyflakes
    # KHÔNG hiểu `noqa` — nó vẫn báo "imported but unused" và làm đỏ cổng 0.
    __import__("guided_diffusion.dist_util")

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
    # 30 phút cho MỘT ảnh (gấp ~300× thời gian thật) thay vì 24 giờ: đây là
    # trần chi phí thật của một lần chạy, xem khối "Guard chi phí" ở đầu file.
    timeout=_clamp_timeout(POLICY.dire_timeout_s),
    max_containers=_clamp_containers(POLICY.dire_max_containers),
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


# ===========================================================================
# 3. Liệt kê nguồn (CPU, gần như $0) + entrypoint CÓ GUARD
# ===========================================================================
@app.function(
    image=cpu_image,
    volumes={"/shards": vol_shards},
    timeout=600,
)
def list_images(src_glob: str = "/shards/test/*/real/*.jpg", limit: int = 200) -> list[str]:
    """Liệt kê ảnh nguồn trên Volume. CPU, không GPU ⇒ gần như không tốn tiền.

    Tách ra khỏi GPU để bước đếm ảnh (quyết định chi phí) không phải bật A100.
    """
    import glob as _glob

    return sorted(_glob.glob(src_glob))[: max(0, int(limit))]


@app.local_entrypoint()
def dire_batch(
    n: int = 200,
    steps: int = 20,
    src_glob: str = "/shards/test/*/real/*.jpg",
    out_dir: str = "/shards/dire_out",
    max_usd: float = 0.0,
    force: bool = False,
    dry_run: bool = False,
) -> None:
    """Chạy DIRE cho tối đa ``n`` ảnh — **có guard chi phí**.

    Thứ tự cố ý: kiểm tiền TRƯỚC, bật container SAU. Vượt trần thì
    ``SystemExit`` ngay ở dòng đầu, chưa có GPU nào được cấp phát.

    ``modal run -m dire_modal.compute_dire::dire_batch --n 200``
    ``modal run -m dire_modal.compute_dire::dire_batch --n 5 --dry-run``
    """
    from pathlib import Path

    cap = max_usd or POLICY.dire_max_usd
    preflight_cost(n, max_usd=cap, force=force)  # ← cửa chặn, đứng trước mọi container
    if dry_run:
        print("[dry-run] không bật container nào.")
        return

    paths = list_images.remote(src_glob, n)  # CPU, không GPU
    if not paths:
        raise SystemExit(f"[!] không thấy ảnh khớp {src_glob}")
    if len(paths) < n:
        # Ước tính lại theo số THẬT (nhỏ hơn nên chắc chắn qua trần).
        print(f"[tiền] chỉ thấy {len(paths)}/{n} ảnh → tính lại")
        preflight_cost(len(paths), max_usd=cap, force=force)

    dsts = [f"{out_dir}/{Path(p).stem}.npy" for p in paths]
    print(f"[chạy] {len(paths)} ảnh · {steps} step · tối đa "
          f"{_clamp_containers(POLICY.dire_max_containers, len(paths))} container")
    done = 0
    for r in DIRECompute(steps=steps).process_one.map(paths, dsts):
        done += 1
        if done % 25 == 0 or done == len(paths):
            print(f"[{done}/{len(paths)}] {r}")
    print(f"[xong] {done} ảnh → {out_dir}")
