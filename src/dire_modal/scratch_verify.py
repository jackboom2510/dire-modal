"""Kiểm chứng **số học DDIM** bằng cách so với `recons/` của tác giả.

Ý tưởng, một câu
-----------------
Tác giả phát hành ảnh gốc (`images/`) **và** ảnh tái tạo (`recons/`) của họ.
Nếu mình tự chạy $R(I(x))$ bằng ADM + DDIM 20 bước trên đúng ảnh đó mà ra
**cùng một ảnh**, thì bước đảo ngược + tái tạo của mình đúng — độc lập với
detector, nhãn, ACC/AP, và với mọi thứ phía sau.

Đây là thứ các cổng hiện có *không* kiểm được: G4 kiểm mốc respacing và thang
giá trị, nhưng không kiểm số học. Một lỗi kiểu "lấy 6 kênh thay vì 3" hay
"fp16 ở chỗ phải fp32" cho ra map trông hợp lý nhưng sai.

Giới hạn phải nói trước
-----------------------
Chỉ đối chiếu được **7 generator** có `recons/` (`real`, `pndm`, `ldm`, `sdv2`,
`vqdiffusion`, `projectedgan`, `midjourney`, `stylegan` — 8 tên, 7 generator +
`real`). `adm` — generator headline của paper — **không có** `recons/`, nên
không kiểm được bằng đường này.

Ngưỡng: KHÔNG đặt trước. Lần chạy đầu **đo** phân bố rồi ghi vào `verify.json`;
các lần sau dùng đúng số đo đó làm cổng. Đặt ngưỡng trước rồi chỉnh theo kết
quả là tự lừa mình.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from statistics import median

from dire_core.recons_pair import HAS_RECONS, pair
from dire_core.scratch import write_json

from .common import VOLUMES, app, gpu_image
from .ddim import Schedule, load_adm

CKPT = "/raw/ckpt/adm_lsun.pt"
OUT = "/ckpt/scratch/verify_recons.json"


def _members(tar_path: Path) -> list[str]:
    import tarfile  # noqa: PLC0415

    with tarfile.open(tar_path, "r:gz") as tf:
        return sorted(m.name for m in tf.getmembers() if m.isfile())


def _load_member(tar_path: Path, name: str):
    import tarfile  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    with tarfile.open(tar_path, "r:gz") as tf:
        fh = tf.extractfile(name)
        if fh is None:
            raise KeyError(f"{name} không phải file trong {tar_path}")
        return Image.open(io.BytesIO(fh.read())).convert("RGB")


@app.function(image=gpu_image, volumes=VOLUMES, gpu="A10G", timeout=3600)
def verify_recons(
    gens: str = ",".join(HAS_RECONS),
    n: int = 10,
    steps: int = 20,
    ckpt: str = CKPT,
    root: str = "/raw",
    out_path: str = OUT,
) -> dict:
    """Chạy recon của mình trên `n` ảnh mỗi generator, so với `recons/` của tác giả."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    model, device = load_adm(ckpt, "cuda")
    schedule = Schedule(1000, steps)

    def _recon(img) -> "torch.Tensor":
        arr = np.asarray(img.resize((256, 256)), dtype=np.float32) / 127.5 - 1.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        from .ddim import invert, reconstruct  # noqa: PLC0415

        xT, _ = invert(model, x, schedule, device)
        rec, _ = reconstruct(model, xT, schedule, device)
        return rec

    report: dict = {"steps": steps, "n_per_gen": n, "generators": {}}
    for gen in [g.strip() for g in gens.split(",") if g.strip()]:
        img_rel, rec_rel = pair(gen)
        img_tar, rec_tar = Path(root) / img_rel, Path(root) / rec_rel
        if not img_tar.exists() or not rec_tar.exists():
            report["generators"][gen] = {"skipped": f"thiếu {img_rel if not img_tar.exists() else rec_rel}"}
            print(f"[bỏ] {gen}: thiếu tar")
            continue

        rec_names = set(_members(rec_tar))
        common = [m for m in _members(img_tar) if m in rec_names][:n]
        if not common:
            report["generators"][gen] = {"skipped": "không có member trùng tên giữa images/ và recons/"}
            print(f"[bỏ] {gen}: member không giao nhau — tar lệch lô?")
            continue

        diffs = []
        for name in common:
            ours = _recon(_load_member(img_tar, name))
            theirs_arr = np.asarray(_load_member(rec_tar, name).resize((256, 256)),
                                    dtype=np.float32) / 127.5 - 1.0
            theirs = torch.from_numpy(theirs_arr).permute(2, 0, 1).unsqueeze(0).to(device)
            diffs.append(float((ours - theirs).abs().mean()))

        diffs.sort()
        p95 = diffs[min(len(diffs) - 1, int(round(0.95 * (len(diffs) - 1))))]
        report["generators"][gen] = {
            "n": len(diffs),
            "median": round(median(diffs), 5),
            "p95": round(p95, 5),
            "max": round(diffs[-1], 5),
            "unit": "mean|x_recon_mình − x_recon_tác_giả| trên [-1,1]",
        }
        print(f"[{gen}] n={len(diffs)} median={median(diffs):.4f} p95={p95:.4f}")

    vals = [v["median"] for v in report["generators"].values() if "median" in v]
    report["summary"] = {
        "n_gens": len(vals),
        "median_of_medians": round(median(vals), 5) if vals else None,
        "note": "ngưỡng cổng PHẢI lấy từ lần chạy đầu, không đặt trước",
    }
    write_json(out_path, report)
    print(f"[xong] {out_path}")
    return report


@app.local_entrypoint()
def main(gens: str = ",".join(HAS_RECONS), n: int = 10, steps: int = 20,
         fail_above: float = -1.0) -> None:
    """`modal run -m dire_modal.scratch_verify::main --n 10`.

    `--fail-above 0.02` để dùng làm CỔNG trong CI/script: vượt ngưỡng ⇒ exit 1.
    Mặc định -1 = không chặn (lần chạy đầu chỉ ĐO).
    """
    rep = verify_recons.remote(gens=gens, n=n, steps=steps)
    med = rep.get("summary", {}).get("median_of_medians")
    if fail_above > 0 and med is not None and med > fail_above:
        raise SystemExit(
            f"[!] median {med} > ngưỡng {fail_above}: bước recon KHÔNG khớp tác giả. "
            "Kiểm: 6 kênh vs 3, fp16, mốc 950 vs 999, JPEG đã nén lại."
        )
    print(json.dumps(rep.get("generators", {}), ensure_ascii=False, indent=2))
