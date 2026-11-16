"""Lệnh chạy ở máy nhà: kiểm cấu hình, xem lịch, chạy end-to-end tí hon.

Ba lệnh không cần GPU cũng không cần checkpoint::

    python -m dire_zero info        # lịch, mốc, NFE, bảng tiền theo sec/step
    python -m dire_zero keys        # đối chiếu 566 key với fixture
    python -m dire_zero selftest    # chạy trọn 4 bước trên U-Net 32×32

`selftest` là thứ nên chạy **trước khi** thuê GPU: nó chạy đúng bốn hàm mà
`modal_app` gọi, chỉ khác kích thước mô hình. Nếu nó xanh, lỗi còn lại chỉ có
thể là lỗi môi trường (thiếu checkpoint, VRAM, phiên bản torch).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

from . import budget, schedule as sch


def cmd_info(args: argparse.Namespace) -> int:
    ts = sch.timesteps(args.train_steps, args.steps)
    gamma = sch.alphas_cumprod(sch.linear_betas(args.train_steps))
    print(f"T={args.train_steps}  S={args.steps}  mốc: {ts[0]} … {ts[-1]}  (số mốc: {len(ts)})")
    print(f"NFE/ảnh = 2S = {sch.nfe_per_image(args.steps)}")
    for t in ts:
        print(f"  t={t:4d}  ᾱ={gamma[t]:.6g}  √(1−ᾱ)={max(0.0, 1 - gamma[t]) ** 0.5:.6g}")
    print("\nChi phí theo sec/step (mỗi ảnh 256², NFE = 2S):")
    print(f"{'ảnh':>6} {'s/step':>7} {'GPU-h':>8} {'USD A10G':>10}")
    for n in (500, 1000, 2000):
        for sec in (0.05, 0.35, 1.0, 2.5):
            est = budget.Estimate(n, 2 * args.steps, sec)
            print(f"{n:6d} {sec:7.2f} {est.gpu_h:8.3f} {est.usd:10.2f}")
    return 0


def cmd_keys(args: argparse.Namespace) -> int:
    fixture = Path(args.fixture)
    if not fixture.exists():
        print(f"không thấy {fixture}", file=sys.stderr)
        return 3
    if importlib.util.find_spec("torch") is None:
        print("cần torch để dựng mô hình (pip install torch)", file=sys.stderr)
        return 3
    import torch  # noqa: PLC0415
    from .checkpoint import check_bijection
    from .unet import ZeroUNet

    doc = json.loads(fixture.read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = ZeroUNet()
    mapping = check_bijection(model, doc["keys"])
    print(f"OK: {len(mapping)} tensor, {model.n_params:,} tham số — khớp {fixture.name}")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """Chạy `gen → dire → train` trên U-Net 32×32 với 8 ảnh ngẫu nhiên."""
    if importlib.util.find_spec("torch") is None:
        print("cần torch (pip install torch)", file=sys.stderr)
        return 3
    from .detector import ResNetDetector, ResNet
    from .diffusion import ModelCounter
    from .pipeline import RunMeta, compute_maps, generate, train
    from .unet import UNetConfig, ZeroUNet

    cfg = UNetConfig(image_size=32, model_channels=32, num_res_blocks=1,
                     attention_ds=(4,), channel_mult=(1, 2), num_heads=2, head_channels=-1)
    model = ZeroUNet(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    f = ModelCounter(model)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        g = generate(f, n=8, out_dir=tmp / "gen", meta=RunMeta("0" * 64, steps=3), seed_base=1, size=32)
        print(f"gen: {g['n']} ảnh, NFE={g['nfe_total']}, {g['seconds']}s")

        items = [(str(tmp / "gen" / nm), 1) for nm in g["names"]]
        d = compute_maps(f, items, tmp / "maps", RunMeta("0" * 64, steps=3), limit=8, size=32)
        print(f"dire: {d['n']} map, NFE={d['nfe_total']}, thang {d['scale']}")

        maps = [{"name": x["name"], "path": str(tmp / "maps" / x["name"]), "label": x["label"]} for x in d["items"]]
        tr = train(lambda: ResNetDetector(ResNet(blocks=(1, 1))), maps, tmp / "train",
                   epochs=1, bs=4, crop=28, device="cpu", patience=99)
        print(f"train: {tr['split']} · test n={tr['test']['n']} acc={tr['test']['acc']:.3f}")
    print("SELFTEST OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dire_zero", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("info", help="lịch nhiễu, mốc, NFE, bảng tiền")
    a.add_argument("--steps", type=int, default=20)
    a.add_argument("--train-steps", type=int, default=sch.T_TRAIN)
    a.set_defaults(func=cmd_info)

    b = sub.add_parser("keys", help="đối chiếu bảng ánh xạ với fixture 566 key")
    b.add_argument("--fixture", default="tests/fixtures/adm_lsun_state_keys.json")
    b.set_defaults(func=cmd_keys)

    c = sub.add_parser("selftest", help="chạy trọn 4 bước trên mô hình tí hon")
    c.set_defaults(func=cmd_selftest)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
