# src/dire_modal/eval.py
"""Eval 1 checkpoint trên nhiều generator. 1 generator = 1 container."""

import modal
from .common import app, gpu_image, vol_shards, vol_ckpt


@app.function(
    image=gpu_image,
    volumes={"/shards": vol_shards, "/ckpt": vol_ckpt},
    gpu="L4",
    timeout=1800,
)
def eval_one(ckpt: str, test_dir: str) -> dict:
    # TODO: port src/eval.py gốc. Trả về {"acc":…, "ap":…}
    raise NotImplementedError


@app.function(image=gpu_image, volumes={"/ckpt": vol_ckpt}, timeout=3600)
def main(ckpt: str = "/ckpt/pretrained/lsun_adm.pth") -> None:
    """modal run -m dire_modal.eval::main --ckpt /ckpt/pretrained/lsun_adm.pth"""
    import glob, json

    test_dirs = sorted(glob.glob("/shards/test/*/*"))
    pairs = [(ckpt, d) for d in test_dirs]
    results = list(eval_one.starmap(pairs, order_outputs=False))
    for r in results:
        print(json.dumps(r))
