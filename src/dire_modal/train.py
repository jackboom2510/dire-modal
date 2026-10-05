# src/dire_modal/train.py
"""Resumable training. Checkpoint mỗi epoch vào /ckpt/{exp}/."""

import modal
from .common import app, gpu_image, vol_shards, vol_ckpt


@app.function(
    image=gpu_image,
    volumes={"/shards": vol_shards, "/ckpt": vol_ckpt},
    gpu="A100",
    timeout=24 * 3600,
    retries=modal.Retries(max_retries=3, initial_delay=60.0),
)
def main(exp: str = "lsun_adm_repro", resume: bool = True) -> None:
    """modal run -m dire_modal.train::main --exp lsun_adm_repro"""
    import json, torch, webdataset as wds
    from pathlib import Path

    # TODO: port src/networks/resnet.py của repo gốc vào package này
    # 1. load config từ configs/train_lsun_adm.yaml
    # 2. build DataLoader từ /shards/train/lsun_bedroom/{real,adm}/shard-*.tar
    # 3. nếu resume & /ckpt/{exp}/last.pt tồn tại → load
    # 4. train loop: lưu /ckpt/{exp}/epoch_{i}.pt + last.pt mỗi epoch
    # 5. vol_ckpt.commit() sau mỗi epoch
    raise NotImplementedError("Port code training của Wang vào đây")
