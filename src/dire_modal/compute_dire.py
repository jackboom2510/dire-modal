# src/dire_modal/compute_dire.py
"""Giai đoạn 2: tự tính DIRE từ images/ bằng ADM DDIM inversion."""

import modal
from .common import app, gpu_image, vol_shards


@app.cls(
    image=gpu_image,
    volumes={"/shards": vol_shards},
    gpu="A100",
    timeout=24 * 3600,
    max_containers=40,
)
class DIRECompute:
    model_path: str = modal.parameter(default="/shards/_models/lsun_bedroom.pt")
    steps: int = modal.parameter(default=20)

    @modal.enter()
    def load(self):
        # TODO: load ADM 1 lần/container
        pass

    @modal.method()
    def process_shard(self, shard_path: str) -> str:
        # TODO: đọc shard → DDIM invert `steps` bước → |x - recon| → ghi shard mới
        raise NotImplementedError
