"""Read the pre-rendered synthetic DDIM illustration shipped with the demo."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "cache" / "ddim-illustration"


def manifest(root: Path = ROOT) -> dict:
    return json.loads((root / "index.json").read_text(encoding="utf-8"))


def frame(steps: int, index: int, root: Path = ROOT) -> tuple[list[tuple[str, str]], str]:
    data = manifest(root)
    rows = data["steps"].get(str(int(steps)))
    if not rows:
        raise ValueError(f"Không có cache cho S={steps}; hiện hỗ trợ S=5/10/20.")
    i = max(0, min(int(index), len(rows) - 1))
    row = rows[i]
    panel = [
        (str(root / row["x_t"]), "Trạng thái x_t — ảnh đã nhiễu / đang khử nhiễu"),
        (str(root / row["x0_hat"]), "Ước lượng x̂₀ — oracle noise của mô phỏng"),
    ]
    if row["phase"].startswith("Thêm"):
        equation = "x_t = √ᾱ_t x₀ + √(1−ᾱ_t) ε"
    else:
        equation = "x_{t−1} = √ᾱ_{t−1} x̂₀ + √(1−ᾱ_{t−1}) ε̂_t  (η = 0)"
    description = (
        f"**{row['phase']}** · bước {i + 1}/{len(rows)} · t={row['t']}"
        + (f" → t={row['t_next']}" if "t_next" in row else "")
        + f"\n\n`{equation}`\n\n"
        "**Minh họa tổng hợp đã lưu cache**: nhiễu oracle được biết trước; không chạy ADM/U-Net, "
        "không phải DIRE map, không phải kết quả thực nghiệm."
    )
    return panel, description
