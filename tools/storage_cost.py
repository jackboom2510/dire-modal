#!/usr/bin/env python3
"""storage_cost.py — máy tính chi phí lưu trữ cho dự án DIRE (thuần stdlib).

So sánh: Modal Volume · Cloudflare R2 (Standard/IA) · Local (ổ tự mua) · Kaggle/HF Hub (free).
Tự đọc `manifests/*.tsv` (sinh bởi `dire_modal.download`) để lấy dung lượng THẬT nếu có.

Ví dụ
-----
    python3 tools/storage_cost.py --size-gb 150
    python3 tools/storage_cost.py --from-manifest manifests/raw.tsv --months 12
    python3 tools/storage_cost.py --size-gb 3000 --egress-gb 500 --json

Ghi chú đơn vị: nhà cung cấp niêm yết lẫn lộn GiB/GB (Modal: GiB, R2: GB).
Tool này coi 1 GiB ≈ 1 GB và ghi rõ giả định — sai số <7%, chấp nhận được khi ra quyết định.
Giá cập nhật: 2026-10-14 (xem PRICES bên dưới; sửa giá ở MỘT chỗ duy nhất này).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

GiB = 1  # đơn vị "GB" trừu tượng; xem ghi chú đơn vị ở docstring

# ----------------------------------------------------------------------------
# BẢNG GIÁ — nguồn & ngày truy cập ghi ngay tại chỗ. Sửa giá ở đây.
# ----------------------------------------------------------------------------
PRICES: dict[str, dict] = {
    "modal_volume": {
        "label": "Modal Volume (Starter)",
        "per_gb_month": 0.09,
        "free_gb": 1024,          # 1 TiB/tháng (modal.com/pricing, 2026-10-14)
        "min_bill_days_after_delete": 4,  # vẫn tính phí tới 4 ngày sau khi xoá
        "source": "https://modal.com/pricing",
    },
    "r2_standard": {
        "label": "Cloudflare R2 Standard",
        "per_gb_month": 0.015,
        "free_gb": 10,            # developers.cloudflare.com/r2/pricing, 2026-10-14
        "source": "https://developers.cloudflare.com/r2/pricing/",
    },
    "r2_ia": {
        "label": "Cloudflare R2 Infrequent Access",
        "per_gb_month": 0.01,
        "free_gb": 0,
        "retrieval_per_gb": 0.01,  # $0.01/GB khi đọc
        "min_storage_days": 30,    # tối thiểu 30 ngày
        "source": "https://developers.cloudflare.com/r2/pricing/",
    },
    "local_nvme": {
        "label": "Local — ổ NVMe 4 TB tự mua (khấu hao 5 năm)",
        # 4 TB NVMe ≈ $180 → 180 / 60 tháng / 4000 GB
        "per_gb_month": round(180 / 60 / 4000, 6),
        "free_gb": 0,
        "source": "giả định giá ổ 4 TB ≈ $180, khấu hao 60 tháng",
    },
    "local_hdd": {
        "label": "Local — ổ HDD 8 TB tự mua (khấu hao 5 năm)",
        "per_gb_month": round(150 / 60 / 8000, 6),
        "free_gb": 0,
        "source": "giả định giá ổ 8 TB ≈ $150, khấu hao 60 tháng",
    },
    "kaggle_hf": {
        "label": "Kaggle Datasets + HF Hub (public, free)",
        "per_gb_month": 0.0,
        "free_gb": 200,  # Kaggle: 200 GB/dataset; HF: ~300 GB/repo
        "quota_note": "Kaggle 200 GB/dataset · HF ~300 GB/repo · file ≤50 GB (HF)",
        "source": "kaggle.com/docs/datasets · discuss.huggingface.co/t/…/14861",
    },
}

EGRESS = {
    # $/GB đi ra Internet (không tính egress nội bộ giữa dịch vụ cùng nhà)
    "modal": {"per_gb": 0.04, "free_gb": 1024, "source": "modal.com/pricing (Starter: 1 TiB/tháng free)"},
    "r2": {"per_gb": 0.0, "free_gb": 0, "source": "developers.cloudflare.com/r2/pricing/ (egress $0)"},
    "aws": {"per_gb": 0.09, "free_gb": 100, "source": "AWS S3 data transfer (giá niêm yết phổ biến)"},
}

STORAGE_OP_COST = {
    # $/triệu thao tác — chỉ R2 tính tiền ops
    "r2_class_a": {"per_million": 4.50, "free_million": 1.0, "source": "R2 pricing (PutObject, multipart…)"},
    "r2_class_b": {"per_million": 0.36, "free_million": 10.0, "source": "R2 pricing (GetObject, HeadObject…)"},
}


@dataclass
class StorageOption:
    key: str
    label: str
    storage_usd: float
    egress_usd: float
    ops_usd: float
    notes: list[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        return self.storage_usd + self.egress_usd + self.ops_usd


def storage_cost(key: str, size_gb: float, months: int = 1) -> float:
    p = PRICES[key]
    billable = max(0.0, size_gb - p["free_gb"])
    return billable * p["per_gb_month"] * months


def egress_cost(provider: str, gb: float) -> float:
    e = EGRESS[provider]
    return max(0.0, gb - e["free_gb"]) * e["per_gb"]


def ops_cost(class_a_millions: float = 0.0, class_b_millions: float = 0.0, on_r2: bool = True) -> float:
    if not on_r2:
        return 0.0
    a = STORAGE_OP_COST["r2_class_a"]
    b = STORAGE_OP_COST["r2_class_b"]
    return (
        max(0.0, class_a_millions - a["free_million"]) * a["per_million"]
        + max(0.0, class_b_millions - b["free_million"]) * b["per_million"]
    )


def break_even_modal_vs_r2() -> float:
    """Dung lượng (GB) mà tại đó Modal Volume = R2 Standard.

    0.09*(X-1024) = 0.015*(X-10)  →  X = 1226.8 GB ≈ 1.23 TB (1.2 TiB)
    """
    mv, r2 = PRICES["modal_volume"], PRICES["r2_standard"]
    num = mv["per_gb_month"] * mv["free_gb"] - r2["per_gb_month"] * r2["free_gb"]
    den = mv["per_gb_month"] - r2["per_gb_month"]
    return num / den


def read_manifest_dir(path: Path) -> float:
    """Đọc manifests/*.tsv → tổng GB (cột 3 = size_bytes, theo `download.py`)."""
    total_bytes = 0
    rows = 0
    files = [path] if path.is_file() else sorted(path.glob("*.tsv"))
    for f in files:
        for line in f.read_text().splitlines()[1:]:  # bỏ header
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            try:
                total_bytes += int(parts[2])
                rows += 1
            except ValueError:
                continue
    if rows == 0:
        raise SystemExit(f"[!] Không đọc được size nào từ {path} — file trống hoặc sai định dạng?")
    return total_bytes / 1e9, rows


def scenario_table(size_gb: float, months: int, egress_gb: float, puts_millions: float, gets_millions: float) -> list[StorageOption]:
    opts = []
    for key in ("modal_volume", "r2_standard", "r2_ia", "kaggle_hf", "local_nvme", "local_hdd"):
        notes = []
        p = PRICES[key]
        st = storage_cost(key, size_gb, months)
        eg = egress_cost("r2" if key.startswith("r2") else "modal", egress_gb) if egress_gb else 0.0
        op = ops_cost(puts_millions, gets_millions, on_r2=key.startswith("r2"))
        if "quota_note" in p:
            notes.append(p["quota_note"])
        if key == "modal_volume" and size_gb > p["free_gb"]:
            notes.append(f"phần vượt {size_gb - p['free_gb']:.0f} GB × $0.09")
        if key == "modal_volume":
            notes.append("vẫn tính phí tới 4 ngày sau khi xoá")
        if key == "kaggle_hf" and size_gb > 200:
            notes.append(f"⚠ vượt 200 GB/dataset — cần chia nhiều dataset/repo")
        if key == "local_nvme" or key == "local_hdd":
            notes.append("chưa tính điện, sao lưu, rủi ro mất ổ")
        if key == "r2_ia":
            notes.append(f"tối thiểu 30 ngày; đọc tốn ${p['retrieval_per_gb']}/GB")
        opts.append(StorageOption(key, p["label"], st, eg, op, notes))
    return opts


def fmt_usd(v: float) -> str:
    return f"${v:,.2f}" if v >= 0.01 else ("$0" if v == 0 else f"${v:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="So sánh chi phí lưu trữ dataset/checkpoint (DIRE).")
    ap.add_argument("--size-gb", type=float, help="tổng dung lượng cần lưu (GB)")
    ap.add_argument("--from-manifest", type=Path, help="đọc manifests/*.tsv để lấy size thật")
    ap.add_argument("--months", type=int, default=1, help="số tháng (mặc định 1)")
    ap.add_argument("--egress-gb", type=float, default=0.0, help="GB đẩy ra Internet mỗi tháng")
    ap.add_argument("--puts", type=float, default=0.0, help="triệu thao tác ghi/tháng (R2 Class A)")
    ap.add_argument("--gets", type=float, default=0.0, help="triệu thao tác đọc/tháng (R2 Class B)")
    ap.add_argument("--json", action="store_true", help="xuất JSON")
    args = ap.parse_args()

    if args.from_manifest:
        size_gb, rows = read_manifest_dir(args.from_manifest)
        print(f"[i] Đọc {rows} dòng từ {args.from_manifest} → {size_gb:.1f} GB ({size_gb/1024:.2f} TiB)")
    elif args.size_gb:
        size_gb = args.size_gb
    else:
        ap.error("cần --size-gb hoặc --from-manifest")

    opts = scenario_table(size_gb, args.months, args.egress_gb, args.puts, args.gets)
    be = break_even_modal_vs_r2()

    if args.json:
        print(json.dumps({
            "size_gb": size_gb,
            "months": args.months,
            "break_even_modal_vs_r2_gb": round(be, 1),
            "options": [
                {"key": o.key, "label": o.label, "storage": round(o.storage_usd, 4),
                 "egress": round(o.egress_usd, 4), "ops": round(o.ops_usd, 4),
                 "total": round(o.total, 4), "notes": o.notes}
                for o in opts
            ],
        }, ensure_ascii=False, indent=2))
        return 0

    print()
    print(f"=== CHI PHÍ LƯU TRỮ · {size_gb:,.1f} GB ({size_gb/1024:.2f} TiB) · {args.months} tháng ===")
    if args.egress_gb:
        print(f"    egress giả định: {args.egress_gb:,.0f} GB/tháng")
    print()
    print(f"{'Phương án':<42} {'Lưu':>10} {'Egress':>10} {'Ops':>8} {'TỔNG':>11}")
    print("-" * 85)
    for o in sorted(opts, key=lambda x: x.total):
        print(f"{o.label:<42} {fmt_usd(o.storage_usd):>10} {fmt_usd(o.egress_usd):>10} "
              f"{fmt_usd(o.ops_usd):>8} {fmt_usd(o.total):>11}")
    print()
    for o in opts:
        if o.notes:
            print(f"  · {o.label}: " + "; ".join(o.notes))
    print()
    print(f"  Điểm hoà vốn Modal Volume vs R2 Standard: {be:,.0f} GB ≈ {be/1000:.2f} TB ({be/1024:.2f} TiB)")
    print(f"  → dưới mốc này Volume rẻ hơn (thực tế: miễn phí tới 1 TiB); trên mốc này R2 rẻ hơn.")
    print(f"  Chi phí GPU chờ I/O tham chiếu: A100 80GB $2.50/h → 1% thời gian chờ trong run 20 h = ${0.01*20*2.50:.2f}")
    print()
    print("  Nguồn giá:", ", ".join(sorted({p.get("source", "") for p in PRICES.values() if p.get("source")})))
    return 0


if __name__ == "__main__":
    sys.exit(main())
