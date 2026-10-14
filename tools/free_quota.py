#!/usr/bin/env python3
"""free_quota.py — sổ theo dõi hạn mức free tier cho dự án DIRE (thuần stdlib).

Trả lời: "tháng/tuần này còn bao nhiêu giờ GPU miễn phí, nên tiêu vào việc gì?"

Ví dụ
-----
    python3 tools/free_quota.py                          # xem toàn bộ còn lại bao nhiêu
    python3 tools/free_quota.py --used modal=8 kaggle=6  # đã dùng 8h A100 ở Modal, 6h ở Kaggle
    python3 tools/free_quota.py --used modal=8 --modal-gpu a100-40
    python3 tools/free_quota.py --json

Nguyên tắc: con số nào không kiểm chứng được thì để `None` và in chữ "chưa xác minh",
KHÔNG bịa số. Nguồn ghi ngay tại chỗ.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta

# ---------------------------------------------------------------------------
# Giá GPU để quy đổi credit (USD/giờ) — nguồn: docs/iaas-options.md §1.1
# ---------------------------------------------------------------------------
MODAL_GPU_USD_PER_H = {
    "t4": 0.59, "l4": 0.80, "a10": 1.10, "l40s": 1.95,
    "a100-40": 2.10, "a100-80": 2.50, "h100": 3.95, "h200": 4.54, "b200": 6.25,
}

# Lightning AI: credit/giờ (xấp xỉ — báo cáo người dùng: 15 credit ≈ 22 h T4 ⇒ ~0.68 credit/h T4)
LIGHTNING_CREDIT_PER_H = {"t4": 0.68, "l40s": 1.36, "a100": 2.04, "h100": 2.72}


@dataclass
class Quota:
    key: str
    label: str
    capacity: float
    unit: str          # "usd" | "gpu_hours" | "credits"
    period: str        # "month" | "week" | "one_time"
    source: str
    verified: bool = True
    note: str = ""


QUOTAS: list[Quota] = [
    Quota("modal", "Modal Starter credit", 30.0, "usd", "month",
          "https://modal.com/pricing", True,
          "reset hàng tháng; KHÔNG bao gồm Volume storage (1 TiB free riêng)"),
    Quota("kaggle", "Kaggle GPU", 30.0, "gpu_hours", "week",
          "https://www.kaggle.com/docs/efficient-gpu-usage", True,
          "GPU P100 16GB, ~9–12h/phiên, không SSH; reset hàng tuần"),
    Quota("lightning", "Lightning AI free", 30.0, "credits", "month",
          "https://lightning.ai/pricing", True,
          "tới 30 credit/tháng (nhiều người chỉ nhận ~15); Studio restart mỗi 4h"),
    Quota("hf_jobs", "HF Jobs credit", 0.0, "credits", "month",
          "https://huggingface.co/docs/hub/jobs-pricing", True,
          "phải nạp credit (PRO $9/th bao gồm credit); đặt số dư thật vào HF_BALANCE_USD"),
]


def modal_usd_for_hours(hours: float, gpu: str) -> float:
    return hours * MODAL_GPU_USD_PER_H[gpu]


def lightning_credits_for_hours(hours: float, gpu: str) -> float:
    return hours * LIGHTNING_CREDIT_PER_H.get(gpu, 1.0)


def parse_used(pairs: list[str]) -> dict[str, float]:
    used: dict[str, float] = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"[!] --used cần dạng key=hours, nhận được {p!r}")
        k, v = p.split("=", 1)
        used[k.strip()] = float(v)
    return used


def report(used: dict[str, float], modal_gpu: str, lightning_gpu: str, json_out: bool,
           hf_balance_usd: float, azure_start: date | None) -> int:
    rows = []
    for q in QUOTAS:
        if q.key == "modal":
            spent_usd = modal_usd_for_hours(used.get("modal", 0.0), modal_gpu)
            remaining = q.capacity - spent_usd
            rows.append({
                "key": q.key, "label": q.label, "period": q.period,
                "used": f"{used.get('modal', 0.0):.1f} h {modal_gpu} = ${spent_usd:.2f}",
                "remaining": f"${remaining:.2f}  (~{remaining / MODAL_GPU_USD_PER_H[modal_gpu]:.1f} h {modal_gpu})",
                "pct_left": max(0.0, remaining) / q.capacity * 100,
                "note": q.note, "source": q.source, "verified": q.verified,
            })
        elif q.key == "kaggle":
            u = used.get("kaggle", 0.0)
            remaining = q.capacity - u
            rows.append({
                "key": q.key, "label": q.label, "period": q.period,
                "used": f"{u:.1f} h",
                "remaining": f"{remaining:.1f} h (tuần này)",
                "pct_left": max(0.0, remaining) / q.capacity * 100,
                "note": q.note, "source": q.source, "verified": q.verified,
            })
        elif q.key == "lightning":
            spent = lightning_credits_for_hours(used.get("lightning", 0.0), lightning_gpu)
            remaining = q.capacity - spent
            hours_left = max(0.0, remaining) / LIGHTNING_CREDIT_PER_H[lightning_gpu]
            rows.append({
                "key": q.key, "label": q.label, "period": q.period,
                "used": f"{used.get('lightning', 0.0):.1f} h {lightning_gpu} ≈ {spent:.1f} credit",
                "remaining": f"{remaining:.1f} credit (~{hours_left:.1f} h {lightning_gpu})",
                "pct_left": max(0.0, remaining) / q.capacity * 100,
                "note": q.note, "source": q.source, "verified": q.verified,
            })
        elif q.key == "hf_jobs":
            remaining = hf_balance_usd - modal_usd_for_hours(used.get("hf", 0.0), "a100-80")
            rows.append({
                "key": q.key, "label": q.label, "period": q.period,
                "used": f"{used.get('hf', 0.0):.1f} h A100 ≈ ${modal_usd_for_hours(used.get('hf', 0.0), 'a100-80'):.2f}",
                "remaining": f"${remaining:.2f}" if hf_balance_usd else "chưa khai báo số dư (--hf-balance-usd)",
                "pct_left": (max(0.0, remaining) / hf_balance_usd * 100) if hf_balance_usd else None,
                "note": q.note, "source": q.source, "verified": q.verified,
            })

    if json_out:
        print(json.dumps({"generated": date.today().isoformat(), "rows": rows}, ensure_ascii=False, indent=2))
        return 0

    print()
    print("=== SỔ HẠN MỨC FREE TIER (DIRE) ===")
    print(f"    ngày: {date.today().isoformat()} · Modal GPU quy đổi: {modal_gpu} · Lightning: {lightning_gpu}")
    print()
    for r in rows:
        pct = f" [{r['pct_left']:.0f}% còn lại]" if r["pct_left"] is not None else ""
        flag = "" if r["verified"] else "  ⚠ chưa xác minh"
        print(f"• {r['label']} ({r['period']}){pct}{flag}")
        print(f"    đã dùng : {r['used']}")
        print(f"    còn lại : {r['remaining']}")
        if r["note"]:
            print(f"    lưu ý   : {r['note']}")
    print()

    if azure_start:
        age = (date.today() - azure_start).days
        left = 30 - age
        print(f"• Azure $200 (one-time, hết hạn sau 30 ngày)")
        print(f"    kích hoạt: {azure_start.isoformat()} → còn {max(0, left)} ngày "
              f"{'⚠ ĐÃ HẾT HẠN' if left <= 0 else ''}")
        print(f"    quy đổi: ~${200 / 3.67:.0f} h A100 (nhưng PHẢI xin quota N-series trước)")
        print()

    # Gợi ý phân bổ
    print("--- Gợi ý phân bổ (theo docs/runbook-free.md) ---")
    print("  1. Dev/smoke test code        → Kaggle (không tốn credit)")
    print("  2. Train detector Tier A      → Modal A100 (~3–8 h ≈ $8–20) — phần đắt nhất, làm sau cùng")
    print("  3. Eval 16 generator song song → Modal (.map), ~$2–4")
    print("  4. Tải + verify dữ liệu        → Modal CPU (~$1–2)")
    print("  5. Demo public                 → HF Spaces CPU (0đ)")
    print()
    print("  Nguồn:", "; ".join(f"{q.key}: {q.source}" for q in QUOTAS))
    print()
    print("  Lưu ý ToS: Colab free CẤM SSH/remote desktop/distributed worker/đa tài khoản.")
    print("             Kaggle không có SSH và không chạy nền — đừng thiết kế pipeline phụ thuộc Kaggle.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Sổ hạn mức free tier cho DIRE")
    ap.add_argument("--used", nargs="*", default=[], help="giờ đã dùng, dạng key=hours (modal, kaggle, lightning, hf)")
    ap.add_argument("--modal-gpu", default="a100-80", choices=sorted(MODAL_GPU_USD_PER_H),
                    help="GPU dùng để quy đổi credit Modal (mặc định a100-80)")
    ap.add_argument("--lightning-gpu", default="t4", choices=sorted(LIGHTNING_CREDIT_PER_H),
                    help="GPU quy đổi credit Lightning (mặc định t4)")
    ap.add_argument("--hf-balance-usd", type=float, default=0.0, help="số dư credit HF Jobs (USD)")
    ap.add_argument("--azure-start", type=str, default=None, help="ngày kích hoạt Azure (YYYY-MM-DD)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    azure_start = None
    if args.azure_start:
        try:
            azure_start = datetime.strptime(args.azure_start, "%Y-%m-%d").date()
        except ValueError:
            raise SystemExit("[!] --azure-start cần định dạng YYYY-MM-DD")

    used = parse_used(args.used)
    unknown = set(used) - {"modal", "kaggle", "lightning", "hf"}
    if unknown:
        print(f"[i] bỏ qua key lạ: {', '.join(sorted(unknown))}", file=sys.stderr)
    return report(used, args.modal_gpu, args.lightning_gpu, args.json, args.hf_balance_usd, azure_start)


if __name__ == "__main__":
    sys.exit(main())
