"""CLI ``dire`` — chạy trên máy nhà, không cần Modal.

Triết lý: **không import torch ở tầng module.** ``dire info`` phải chạy được trên
máy chưa cài torch (ví dụ trong CI, hoặc khi chỉ muốn xem cấu hình), và lỗi
"thiếu torch" phải là một câu tiếng Việt nói rõ cần cài gì — không phải
``ModuleNotFoundError``.

Cùng lõi với cloud: mọi phép tính đi qua ``dire_core`` (tiền xử lý, cách đọc
checkpoint, cách đổi logit → P(fake)). CLI không có bản sao nào.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

WEIGHTS_HINT = """\
Không tìm thấy checkpoint. Ba cách có nó:
  * trỏ thẳng:            --weights /duong/dan/lsun_adm.pth
  * biến môi trường:      DIRE_WEIGHTS=/duong/dan/lsun_adm.pth
  * chưa tải bao giờ:     python -m dire_modal.local_fetch --tiers e2min --mode fetch
"""

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

_TRUE_WORDS = {"1", "true", "yes", "y", "fake", "diffusion", "ai"}
_FALSE_WORDS = {"0", "false", "no", "n", "real"}


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------
def default_weights() -> str:
    """Thứ tự tìm checkpoint: ``--weights`` → ``$DIRE_WEIGHTS`` → ``checkpoints/``."""
    import os

    env = os.environ.get("DIRE_WEIGHTS")
    if env:
        return env
    for cand in (
        Path("checkpoints/lsun_adm.pth"),
        Path("data/raw/checkpoints/lsun_adm.pth"),
    ):
        if cand.exists():
            return str(cand)
    return "checkpoints/lsun_adm.pth"


def _iter_images(path: Path):
    if path.is_file():
        yield path
        return
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            yield p


def _load_pil(path: Path):
    from PIL import Image

    return Image.open(path).convert("RGB")


def _emit(obj, as_json: bool) -> None:
    if as_json:
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    else:
        print(f"{obj['path']}  P(fake)={obj['p_fake']:.6f}  → {obj['verdict']}")


def _verdict(p: float, threshold: float) -> str:
    return "fake" if p > threshold else "real"


# ---------------------------------------------------------------------------
# Lệnh
# ---------------------------------------------------------------------------
def cmd_info(args) -> int:
    from dire_core import CROP, MEAN, PREPROC, RESIZE, STD

    info = {
        "preproc": PREPROC,
        "resize": RESIZE,
        "crop": CROP,
        "mean": list(MEAN),
        "std": list(STD),
        "weights": args.weights or default_weights(),
        "weights_exists": Path(args.weights or default_weights()).exists(),
        "torch": _torch_version(),
    }
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


def _torch_version() -> str | None:
    try:  # không import torch ở tầng module — xem docstring
        import torch

        return torch.__version__
    except Exception:  # noqa: BLE001
        return None


def _detector(weights: str | None):
    from dire_core import Detector

    w = weights or default_weights()
    if not Path(w).exists():
        print(f"[!] {WEIGHTS_HINT}", file=sys.stderr)
        raise SystemExit(2)
    try:
        return Detector.load(w)
    except ImportError as exc:  # torch/torchvision/PIL thiếu
        raise SystemExit(
            "[!] Chạy `detect`/`batch` cần torch + torchvision + Pillow:\n"
            "    python -m pip install torch torchvision pillow\n"
            f"    (chi tiết: {exc})"
        ) from exc


def cmd_detect(args) -> int:
    det = _detector(args.weights)
    path = Path(args.image)
    if not path.exists():
        raise SystemExit(f"[!] không thấy file: {path}")
    p = det.score_pils([_load_pil(path)])[0]
    out = {
        "path": str(path),
        "p_fake": p,
        "verdict": _verdict(p, args.threshold),
        "threshold": args.threshold,
        **{k: det.info()[k] for k in ("preproc", "n_out", "prob", "device")},
    }
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(f"P(fake) = {p:.6f}   → {out['verdict']}   (ngưỡng {args.threshold})")
        print(f"preproc={out['preproc']}  n_out={out['n_out']}  prob={out['prob']}  device={out['device']}")
    return 0 if out["verdict"] == "fake" else 1 if args.exit_code else 0


def cmd_batch(args) -> int:
    det = _detector(args.weights)
    root = Path(args.path)
    if not root.exists():
        raise SystemExit(f"[!] không thấy đường dẫn: {root}")
    files = list(_iter_images(root))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"[!] không có ảnh nào trong {root} ({', '.join(IMAGE_SUFFIXES)})")

    rows, batch = [], []
    for f in files:
        try:
            batch.append((f, _load_pil(f)))
        except Exception as exc:  # noqa: BLE001 — một ảnh hỏng không giết cả batch
            rows.append({"path": str(f), "p_fake": None, "verdict": "error", "error": str(exc)})
            continue
        if len(batch) >= args.batch_size:
            rows.extend(_score_batch(det, batch, args.threshold))
            batch = []
    if batch:
        rows.extend(_score_batch(det, batch, args.threshold))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["path", "p_fake", "verdict", "error"], extrasaction="ignore"
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    ok = [r for r in rows if r["p_fake"] is not None]
    n_fake = sum(1 for r in ok if r["verdict"] == "fake")
    print(
        f"{len(ok)} ảnh · {n_fake} fake · {len(ok) - n_fake} real · "
        f"{len(rows) - len(ok)} lỗi  → {out}"
    )
    return 0


def _score_batch(det, batch, threshold):
    scores = det.score_pils([img for _, img in batch])
    return [
        {"path": str(f), "p_fake": s, "verdict": _verdict(s, threshold), "error": ""}
        for (f, _), s in zip(batch, scores)
    ]


def cmd_serve(args) -> int:
    """Chạy HTTP API tại máy — dùng CHÍNH `dire_local.server` mà Modal bọc lại."""
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "[!] `serve` cần fastapi + uvicorn:\n"
            "    python -m pip install fastapi uvicorn\n"
            f"    (chi tiết: {exc})"
        ) from exc
    from .server import build_app

    app = build_app(weights=args.weights, threshold=args.threshold)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dire",
        description="DIRE trên máy nhà — cùng lõi với đường Modal (dire_core).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def _common(sp):
        sp.add_argument("--weights", default=None,
                        help="checkpoint .pth (mặc định $DIRE_WEIGHTS hoặc checkpoints/lsun_adm.pth)")

    sp = sub.add_parser("info", help="in cấu hình tiền xử lý + kiểm môi trường (không cần torch)")
    _common(sp)
    sp.set_defaults(func=cmd_info)

    sp = sub.add_parser("detect", help="chấm 1 ảnh")
    _common(sp)
    sp.add_argument("image")
    sp.add_argument("--threshold", type=float, default=0.5,
                    help="ngưỡng ACC (mặc định 0.5, đúng như paper)")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--exit-code", action="store_true",
                    help="trả mã 1 khi kết luận fake (tiện cho shell)")
    sp.set_defaults(func=cmd_detect)

    sp = sub.add_parser("batch", help="chấm cả thư mục → CSV")
    _common(sp)
    sp.add_argument("path")
    sp.add_argument("--out", default="scores.csv")
    sp.add_argument("--batch-size", type=int, default=64)
    sp.add_argument("--threshold", type=float, default=0.5)
    sp.add_argument("--limit", type=int, default=0)
    sp.set_defaults(func=cmd_batch)

    sp = sub.add_parser("serve", help="HTTP API tại máy (fastapi + uvicorn)")
    _common(sp)
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--threshold", type=float, default=0.5)
    sp.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
