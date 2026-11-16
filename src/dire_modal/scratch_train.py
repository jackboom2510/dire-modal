"""R2: tự train detector ResNet-50 trên DIRE map **của mình**.

Vì sao không dùng `train.py`
----------------------------
`train.py` là đường đối chiếu (FR-8): nó nạp DIRE map **đã phát hành** và nhằm
trả lời "trọng số tác giả có tái tạo được không". Đường scratch hỏi câu khác:
"từ ảnh mình sinh, map mình tính, có train ra detector hoạt động được không".
Đầu vào khác, câu trả lời khác, nên module riêng — và không ghi vào
`/ckpt/results/` của đường kia.

Giao thức: chép từ `vendor/utils/config.py`, không đoán
------------------------------------------------------
* ResNet-50, **1 logit** + `BCEWithLogitsLoss` (không phải 2 lớp + CE);
* Adam `lr = 1e-4`, batch 64;
* 256 → `RandomCrop(224)`, hflip 0,5 khi train; `CenterCrop(224)` khi eval;
* chuẩn hoá ImageNet sau khi đưa map (3 kênh) về `[0, 1]`;
* chọn epoch **chỉ trên val**; `lr/10` sau 5 epoch không cải thiện, tối đa 3 lần.

Khác một điểm có chủ ý: **lưu cả seed split vào checkpoint**. Một checkpoint
không biết nó train trên seed nào là checkpoint không kiểm chứng được.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from dire_core.scratch import assert_no_published, sha256_file, write_json

from .common import VOLUMES, app, gpu_image

MAPS = "/raw/scratch/maps"
CKPT_OUT = "/ckpt/scratch"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _pairs(maps_root: Path, split: dict | None, side: str) -> list[Path]:
    d = maps_root / side
    return sorted(d.glob("*.npy")) if d.is_dir() else []


def _is_test(name: str, test_seeds: list[int]) -> bool:
    """Tên file map là `<idx>_s<seed>.npy` (xem `scratch_gen.dire`)."""
    if not test_seeds:
        return False
    try:
        seed = int(name.split("_s")[1].split(".")[0])
    except (IndexError, ValueError):
        return False
    return seed in set(test_seeds)


def _dataset_tensors(maps_root: Path, test_seeds: list[int], which: str):
    """Trả `(tensor N×3×256×256 float32, nhãn N)`; nhãn 0 = thật, 1 = giả."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    xs, ys = [], []
    for side, label in (("real", 0), ("fake", 1)):
        for p in _pairs(maps_root, None, side):
            is_test = _is_test(p.name, test_seeds)
            if (which == "test") != is_test:
                continue
            arr = np.load(p).astype("float32")   # (3, 256, 256), [0, 2]
            xs.append(torch.from_numpy(arr))
            ys.append(label)
    if not xs:
        raise SystemExit(f"[!] không có map nào cho tập {which} trong {maps_root}")
    return torch.stack(xs), torch.tensor(ys, dtype=torch.float32)


def _model(torch, resnet50, device):
    m = resnet50(weights=None)
    m.fc = torch.nn.Linear(2048, 1)
    return m.to(device)


def _augment(torch, x, train: bool):
    """Map `[0, 2]` → chuẩn hoá ImageNet, crop 224, hflip."""
    x = (x / 2.0).clamp(0, 1)
    if train:
        i = torch.randint(0, 33, (1,)).item()
        j = torch.randint(0, 33, (1,)).item()
        x = x[:, :, i:i + 224, j:j + 224]
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[3])
    else:
        x = x[:, :, 16:240, 16:240]
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def _acc(logits, y, thr: float = 0.5) -> float:
    import torch  # noqa: PLC0415

    pred = (torch.sigmoid(logits).view(-1) >= thr).float()
    return (pred == y).float().mean().item()


@app.function(image=gpu_image, volumes=VOLUMES, gpu="A10G", timeout=12 * 3600)
def train(
    exp: str = "scratch_adm",
    epochs: int = 40,
    batch: int = 64,
    lr: float = 1e-4,
    seed: int = 3407,
    maps_root: str = MAPS,
    out_root: str = CKPT_OUT,
) -> dict:
    """Train detector trên map của mình; chọn epoch trên **val**, test để cuối."""
    import torch  # noqa: PLC0415
    from torchvision.models import resnet50  # noqa: PLC0415

    root = Path(maps_root)
    assert_no_published([str(p) for p in root.rglob("*.npy")])

    gen_meta_path = Path("/raw/scratch/gen/adm/gen.json")
    test_seeds: list[int] = []
    if gen_meta_path.exists():
        g = json.loads(gen_meta_path.read_text(encoding="utf-8"))
        test_seeds = list(g.get("split", {}).get("test", []))

    torch.manual_seed(seed)
    device = "cuda"

    x_tr, y_tr = _dataset_tensors(root, test_seeds, "train")
    x_te, y_te = _dataset_tensors(root, test_seeds, "test")
    # val: 15% của train, tách theo chỉ số (cố định, ghi vào checkpoint)
    n_val = max(1, int(0.15 * len(x_tr)))
    x_va, y_va = x_tr[-n_val:], y_tr[-n_val:]
    x_tr, y_tr = x_tr[:-n_val], y_tr[:-n_val]
    print(f"[dữ liệu] train {len(x_tr)} · val {len(x_va)} · test {len(x_te)}")

    model = _model(torch, resnet50, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best, best_state, patience, drops = -1.0, None, 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(x_tr))
        tot, seen = 0.0, 0
        for k in range(0, len(x_tr), batch):
            idx = perm[k:k + batch]
            xb = _augment(torch, x_tr[idx].to(device), True)
            yb = y_tr[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb).view(-1), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            seen += len(idx)
        model.eval()
        with torch.inference_mode():
            va = _acc(model(_augment(torch, x_va.to(device), False)).view(-1), y_va.to(device))
        print(f"[epoch {ep:02d}] loss {tot / seen:.4f} · val ACC {va:.4f}")
        if va > best:
            best, patience = va, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 5 and drops < 3:
                for g in opt.param_groups:
                    g["lr"] *= 0.1
                drops += 1
                patience = 0
                print(f"[lr] giảm còn {opt.param_groups[0]['lr']:.1e}")
            elif patience >= 5:
                print("[dừng] hết lượt giảm lr")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    out = Path(out_root) / exp
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "detector.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "n_out": 1,
            "mode": "scratch",
            "claim": "R1+R2+R3",
            "best_val_acc": best,
            "test_seeds": test_seeds,
            "maps_sha": {p.name: sha256_file(p) for p in
                         sorted((root / "fake").glob("*.npy"))[:20]},
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "hyper": {"epochs": epochs, "batch": batch, "lr": lr, "seed": seed,
                      "optimizer": "adam", "loss": "bce_with_logits", "n_out": 1},
        },
        ckpt,
    )

    model.eval()
    with torch.inference_mode():
        test_acc = _acc(model(_augment(torch, x_te.to(device), False)).view(-1),
                        y_te.to(device))
    doc = {
        "exp": exp,
        "n_train": int(len(x_tr)),
        "n_val": int(len(x_va)),
        "n_test": int(len(x_te)),
        "best_val_acc": best,
        "test_acc_at_0.5": test_acc,
        "ckpt": str(ckpt),
        "note": "test chỉ chấm MỘT lần; ngưỡng chốt trên val từ trước (G6)",
    }
    write_json(out / "train.json", doc)
    print(f"[xong] {ckpt} · val {best:.4f} · test {test_acc:.4f}")
    return doc
