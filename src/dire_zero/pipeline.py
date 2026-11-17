"""Bốn bước của phép đo, viết thành hàm thuần để test được với mô hình tí hon.

::

    gen    → ảnh giả (R3)            : sinh từ checkpoint công khai, seed mình chọn
    dire   → map cho cả hai phía (R1): |x − R(I(x))|, cùng một lịch, cùng một S
    train  → detector (R2)           : ResNet-50, ngưỡng chọn trên val (G6)
    report → số + khoảng tin cậy     : ACC@0,5, AP, AUC, CI bootstrap

Mỗi hàm nhận `model` (hoặc factory) làm tham số, **không** tự nạp checkpoint.
Nhờ vậy `tests/test_zero_pipeline.py` chạy trọn bốn bước trên U-Net 32×32 với
8 ảnh trong 20 giây, không cần GPU và không cần tải 2,5 GB.
"""

from __future__ import annotations

import datetime as _dt
import json
import time
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch import Tensor

from . import data as data_mod
from . import metrics
from .diffusion import ModelCounter, Schedule, dire_map, from_uint8, sample, to_uint8

__all__ = [
    "write_json",
    "load_image",
    "to_hwc_uint8",
    "save_png",
    "generate",
    "compute_maps",
    "train",
    "evaluate",
    "RunMeta",
]


def write_json(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_image(path: str | Path) -> Tensor:
    """Ảnh trên đĩa → tensor ``(1, 3, H, W)`` trong ``[−1, 1]``."""
    from PIL import Image  # noqa: PLC0415

    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
        arr = arr.view(im.size[1], im.size[0], 3)
    return from_uint8(arr.numpy())


def to_hwc_uint8(arr: Tensor) -> "object":
    """Tensor ảnh → mảng uint8 ``(H, W[, 3])`` **trên CPU** — mọi đường ghi đĩa đi qua đây.

    Ba bước, theo đúng thứ tự này:

    1. ``.detach()`` — ảnh sinh ra trong ``inference_mode`` nhưng hàm này cũng được
       gọi từ chỗ khác, và ``.numpy()`` trên tensor cần grad sẽ ném;
    2. ``.cpu()`` — **bắt buộc**, không phải tối ưu: ``to_uint8`` giữ nguyên thiết bị
       của đầu vào, mà ``gen`` chạy trên A10G. Thiếu bước này, ``.numpy()`` ném
       ``TypeError: can't convert cuda:0 device type tensor to numpy`` — sau khi đã
       trả tiền cho checkpoint và bước sinh đầu tiên (xem §3.7 của
       ``docs/from-zero-clean-room.md``);
    3. ``.contiguous()`` — ``permute`` trả view không liền mạch; Pillow cần bộ đệm
       liền mạch, và ``.numpy()`` trên view lạ có thể cho mảng có stride lạ.

    Nhận cả ``(1, C, H, W)`` và ``(C, H, W)``; trả ``(H, W)`` khi ``C = 1``.
    """
    u8 = to_uint8(arr).detach().cpu()
    if u8.dim() == 4:
        if u8.shape[0] != 1:
            raise ValueError(f"save_png nhận đúng một ảnh, nhận batch {tuple(u8.shape)}")
        u8 = u8.squeeze(0)
    if u8.dim() != 3:
        raise ValueError(f"cần tensor (1, C, H, W) hoặc (C, H, W), nhận {tuple(u8.shape)}")
    hwc = u8.permute(1, 2, 0).contiguous().numpy()
    return hwc[..., 0] if hwc.shape[-1] == 1 else hwc


def save_png(arr: Tensor, path: str | Path) -> None:
    from PIL import Image  # noqa: PLC0415

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_hwc_uint8(arr)).save(path, format="PNG")


class RunMeta:
    """Giữ mọi thứ cần để người khác dựng lại: checkpoint, bước, seed, phiên bản."""

    def __init__(self, ckpt_sha256: str, steps: int, sampler: str = "ddim", eta: float = 0.0) -> None:
        self.ckpt_sha256 = ckpt_sha256
        self.steps = steps
        self.sampler = sampler
        self.eta = eta

    def as_dict(self) -> dict:
        return {
            "ckpt_sha256": self.ckpt_sha256,
            "sampler": self.sampler,
            "eta": self.eta,
            "steps": self.steps,
            "t_max": Schedule(steps=self.steps).timesteps[-1],
            "nfe_per_image": 2 * self.steps,
            "created": _now(),
        }


def generate(
    f: ModelCounter,
    n: int,
    out_dir: str | Path,
    meta: RunMeta,
    seed_base: int = 1000,
    size: int = 256,
    split_seed: int = 3407,
    progress: Callable[[int], None] | None = None,
) -> dict:
    """R3 — sinh ``n`` ảnh, ghi PNG + `gen.json` (đủ trường để dựng lại)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sched = Schedule(steps=meta.steps, device=str(next(f.model.parameters()).device))
    f.reset()
    names: list[str] = []
    t0 = time.time()
    for i in range(n):
        seed = seed_base + i
        img = sample(f, (1, 3, size, size), sched, seed=seed)
        name = f"{seed:06d}.png"
        save_png(img, out / name)
        names.append(name)
        if progress and (i + 1) % 50 == 0:
            progress(i + 1)

    split = data_mod.split_indices(names, seed=split_seed, frac_train=0.8)
    payload = meta.as_dict()
    payload.update(
        n=n,
        seed_base=seed_base,
        seeds=[seed_base + i for i in range(n)],
        names=names,
        seed_test=split_seed,
        split={"train": [names[i] for i in split["train"]], "test": [names[i] for i in split["test"]]},
        nfe_total=f.nfe,
        seconds=round(time.time() - t0, 2),
    )
    write_json(out / "gen.json", payload)
    return payload


def map_rel_path(item) -> Path:
    """Đường dẫn **tương đối** của map cho một item: ``<split>/<generator>/<tên>.npy``.

    Một hàm duy nhất cho cả ba chỗ cần biết "map này nằm ở đâu": lúc ghi
    (`compute_maps`), lúc **đếm cái đã có** (`--skip-existing`), và lúc đối chiếu.
    Ba bản sao của cùng một quy tắc đặt tên là ba cơ hội để chúng lệch nhau.
    """
    it = _as_item(item)
    return Path(it["split"] or "all") / it["generator"] / (Path(it["path"]).stem + ".npy")


def _index_run(index_path: Path, run: dict, items: list[dict]) -> dict:
    """Gộp lần chạy vào **sổ** `dire.json` thay vì ghi đè nó.

    Lỗi thật: chạy `::dire` lần hai (ví dụ để thêm map cho ảnh vừa sinh) ghi đè
    `dire.json`, nên bản ghi của lần đầu — số ảnh, NFE, số giây — **biến mất**,
    trong khi các `.npy` của lần đầu vẫn nằm đó. Kết quả là một cây map không còn
    tài liệu nào giải thích, tức là đúng thứ mà nhãn `R1` phải tránh.

    Giữ nguyên các khoá cũ ở cấp ngoài (`n`, `steps`, `nfe_total`, `seconds`,
    `items`) để chỗ đọc cũ không hỏng; thêm `runs` (mỗi lần chạy một mục) và
    `dire-runs/<created>.json` (bản đầy đủ, không sửa nữa). Nếu các lần chạy dùng
    `steps` khác nhau thì `steps` ở cấp ngoài là **null** kèm `steps_mixed`, chứ
    không lấy một giá trị bất kỳ làm đại diện.
    """
    old: dict = {}
    if index_path.exists():
        try:
            old = json.loads(index_path.read_text(encoding="utf-8"))
        except ValueError:
            old = {}

    runs = [dict(r) for r in (old.get("runs") or []) if isinstance(r, dict)]
    if not runs and old.get("n"):
        # `dire.json` của bản cũ: không có `runs`, nhưng số liệu vẫn là thật.
        runs.append({k: old.get(k) for k in ("created", "steps", "n", "nfe_total", "seconds", "ckpt_sha256")})
    runs.append(
        {
            k: run.get(k)
            for k in ("created", "steps", "n", "n_new", "n_reused", "nfe_total", "seconds",
                      "ckpt_sha256", "generators", "splits", "skip_existing")
        }
    )

    by_path: dict[str, dict] = {}
    for it in old.get("items") or []:
        if isinstance(it, dict) and it.get("path"):
            by_path[it["path"]] = it
    for it in items:
        by_path[it["path"]] = it

    steps = sorted({r["steps"] for r in runs if r.get("steps") is not None})
    merged = {
        "n": len(by_path),
        "n_new": sum(int(r.get("n_new") or r.get("n") or 0) for r in runs),
        "n_reused": sum(int(r.get("n_reused") or 0) for r in runs),
        "steps": steps[0] if len(steps) == 1 else None,
        "steps_mixed": steps,
        "nfe_total": sum(int(r.get("nfe_total") or 0) for r in runs),
        "seconds": round(sum(float(r.get("seconds") or 0) for r in runs), 2),
        "scale": run.get("scale"),
        "ckpt_sha256": run.get("ckpt_sha256"),
        "generators": sorted({it["generator"] for it in by_path.values()}),
        "splits": sorted({str(it.get("split") or "") for it in by_path.values()}),
        "runs": runs,
        "items": sorted(by_path.values(), key=lambda d: d["path"]),
    }
    write_json(index_path, merged)
    # số thứ tự trong tên: hai lần chạy trong cùng một giây (test, hoặc chạy lại
    # ngay sau khi sửa) không được ghi đè bản ghi của nhau.
    archive = index_path.parent / "dire-runs" / f"{run.get('created') or _now()}-{len(runs):03d}.json"
    write_json(archive, {**run, "items": items})
    return merged


def compute_maps(
    f: ModelCounter,
    items: Sequence[tuple],
    out_dir: str | Path,
    meta: RunMeta,
    limit: int | None = None,
    size: int = 256,
    progress: Callable[[int], None] | None = None,
    skip_existing: bool = False,
) -> dict:
    """R1 — tính DIRE map cho ``items``; mỗi item là dict hoặc tuple.

    Tuple: ``(path, label[, split[, generator]])``. Split **được ghi lại**, không
    tính lại: cây DiffusionForensics đã có ``train``/``val``/``test`` do người
    đóng gói quyết định, chia lại theo hash là bỏ quyết định đó và tự tạo rò rỉ.

    Lưu float16, thang ``[0, 2]``.

    Lưu float16 chứ không uint8: thang ``[0, 2]`` chỉ có 256 mức trong uint8, và
    DIRE sống ở **biên độ nhỏ** — lượng tử hoá 8-bit làm mất đúng phần tín hiệu
    mà detector cần. Giá phải trả là gấp đôi dung lượng; đó là lựa chọn có ý thức.

    ``skip_existing=True``: map nào đã có (file khác rỗng) thì **không tính lại**.
    Không có cờ này, mỗi lần thêm một generator mới là mỗi lần trả tiền tính lại
    toàn bộ ảnh `real` — chúng luôn nằm trong danh sách, và đường dẫn map của
    chúng không đổi, nên lần chạy sau ghi đè đúng file vừa trả tiền để tạo.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = next(f.model.parameters()).device
    sched = Schedule(steps=meta.steps, device=str(device))
    f.reset()
    used = items[:limit] if limit else items
    t0 = time.time()
    made = []
    reused = []
    skipped = 0
    for i, raw in enumerate(used, start=1):
        item = _as_item(raw)
        rel = map_rel_path(item)
        dst = out / rel
        if skip_existing and dst.exists() and dst.stat().st_size > 0:
            reused.append(
                {
                    "name": rel.name,
                    "path": rel.as_posix(),
                    "abs": str(dst),
                    "label": int(item["label"]),
                    "split": item["split"],
                    "generator": item["generator"],
                    "source": item["path"],
                    "reused": True,
                }
            )
            if progress and i % 50 == 0:
                progress(i)
            continue
        img = load_image(item["path"])
        if img.shape[-1] != size or img.shape[-2] != size:
            skipped += 1
            continue
        # `.to(device)` là **bắt buộc**, không phải tối ưu: `load_image` trả tensor
        # CPU, còn mô hình nằm trên GPU ở chế độ thật. Thiếu bước này, `timesteps`
        # được tạo trên `x.device` (CPU) ⇒ `time_embedding` ra CPU ⇒ `time_mlp`
        # (CUDA) chết ở `addmm` với "Expected all tensors to be on the same device".
        # Test CPU không bắt được vì ở đó hai bên cùng CPU; `selfcheck` cũng không
        # bắt được vì nó tự gọi `.cuda()` trước khi vào `dire_map`. Đặt **sau** kiểm
        # kích thước để ảnh sai cỡ không bị chuyển lên GPU vô ích.
        dm, _recon = dire_map(f, img.to(device), sched)
        # <split>/<generator>/<tên>.npy — nhãn và split đi cùng dữ liệu, không
        # suy lại từ tên file (tên PNG là sha, không mang thông tin)
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        import numpy as np  # noqa: PLC0415

        # `.cpu()` trước `.numpy()`: map nằm trên GPU, còn `np.save` cần bộ nhớ CPU.
        np.save(out / rel, dm.squeeze(0).detach().to("cpu", torch.float16).numpy())
        made.append(
            {
                "name": rel.name,
                "path": rel.as_posix(),
                "abs": str(out / rel),
                "label": int(item["label"]),
                "split": item["split"],
                "generator": item["generator"],
                "source": item["path"],
            }
        )
        if progress and i % 50 == 0:
            progress(i)

    payload = meta.as_dict()
    payload.update(
        n=len(made) + len(reused),
        n_new=len(made),
        n_reused=len(reused),
        skipped=skipped,
        nfe_total=f.nfe,
        seconds=round(time.time() - t0, 2),
        generators=sorted({it["generator"] for it in made + reused}),
        splits=sorted({str(it["split"] or "") for it in made + reused}),
        skip_existing=skip_existing,
        scale="[0, 2] float16, không nhân 4, không clip",
        items=made + reused,
    )
    _index_run(out / "dire.json", payload, made + reused)
    return payload


def _as_item(raw) -> dict:
    """Chuẩn hoá item: dict, hoặc tuple ``(path, label[, split[, generator]])``."""
    if isinstance(raw, dict):
        d = dict(raw)
    else:
        seq = tuple(raw)
        d = {
            "path": seq[0],
            "label": seq[1],
            "split": seq[2] if len(seq) > 2 else "",
            "generator": seq[3] if len(seq) > 3 else "",
        }
    d.setdefault("split", "")
    if not d.get("generator"):
        d["generator"] = "real" if int(d.get("label", 0)) == 0 else "fake"
    return d


def _split_maps(items: list[dict], seed: int = 3407, frac_train: float = 0.8) -> dict[str, list[dict]]:
    """Ưu tiên split **đã ghi** trong `dire.json`; chỉ chia theo hash khi không có."""
    labelled = [it for it in items if it.get("split") in ("train", "val", "test")]
    if labelled and len(labelled) == len(items):
        return {k: [it for it in items if it["split"] == k] for k in ("train", "val", "test")}
    names = [it["name"] for it in items]
    idx = data_mod.split_indices(names, seed=seed, frac_train=frac_train)
    return {"train": [items[i] for i in idx["train"]], "val": [], "test": [items[i] for i in idx["test"]]}


def _batch(maps: list[Path], labels: list[int], bs: int, crop: int, train: bool, gen: torch.Generator):
    """Sinh batch: cắt ngẫu nhiên (train) hoặc giữa ảnh (test) + lật ngang."""
    import numpy as np  # noqa: PLC0415

    order = torch.randperm(len(maps), generator=gen).tolist()
    for start in range(0, len(order), bs):
        chunk = order[start:start + bs]
        xs, ys = [], []
        for i in chunk:
            arr = np.load(maps[i]).astype("float32")
            t = torch.from_numpy(arr)
            if train:
                top = int(torch.randint(0, t.shape[1] - crop + 1, (1,), generator=gen))
                left = int(torch.randint(0, t.shape[2] - crop + 1, (1,), generator=gen))
                t = t[:, top:top + crop, left:left + crop]
                if torch.rand(1, generator=gen).item() < 0.5:
                    t = torch.flip(t, dims=[2])
            else:
                top = (t.shape[1] - crop) // 2
                left = (t.shape[2] - crop) // 2
                t = t[:, top:top + crop, left:left + crop]
            xs.append(t)
            ys.append(float(labels[i]))
        yield torch.stack(xs), torch.tensor(ys, dtype=torch.float32)


@torch.no_grad()
def _score(model, maps, labels, crop, device, bs=32) -> tuple[list[float], list[int]]:
    model.eval()
    scores, ys = [], []
    gen = torch.Generator().manual_seed(0)
    for xb, yb in _batch(maps, labels, bs, crop, False, gen):
        p = model(xb.to(device)).float().cpu().tolist()
        scores.extend(p)
        ys.extend(int(v) for v in yb.tolist())
    return scores, ys


def train(
    model_factory: Callable[[], torch.nn.Module],
    maps: list[dict],
    out_dir: str | Path,
    epochs: int = 40,
    lr: float = 1e-4,
    bs: int = 32,
    crop: int = 224,
    device: str = "cpu",
    seed: int = 3407,
    split_seed: int = 3407,
    patience: int = 5,
    max_lr_drops: int = 3,
    init: str = "scratch",
    max_minutes: float | None = None,
    extra: dict | None = None,
) -> dict:
    """R2 — train detector. Ngưỡng chọn trên **val**, test chấm **một lần** (G6).

    Early stop theo val AP (không theo val ACC): AP dùng được cả khi lớp mất cân
    bằng và không phụ thuộc một ngưỡng cứng — chọn mô hình bằng ACC rồi báo AP
    là trộn hai tiêu chí.
    """
    torch.manual_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    parts = _split_maps(maps, split_seed)
    tr, te = parts["train"], parts["test"]
    if parts["val"]:
        va = parts["val"]                       # split của người đóng gói: dùng nguyên
    else:
        pool = tr                               # không có val ⇒ lấy một phần train
        inner = data_mod.split_indices([m["name"] for m in pool], seed=split_seed + 1, frac_train=0.85)
        tr = [pool[i] for i in inner["train"]]
        va = [pool[i] for i in inner["test"]]

    model = model_factory().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)

    def packs(rows):
        return [r["path"] for r in rows], [r["label"] for r in rows]

    tr_p, tr_y = packs(tr)
    va_p, va_y = packs(va)
    te_p, te_y = packs(te)

    best = {"ap": -1.0, "epoch": -1, "state": None}
    drops = 0
    history = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        tot, seen = 0.0, 0
        for xb, yb in _batch(tr_p, tr_y, bs, crop, True, gen):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(yb)
            seen += len(yb)

        vs, vy = _score(model, va_p, va_y, crop, device)
        vap = metrics.average_precision(vs, vy)
        history.append({"epoch": epoch, "loss": round(tot / max(seen, 1), 5), "val_ap": round(vap, 5)})
        if vap > best["ap"] + 1e-4:
            best = {"ap": vap, "epoch": epoch, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            drops = 0
        else:
            if drops < max_lr_drops:
                for g in opt.param_groups:
                    g["lr"] = g["lr"] / 10
                drops += 1
            if epoch - best["epoch"] >= patience:
                break
        if max_minutes and (time.time() - t0) / 60 > max_minutes:
            history.append({"stopped": f"quá {max_minutes} phút"})
            break

    if best["state"] is None:
        # val chỉ có một lớp (thường vì quá ít ảnh) ⇒ AP = NaN, không có gì để chọn.
        # Vẫn dùng mô hình cuối, nhưng ghi rõ vào `train.json` thay vì giả vờ đã chọn.
        best = {"ap": float("nan"), "epoch": epoch, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        history.append({"warning": "val có một lớp — không chọn được mô hình theo val AP"})

    model.load_state_dict(best["state"])
    vs, vy = _score(model, va_p, va_y, crop, device)
    # Ngưỡng chọn trên **val** (G6) theo balanced accuracy — không theo ACC thường,
    # và không bao giờ nhìn test. Val ít mẫu thì lùi về 0,5 và ghi rõ nguồn.
    best_thr, thr_source = metrics.pick_threshold(vs, vy)
    best_acc = metrics.accuracy(vs, vy, best_thr)

    ts_, ty_ = _score(model, te_p, te_y, crop, device)
    test = metrics.evaluate(ts_, ty_, threshold=best_thr, seed=seed)

    torch.save(model.state_dict(), out / "detector.pth")
    payload = {
        "label": "R1+R2+R3",
        "init": init,
        "epochs_run": len(history),
        "best_epoch": best["epoch"],
        "best_val_ap": round(best["ap"], 5),
        "threshold_from_val": best_thr,
        "threshold_source": thr_source,
        "val_acc_at_threshold": round(best_acc, 5),
        "test": test,
        "split": {"n_train": len(tr), "n_val": len(va), "n_test": len(te), "seed": split_seed},
        "hparams": {"lr": lr, "batch": bs, "crop": crop, "seed": seed, "patience": patience},
        "history": history,
        "seconds": round(time.time() - t0, 2),
        "honesty": (
            "test chấm đúng một lần sau khi đã chốt mô hình và ngưỡng trên val; "
            "con số này KHÔNG so được với Table 2 của bài báo (khác init, khác seed, "
            "khác generator)."
        ),
    }
    # `extra`: bối cảnh mà hàm này không biết (ví dụ: đã lọc bài toán theo
    # generator nào). Ghi cùng chỗ với số liệu, để đọc `train.json` là đủ.
    payload.update(extra or {})
    write_json(out / "train.json", payload)
    return payload


def evaluate(scores: Sequence[float], labels: Sequence[int], threshold: float = 0.5, seed: int = 0) -> dict:
    return metrics.evaluate(scores, labels, threshold=threshold, seed=seed)


def collect_maps(root: str | Path) -> list[dict]:
    """Gom ``.npy`` dưới ``root/<split>/<generator>/`` thành item cho `train`.

    Nhãn đọc từ **tên thư mục generator**: ``real`` ⇒ 0, còn lại ⇒ 1. Nhờ vậy
    một cây chứa nhiều generator (``adm``, ``pndm``, ``ours_adm``) vẫn train được
    mà không truyền nhãn riêng — và không thể lẫn nhãn.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    out: list[dict] = []
    for p in sorted(root.rglob("*.npy")):
        rel = p.relative_to(root)
        generator = rel.parts[-2] if len(rel.parts) >= 2 else "fake"
        split = rel.parts[0] if len(rel.parts) >= 3 else ""
        out.append(
            {
                "name": p.name,
                "path": str(p),
                "label": 0 if generator == "real" else 1,
                "split": split,
                "generator": generator,
            }
        )
    return out
