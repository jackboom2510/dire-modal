# src/dire_modal/eval.py
"""Eval 1 checkpoint DIRE trên từng cặp (real, generator) — mốc **M1** của SPEC-001.

Vì sao đọc tar gốc trong ``/raw`` mà không đọc shard trong ``/shards``
---------------------------------------------------------------------
Ba lý do, xếp theo mức độ nghiêm trọng:

1. **``pack`` có lỗi làm mất nhánh.** ``pack_one`` đặt shard vào
   ``/shards/{split}/{stem}/`` với ``stem`` là *tên tar*, nên
   ``dire/test/lsun_bedroom/lsun_bedroom/real.tar.gz`` và
   ``.../imagenet/real.tar.gz`` rơi vào **cùng** ``/shards/test/real/``. Eval mà
   đọc shard sẽ gộp hai nhánh — đúng cái ``probe_pair`` đã cảnh báo: đếm trùng
   1000 ảnh real và kéo mean AP về phía LSUN.
2. **Pack là để train.** Docstring của ``pack.py`` đã ghi "pack chỉ phục vụ
   huấn luyện". Eval đọc tuần tự, không cần shard, không cần shuffle.
3. **Ít bước hơn = ít chỗ sai hơn**: eval trên tar gốc không phụ thuộc trạng
   thái ``_DONE``/``meta.json`` của một stage khác.

Nên: eval đọc ``/raw/dire/<split>/<branch>/<sub>/<name>.tar.gz``.

Bộ ba tiền xử lý — sai một bước là lệch số
------------------------------------------
Theo ``utils/datasets.py`` của tác giả (nhánh eval)::

    Resize(256) → CenterCrop(224) → ToTensor() → Normalize(ImageNet)

``demo.py`` của tác giả **thiếu ``Normalize``**; đó là lý do điểm trong demo
không khớp điểm trong paper. Ở đây luôn chuẩn hoá, và ``PREPROC`` được ghi vào
CSV để một lần chạy không thể lẫn với lần chạy khác.
"""

import csv
import datetime
import json
import os
import tarfile
from pathlib import Path

from .common import app, gpu_image, vol_ckpt, vol_raw

# Định danh pipeline tiền xử lý — ghi vào CSV + run.json để so sánh được.
PREPROC = "resize256_centercrop224_normalize"

RESIZE = 256
CROP = 224
# Hằng số ImageNet, đúng như `utils/datasets.py` gốc.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Nhãn: dataset DIRE dùng hai lớp `0_real` / `1_fake` (mode binary của ImageFolder).
REAL, FAKE = 0, 1

# Ngưỡng ACC. Giữ 0.5 cho khớp `accuracy_score(y_true, y_pred > 0.5)` của tác giả;
# ngưỡng hiệu chỉnh trên val là việc của FR-5, không phải ở đây.
ACC_THRESHOLD = 0.5
# Metric AEROBLADE dùng, DIRE paper không dùng — nhưng không có nó thì không so
# được hai detector trên cùng một trục.
FPR_TARGET = 0.05

RESULTS_ROOT = Path("/ckpt/results")


# ============================================================================
# Metric — pure Python, không numpy/sklearn
# ============================================================================
# Viết tay thay vì gọi sklearn vì hai lý do: (a) test phải chạy được trong
# sandbox không có numpy, (b) `average_precision_score` xử lý score TRÙNG nhau
# theo cách riêng — cần khớp nó chứ không phải "gần đúng". Cách làm ở đây gộp
# các score bằng nhau vào cùng một điểm ngưỡng, đúng như sklearn.
def _average_precision(y_true: list[int], y_score: list[float]) -> float:
    """AP = Σ (R_i − R_{i−1}) · P_i trên các ngưỡng phân biệt, giảm dần."""
    n_pos = sum(1 for y in y_true if y == FAKE)
    if n_pos == 0:
        return 0.0
    order = sorted(range(len(y_score)), key=lambda i: -y_score[i])
    ap = 0.0
    tp = fp = 0
    prev_recall = 0.0
    i = 0
    n = len(order)
    while i < n:
        j = i
        score = y_score[order[i]]
        while j < n and y_score[order[j]] == score:
            if y_true[order[j]] == FAKE:
                tp += 1
            else:
                fp += 1
            j += 1
        recall = tp / n_pos
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j
    return ap


def _roc_points(y_true: list[int], y_score: list[float]) -> list[tuple[float, float]]:
    """Các điểm (FPR, TPR) tại mọi ngưỡng phân biệt, kèm (0,0) và (1,1)."""
    n_pos = sum(1 for y in y_true if y == FAKE)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return [(0.0, 0.0), (1.0, 1.0)]
    order = sorted(range(len(y_score)), key=lambda i: -y_score[i])
    pts = [(0.0, 0.0)]
    tp = fp = 0
    i, n = 0, len(order)
    while i < n:
        score = y_score[order[i]]
        while i < n and y_score[order[i]] == score:
            if y_true[order[i]] == FAKE:
                tp += 1
            else:
                fp += 1
            i += 1
        pts.append((fp / n_neg, tp / n_pos))
    return pts


def _tpr_at_fpr(y_true: list[int], y_score: list[float], target: float = FPR_TARGET) -> float:
    """TPR lớn nhất mà FPR ≤ ``target``.

    Đây là metric AEROBLADE báo cáo. Ba chi tiết, mỗi cái từng làm lệch số:

    * ``max`` chứ không phải "điểm cuối": ROC không đơn điệu theo FPR khi có
      score trùng, nên phải quét hết các điểm.
    * Dùng TOÀN BỘ điểm ROC, tức tương đương
      ``sklearn.metrics.roc_curve(..., drop_intermediate=False)``. Mặc định của
      sklearn là ``True`` và nó **bỏ các điểm thẳng hàng**; tại một ngân sách FPR
      cố định, điểm bị bỏ có thể chính là điểm tốt nhất ⇒ TPR thấp hơn sự thật.
    * Ngưỡng là "score ≥ t" như sklearn. Cần biết vì có thể lệch 1 điểm dữ liệu
      ở biên.

    Đã đối chiếu 500 ca ngẫu nhiên (kể cả score trùng) với sklearn: khớp tuyệt
    đối cả AP và TPR@5%FPR.
    """
    best = 0.0
    for fpr, tpr in _roc_points(y_true, y_score):
        if fpr <= target + 1e-12 and tpr > best:
            best = tpr
    return best


def _accuracy(y_true: list[int], y_score: list[float], threshold: float = ACC_THRESHOLD) -> float:
    if not y_true:
        return 0.0
    hits = sum(1 for y, s in zip(y_true, y_score) if (s > threshold) == (y == FAKE))
    return hits / len(y_true)


def _subset_accuracy(y_true, y_score, label: int, threshold: float = ACC_THRESHOLD) -> float:
    pairs = [(y, s) for y, s in zip(y_true, y_score) if y == label]
    if not pairs:
        return 0.0
    return sum(1 for y, s in pairs if (s > threshold) == (y == FAKE)) / len(pairs)


def metrics(y_true: list[int], y_score: list[float]) -> dict:
    """Bộ metric đầy đủ. ``n_real``/``n_fake`` để bắt lỗi nhãn IM LẶNG.

    ``n_fake == 0`` là lỗi dữ liệu nghiêm trọng nhất có thể xảy ra ở đây: AP sẽ
    là 0 (hoặc 1 nếu tính sai) mà không có gì báo. Nên đếm được là điều kiện
    của mọi dòng CSV (AC-1.2 của spec).
    """
    n_fake = sum(1 for y in y_true if y == FAKE)
    n_real = len(y_true) - n_fake
    return {
        "n_real": n_real,
        "n_fake": n_fake,
        "acc": _accuracy(y_true, y_score),
        "ap": _average_precision(y_true, y_score) if n_fake else 0.0,
        "tpr_at_5fpr": _tpr_at_fpr(y_true, y_score) if n_fake and n_real else 0.0,
        "r_acc": _subset_accuracy(y_true, y_score, REAL),
        "f_acc": _subset_accuracy(y_true, y_score, FAKE),
    }


# ============================================================================
# Đọc dữ liệu
# ============================================================================
def label_of(member_name: str, archive_name: str = "") -> int:
    """Nhãn của một member trong tar.

    Ưu tiên cấu trúc ``0_real``/``1_fake`` bên trong member (đó là convention
    ImageFolder của dataset DIRE). Không thấy thì đoán theo TÊN ARCHIVE — vì một
    tar một-generator thường chỉ có ảnh giả trong một thư mục.
    """
    low = member_name.replace("\\", "/").lower()
    if "0_real" in low or "/real/" in low or low.startswith("real/"):
        return REAL
    if "1_fake" in low or "/fake/" in low or low.startswith("fake/"):
        return FAKE
    base = Path(archive_name).name.lower()
    return REAL if base.startswith("real") else FAKE


def iter_images(archive: Path):
    """Yield ``(member_name, bytes)`` cho từng file ảnh trong tar.

    ``"r:*"`` chứ KHÔNG ``"r:gz"``: dữ liệu DIRE là tar **không nén** dù tên
    ``.tar.gz`` (xác nhận 2026-10-21 bằng sha256 + `tarfile`).
    """
    with tarfile.open(archive, "r:*") as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = member.name.lower()
            if not name.endswith((".png", ".jpg", ".jpeg")):
                continue
            data = tf.extractfile(member)
            if data is None:
                continue
            yield member.name, data.read()


def pairs_from_raw(root: Path, split: str = "test") -> list[dict]:
    """Mỗi thư mục chứa ``real.tar.gz`` là một đơn vị eval.

    ``/raw/dire/<split>/<branch>[/<sub>]/`` — độ sâu KHÔNG cố định:
    ``lsun_bedroom/lsun_bedroom/`` có 2 tầng, ``celebahq/`` chỉ 1. Nên nhóm theo
    thư mục cha thật thay vì đoán theo cấu trúc.

    Trả list dict ``{group, branch, generator, real, fake}`` đã sắp xếp theo
    ``branch``/``generator`` để kết quả tất định.
    """
    base = Path(root) / "dire" / split
    out: list[dict] = []
    if not base.is_dir():
        return out
    groups: dict[Path, list[Path]] = {}
    for tar in sorted(base.rglob("*.tar.gz")):
        groups.setdefault(tar.parent, []).append(tar)
    for parent, tars in groups.items():
        real = next((t for t in tars if t.name == "real.tar.gz"), None)
        if real is None:
            continue
        branch = parent.relative_to(base).as_posix().replace("/", "__")
        for fake in sorted(t for t in tars if t is not real):
            out.append(
                {
                    "group": parent.relative_to(base).as_posix(),
                    "branch": branch,
                    "generator": fake.name[: -len(".tar.gz")],
                    "real": real,
                    "fake": fake,
                }
            )
    out.sort(key=lambda d: (d["branch"], d["generator"]))
    return out


# ============================================================================
# Ghi kết quả
# ============================================================================
CSV_FIELDS = [
    "experiment_id", "detector", "branch", "generator", "split", "transform",
    "n_real", "n_fake", "acc", "ap", "tpr_at_5fpr", "threshold",
    "git_sha", "created_at",
]


def write_csv(rows: list[dict], path: Path) -> None:
    """Ghi CSV theo schema §10.1 của SPEC-001.

    Cột thừa trong ``rows`` bị bỏ, cột thiếu thành ô rỗng — nhưng ``write_csv``
    KHÔNG tự bịa giá trị: dòng nào thiếu metric thì để trống, và người đọc thấy
    ngay (NFR-6: không bịa số).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def run_json(path: Path, *, experiment_id: str, tiers: str, n_images: dict,
             rows: int, sysinfo: dict | None = None, extra: dict | None = None) -> dict:
    """Metadata để người khác tái lập mà không phải hỏi (FR-7).

    ``git_sha`` lấy từ env ``GIT_SHA`` (Modal truyền vào lúc chạy) — không đọc
    git trong container vì image không có ``.git``. Thiếu thì ghi
    ``"unknown"``, KHÔNG bịa.
    """
    doc = {
        "spec": "SPEC-001",
        "experiment_id": experiment_id,
        "detector": "dire",
        "git_sha": os.environ.get("GIT_SHA") or "unknown",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "preproc": PREPROC,
        "tiers": [t.strip() for t in str(tiers).split(",") if t.strip()],
        "n_rows": rows,
        "n_images": n_images,
        "acc_threshold": ACC_THRESHOLD,
        "fpr_target": FPR_TARGET,
        "versions": sysinfo or {},
    }
    if extra:
        doc.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc


# ============================================================================
# Phần cần GPU (torch)
# ============================================================================
def preprocess(img, torch, transforms):
    """Resize 256 → CenterCrop 224 → ToTensor → Normalize(ImageNet).

    Trả tensor CHW đã chuẩn hoá. Một chỗ duy nhất, để không có hai bản pipeline
    lệch nhau như ``demo.py`` vs ``datasets.py`` của tác giả.
    """
    tf = transforms.Compose(
        [
            transforms.Resize(RESIZE),
            transforms.CenterCrop(CROP),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return tf(img)


def load_detector(ckpt: str, torch, resnet50, device):
    """ResNet50 nhị phân của DIRE.

    File checkpoint gốc lưu cả ``optimizer``/``epoch`` (269.4 MB so với 97.8 MB
    của ResNet50 fp32), nên phải bóc đúng lớp: ``model``/``state_dict``/``net``,
    rồi bỏ tiền tố ``module.`` do DataParallel.
    """
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict):
        for key in ("model", "state_dict", "net", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    state = {k[len("module."):] if k.startswith("module.") else k: v
             for k, v in state.items()}
    model = resnet50(num_classes=2)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def score_pair(
    model, real: Path, fake: Path, torch, transforms, device,
    batch_size: int = 64, limit: int = 0,
) -> dict:
    """Chạy cả hai archive, trả ``y_true``/``y_score`` và metric.

    ``limit`` > 0 cắt mỗi archive còn ``limit`` ảnh — dùng cho smoke test có
    kiểm soát chi phí (FR-2 của spec dùng đúng đường này với 50+50 ảnh).
    """
    from io import BytesIO

    from PIL import Image  # noqa: PLC0415 — chỉ cần trong container GPU

    y_true: list[int] = []
    y_score: list[float] = []
    skipped = 0
    with torch.inference_mode():
        for archive in (real, fake):
            if not Path(archive).exists():
                continue
            buf, labels = [], []
            for name, data in iter_images(Path(archive)):
                try:
                    img = Image.open(BytesIO(data)).convert("RGB")
                except Exception:  # noqa: BLE001 — một ảnh hỏng không được giết cả eval
                    skipped += 1
                    continue
                buf.append(preprocess(img, torch, transforms))
                labels.append(label_of(name, Path(archive).name))
                if limit and len(labels) >= limit:
                    break
                if len(buf) >= batch_size:
                    probs = model(torch.stack(buf).to(device)).softmax(dim=1)[:, FAKE]
                    y_score.extend(probs.cpu().tolist())
                    y_true.extend(labels)
                    buf, labels = [], []
            if buf:
                probs = model(torch.stack(buf).to(device)).softmax(dim=1)[:, FAKE]
                y_score.extend(probs.cpu().tolist())
                y_true.extend(labels)

    res = metrics(y_true, y_score)
    res["skipped"] = skipped
    return res


# ============================================================================
# Modal functions
# ============================================================================
@app.function(
    image=gpu_image,
    volumes={"/raw": vol_raw, "/ckpt": vol_ckpt},
    gpu="L4",
    timeout=1800,
    max_containers=4,
)
def eval_one(ckpt: str, real: str, fake: str, limit: int = 0,
             batch_size: int = 64) -> dict:
    """Eval MỘT cặp (real, generator). 1 cặp = 1 container.

    ``modal run -m dire_modal.eval::eval_one --ckpt … --real … --fake …``
    """
    import torch
    import torchvision.transforms as transforms

    from .vendor.networks.resnet import resnet50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_detector(ckpt, torch, resnet50, device)
    r = score_pair(model, Path(real), Path(fake), torch, transforms, device,
                   batch_size=batch_size, limit=limit)
    return {"real": real, "fake": fake, **r}


def _sysinfo() -> dict:
    import sys

    info = {"python": sys.version.split()[0]}
    for mod in ("torch", "torchvision", "sklearn", "PIL"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 — thiếu module thì ghi thiếu, không bịa
            pass
    return info


@app.function(
    image=gpu_image,
    volumes={"/raw": vol_raw, "/ckpt": vol_ckpt},
    timeout=4 * 3600,
)
def main(
    ckpt: str = "/ckpt/pretrained/lsun_adm.pth",
    split: str = "test",
    root: str = "/raw",
    exp_id: str = "",
    tiers: str = "e2",
    limit: int = 0,
    batch_size: int = 64,
    only: str = "",
) -> None:
    """Eval checkpoint trên MỌI cặp tìm được, ghi CSV + run.json.

    ``modal run -m dire_modal.eval::main --tiers e2``
    ``modal run -m dire_modal.eval::main --only lsun_bedroom__lsun_bedroom --limit 50``
    """
    if not Path(ckpt).exists():
        raise SystemExit(
            f"[!] Không thấy checkpoint tại {ckpt}.\n"
            "    Cần: modal volume cp /raw/checkpoints/lsun_adm.pth "
            "dire-ckpt:/pretrained/lsun_adm.pth\n"
            "    (xem scripts/04_eval_pretrained.sh)"
        )

    pairs = pairs_from_raw(Path(root), split)
    if only:
        pairs = [p for p in pairs if p["branch"] == only or p["generator"] == only]
    if not pairs:
        raise SystemExit(
            f"[!] Không tìm thấy cặp (real, generator) nào trong {root}/dire/{split}.\n"
            "    → chạy `download::main --mode fetch --tiers e2` trước."
        )

    exp = exp_id or f"eval_{datetime.datetime.now(datetime.timezone.utc):%Y%m%d_%H%M%S}"
    out_dir = RESULTS_ROOT / exp
    print(f"[eval] {len(pairs)} cặp · ckpt={ckpt}\n[eval] → {out_dir}")

    payloads = [(ckpt, str(p["real"]), str(p["fake"]), limit, batch_size) for p in pairs]
    results = list(
        eval_one.starmap(payloads, order_outputs=False)
    )

    created = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    git_sha = os.environ.get("GIT_SHA") or "unknown"
    by_fake = {str(p["fake"]): p for p in pairs}
    rows: list[dict] = []
    for r in results:
        meta = by_fake.get(r["fake"], {})
        rows.append(
            {
                "experiment_id": exp,
                "detector": "dire",
                "branch": meta.get("branch", ""),
                "generator": meta.get("generator", ""),
                "split": split,
                "transform": PREPROC,
                "n_real": r["n_real"],
                "n_fake": r["n_fake"],
                "acc": round(r["acc"], 6),
                "ap": round(r["ap"], 6),
                "tpr_at_5fpr": round(r["tpr_at_5fpr"], 6),
                "threshold": ACC_THRESHOLD,
                "git_sha": git_sha,
                "created_at": created,
            }
        )
    rows.sort(key=lambda d: (d["branch"], d["generator"]))

    n_images = {
        "real_total": sum(r["n_real"] for r in results),
        "fake_total": sum(r["n_fake"] for r in results),
        "limit_per_archive": limit,
        "skipped_unreadable": sum(r.get("skipped", 0) for r in results),
    }
    write_csv(rows, out_dir / "dire.csv")
    run_json(
        out_dir / "run.json",
        experiment_id=exp,
        tiers=tiers,
        n_images=n_images,
        rows=len(rows),
        sysinfo=_sysinfo(),
        extra={"ckpt": ckpt, "split": split, "root": root, "only": only},
    )

    # Bảng in ra: đọc được ngay trong log của Modal, không phải tải CSV về.
    print(f"\n{'branch':<28} {'generator':<18} {'n':>6} {'ACC':>7} {'AP':>7} {'T@5%':>7}")
    print("-" * 78)
    for d in rows:
        print(f"{d['branch']:<28} {d['generator']:<18} "
              f"{d['n_fake']:>6} {d['acc']:>7.4f} {d['ap']:>7.4f} {d['tpr_at_5fpr']:>7.4f}")
    broken = [d for d in rows if d["n_real"] == 0 or d["n_fake"] == 0]
    if broken:
        print("\n[!] Cặp có một lớp RỖNG — nhãn hoặc đường dẫn sai, số ở trên vô nghĩa:")
        for d in broken:
            print(f"    {d['branch']}/{d['generator']}: real={d['n_real']} fake={d['n_fake']}")
    print(f"\n[eval] → {out_dir}/dire.csv · {out_dir}/run.json")
    vol_ckpt.commit()


@app.local_entrypoint()
def local_pairs(root: str = ".", split: str = "test") -> None:
    """Liệt kê cặp sẽ eval, KHÔNG cần Modal/GPU: ``python -m dire_modal.eval``.

    Dùng để soát trước khi tiêu tiền GPU: nếu danh sách này sai thì kết quả
    cũng sai, mà đọc nó miễn phí.
    """
    pairs = pairs_from_raw(Path(root), split)
    print(f"{len(pairs)} cặp trong {root}/dire/{split}")
    for p in pairs:
        print(f"  {p['branch']:<28} {p['generator']:<18} "
              f"{p['real'].name} + {p['fake'].name}")
    branches = sorted({p["branch"] for p in pairs})
    print(f"\n{len(branches)} nhánh: {', '.join(branches)}")
    print("\n⚠ Nhánh `lsun_bedroom__imagenet` là CROSS-GENERATOR (dùng chung 1000 ảnh"
          "\n  real với `lsun_bedroom__lsun_bedroom`) ⇒ báo cáo riêng, đừng cộng"
          "\n  vào mean AP.")
