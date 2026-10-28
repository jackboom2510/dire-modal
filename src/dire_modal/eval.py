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
import math as _math
import os
import tarfile
from pathlib import Path

from .common import app, cpu_image, gpu_image, vol_ckpt, vol_raw

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

# Chỗ tìm checkpoint. `/raw/checkpoints/...` đứng ĐẦU vì đó là nơi `download`
# đổ file về — bắt người dùng chạy tay `modal volume cp` sang Volume thứ hai chỉ
# để eval một checkpoint là một bước thừa dễ quên, và nó từng là lý do
# `compute_dire` báo "không thấy checkpoint" dù file đã tải xong.
CKPT_CANDIDATES = (
    "/raw/checkpoints/lsun_adm.pth",
    "/ckpt/pretrained/lsun_adm.pth",
)

# Đường dẫn tới bảng số THAM CHIẾU của tác giả (Table 2, khối DIRE). Nằm trong
# package chứ không ở `manifests/` vì Modal chỉ đóng gói thư mục module — cùng
# lý do `tiers.yaml` nằm trong package.
PAPER_TABLE = Path(__file__).with_name("paper_table2.tsv")


def resolve_ckpt(explicit: str = "", name: str = "lsun_adm.pth") -> str:
    """Đường dẫn checkpoint thật, theo thứ tự ưu tiên; ``""`` nếu không có.

    Không bịa đường dẫn: trả rỗng để caller in hướng dẫn cụ thể.
    """
    for cand in (explicit, f"/raw/checkpoints/{name}", f"/ckpt/pretrained/{name}"):
        if cand and Path(cand).exists():
            return cand
    return ""


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
    # Cột cuối: cặp KHÔNG chấm được (checkpoint sai kiểu, tar hỏng, …). Để trống
    # nghĩa là "chạy xong"; có chữ nghĩa là "cặp này không có số" — không được để
    # một dòng toàn ô rỗng trông giống "đã chấm nhưng ra 0".
    "error",
]


def row_for_result(exp: str, r: dict, meta: dict, split: str,
                   git_sha: str, created: str) -> dict:
    """Một dòng CSV từ kết quả của ``eval_one`` — kể cả kết quả LỖI.

    Cặp lỗi phải để **trống** metric, không được ghi 0: dòng ``acc = 0`` đọc như
    "detector đoán sai hết", còn dòng trống + cột ``error`` đọc đúng là "cặp này
    không chấm được". NFR-6: không bịa số.
    """
    err = r.get("error") or ""
    if err:
        return {
            "experiment_id": exp, "detector": "dire",
            "branch": meta.get("branch", ""), "generator": meta.get("generator", ""),
            "split": split, "transform": PREPROC,
            "n_real": "", "n_fake": "", "acc": "", "ap": "", "tpr_at_5fpr": "",
            "threshold": ACC_THRESHOLD, "git_sha": git_sha, "created_at": created,
            "error": err,
        }
    return {
        "experiment_id": exp, "detector": "dire",
        "branch": meta.get("branch", ""), "generator": meta.get("generator", ""),
        "split": split, "transform": PREPROC,
        "n_real": r["n_real"], "n_fake": r["n_fake"],
        "acc": round(r["acc"], 6), "ap": round(r["ap"], 6),
        "tpr_at_5fpr": round(r["tpr_at_5fpr"], 6),
        "threshold": ACC_THRESHOLD, "git_sha": git_sha, "created_at": created,
        "error": "",
    }


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


def fc_out_features(state: dict) -> int:
    """Số unit của lớp cuối, đọc từ **chính checkpoint** — không đoán theo tên.

    Vì sao phải đọc thay vì chọn một con số: DIRE gốc dùng
    ``resnet(num_classes=1)`` + ``BCEWithLogitsLoss`` (một logit, ``sigmoid`` >
    0.5 là fake), trong khi ResNet của torchvision/timm quen dùng 2 lớp +
    ``CrossEntropyLoss``. Hai quy ước cho ra ``fc.weight`` shape ``[1, 2048]`` và
    ``[2, 2048]``; chọn sai thì ``load_state_dict`` nổ:

        size mismatch for fc.weight: copying a param with shape torch.Size([1, 2048])
        from checkpoint, the shape in current model is torch.Size([2, 2048])

    và đó là lỗi chặn TOÀN BỘ eval (lần chạy 2026-10-28) — sau khi đã trả tiền
    bật container GPU.

    Trả ``0`` nếu không thấy ``fc.weight`` (checkpoint lạ) để caller báo rõ.
    """
    for key in ("fc.weight", "module.fc.weight", "_orig_mod.fc.weight"):
        w = state.get(key)
        if w is None:
            continue
        shape = getattr(w, "shape", None)
        if shape is None:
            continue
        try:
            if len(shape) == 2:
                return int(shape[0])
        except TypeError:      # shape không phải chuỗi số
            continue
    return 0


def prob_fake(rows, n_out: int = 1):
    """Logit thô → ``P(fake)``, theo đúng quy ước mà checkpoint được train.

    ``n_out == 1``: ``sigmoid(z)`` — DIRE gốc, ``1_fake`` là lớp dương.
    ``n_out >= 2``: ``softmax(z)[FAKE]``.

    Nhận list ``float`` (một logit/ảnh) hoặc list ``list[float]`` (nhiều logit).
    Tính bằng toán Python thuần trên list đã qua ``.tolist()``: batch 64 nên chi
    phí không đáng kể, và đổi lại hàm này test được **không cần torch** — đúng
    loại logic đã sai âm thầm một lần.
    """
    out: list[float] = []
    for row in rows:
        if n_out == 1:
            z = float(row[0] if isinstance(row, (list, tuple)) else row)
            # Chặn tràn số: sigmoid(-800) = 0.0 là đúng, math.exp sẽ raise.
            if z >= 0:
                out.append(1.0 / (1.0 + _math.exp(-z)))
            else:
                e = _math.exp(z)
                out.append(e / (1.0 + e))
            continue
        vals = list(row) if isinstance(row, (list, tuple)) else [float(row)]
        if len(vals) <= FAKE:
            # Không đủ cột để lấy lớp fake — thà báo lỗi còn hơn bịa điểm.
            raise ValueError(f"model trả {len(vals)} logit, cần > {FAKE}")
        m = max(vals)
        exps = [_math.exp(v - m) for v in vals]
        out.append(exps[FAKE] / sum(exps))
    return out


def load_detector(ckpt: str, torch, resnet50, device):
    """ResNet50 nhị phân của DIRE → ``(model, n_out)``.

    ``n_out`` là số unit lớp cuối **đọc từ checkpoint** (xem
    :func:`fc_out_features`): 1 với checkpoint gốc của tác giả, 2 với checkpoint
    huấn luyện theo kiểu CrossEntropy. Nhờ vậy một checkpoint thuộc quy ước nào
    cũng nạp được, và không có chỗ nào phải "đoán 2 cho chắc".

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

    n_out = fc_out_features(state)
    if n_out not in (1, 2):
        raise SystemExit(
            f"[!] Không đọc được lớp cuối của checkpoint: {ckpt}\n"
            f"    Có {len(state)} tensor, không thấy `fc.weight` shape [n, 2048].\n"
            f"    Khoá đầu tiên: {list(state)[:5]}\n"
            "    Kiểm file có đúng là ResNet50 của DIRE không "
            "(`modal run -m dire_modal.download::verify` in sha256 để đối chiếu)."
        )

    model = resnet50(num_classes=n_out)
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise SystemExit(
            f"[!] Checkpoint không khớp kiến trúc ResNet50: {ckpt}\n"
            f"    n_out đọc từ checkpoint = {n_out}\n"
            f"    {exc}\n"
            "    Gợi ý: `fc.weight` [1,2048] = DIRE gốc (BCE, 1 logit); [2,2048] = 2 lớp."
        ) from exc
    model.to(device).eval()
    return model, n_out


def score_pair(
    model, real: Path, fake: Path, torch, transforms, device,
    batch_size: int = 64, limit: int = 0, n_out: int = 1,
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
                    out = model(torch.stack(buf).to(device))
                    y_score.extend(prob_fake(out.cpu().tolist(), n_out))
                    y_true.extend(labels)
                    buf, labels = [], []
            if buf:
                out = model(torch.stack(buf).to(device))
                y_score.extend(prob_fake(out.cpu().tolist(), n_out))
                y_true.extend(labels)

    res = metrics(y_true, y_score)
    res["skipped"] = skipped
    return res


# ============================================================================
# Modal functions
# ============================================================================
# ============================================================================
# Báo cáo: gộp nhiều lần eval thành bảng so với paper
# ============================================================================
# Quy trình tái lập cần một câu trả lời dạng "chúng ta được 100/100, paper 100/100"
# cho 4 detector × 8 generator — đúng khối DIRE của Table 2. Làm việc gộp này
# trong Python (không phải bash) vì phải đọc CSV + run.json và tính Δ.


def load_paper_table(path: Path | str | None = None) -> dict[str, dict[str, dict[str, float]]]:
    """Đọc `paper_table2.tsv` → ``{ckpt: {generator: {"acc": %, "ap": %}}}``.

    Đơn vị là **phần trăm** như paper in. Hàng `avg` được giữ nguyên (paper in
    sẵn) và `tests/test_paper_table.py` kiểm nó khớp trung bình 8 ô.
    """
    p = Path(path) if path else PAPER_TABLE
    out: dict[str, dict[str, dict[str, float]]] = {}
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("ckpt\t"):
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                raise ValueError(f"dòng hỏng trong {p.name}: {line!r}")
            ckpt, gen, acc, ap = parts
            out.setdefault(ckpt, {})[gen] = {"acc": float(acc), "ap": float(ap)}
    return out


def ckpt_stem(ckpt: str | Path) -> str:
    """``/raw/checkpoints/lsun_adm.pth`` → ``lsun_adm`` (khoá để so với paper)."""
    return Path(str(ckpt)).name.rsplit(".", 1)[0]


def _pct(v) -> float | None:
    """CSV lưu 0..1; paper in %. Trả ``None`` khi ô trống (NFR-6: không bịa số)."""
    try:
        if v is None or str(v).strip() == "":
            return None
        return round(float(v) * 100.0, 4)
    except (TypeError, ValueError):
        return None


def collect_results(results_root: Path | str, exp_ids: list[str] | None = None
                    ) -> dict[str, dict[str, dict[str, float | None]]]:
    """Gom mọi ``results/<exp>/dire.csv`` → ``{ckpt: {generator: {acc, ap}}}``.

    Nhóm theo **stem của checkpoint** đọc từ ``run.json → extra.ckpt``, không
    theo tên thư mục: tên thư mục do người dùng đặt (`--exp-id spec001_fr1`),
    còn ckpt mới là thứ paper phân biệt (4 hàng = 4 detector).
    """
    root = Path(results_root)
    if exp_ids:
        dirs = [root / e for e in exp_ids]
    else:
        dirs = sorted(d for d in root.iterdir() if (d / "dire.csv").exists()) if root.is_dir() else []

    out: dict[str, dict[str, dict[str, float | None]]] = {}
    for d in dirs:
        csv_path = d / "dire.csv"
        if not csv_path.exists():
            print(f"[skip] {d.name}: không có dire.csv")
            continue
        ckpt = d.name
        meta = d / "run.json"
        if meta.exists():
            try:
                info = json.loads(meta.read_text(encoding="utf-8"))
                ckpt = ckpt_stem(
                    (info.get("extra") or {}).get("ckpt")
                    or (info.get("data") or {}).get("ckpt")
                    or ckpt
                )
            except Exception:  # noqa: BLE001 — run.json hỏng thì dùng tên thư mục
                pass
        with csv_path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if (row.get("split") or "test") != "test":
                    continue
                # Chỉ nhánh test của LSUN-Bedroom có 8 cột paper. Nhánh
                # `imagenet__*` là cross-dataset (Table 3) — vẫn hiện, nhưng
                # không so Δ với Table 2.
                out.setdefault(ckpt, {})[row.get("generator") or "?"] = {
                    "acc": _pct(row.get("acc")),
                    "ap": _pct(row.get("ap")),
                    "branch": row.get("branch") or "",
                    "n_fake": row.get("n_fake") or "",
                }
    return out


def compare_to_paper(
    ours: dict, paper: dict, tol: float = 0.5
) -> dict:
    """So từng ô có ở cả hai bên. Trả ``cells`` / ``missing`` / ``worst``.

    ``tol`` = dung sai tuyệt đối (điểm %) để coi là "khớp". 0.5 điểm là hợp lý:
    paper in 1 chữ số thập phân, còn lệch thật sự giữa hai lần chạy phải là 0
    (eval tất định, AC-1.4).
    """
    cells: list[dict] = []
    missing: list[dict] = []
    for ckpt, gens in paper.items():
        for gen, ref in gens.items():
            if gen == "avg":
                continue
            got = (ours.get(ckpt) or {}).get(gen)
            if got is None or got.get("acc") is None:
                missing.append({"ckpt": ckpt, "generator": gen})
                continue
            d_acc = round(float(got["acc"]) - ref["acc"], 4)
            d_ap = (round(float(got["ap"]) - ref["ap"], 4)
                    if got.get("ap") is not None else None)
            cells.append({
                "ckpt": ckpt, "generator": gen,
                "acc": got["acc"], "acc_paper": ref["acc"], "d_acc": d_acc,
                "ap": got.get("ap"), "ap_paper": ref["ap"], "d_ap": d_ap,
                "ok": abs(d_acc) <= tol and (d_ap is None or abs(d_ap) <= tol),
            })
    worst = max(cells, key=lambda c: abs(c["d_acc"]), default=None)
    return {
        "cells": cells,
        "missing": missing,
        "worst": worst,
        "n_cells": len(cells),
        "n_ok": sum(1 for c in cells if c["ok"]),
    }


def _delta(ours: float | None, paper: float | None, width: int = 7) -> str:
    """Chuỗi ΔACC/ΔAP. Làm tròn về 0.0 để không in ``-0.0`` (gây hiểu nhầm âm)."""
    if ours is None or paper is None:
        return " " * width + "-"
    d = round(float(ours) - float(paper), 1)
    if abs(d) < 0.05:
        d = 0.0
    return f"{d:+{width}.1f}"


def render_report(
    ours: dict, paper: dict | None = None, tol: float = 0.5
) -> list[str]:
    """Render bảng so sánh. Tách khỏi I/O để test được không cần Modal."""
    lines: list[str] = []
    order = ["adm", "ddpm", "iddpm", "pndm", "sdv1", "sdv2", "ldm", "vqdiffusion"]
    for ckpt in sorted(ours):
        gens = ours[ckpt]
        paper_row = (paper or {}).get(ckpt) or {}
        # Chỉ in 8 cột paper trước, rồi các generator mở rộng sau — để bảng đọc
        # được như Table 2 và phần mở rộng không chen vào giữa.
        known = [g for g in order if g in gens]
        extra = sorted(g for g in gens if g not in order)
        lines.append("")
        lines.append("=" * 78)
        lines.append(f"detector: {ckpt}   (ckpt của tác giả, KHÔNG train lại)")
        lines.append("=" * 78)
        lines.append(f"  {'generator':<14} {'ACC':>7} {'paper':>7} {'Δ':>7} "
                     f"{'AP':>7} {'paper':>7} {'Δ':>7}   n_fake")
        for gen in [*known, *extra]:
            g = gens[gen]
            ref = paper_row.get(gen) or {}
            pa = ref.get("acc")
            pap = ref.get("ap")
            d_acc = _delta(g.get("acc"), pa)
            d_ap = _delta(g.get("ap"), pap)
            a = f"{g['acc']:7.1f}" if g.get("acc") is not None else "      -"
            ap = f"{g['ap']:7.1f}" if g.get("ap") is not None else "      -"
            pas = f"{pa:7.1f}" if pa is not None else "      -"
            paps = f"{pap:7.1f}" if pap is not None else "      -"
            lines.append(f"  {gen:<14} {a} {pas} {d_acc} {ap} {paps} {d_ap}   "
                         f"{g.get('n_fake') or '-'}")
        # Trung bình trên 8 cột paper (đúng định nghĩa `Avg.` của Table 2)
        vals = [gens[g]["acc"] for g in order
                if g in gens and gens[g].get("acc") is not None]
        if vals:
            ref_avg = (paper_row.get("avg") or {}).get("acc")
            t = f"{sum(vals) / len(vals):7.1f}"
            r = f"{ref_avg:7.1f}" if ref_avg is not None else "      -"
            d = _delta(sum(vals) / len(vals), ref_avg)
            lines.append(f"  {'Avg(8 cột)':<14} {t} {r} {d}")
        if extra:
            lines.append(f"  ↳ {len(extra)} generator MỞ RỘNG (ngoài Table 2, "
                         f"không có số paper): {', '.join(extra)}")

    if paper:
        cmp = compare_to_paper(ours, paper, tol=tol)
        lines.append("")
        lines.append("-" * 78)
        if cmp["n_cells"] == 0:
            lines.append("  [i] chưa so được ô nào — xem danh sách thiếu ở dưới.")
        else:
            lines.append(f"  so với paper: {cmp['n_ok']}/{cmp['n_cells']} ô "
                         f"khớp trong ±{tol:g} điểm")
            if cmp["worst"]:
                w = cmp["worst"]
                lines.append(f"  lệch nhiều nhất: {w['ckpt']}/{w['generator']} "
                             f"ΔACC={w['d_acc']:+.2f}")
        if cmp["missing"]:
            from collections import Counter
            by_ckpt = Counter(m["ckpt"] for m in cmp["missing"])
            lines.append(f"  thiếu dữ liệu: {len(cmp['missing'])} ô "
                         f"({', '.join(f'{k}: {v}' for k, v in sorted(by_ckpt.items()))})")
            lines.append("  → cần `--tiers e2,ckpts` nếu thiếu ckpt; "
                         "`--tiers e2` nếu thiếu generator.")
    return lines


@app.function(
    image=cpu_image,
    volumes={"/ckpt": vol_ckpt, "/raw": vol_raw},
    timeout=1800,
)
def report(exp_ids: str = "", paper: bool = True, tol: float = 0.5) -> dict:
    """Gộp các lần eval thành bảng so với Table 2 của paper.

    ``modal run -m dire_modal.eval::report``
    ``modal run -m dire_modal.eval::report --exp-ids eval_20261025_111500,eval_...``

    Đọc ``/ckpt/results/*/dire.csv`` + ``run.json``, nhóm theo **stem của
    checkpoint**, in 1 khối/detector với cột Δ so với số tác giả. Không chạy
    model, không cần GPU ⇒ gần 0 credit.
    """
    ours = collect_results(
        RESULTS_ROOT, [e for e in exp_ids.split(",") if e.strip()] or None
    )
    if not ours:
        raise SystemExit(
            f"[!] Không có kết quả nào trong {RESULTS_ROOT}.\n"
            "    Chạy trước: bash scripts/01_repro.sh --stage eval"
        )
    ref = load_paper_table() if paper else None
    for line in render_report(ours, ref, tol):
        print(line)
    return {"detectors": sorted(ours), "n_cells": sum(len(v) for v in ours.values())}


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
    # `.map()` của Modal HUỶ mọi task anh em khi một task raise ⇒ trước đây một
    # checkpoint sai kiểu (fc [1,2048] vs [2,2048], 2026-10-28) làm chết cả 4
    # detector × 15 cặp và KHÔNG ghi được CSV nào. Cùng bài học với `fetch_one`:
    # không bao giờ raise ra ngoài, trả dict để `main` ghi lại và báo cáo.
    try:
        model, n_out = load_detector(ckpt, torch, resnet50, device)
        r = score_pair(model, Path(real), Path(fake), torch, transforms, device,
                       batch_size=batch_size, limit=limit, n_out=n_out)
    except SystemExit as exc:      # thông điệp đã rõ ràng, giữ nguyên văn
        return {"real": real, "fake": fake, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 — một cặp hỏng không được giết cả lượt
        import traceback

        traceback.print_exc()
        return {"real": real, "fake": fake,
                "error": f"{type(exc).__name__}: {exc}"}
    return {"real": real, "fake": fake, "n_out": n_out, **r}


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
    ckpt: str = "",
    split: str = "test",
    root: str = "/raw",
    exp_id: str = "",
    tiers: str = "e2",
    limit: int = 0,
    batch_size: int = 64,
    only: str = "",
) -> str:
    """Eval checkpoint PRE-TRAINED trên mọi cặp tìm được, ghi CSV + run.json.

    **Không train gì.** Đây là đường tái lập của dự án: detector ResNet50 lấy
    nguyên từ checkpoint tác giả (`lsun_adm.pth`), chỉ chạy forward.

    ``ckpt`` rỗng ⇒ tự tìm (`resolve_ckpt`): `/raw/checkpoints/lsun_adm.pth`
    trước, rồi `/ckpt/pretrained/lsun_adm.pth`.

    ``modal run -m dire_modal.eval::main --tiers e2``
    ``modal run -m dire_modal.eval::main --only lsun_bedroom__lsun_bedroom --limit 100``
    """
    found = resolve_ckpt(ckpt)
    if not found:
        raise SystemExit(
            "[!] Không thấy checkpoint. Đã thử:\n"
            + "".join(f"    - {c}\n" for c in (ckpt, *CKPT_CANDIDATES) if c)
            + "    Cần: modal run -m dire_modal.download::main --tiers e2min --mode fetch\n"
            "    (hoặc --paths checkpoints/lsun_adm.pth)"
        )
    ckpt = found

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
    failed: list[dict] = [r for r in results if r.get("error")]
    rows: list[dict] = [
        row_for_result(exp, r, by_fake.get(r["fake"], {}), split, git_sha, created)
        for r in results
    ]
    rows.sort(key=lambda d: (d["branch"], d["generator"]))

    scored = [r for r in results if not r.get("error")]
    n_images = {
        "real_total": sum(r["n_real"] for r in scored),
        "fake_total": sum(r["n_fake"] for r in scored),
        "limit_per_archive": limit,
        "skipped_unreadable": sum(r.get("skipped", 0) for r in scored),
        "pairs_failed": len(failed),
    }
    write_csv(rows, out_dir / "dire.csv")
    run_json(
        out_dir / "run.json",
        experiment_id=exp,
        tiers=tiers,
        n_images=n_images,
        rows=len(rows),
        sysinfo=_sysinfo(),
        extra={"ckpt": ckpt, "split": split, "root": root, "only": only,
               "pairs_failed": [{"fake": r["fake"], "error": r["error"]}
                                for r in failed]},
    )

    # Bảng in ra: đọc được ngay trong log của Modal, không phải tải CSV về.
    print(f"\n{'branch':<28} {'generator':<18} {'n':>6} {'ACC':>7} {'AP':>7} {'T@5%':>7}")
    print("-" * 78)
    for d in rows:
        if d.get("error"):
            continue
        print(f"{d['branch']:<28} {d['generator']:<18} "
              f"{d['n_fake']:>6} {d['acc']:>7.4f} {d['ap']:>7.4f} {d['tpr_at_5fpr']:>7.4f}")
    if failed:
        print(f"\n[!] {len(failed)}/{len(results)} cặp KHÔNG chấm được "
              "(metric để trống trong CSV, xem cột `error`):")
        seen: set[str] = set()
        for r in failed:
            head = r["error"].strip().splitlines()[0] if r["error"].strip() else "?"
            if head in seen:
                continue
            seen.add(head)
            print(f"    {head}")
        print("    (gộp theo thông điệp — sửa một lần là hết cả lượt)")
    broken = [d for d in rows if d.get("error") is None and (
        d["n_real"] == 0 or d["n_fake"] == 0)]
    if broken:
        print("\n[!] Cặp có một lớp RỖNG — nhãn hoặc đường dẫn sai, số ở trên vô nghĩa:")
        for d in broken:
            print(f"    {d['branch']}/{d['generator']}: real={d['n_real']} fake={d['n_fake']}")
    # Cổng S2 của SPEC-001: ACC trên generator `adm` phải ≥ 0.99. In ngay tại
    # đây để người chạy thấy kết quả mà không phải mở CSV.
    adm = [d for d in rows if d["generator"] == "adm"]
    if adm:
        gate = "ĐẠT" if max(d["acc"] for d in adm) >= 0.99 else "KHÔNG ĐẠT"
        print(f"\n[CỔNG S2] ACC(adm) = {max(d['acc'] for d in adm):.4f} "
              f"→ {gate} (ngưỡng 0.99)")

    print(f"\n[eval] → {out_dir}/dire.csv · {out_dir}/run.json")
    vol_ckpt.commit()
    if failed:
        # Ném SAU khi đã ghi + commit: script dừng ở detector hỏng thay vì chạy
        # tiếp 3 detector nữa với cùng lỗi (và cùng hoá đơn).
        raise SystemExit(
            f"[!] {len(failed)}/{len(results)} cặp không chấm được — kết quả đã "
            f"ghi ở {out_dir} để còn đối chiếu."
        )
    return str(out_dir)


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
