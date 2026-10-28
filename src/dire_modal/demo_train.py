# src/dire_modal/demo_train.py
"""Train ResNet50 của DIRE ở quy mô **demo** — để MÔ PHỎNG cơ chế, không để báo cáo.

Vì sao file này tồn tại, và vì sao nó KHÔNG phải `train.py`
-----------------------------------------------------------
`train.py` (resumable, A100, 400 epoch) là đường tái lập thật; nó vẫn là
``NotImplementedError`` vì dự án chốt **không train lại** (ràng buộc N1). File
này phục vụ một mục đích khác hẳn: có **một đường loss thật** để vẽ/animate khi
giải thích DIRE hoạt động ra sao, thay vì vẽ một đường cong bịa.

Ranh giới đó phải cứng, nếu không nó sẽ trôi thành "số của dự án":

1. **Không ghi vào** ``/ckpt/results/**`` và **không đụng** ``/ckpt/pretrained/``
   hay ``/raw/checkpoints/*.pth``. Mọi thứ ra ``/ckpt/demo_train/<exp>/``. Nhờ
   vậy ``eval::report`` không bao giờ nhìn thấy checkpoint demo và không thể
   trộn nó vào bảng Δ so với paper.
2. **Mọi artefact mang nhãn** ``DEMO_LABEL`` (trong ``loss_log.json``) và
   ``"demo": True`` (trong checkpoint). ACC của 2 000 ảnh × 5 epoch **thấp hơn
   99.9% của paper rất nhiều**; hai con số đặt cạnh nhau mà không chú thích là
   nói dối bằng hình ảnh.
3. **Mặc định là ``--mode plan``**: in ước lượng tiền/thời gian rồi thoát. Phải
   gõ ``--mode run`` mới thật sự train — cùng quy ước ``download::main``.

Vì sao không gọi `vendor/trainer.py` của tác giả
------------------------------------------------
Trainer gốc gắn cứng ``exp_dir``/``logs_path``/``nepoch=400``/visdom và tự tạo
layout ``data/exp/<name>``. Dùng nó nghĩa là kéo theo cả một hệ thống file và
một vòng lặp 400 epoch vào một việc cần 5 phút. Ở đây chỉ cần ~``resnet50`` +
``BCEWithLogitsLoss`` + Adam, tức là **đúng phần toán** của tác giả, không phải
phần hạ tầng của họ.

Siêu tham số: chép, không đoán
------------------------------
Giá trị nằm ở ``demo_train.yaml`` **cạnh module** (Modal chỉ đóng gói thư mục
module — cùng lý do ``tiers.yaml`` nằm ở đây), và ``tests/test_demo_train.py``
đối chiếu từng số với ``vendor/utils/config.py`` + ``vendor/utils/trainer.py``.
Sửa một bên mà quên bên kia là test đỏ.

Vì sao 1 logit
--------------
Tác giả dùng ``resnet(num_classes=1)`` + ``BCEWithLogitsLoss`` ⇒ ``fc.weight``
là ``[1, 2048]``. Đây đúng là thứ làm eval nổ ngày 2026-10-28
(``size mismatch for fc.weight [1,2048] vs [2,2048]``). Checkpoint do file này
sinh ra vì thế là một phép thử khép vòng: ``eval.load_detector`` **phải** nạp
được nó, và ``prob_fake`` phải dùng ``sigmoid``.

Chạy::

    # 0 credit — in tiền/thời gian, không tải, không train
    modal run -m dire_modal.demo_train::main

    # train thật (cần tier `train` đã có trên /raw)
    modal run -m dire_modal.demo_train::main --mode run --images 2000 --epochs 5
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .common import app, gpu_image, vol_ckpt, vol_raw


def _eval():
    """Import `eval` theo yêu cầu — KHÔNG ở tầng module.

    Hai lý do:

    1. **Đăng ký Modal.** ``eval.py`` có hàm ``main`` trong cùng ``app``; import
       nó ở tầng module làm hai ``main`` tranh tên nhau (Modal in
       ``function name 'main' collision!``). Ở đây ``::main`` phải là của
       ``demo_train`` một cách không mơ hồ.
    2. **Import nhẹ.** ``plan``/test không cần kéo cả ``eval`` vào.

    Giá trị vẫn dùng CHUNG một nguồn (``E.RESIZE``, ``E.MEAN``, ``E.PREPROC``…)
    nên train và eval không thể lệch nhau — trì hoãn, không sao chép.
    """
    from . import eval as E  # noqa: PLC0415

    return E


# ---------------------------------------------------------------------------
# Siêu tham số — giá trị CHUẨN ở demo_train.yaml, đây chỉ là bản dự phòng
# ---------------------------------------------------------------------------
SEED = 3407                 # DefaultConfigs.seed
ARCH = "resnet50"           # DefaultConfigs.arch
N_OUT = 1                   # 1 logit + BCEWithLogitsLoss (KHÔNG phải 2 lớp)
BATCH = 64                  # DefaultConfigs.batch_size
# 256/224 — TRÙNG `eval.RESIZE`/`eval.CROP`; test đối chiếu với eval nên không
# thể lệch. Ghi literal ở đây để tầng module không phải import eval.
LOAD_SIZE = 256
CROP = 224
LR = 1e-4                   # DefaultConfigs.lr
BETA1 = 0.9                 # DefaultConfigs.beta1
EPOCHS = 5                  # demo; tác giả 400 + earlystop 5
IMAGES = 2000               # tổng hai lớp (1000 real + 1000 fake)
VAL_IMAGES = 200            # đủ để thấy ACC, không đủ để kết luận gì
MAX_MINUTES = 15
LOG_EVERY = 10              # step
GPU = "L4"
USD_PER_GPU_HOUR = 0.80     # Modal L4, docs/iaas-options.md §1.1
# Ước lượng, CHƯA đo: ResNet50 @224, batch 64, forward+backward trên L4.
# Truyền --sec-per-step sau khi đo thật để con số tiền có nghĩa.
SEC_PER_STEP = 0.35
# Val chỉ forward ⇒ ~1/3 chi phí một step train.
VAL_FORWARD_FRACTION = 1.0 / 3.0

# Nhãn dán vào MỌI artefact. Không có nó thì một hình đẹp sẽ bị đọc thành kết quả.
DEMO_LABEL = "demo — KHÔNG phải số của paper"

CONFIG_NAME = "demo_train.yaml"
CONFIG_PATH = Path(__file__).with_name(CONFIG_NAME)

# Dữ liệu: đúng split TÁC GIẢ dùng để train detector (tier `train` của manifest),
# không phải test split. Train trên test split là leakage — đường này không có.
TRAIN_REAL = "/raw/dire/train/lsun_bedroom/real.tar.gz"
TRAIN_FAKE = "/raw/dire/train/lsun_bedroom/adm.tar.gz"
VAL_REAL = "/raw/dire/val/lsun_bedroom/real.tar.gz"
VAL_FAKE = "/raw/dire/val/lsun_bedroom/adm.tar.gz"

# Gốc ghi kết quả. Tên thư mục CỐ ĐỊNH để test khẳng định không đụng `results/`.
OUTPUT_ROOT = "/ckpt/demo_train"


# ---------------------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------------------
def load_config(path: Path | str | None = None) -> dict:
    """Đọc ``demo_train.yaml``. Thiếu/hỏng ⇒ dùng hằng số ở trên, không raise.

    Cùng triết lý ``manifest.load_policy``: một cái file cấu hình không được
    phép làm chết một job đã trả tiền bật GPU.
    """
    p = Path(path) if path else CONFIG_PATH
    try:
        import yaml  # pyyaml có trong cpu_image
    except Exception:  # noqa: BLE001
        return _fallback_config()
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 — thiếu file / YAML hỏng
        return _fallback_config()
    if not isinstance(data, dict):
        return _fallback_config()
    base = _fallback_config()
    base.update({k: v for k, v in data.items() if v is not None})
    return base


def _fallback_config() -> dict:
    return {
        "seed": SEED,
        "arch": ARCH,
        "pretrained": True,
        "n_out": N_OUT,
        "batch_size": BATCH,
        "load_size": LOAD_SIZE,
        "crop_size": CROP,
        "optim": "adam",
        "lr": LR,
        "beta1": BETA1,
        "loss": "bce_with_logits",
        "epochs": EPOCHS,
        "images": IMAGES,
        "val_images": VAL_IMAGES,
        "max_minutes": MAX_MINUTES,
        "log_every": LOG_EVERY,
        "gpu": GPU,
        "usd_per_gpu_hour": USD_PER_GPU_HOUR,
        "sec_per_step": SEC_PER_STEP,
        "aug": {
            "resize": True,
            "crop": True,
            "flip": True,
            "blur_prob": 0.0,
            "jpg_prob": 0.0,
            "gray_prob": 0.0,
        },
    }


# ---------------------------------------------------------------------------
# Guard chi phí — phần duy nhất ở file này có thể tiêu tiền
# ---------------------------------------------------------------------------
def estimate(
    images: int = IMAGES,
    epochs: int = EPOCHS,
    batch: int = BATCH,
    sec_per_step: float = SEC_PER_STEP,
    usd_per_gpu_hour: float = USD_PER_GPU_HOUR,
    val_images: int = VAL_IMAGES,
) -> dict:
    """Ước lượng GPU-giây và USD cho ``images`` ảnh × ``epochs``.

    Trả **một** con số kỳ vọng (không có "trần xấu nhất" kiểu
    ``compute_dire.estimate_cost``) vì đường này là một container duy nhất, chạy
    vài phút: rủi ro không phải container treo 30 phút mà là *chọn sai epochs*.

    Mọi tham số đều nhận từ ngoài để test tính tay được, không phải đọc từ môi
    trường.
    """
    n = max(0, int(images))
    e = max(1, int(epochs))
    b = max(1, int(batch))
    v = max(0, int(val_images))
    train_steps = math.ceil(n / b) * e if n else 0
    val_steps = math.ceil(v / b) * e if v else 0
    train_s = train_steps * float(sec_per_step)
    val_s = val_steps * float(sec_per_step) * VAL_FORWARD_FRACTION
    gpu_s = train_s + val_s
    return {
        "images": n,
        "epochs": e,
        "batch": b,
        "train_steps": train_steps,
        "val_steps": val_steps,
        "gpu_seconds": round(gpu_s, 3),
        "gpu_minutes": round(gpu_s / 60.0, 3),
        "gpu_hours": round(gpu_s / 3600.0, 5),
        "usd": round(gpu_s / 3600.0 * float(usd_per_gpu_hour), 5),
        "usd_per_gpu_hour": float(usd_per_gpu_hour),
        "sec_per_step": float(sec_per_step),
    }


def enforce(est: dict, max_images: int = IMAGES, max_minutes: float = MAX_MINUTES) -> dict:
    """Chặn TRƯỚC khi bật GPU nếu vượt trần. Raise ``SystemExit``, không cảnh báo.

    Cảnh báo thì không ai đọc; đây là chỗ duy nhất trong file được phép tiêu
    tiền nên phải dừng hẳn — cùng cách ``compute_dire.preflight_cost`` làm.
    """
    if est["images"] > int(max_images):
        raise SystemExit(
            f"[!] {est['images']} ảnh > trần --max-images {int(max_images)}.\n"
            f"    Đây là demo cơ chế, không phải train lại: 2 000 ảnh đã đủ một "
            f"đường loss.\n    Giảm --images hoặc nâng --max-images một cách có ý thức."
        )
    if est["gpu_minutes"] > float(max_minutes):
        raise SystemExit(
            f"[!] Ước lượng {est['gpu_minutes']:.1f} phút > trần --max-minutes "
            f"{float(max_minutes):.0f} phút (≈ ${est['usd']:.3f}).\n"
            f"    Giảm --epochs/--images, hoặc nâng trần nếu thật sự cần."
        )
    return est


def describe_plan(est: dict, max_images: int = IMAGES,
                  max_minutes: float = MAX_MINUTES) -> dict:
    """Bản ``--mode plan``: in tiền/thời gian + nhãn demo, KHÔNG chạm mạng/GPU."""
    enforce(est, max_images=max_images, max_minutes=max_minutes)
    print(f"[tiền] {est['images']} ảnh × {est['epochs']} epoch "
          f"= {est['train_steps']} step (+{est['val_steps']} step val) "
          f"≈ {est['gpu_minutes']:.1f} phút ≈ ${est['usd']:.3f}")
    print(f"[tiền] L4 @ ${est['usd_per_gpu_hour']:.2f}/h · "
          f"{est['sec_per_step']:.2f} s/step — ước lượng, chưa đo")
    print(f"[demo] {DEMO_LABEL}")
    print("[plan] chưa train gì. Thêm --mode run để chạy thật.")
    return {"verdict": "ok", "demo": True, "label": DEMO_LABEL, **est}


# ---------------------------------------------------------------------------
# Dữ liệu
# ---------------------------------------------------------------------------
def _open_rgb(data: bytes):
    """PNG/JPEG bytes → ảnh RGB, hoặc ``None`` nếu hỏng.

    Trả ``None`` chứ không raise: một ảnh hỏng trong 2 000 ảnh không được phép
    giết cả lần train (bài học ``fetch_one``/``eval_one``).
    """
    try:
        from io import BytesIO

        from PIL import Image

        return Image.open(BytesIO(data)).convert("RGB")
    except Exception:  # noqa: BLE001
        return None


def iter_labeled(archives, per_class: int = 0, opener=None, transform=None,
                 stats: dict | None = None):
    """Yield ``(mẫu, nhãn)`` từ các tar, đọc/giải mã tuần tự.

    ``archives``: ``[(path, label)]``. ``label`` là ``eval.REAL``/``eval.FAKE``
    (0/1) — dùng lại hằng của ``eval`` để nhãn không lệch giữa train và eval.

    ``opener``/``transform`` được tiêm từ ngoài để test chạy được **không cần
    PIL/torch**: test truyền ``opener=lambda b: b`` và đếm nhãn, không cần giải
    mã ảnh thật.
    """
    E = _eval()
    open_ = opener or _open_rgb
    tf = transform or (lambda img: img)
    st = stats if stats is not None else {}
    st.setdefault("read", 0)
    st.setdefault("skipped", 0)
    st.setdefault("per_label", {})
    for path, label in archives:
        p = Path(path)
        if not p.exists():
            continue
        n = 0
        for _name, data in E.iter_images(p):
            if per_class and n >= int(per_class):
                break
            img = open_(data)
            if img is None:
                st["skipped"] += 1
                continue
            st["read"] += 1
            st["per_label"][label] = st["per_label"].get(label, 0) + 1
            n += 1
            yield tf(img), label


def batches(iterable, size: int):
    """Gom iterator thành list ``size`` — giữ cả phần đuôi (không bỏ rác cuối)."""
    n = max(1, int(size))
    buf: list = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


# ---------------------------------------------------------------------------
# Log + checkpoint
# ---------------------------------------------------------------------------
def log_entry(step: int, epoch: int, loss: float, acc: float, n: int) -> dict:
    """Một dòng log. Từ chối NaN/inf — không ghi số không đọc được (NFR-6)."""
    loss = float(loss)
    acc = float(acc)
    if not (math.isfinite(loss) and math.isfinite(acc)):
        raise ValueError(f"loss/acc không hữu hạn tại step {step}: {loss!r}, {acc!r}")
    return {
        "step": int(step),
        "epoch": int(epoch),
        "loss": round(loss, 6),
        "acc": round(acc, 4),
        "n": int(n),
    }


def summarize(steps: list[dict], epochs_log: list[dict] | None = None) -> dict:
    """Tóm tắt để scene manim đọc: điểm đầu, điểm cuối, val tốt nhất.

    ``best_val_acc`` là ``None`` khi chưa có val — **không** thay bằng 0.
    """
    ep = epochs_log or []
    vals = [e["val_acc"] for e in ep if e.get("val_acc") is not None]
    return {
        "n_steps": len(steps),
        "first_loss": steps[0]["loss"] if steps else None,
        "last_loss": steps[-1]["loss"] if steps else None,
        "first_acc": steps[0]["acc"] if steps else None,
        "last_acc": steps[-1]["acc"] if steps else None,
        "best_val_acc": max(vals) if vals else None,
        "epochs": len(ep),
    }


def ckpt_dir(exp: str = "demo_train", base: str = OUTPUT_ROOT) -> Path:
    """``/ckpt/demo_train/<exp>/`` — và từ chối nếu ai đó trỏ vào ``results``.

    Đây là hàng rào giữa "hình minh hoạ" và "số của dự án": ``eval::report``
    quét ``/ckpt/results/**``, nên checkpoint demo không bao giờ lọt vào bảng Δ.
    """
    d = Path(base) / str(exp)
    if "results" in d.parts:
        raise SystemExit(
            f"[!] Không ghi demo vào {d}: `results/` là chỗ của số THẬT "
            f"(eval::report quét thư mục đó). Dùng {OUTPUT_ROOT}/<exp>/."
        )
    return d


def state_for_save(model_state, epoch: int, images: int, n_out: int = N_OUT) -> dict:
    """Checkpoint cùng khuôn với bản của tác giả, kèm cờ ``demo``.

    Khoá ``model`` là khoá mà ``eval.load_detector`` đã biết bóc, nên checkpoint
    này nạp được bằng **đúng** đường code của eval — không có nhánh riêng cho
    demo (nhánh riêng là chỗ lỗi ẩn nấp).
    """
    return {
        "model": model_state,
        "arch": ARCH,
        "n_out": int(n_out),
        "epoch": int(epoch),
        "images": int(images),
        "seed": SEED,
        "preproc": _eval().PREPROC,
        "demo": True,
        "label": DEMO_LABEL,
    }


def log_payload(exp: str, steps: list[dict], epochs_log: list[dict],
                stats: dict, cfg: dict | None = None) -> dict:
    """Nội dung ``loss_log.json`` — nguồn DUY NHẤT cho mọi scene manim.

    ``stats`` là số ảnh **cộng dồn qua các epoch** (2 000 ảnh × 5 epoch =
    10 000 lượt đọc), không phải 2 000. Gọi tên khác đi sẽ khiến scene vẽ sai.
    """
    return {
        "label": DEMO_LABEL,
        "demo": True,
        "exp": str(exp),
        "seed": SEED,
        "config": cfg or _fallback_config(),
        "steps": list(steps),
        "epochs": list(epochs_log),
        "stats": dict(stats),
        "summary": summarize(steps, epochs_log),
    }


def write_log(path: Path | str, payload: dict) -> Path:
    """Ghi JSON UTF-8. Không bao giờ ghi theo locale của máy (Windows cp1252)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Train — phần duy nhất cần torch
# ---------------------------------------------------------------------------
def train_transforms(transforms):
    """Aug của tác giả: resize 256 → crop ngẫu nhiên 224 → flip.

    Ba phép ``blur``/``jpg``/``gray`` của họ đều **0.0** trong ``DefaultConfigs``
    nên không đưa vào: giữ đúng, không thêm.

    Chuẩn hoá lấy **trực tiếp** ``eval.MEAN``/``eval.STD``: hai bản hệ số
    ImageNet chép tay là hai cơ hội để lệch nhau.
    """
    E = _eval()
    return transforms.Compose([
        transforms.Resize(LOAD_SIZE),
        transforms.RandomCrop(CROP),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(E.MEAN, E.STD),
    ])


def _val_accuracy(model, torch, transforms, device, per_class: int = VAL_IMAGES // 2) -> float | None:
    """ACC trên val bằng **đúng** ``eval.preprocess`` ⇒ so được với eval."""
    from io import BytesIO

    from PIL import Image

    E = _eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    model.eval()
    with torch.inference_mode():
        for path, label in ((VAL_REAL, E.REAL), (VAL_FAKE, E.FAKE)):
            if not Path(path).exists():
                continue
            n = 0
            for _name, data in E.iter_images(Path(path)):
                if n >= per_class:
                    break
                try:
                    img = Image.open(BytesIO(data)).convert("RGB")
                except Exception:  # noqa: BLE001
                    continue
                x = E.preprocess(img, torch, transforms).unsqueeze(0).to(device)
                y_true.append(label)
                y_pred.append(E.FAKE if float(model(x).item()) > 0 else E.REAL)
                n += 1
    model.train()
    if not y_true:
        return None
    return sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)


def run(images: int = IMAGES, epochs: int = EPOCHS, exp: str = "demo_train",
        sec_per_step: float = SEC_PER_STEP, max_images: int = IMAGES,
        max_minutes: float = MAX_MINUTES) -> dict:
    """Vòng train thật. Gọi bởi :func:`main` khi ``--mode run``.

    Trả ``{"loss_log": path, "ckpt": path, "summary": {...}}`` để ``main`` in ra
    và ``vol_ckpt.commit()``.
    """
    import torch
    from torchvision import models, transforms

    E = _eval()
    est = enforce(estimate(images, epochs, sec_per_step=sec_per_step), max_images, max_minutes)
    cfg = load_config()

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for p in (TRAIN_REAL, TRAIN_FAKE):
        if not Path(p).exists():
            raise SystemExit(
                f"[!] Thiếu {p}\n"
                "    Cần tier `train` (4 file · 5.74 GB):\n"
                "    modal run -m dire_modal.download::main --tiers train --mode fetch"
            )

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, N_OUT)   # [1, 2048]
    model.to(device).train()

    criterion = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(BETA1, 0.999))

    per_class = max(1, int(images) // 2)
    # Compose MỘT lần: dựng transform cho từng ảnh trong 2 000 ảnh là lãng phí
    # đúng vào chỗ đang tính tiền.
    tf_train = train_transforms(transforms)
    steps_log: list[dict] = []
    epochs_log: list[dict] = []
    stats: dict = {}
    step = 0
    for epoch in range(1, int(epochs) + 1):
        loader = iter_labeled(
            [(TRAIN_REAL, E.REAL), (TRAIN_FAKE, E.FAKE)],
            per_class=per_class,
            transform=tf_train,
            stats=stats,
        )
        for batch in batches(loader, BATCH):
            xs = torch.stack([x for x, _ in batch]).to(device)
            ys = torch.tensor([float(y) for _, y in batch], device=device)
            logits = model(xs).squeeze(1)
            loss = criterion(logits, ys)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % LOG_EVERY == 0 or step == 1:
                acc = float(((logits > 0) == (ys > 0.5)).float().mean().item())
                steps_log.append(log_entry(step, epoch, float(loss.item()), acc, len(batch)))
        val_acc = _val_accuracy(model, torch, transforms, device)
        epochs_log.append({"epoch": epoch, "val_acc": val_acc})
        print(f"[demo] epoch {epoch}/{epochs} · val_acc = "
              f"{'—' if val_acc is None else f'{val_acc:.3f}'}")

    out = ckpt_dir(exp)
    out.mkdir(parents=True, exist_ok=True)
    payload = log_payload(exp, steps_log, epochs_log, stats, cfg)
    log_path = write_log(out / "loss_log.json", payload)
    ckpt_path = out / "demo_resnet50.pth"
    torch.save(state_for_save(model.state_dict(), int(epochs), images), ckpt_path)
    print(f"[demo] {DEMO_LABEL}")
    print(f"[demo] log  → {log_path}")
    print(f"[demo] ckpt → {ckpt_path}  (fc.weight [1, {model.fc.in_features}])")
    return {"loss_log": str(log_path), "ckpt": str(ckpt_path),
            "summary": payload["summary"], "est": est}


# ---------------------------------------------------------------------------
# Modal entrypoint
# ---------------------------------------------------------------------------
@app.function(
    image=gpu_image,
    volumes={"/raw": vol_raw, "/ckpt": vol_ckpt},
    gpu=GPU,
    timeout=3600,
)
def main(
    mode: str = "plan",
    images: int = IMAGES,
    epochs: int = EPOCHS,
    exp: str = "demo_train",
    sec_per_step: float = SEC_PER_STEP,
    max_images: int = IMAGES,
    max_minutes: float = MAX_MINUTES,
) -> None:
    """``plan`` (mặc định, 0 credit) hoặc ``run`` (train thật trên L4).

    ``modal run -m dire_modal.demo_train::main --mode run --images 2000 --epochs 5
    ``modal volume get dire-ckpt demo_train/<exp> ./demo_train_out``
    """
    if mode.strip().lower() not in {"run", "yes"}:
        describe_plan(estimate(images, epochs, sec_per_step=sec_per_step),
                      max_images=max_images, max_minutes=max_minutes)
        return
    res = run(images=images, epochs=epochs, exp=exp, sec_per_step=sec_per_step,
              max_images=max_images, max_minutes=max_minutes)
    vol_ckpt.commit()   # ghi trước khi container chết — bài học eval 2026-10-28
    print(f"[demo] xong. summary = {json.dumps(res['summary'], ensure_ascii=False)}")
