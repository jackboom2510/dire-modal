"""HF Spaces demo cho DIRE — chạy trên CPU free (0đ).

Tab 1 — Detector: upload ảnh → xác suất real/fake (ResNet-50 đã fine-tune trên DIRE).
Tab 2 — DIRE viz : so sánh ảnh gốc | ảnh tái tạo | |x − recon| (từ gallery tính trước).

Nguyên tắc: KHÔNG bịa kết quả. Nếu chưa có checkpoint/gallery, app hiện hướng dẫn
thay vì trả về số ngẫu nhiên.

Biến môi trường (đặt trong Space Settings → Variables and secrets):
    DIRE_MODEL_REPO   ví dụ "jackboom2510/dire-detector" (mặc định: suy ra từ MODEL_REPO)
    DIRE_MODEL_FILE   tên file trong repo (mặc định "dire_resnet50.pth")
    DIRE_GALLERY_DIR  thư mục chứa gallery (mặc định "gallery")
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import gradio as gr
from PIL import Image

MODEL_REPO = os.environ.get("DIRE_MODEL_REPO", os.environ.get("MODEL_REPO", "jackboom2510/dire-detector"))
MODEL_FILE = os.environ.get("DIRE_MODEL_FILE", "dire_resnet50.pth")
GALLERY_DIR = Path(os.environ.get("DIRE_GALLERY_DIR", "gallery"))
# Tiền xử lý của tác giả (`utils/datasets.py`, nhánh eval):
# Resize(256) → CenterCrop(224) → ToTensor → Normalize(ImageNet).
RESIZE = 256
CROP = 224

_state: dict = {"model": None, "error": None}


# ---------------------------------------------------------------------------
# Nạp model (một lần, lúc khởi động)
# ---------------------------------------------------------------------------
def _load_model():
    import torch
    import torchvision

    try:
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    except Exception as exc:  # repo chưa có / chưa public / chưa train xong
        _state["error"] = f"Không tải được checkpoint từ {MODEL_REPO}/{MODEL_FILE}: {exc}"
        print("[warn]", _state["error"])
        return

    try:
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        # bỏ tiền tố "module." nếu train bằng DataParallel
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

        # Số unit lớp cuối phải ĐỌC TỪ CHECKPOINT, không được chọn sẵn.
        # DIRE gốc: `resnet(num_classes=1)` + BCEWithLogitsLoss ⇒ `fc.weight` [1, 2048].
        # Bản trước hard-code `num_classes=2` + `strict=False` nên lớp fc KHÔNG
        # được nạp (chỉ in warn) và demo trả xác suất rác mà trông vẫn "chạy" —
        # cùng gốc với lỗi eval 2026-10-28:
        #   size mismatch for fc.weight: [1,2048] vs [2,2048]
        w = state.get("fc.weight")
        n_out = int(w.shape[0]) if w is not None and len(w.shape) == 2 else 0
        if n_out not in (1, 2):
            raise ValueError(f"không đọc được `fc.weight` (shape {getattr(w, 'shape', None)})")

        model = torchvision.models.resnet50(num_classes=n_out)
        missing, unexpected = model.load_state_dict(state, strict=False)
        # `fc.weight`/`fc.bias` mà thiếu = lớp quyết định chưa được nạp ⇒ demo
        # đoán bừa. Phải là lỗi, không phải cảnh báo.
        if any(k.startswith("fc.") for k in missing):
            raise ValueError(f"lớp fc không được nạp: missing={[m for m in missing if m.startswith('fc.')]}")
        _state["n_out"] = n_out
        model.eval()
        _state["model"] = model
        print(f"[ok] đã nạp {MODEL_REPO}/{MODEL_FILE} (n_out={n_out}, "
              f"missing={len(missing)}, unexpected={len(unexpected)})")
    except Exception as exc:
        _state["error"] = f"Checkpoint tải được nhưng load lỗi: {exc}"
        print("[warn]", _state["error"])


# ---------------------------------------------------------------------------
# Tab 1 — detector
# ---------------------------------------------------------------------------
def detect(image: Image.Image | None):
    if image is None:
        return {"Chưa có ảnh": 1.0}, "Hãy upload một ảnh."
    if _state["model"] is None:
        return (
            {"Chưa có model": 1.0},
            "⚠ Model chưa sẵn sàng.\n\n"
            f"Chi tiết: {_state['error']}\n\n"
            "Cách khắc phục: dùng checkpoint tác giả (`bash scripts/01_repro.sh --quick`), "
            "rồi upload `dire_resnet50.pth` lên HF Hub và đặt `DIRE_MODEL_REPO` trong Space Settings.",
        )

    import torch
    import torchvision.transforms as T

    # Tiền xử lý phải khớp lúc train (`utils/datasets.py` của tác giả):
    # Resize(256) → CenterCrop(224) → Normalize(ImageNet). Bản trước crop thẳng
    # 224 và chuẩn hoá [0.5] — lệch cả hai bước, tức là lệch số âm thầm.
    tf = T.Compose([T.Resize(RESIZE), T.CenterCrop(CROP), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    x = tf(image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        logits = _state["model"](x)
        if _state.get("n_out") == 1:
            # 1 logit + BCE ⇒ sigmoid chính là P(fake) (nhãn dương = 1_fake).
            p_fake = float(torch.sigmoid(logits)[0][0])
            p_real = 1.0 - p_fake
        else:
            probs = torch.softmax(logits, dim=1)[0].tolist()
            p_real, p_fake = probs[0], probs[1]
    verdict = "FAKE (ảnh do AI sinh)" if p_fake > p_real else "REAL (ảnh thật)"
    note = (
        f"**Kết luận: {verdict}**\n\n"
        f"- P(real) = {p_real:.4f}\n- P(fake) = {p_fake:.4f}\n\n"
        "_Đầu vào của detector là ảnh DIRE (sai số tái tạo qua ADM). Bản demo này chạy trực tiếp "
        "trên ảnh gốc với model ResNet-50 đã train trên đặc trưng DIRE → độ chính xác thấp hơn "
        "cấu hình trong paper (99% ACC cần ảnh DIRE từ `compute_dire.py`)._"
    )
    return {"real": p_real, "fake": p_fake}, note


# ---------------------------------------------------------------------------
# Tab 2 — DIRE viz (từ gallery tính trước)
# ---------------------------------------------------------------------------
def _gallery_items() -> list[tuple[str, str, str, str]]:
    """Mỗi mẫu gồm 3 panel: original / reconstruction / |diff|.

    Quy ước tên file: <tên>_orig.png, <tên>_recon.png, <tên>_diff.png (+ <tên>.txt mô tả tuỳ chọn).
    """
    items = []
    for orig in sorted(glob.glob(str(GALLERY_DIR / "*_orig.png"))):
        stem = orig[: -len("_orig.png")]
        recon, diff = f"{stem}_recon.png", f"{stem}_diff.png"
        if Path(recon).exists() and Path(diff).exists():
            caption_file = Path(f"{stem}.txt")
            caption = caption_file.read_text().strip() if caption_file.exists() else Path(stem).name
            items.append((orig, recon, diff, caption))
    return items


def visualize(name: str | None):
    items = _gallery_items()
    if not items:
        msg = (
            "⚠ **Gallery chưa được tạo.**\n\n"
            "Tab này cần ảnh tính trước vì ADM DDIM inversion 256×256 quá nặng cho CPU free.\n\n"
            "Cách tạo (trên GPU, ~1 phút cho 12 mẫu):\n"
            "```bash\n"
            "modal run -m dire_modal.compute_dire::make_gallery --n 12 --steps 20 --out gallery/\n"
            "```\n"
            "rồi commit thư mục `gallery/` (mỗi mẫu cần 3 file ảnh: `_orig.png`, `_recon.png`, `_diff.png`).\n\n"
            "Ý nghĩa: ảnh do diffusion sinh ra thì **tái tạo được** (|diff| nhỏ), ảnh thật thì không."
        )
        return None, None, None, msg

    labels = [c for _, _, _, c in items]
    idx = labels.index(name) if name in labels else 0
    o, r, d, c = items[idx]
    return Image.open(o), Image.open(r), Image.open(d), f"### {c}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
INTRO = """
# DIRE — Diffusion Reconstruction Error

Demo cho bài toán **phát hiện ảnh do diffusion model sinh ra** (DIRE, ICCV 2023).

| Tab | Nội dung | Nguồn |
|---|---|---|
| **Detector** | Upload ảnh → P(real)/P(fake) | ResNet-50 fine-tune trên đặc trưng DIRE |
| **DIRE viz** | Ảnh gốc \\| ảnh tái tạo \\| sai số \\|x−recon\\| | Gallery tính trước trên GPU |

Ý tưởng cốt lõi: ảnh do diffusion sinh ra có thể **tái tạo gần đúng** bằng chính diffusion model
(DDIM inversion) → sai số nhỏ; ảnh thật thì không → sai số lớn.
"""

with gr.Blocks(title="DIRE — Diffusion Reconstruction Error") as demo:
    gr.Markdown(INTRO)

    with gr.Tab("Detector"):
        with gr.Row():
            inp = gr.Image(type="pil", label="Ảnh đầu vào")
            out_label = gr.Label(num_top_classes=2, label="Dự đoán")
        with gr.Accordion("Chi tiết", open=False):
            out_text = gr.Markdown()
        gr.Button("Phân tích", variant="primary").click(detect, inputs=inp, outputs=[out_label, out_text])

    with gr.Tab("DIRE viz"):
        items = _gallery_items()
        choices = [c for _, _, _, c in items] if items else ["(chưa có gallery)"]
        dd = gr.Dropdown(choices=choices, value=choices[0], label="Chọn mẫu")
        with gr.Row():
            img_o = gr.Image(label="Ảnh gốc", type="pil")
            img_r = gr.Image(label="Tái tạo (ADM DDIM)", type="pil")
            img_d = gr.Image(label="|x − recon| (DIRE)", type="pil")
        caption = gr.Markdown()
        dd.change(visualize, inputs=dd, outputs=[img_o, img_r, img_d, caption])
        demo.load(lambda: visualize(choices[0]), outputs=[img_o, img_r, img_d, caption])

    gr.Markdown(
        "_Chạy trên CPU free của HF Spaces. Nguồn dataset: DiffusionForensics (Wang et al., ICCV 2023) — "
        "dùng cho mục đích nghiên cứu._"
    )


if __name__ == "__main__":
    _load_model()
    demo.launch()
