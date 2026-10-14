---
title: DIRE — Diffusion Reconstruction Error
emoji: 🔍
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
short_description: Phát hiện ảnh do diffusion model sinh ra bằng sai số tái tạo (DIRE)
---

# DIRE demo (HF Spaces, CPU free)

Demo cho **DIRE — Diffusion Reconstruction Error** (Wang et al., ICCV 2023).

- **Tab Detector**: upload ảnh → P(real) / P(fake) bằng ResNet-50 fine-tune trên đặc trưng DIRE.
- **Tab DIRE viz**: ảnh gốc | ảnh tái tạo (ADM DDIM-20) | sai số |x − recon|, lấy từ gallery tính trước.

## Cách deploy (0đ)

```bash
# 1. Tạo Space (SDK: Gradio, hardware: CPU basic — free)
pip install -U huggingface_hub
huggingface-cli repo create dire-demo --type space --space_sdk gradio

# 2. Đẩy code này lên
cd platforms/hf-spaces
git clone https://huggingface.co/spaces/<user>/dire-demo space && cd space
cp ../app.py ../requirements.txt ../README.md .
git add -A && git commit -m "DIRE demo" && git push
```

## Cấu hình (Space Settings → Variables)

| Biến | Ý nghĩa | Mặc định |
|---|---|---|
| `DIRE_MODEL_REPO` | repo HF chứa checkpoint detector | `jackboom2510/dire-detector` |
| `DIRE_MODEL_FILE` | tên file checkpoint | `dire_resnet50.pth` |
| `DIRE_GALLERY_DIR` | thư mục gallery | `gallery` |

Chưa có checkpoint/gallery → app **hiện hướng dẫn**, không trả kết quả giả.

## Trạng thái hiện tại

| Thành phần | Trạng thái |
|---|---|
| Detector | ⏳ chờ checkpoint từ `./scripts/05_train_repro.sh` |
| DIRE viz gallery | ⏳ chờ `modal run -m dire_modal.compute_dire::make_gallery` |

## Lưu ý về độ chính xác

Paper báo ~99% ACC **khi detector nhận đầu vào là ảnh DIRE** (sai số tái tạo), không phải ảnh gốc.
Bản demo chạy detector trên ảnh gốc nên độ chính xác **thấp hơn** — đây là giới hạn của việc
tính DIRE ngay trên CPU free, được ghi rõ trong UI để tránh hiểu nhầm.

## Nguồn & bản quyền

- Bài báo / code gốc: <https://github.com/ZhendongWang6/DIRE> (repo gốc đã archive 2025-07-08)
- Dataset DiffusionForensics dùng cho mục đích nghiên cứu; xem `docs/runbook-free.md` §License.
