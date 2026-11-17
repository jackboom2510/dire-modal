---
title: DIRE — sai số tái tạo
emoji: 🔍
colorFrom: indigo
colorTo: gray
sdk: streamlit
sdk_version: 1.67.0
app_file: app.py
pinned: false
license: mit
short_description: DIRE map, nguồn gốc từng con số, và một phép đo chạy thật
---

# DIRE — demo thuyết trình (HF Spaces, CPU free)

Demo cho **DIRE — Diffusion Reconstruction Error** (Wang et al., ICCV 2023). Repo
gốc: <https://github.com/ZhendongWang6/DIRE> (đã archive 2025-07-08).

## Demo này làm gì

| Trang | Nội dung | Chạy ở đâu | Nhãn |
|---|---|---|---|
| 1 | Demo làm được gì, không làm được gì | tĩnh | — |
| 2 | **Gallery**: ảnh gốc · DIRE map · điểm detector, kèm `.json` nguồn gốc | ảnh tính trước | `R0` / `R1+R2` |
| 3 | **Tính DIRE thật**: chạy đúng hai vòng lặp trên ADM, hiện `x̂₀` từng bước | ngay trên Space, **mặc định tắt** | không áp dụng |
| 4 | **Thí nghiệm 2 chiều**: mô hình khuếch tán có score chính xác, đo DIRE | ngay trên Space, 61 ms | minh hoạ lý thuyết |
| 5 | Lịch nhiễu `ᾱ_t`, NFE, và chi phí | ngay trên Space | công thức |
| 6 | Bảng 2 của bài + đường tái lập + hạn chế | tĩnh | số của paper |

## Demo này **không** làm gì

- **Không có ô upload ảnh.** Một ResNet-50 học trên DIRE map không có nghĩa gì khi
  nhận ảnh RGB thô. Bản Gradio trước đây trả `P(fake)` cho ảnh upload; đó là lỗi bị
  ghi ở `docs/demo-web-plan.md` §2 và nó đã bị bỏ, không được dựng lại.
- **Không** ghi "độ chính xác 99 %" ở bất kỳ đâu. Số của bài luôn kèm chữ
  *"số của paper (ICCV 2023), không phải của demo này"*.
- **Không** hiện `p_fake` khi chưa qua cổng kiểm: `detector.gate` phải là `passed`
  và ngưỡng phải chọn trên `val`.
- **Không** tính DIRE lúc chạy **theo mặc định**: một lượt forward ADM 256² là
  **10,1 s** trên 2 vCPU (đo), nên `S = 5` là **101 s**/ảnh và `S = 20` là 404 s.
  Trang 3 làm được nhưng phải bật thủ công (`DIRE_CKPT` + torch) và phải tick xác
  nhận thời gian chờ; lý do và số đo ở `docs/demo-hf-spaces-plan.md` §13.

## Cấu trúc

```
app.py                 # UI Streamlit, 5 trang
demolib/toy2d.py       # mô hình 2 chiều, score chính xác (thuần math)
demolib/provenance.py  # đọc .json của gallery, quyết định khi nào hiện điểm
demolib/charts.py      # SVG thuần, không matplotlib
gallery/               # <tên>_orig.png  <tên>_diff.png  [<tên>_recon.png]  <tên>.json
requirements.txt       # đúng một dòng: streamlit
```

`gallery/` đang **rỗng** (chỉ `.gitkeep`). Đó là trạng thái hợp lệ: trang 2 hiện
lệnh dựng gallery chứ không hiện ba khung xám giả.

## Dựng gallery (khi đã có DIRE map)

```bash
python tools/make_gallery.py \
    --maps  <volume>/maps/test \
    --orig  <volume>/images/test \
    --out   platforms/hf-spaces/gallery \
    --n 8 --steps 20 --label R1+R2 \
    --map-source "maps/test/*.npy · ::dire --steps 20"
```

Thêm `--scores scores.json --gate passed` chỉ khi cổng 2 (`docs/demo-web-plan.md`
§5) đã chạy; mặc định `--gate failed` nên **không** điểm nào được hiện.

## Deploy (0 ₫)

```bash
# 1. Space mới (SDK streamlit, hardware CPU basic)
hf repo create dire-demo --repo-type space --space-sdk streamlit

# 2. Đẩy nội dung thư mục này lên
git clone https://huggingface.co/spaces/<user>/dire-demo /tmp/space && cd /tmp/space
cp -r <repo>/platforms/hf-spaces/. .
git add -A && git commit -m "DIRE demo (streamlit, CPU free)" && git push
```

Chi tiết, số đo cold start, cổng kiểm và phương án rollback:
**`docs/demo-hf-spaces-plan.md`**.

## Cấu hình

| Biến | Ý nghĩa | Mặc định |
|---|---|---|
| `DIRE_GALLERY_DIR` | thư mục gallery, tuyệt đối hoặc tương đối với `app.py` | `gallery` |
| `DIRE_CKPT` | đường dẫn checkpoint ADM (bật trang 3); cần thêm `torch==2.4.0` vào `requirements.txt` | `ckpt/adm_lsun.pt` |

Không có secret nào. Space không gọi mạng ngoài, không lưu ảnh người dùng.

## Nguồn & bản quyền

- Bài báo / code gốc: <https://github.com/ZhendongWang6/DIRE>
- Ảnh trong gallery (khi có): DiffusionForensics (Wang et al., ICCV 2023),
  chỉ dùng cho mục đích nghiên cứu; xem `docs/runbook-free.md` §License.
