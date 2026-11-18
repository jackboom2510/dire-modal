---
title: DIRE — sai số tái tạo
emoji: 🔍
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
app_file: gradio_app.py
pinned: false
license: mit
short_description: DIRE map, nguồn gốc từng con số, và một phép đo chạy thật
---

# DIRE — demo thuyết trình (HF Spaces, CPU free)

Demo cho **DIRE — Diffusion Reconstruction Error** (Wang et al., ICCV 2023). Repo
gốc: <https://github.com/ZhendongWang6/DIRE> (đã archive 2025-07-08).

## Hai giao diện, một thư viện

| Tệp | SDK | Dùng khi |
|---|---|---|
| **`gradio_app.py`** | `gradio` (mặc định của Space này) | bản chính: hàng đợi, trạng thái phiên và nút *Restart* do Gradio lo |
| `app.py` | `streamlit` | bản dự phòng, giữ nguyên và vẫn chạy được (`streamlit run app.py`) |

Hai bản dùng chung `demolib/` (`toy2d`, `provenance`, `charts`, `cost`, `remote`),
nên **không có con số nào khác nhau** giữa hai giao diện: chúng chỉ khác cách bày.
Test của hai bản nằm ở `tests/test_hf_spaces_demo.py` (Streamlit) và
`tests/test_hf_spaces_gradio.py` (Gradio).

Cả hai có **cùng một tab gallery** (ảnh gốc · DIRE map · điểm detector, kèm `.json`
nguồn gốc), tab thí nghiệm 2 chiều, tab lịch/chi phí, và tab Table 2.

## Demo này làm gì

| Tab (Gradio) | Nội dung | Chạy ở đâu | Nhãn |
|---|---|---|---|
| 1 | **Gallery**: ảnh gốc · DIRE map · điểm detector, kèm `.json` nguồn gốc | ảnh tính trước | `R0` / `R1+R2` |
| 2 | **Tính DIRE thật**: upload ảnh → **DIRE map** (không trả `P(fake)`), qua endpoint Modal | GPU Modal, **mặc định tắt** | `R1` |
| 3 | **Thí nghiệm 2 chiều**: mô hình khuếch tán có score chính xác, đo DIRE | ngay trên Space, 61 ms | minh hoạ lý thuyết |
| 4 | Lịch nhiễu `ᾱ_t`, NFE, và chi phí | ngay trên Space | công thức |
| 5 | Bảng 2 của bài + đường tái lập + hạn chế | tĩnh | số của paper |

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
gradio_app.py          # UI Gradio, 5 tab  (mặc định của Space)
app.py                 # UI Streamlit, 6 trang (bản dự phòng)
demolib/toy2d.py       # mô hình 2 chiều, score chính xác (thuần math)
demolib/provenance.py  # đọc .json của gallery, quyết định khi nào hiện điểm
demolib/charts.py      # SVG thuần, không matplotlib
gallery/               # <tên>_orig.png  <tên>_diff.png  [<tên>_recon.png]  <tên>.json
requirements.txt       # gradio (UI chính) — streamlit nằm ở requirements-streamlit.txt
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
# 1. Space mới (SDK gradio, hardware CPU basic)
hf repo create dire-demo --repo-type space --space-sdk gradio

# 2. Đẩy nội dung thư mục này lên
git clone https://huggingface.co/spaces/<user>/dire-demo /tmp/space && cd /tmp/space
cp -r <repo>/platforms/hf-spaces/. .
git add -A && git commit -m "DIRE demo (gradio, CPU free)" && git push
```

`README.md` của thư mục này là front-matter của Space: `sdk: gradio`,
`app_file: gradio_app.py`. Nếu HF báo `sdk_version 6.32.0` không hợp lệ, chọn bản
6.x mới nhất trong *Settings → Gradio* — app chỉ dùng API ổn định từ 5.x
(`Blocks`, `Tab`, `Gallery`, `Image(sources=…)`, `.release/.change`, `gr.JSON`),
nên không phụ thuộc một bản vá cụ thể.

Muốn Space chạy **bản Streamlit** thay thế: sửa front-matter thành
`sdk: streamlit` + `app_file: app.py`, và thêm `streamlit==1.67.0` vào
`requirements.txt`.

Chi tiết, số đo cold start, cổng kiểm và phương án rollback:
**`docs/demo-hf-spaces-plan.md`**.

## Cấu hình

| Biến | Ý nghĩa | Mặc định |
|---|---|---|
| `DIRE_GALLERY_DIR` | thư mục gallery, tuyệt đối hoặc tương đối với `app.py` | `gallery` |
| `DIRE_CKPT` | đường dẫn checkpoint ADM (bật tab DIRE **tại chỗ** của bản Streamlit); cần thêm `torch==2.4.0` | `ckpt/adm_lsun.pt` |
| `DEMO_URL` | địa chỉ endpoint GPU trên Modal (`https://<ws>--dire-demo-api.modal.run`) — bật trang 3 **qua GPU** | *(trống = tắt)* |
| `DEMO_TOKEN` | token của endpoint; **chỉ đặt trong Secrets**, không bao giờ gửi xuống trình duyệt | *(trống)* |

**Không có secret nào là bắt buộc.** Thiếu `DEMO_URL`/`DEMO_TOKEN` thì Space chạy
y như trước: tab *Tính DIRE thật* hiện lời giải thích thay vì nút chạy, 0 đồng,
không gọi mạng ngoài.

Đặt `DEMO_URL` + `DEMO_TOKEN` (Settings → Secrets) thì trang 3 gọi endpoint GPU:
`S = 5` còn **0,92 s** thay vì 101 s. Trình duyệt **không** thấy token — nó gọi
server của Space, server gọi Modal. Dựng endpoint: `docs/demo-modal-plan.md`
(`modal secret create dire-demo …` rồi `modal deploy -m dire_zero.demo_api`).

Ảnh người dùng vẫn không bị lưu ở đâu: Space chỉ chuyển bytes sang Modal, endpoint
tính rồi trả PNG, không ghi đĩa.

## Nguồn & bản quyền

- Bài báo / code gốc: <https://github.com/ZhendongWang6/DIRE>
- Ảnh trong gallery (khi có): DiffusionForensics (Wang et al., ICCV 2023),
  chỉ dùng cho mục đích nghiên cứu; xem `docs/runbook-free.md` §License.
