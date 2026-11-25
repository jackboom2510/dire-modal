# Prototype trực quan DDIM → reconstruction → DIRE

## Context
Repo đã có implementation phù hợp trong `platforms/hf-spaces/demolib/live.py`: `run_steps()` chạy hai vòng DDIM thật, gọi callback sau mỗi forward và trả ảnh reconstruction, DIRE map, NFE, thời gian và thống kê. Cần tạo một prototype trực quan để người xem thấy quá trình từ ảnh gốc qua các bước nhiễu/inversion, rồi đi ngược để tái tạo và tính DIRE; không biến nó thành detector và không gọi Modal.

## Mục tiêu
Tạo một prototype local/CPU dùng ảnh mẫu hoặc ảnh người dùng chọn, với điều khiển số mốc `S` nhỏ (1–5 mặc định để không quá chậm), hiển thị từng frame của hai pha:

1. **DDIM inversion / đi lên:** ảnh gốc `x0` → các trạng thái t tăng → `x_T`.
2. **DDIM reconstruction / đi xuống:** `x_T` → các trạng thái t giảm → `x̂0`.
3. **DIRE:** tính và hiển thị `|x0 − x̂0|` theo từng pixel/kênh, cùng mean/p99/max và `2S NFE`.

Mỗi frame phải ghi `phase`, index, timestep, NFE và trạng thái; reconstruction và DIRE phải được tạo từ cùng run, không dùng animation giả. Giao diện phải ghi rõ đây là DIRE map, không phải `P(fake)`; nếu chạy toy 2D thì ghi toy, nếu chạy ADM checkpoint thì ghi checkpoint/provenance.

## Phương án đề xuất
- Tận dụng `demolib.live.run_steps()` và callback `on_frame`, không viết lại thuật toán DDIM.
- Tạo giao diện prototype riêng tại `platforms/hf-spaces/ddim_prototype.py` (Streamlit hoặc Gradio theo dependency sẵn có); ưu tiên chạy local với checkpoint được cung cấp, không tải checkpoint tự động.
- Nếu không có checkpoint ADM local, cung cấp chế độ **toy 2D** dựa trên `demolib.toy2d` để vẫn minh họa đầy đủ hai vòng lặp; nhãn rõ “toy 2D, không phải ảnh/ADM”. Không giả vờ kết quả toy là ảnh thật.
- Có hai chế độ xem: `step-by-step` (frame slider) và `replay` (tự chuyển frame), cùng panel so sánh ảnh gốc / ảnh cuối / DIRE.
- Dùng ảnh từ `modal-runs/gallery` làm gallery chỉ khi metadata `.json` đi kèm; nhưng gallery PNG chỉ là display-only và không được dùng làm authoritative map. Prototype tính toán mới phải đánh dấu `self-computed` và ghi S/checkpoint.

## Hiển thị tối thiểu
- Header: `DIRE — DDIM inversion và reconstruction`.
- Pipeline diagram: `x0 → inversion (t tăng) → xT → reconstruction (t giảm) → x̂0 → |x0−x̂0|`.
- Step cards: phase, `t`, frame image, `eps_rms`, map mean.
- Summary: `S`, `2S NFE`, seconds, DIRE mean/p99/max, source/provenance.
- Warnings: cùng model/schedule/eta=0 cho hai chiều; DIRE không phải detector score; PNG map chỉ để xem; `S` là số mốc respacing.

## Verification
- Unit test callback nhận đủ `2S` frame và thứ tự `invert` tăng t / `reconstruct` giảm t.
- Kiểm tra frame cuối reconstruction khớp `recon_png` và map stats khớp output.
- Chạy toy 2D/CPU benchmark, không Modal/GPU.
- Chạy `pytest` nhóm demo/DDIM hiện có và static checks; ghi rõ nếu ADM checkpoint không có nên chỉ kiểm toy mode.
