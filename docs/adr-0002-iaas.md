# ADR-0002 — Chiến lược đa nền tảng (Modal-first, dữ liệu neo ở R2/HF)

- **Trạng thái:** Accepted
- **Ngày:** 2026-10-14
- **Liên quan:** `docs/iaas-options.md`, ADR-0001, `docs/runbook-free.md`

## Bối cảnh

Repo hiện gắn chặt vào Modal: `modal.Image`, `@app.function`, `.map()`, `volumes=`, `modal.Cluster`…
Mọi file compute (`download.py`, `pack.py`, `train.py`, `eval.py`, `compute_dire.py`) chỉ chạy được trên Modal.

Ràng buộc thực tế của chủ repo:
- **Ngân sách 0 ₫** (chỉ free tier), nhưng **có thẻ quốc tế** → dùng được Modal Starter ($30/tháng credit).
- Ưu tiên **có kết quả nhanh**.
- Cần **demo công khai**.
- Máy local **mỏng** → chỉ làm control plane.

Sự thật định lượng (xem `docs/iaas-options.md` §1): **$30 credit ≈ 12 giờ A100 80GB**.
Ước lượng nhu cầu: train Tier A ~3–8 A100-h ✅, eval ~1–3 h ✅, **stage-2 (tính DIRE cho 200k ảnh) 30–60 h ❌**.

## Quyết định

**Modal là engine chính, nhưng KHÔNG phải nơi duy nhất — và giới hạn của nó được ghi thành ngưỡng cứng:**

| Vai trò | Nền tảng | Lý do |
|---|---|---|
| Orchestration, tải dữ liệu, pack, train, eval | **Modal** | $30 credit + Volume 1 TiB free + `.map()` + code đã viết |
| Dev loop / smoke test | **Kaggle** (chính) + Colab (tương tác) | ~120–330 GPU-h/tháng, **không** tiêu credit Modal |
| Bulk GPU khi vượt credit | **Thunder $1.09/h** → Vast → RunPod | Rẻ hơn Modal 30–55%/giờ |
| Demo công khai | **HF Spaces CPU free** | URL công khai, ResNet-50 chạy được trên CPU |
| Dữ liệu (neo trung lập) | **R2 + HF Hub + Kaggle** | Xem ADR-0001 — không phụ thuộc nhà cung cấp compute nào |

**Ngưỡng rời Modal (bắt buộc tuân theo):**

1. Job ước tính **> 12 A100-giờ** → không chạy hết trên Modal; chia nhỏ, hoặc chạy trên Thunder/Vast.
2. Cùng một job chạy **> 3 lần** (đã ổn định) và > 4 GPU-h → chuyển sang Vast/Thunder để tiết kiệm.
3. Bất kỳ job nào > 1 h → **phải** checkpoint ra Volume **và** R2 trước khi chạy (để resume ở nơi khác).
4. Job cần SSH/root hoặc preemption không kiểm soát → dùng VM (RunPod/Vast), không dùng Modal.
5. **Không bao giờ** đặt kết quả duy nhất trên nền tảng free không có storage bền (Kaggle/Colab).

## Hệ quả

**Tích cực**
- Chi phí = $0 cho tới khi thật sự cần: dev trên Kaggle, train trong $30 credit.
- Không bị khoá vào Modal: dữ liệu ở R2/HF, code chạy được ở nơi khác (đã có `platforms/runpod`, `platforms/vast`, `platforms/kaggle`).
- Có đường nâng cấp rẻ nếu hết credit, không phải viết lại pipeline.

**Tiêu cực / chấp nhận**
- Phải viết code một lần, chạy nhiều nơi (`platforms/*`) → tốn công hơn thuần Modal.
- Trải job dài qua Lightning free (restart 4 h) đòi hỏi resume thật chắc — rủi ro nếu checkpoint hỏng.
- Học 3–8 h/nền tảng mới (đã tính vào chi phí ở `docs/iaas-options.md` §2 G4).

**Việc phải làm kèm theo**
- `train.py` **phải** lưu `last.pt` mỗi epoch + `vol_ckpt.commit()` (điều kiện để resume ở nơi khác).
- Trước khi chạy job lớn: `python3 tools/free_quota.py --used modal=<đã dùng>`.
- Sau khi port `compute_dire.py`: chạy `bash tools/measure_dire.sh` và **cập nhật §4.3 của `iaas-options.md`** bằng số thật.

## Phương án đã cân nhắc & loại

| Phương án | Vì sao loại |
|---|---|
| **Thuần Modal** | $30 không đủ cho stage-2; lock-in cao; phí | 
| **Thuần VM (Vast/RunPod)** | Rẻ hơn/giờ nhưng mất scale-to-zero, phải tự schedule 15 job eval, mất Volume |
| **Thuần Kaggle/Colab (0 ₫ tuyệt đối)** | Không SSH, mất máy khi hết phiên, P100 16 GB, quota 30 h/tuần → không train nổi Tier A |
| **Azure $200** | $200 = ~50 A100-h **nhưng** phải xin quota N-series, credit hết hạn 30 ngày, giá/giờ đắt (chủ repo đã chọn bỏ) |
| **GCP $300** | Trial **không gắn được GPU vào Compute Engine VM** → vô dụng cho mục tiêu này |
