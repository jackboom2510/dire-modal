# ADR-0004 — Nới N1: train lại **để đối chiếu** trọng số tác giả, không để cải thiện

- **Trạng thái:** Accepted
- **Liên quan:** `spec.md` §3 N1, §6 C3, §7 FR-8, §12.1; `src/dire_modal/train.py`,
  `docs/adr-0003-demo-training.md`, `docs/runbook-free.md` §7

## Bối cảnh

SPEC-001 bản 1.0 đặt **N1 = "không train lại detector DIRE"**, kéo theo **C3 =
cải thiện phải training-free**. Quy tắc đó có lý do tốt: mục tiêu của dự án là
*đo đúng và cải thiện có kiểm chứng*, không phải train lại một ResNet-50.

Nhưng nó tạo ra một lỗ hổng giải thích. Khi báo cáo nói "DIRE đạt 99.9% ACC",
người đọc có quyền hỏi: **trọng số `lsun_adm.pth` đó học được gì?** Với N1 tuyệt
đối, dự án không có cách nào trả lời ngoài việc trích paper — tức là lặp lại lời
tác giả, không kiểm chứng được gì. Và nếu checkpoint có vấn đề (train trên split
lẫn test, tar thiếu ảnh, epoch dừng sớm), ta sẽ không bao giờ biết.

Cần phân biệt hai câu khác nhau:

| Câu hỏi | Trả lời bằng | Được phép không |
|---|---|---|
| "DIRE đạt bao nhiêu?" | checkpoint **tác giả** | ✅ đây là tái lập |
| "Trọng số tác giả có tái tạo được không, lệch ở đâu?" | **train lại rồi so** | ✅ đây là đối chiếu |
| "Mô hình **của chúng tôi** tốt hơn à?" | trọng số tự train | ❌ không phải mục tiêu, và không công bằng |

## Quyết định

**N1 được nới thành quy tắc mềm.** Train lại **được phép**, với ba điều kiện:

1. **Chỉ để đối chiếu.** Kết quả train lại là bằng chứng *về* trọng số tác giả,
   không phải kết quả *của* dự án. Câu "DIRE đạt X%" chỉ được nói bằng checkpoint
   tác giả (FR-1). Trọng số tự train không thay `lsun_adm.pth`, không vào bảng Δ
   của paper, không dùng chọn ngưỡng cho FR-5.
2. **Đúng giao thức của tác giả.** Siêu tham số chép từ `vendor/utils/config.py`
   (§5 F1): 1 logit + BCEWithLogits, batch 64, Adam 1e-4 β1 0.9, 256 → crop 224,
   hflip 0,5, seed 3407, 400 epoch, earlystop 5 kèm `lr/10`. Chọn epoch **chỉ**
   trên `val`; test không được chạm trong lúc train.
3. **Báo cáo hai dòng cạnh nhau trên cùng test.** `dire` (tác giả) và
   `dire_retrained` (của ta), cùng schema CSV, `report.py` in thành **hai bảng
   riêng**. Không được chỉ in một dòng.

Đổi lại, **C3 đổi nghĩa** chứ không bị bỏ: "cải thiện phải training-free" vẫn
đúng nguyên vẹn cho FR-2…FR-6. Chỉ đường **đối chiếu** (FR-8) mới được train.

## Vì sao rẻ hơn tưởng

Đầu vào của detector là **DIRE map đã phát hành** (`dire/train/lsun_bedroom/`),
không phải ảnh gốc. Nên FR-8:

- **không** cần `compute_dire`,
- **không** cần weight ADM `lsun_bedroom.pt` (5.98 GB, bị `NEVER` chặn),
- **không** phụ thuộc FR-2,
- chỉ cần tier `train` — **4 file / 5.74 GB**.

Đây đúng lý do `demo_train` (ADR-0003) chạy được trên L4: train ResNet-50 trên
ảnh 224² không cần diffusion.

## Chi phí — phần phải nói thẳng

Ước tính từ paper + code (80k ảnh, batch 64 ⇒ 1 250 step/epoch), **chưa đo**:

| Kịch bản | GPU | Chi phí |
|---|---|---|
| Đủ 400 epoch | Modal A100-80 | **$32–56** |
| Đủ 400 epoch | Vast/RunPod 4090 | **~$4–7** |
| Early-stop 40–80 epoch | Modal A100-80 | $3–11 |
| Subset 8k/lớp × 40 epoch | Modal A100-80 | ~$2.5 |
| Đo 1 epoch (`sec/epoch` thật) | bất kỳ | ~$0.1 |

⇒ **M6 nằm ngoài NFR-1 ($20)**, và đủ 400 epoch trên Modal A100 **vượt C5 ($30)**.
Vì thế quy trình bắt buộc: `--mode plan` (0 credit) → train 1 epoch để đo → **rồi
mới** quyết định. Không cam kết $ trước khi có số `sec/epoch`.

## Hệ quả

**Được:**
- Trả lời được "trọng số tác giả học được gì" bằng **số của chính dự án**.
- Mở khoá **Table 4/5/6** của paper (ablation số bước DDIM, loại đầu vào, ABS) —
  trước đây bị loại chỉ vì cần train lại trên subset ADM.
- Nếu retrain lệch nhiều, đó là **phát hiện có giá trị** (dữ liệu phát hành khác
  paper, hoặc checkpoint train bằng cấu hình khác), không phải thất bại.

**Không được:**
- Gọi retrain là "tái lập paper". Tái lập là `01_repro.sh` với checkpoint tác giả.
- Dùng ACC của retrain để nói dự án "tốt hơn" hay "kém hơn" paper.
- Bỏ nhãn `provenance: retrained` khỏi CSV/`run.json`.

## Phương án đã loại

| Phương án | Vì sao loại |
|---|---|
| Giữ N1 tuyệt đối | Không kiểm chứng được gì về trọng số; lặp lại lời paper |
| Cho train lại **và** dùng làm số chính | Trộn hai loại bằng chứng; không công bằng với paper; dễ tự khen |
| Chỉ train **head** (linear probe) | Rẻ nhưng không trả lời "trọng số ResNet-50 học được gì" — chỉ đo feature có sẵn |
| Train lại **rồi** dùng nó thay `lsun_adm.pth` trong eval | Phá S7 (Δ so với Table 2) và làm mọi so sánh trước đó vô nghĩa |

## Cách kiểm (để quy tắc không thành lời nói)

| Kiểm | Ở đâu |
|---|---|
| Retrain ghi `detector=dire_retrained` + `provenance` | AC-8.5, `run.json` |
| Báo cáo in hai dòng trên cùng test | AC-8.3, cổng **S8** |
| Lệch > 2 điểm ⇒ kết luận "chưa khớp" | AC-8.4 |
| Chọn epoch chỉ trên val | AC-8.2 |
| Trần $5/lần chạy + preflight | AC-8.6, NFR-2 |
| Subset phải mang nhãn `subset` | AC-8.7 |
