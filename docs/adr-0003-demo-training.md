# ADR-0003 — `demo_train`: train ở quy mô demo để MÔ PHỎNG, không để báo cáo

- **Trạng thái:** Accepted
- **Ngày:** 2026-10-28
- **Liên quan:** `src/dire_modal/demo_train.py`, `src/dire_modal/demo_train.yaml`,
  `docs/runbook-free.md` §7, ADR-0002, `tests/test_demo_train.py`

## Bối cảnh

Cần giải thích DIRE hoạt động ra sao (video manim: lý thuyết → pipeline →
bảng số → đường đi dữ liệu). Scene "training" cần **một đường loss** để animate.

Ba cách có thể làm, và hai cách bị loại:

1. **Vẽ đường cong cho đẹp** — bị loại. Đây đúng loại lỗi repo này đã sửa nhiều
   lần: một hình trông như dữ liệu nhưng không phải dữ liệu. Người xem không có
   cách nào phân biệt.
2. **Train lại đúng 400 epoch như tác giả** — bị loại. Ràng buộc **N1** của dự án
   là *không train lại*; ngoài ra 400 epoch trên A100 cho một video 4 phút là
   tiêu tiền sai chỗ, và kết quả vẫn không thể so với paper (thiếu nhiều thứ
   khác của pipeline tác giả).
3. **Train thật, quy mô nhỏ, có nhãn** — chọn cách này.

Một chi tiết kỹ thuật quyết định chi phí: **train ResNet của DIRE không cần
diffusion**. Đầu vào là **DIRE map đã tính sẵn** (tar `0_real`/`1_fake` của tác
giả), không phải ảnh gốc. Nên đây chỉ là fine-tune ResNet50 trên ảnh: **L4 là
đủ**, không cần ADM 2.5 GB, không cần `compute_dire`.

## Quyết định

Thêm `demo_train` — một module **tách biệt**, tự viết vòng lặp, có nhãn ở mọi
artefact, và bị cấm ghi vào chỗ của số thật.

### Ranh giới (là hợp đồng, có test)

| Quy tắc | Vì sao | Test |
|---|---|---|
| Mặc định `--mode plan`, phải gõ `--mode run` mới train | gõ thiếu cờ không được tiêu tiền | `test_main_defaults_to_plan`, `test_plan_mode_prints_label_and_trains_nothing` |
| Ghi **chỉ** vào `/ckpt/demo_train/<exp>/` | `eval::report` quét `/ckpt/results/**` ⇒ demo không bao giờ lọt vào bảng Δ | `test_output_dir_is_never_the_results_tree` |
| Mọi artefact mang `DEMO_LABEL` + `"demo": true` | ACC của 2 000 ảnh × 5 epoch **thấp hơn 99.9% của paper rất nhiều** | `test_log_payload_is_the_only_source_for_scenes`, `test_checkpoint_carries_one_logit_and_eval_can_read_it` |
| Trần `--max-images` + `--max-minutes`, in `[tiền]` trước | đây là chỗ duy nhất trong module tiêu tiền | `test_enforce_rejects_too_many_images`, `…_too_many_minutes` |
| Không nằm trong `scripts/` | repo chốt **đúng 2 quy trình** (`01_repro`, `02_improve`) | `tests/test_scripts_layout.py` |
| Không import `torch` ở tầng module | `plan`/test không được kéo torch vào | `test_module_does_not_import_torch_at_top` |
| Log chỉ chứa số hữu hạn | `NaN` trong JSON là số không đọc được (NFR-6) | `test_log_entry_refuses_nan` |
| Ô trống giữ `null`, không thay `0` | "chưa có val" ≠ "val = 0" | `test_summarize_keeps_missing_val_empty` |

### Dữ liệu: đúng split của tác giả

`/raw/dire/train/lsun_bedroom/{real,adm}.tar.gz` (train) và
`/raw/dire/val/lsun_bedroom/{real,adm}.tar.gz` (val). **Không** dùng test split:
train trên test là leakage và mọi con số sau đó vô nghĩa — kể cả con số dùng để
minh hoạ.

Tier `train` của manifest = **4 file · 5.74 GB**. Ở trần 76.6 KiB/s của tài khoản
Baidu free, đó là **~20–22 giờ** tải (≈ 0 credit vì chạy trên Modal CPU). Đo lại
bằng `modal run -m dire_modal.download::plan --tiers train` trước khi tin con số
này — tier chỉ lưu tổng, không lưu phần lẻ từng file.

### Siêu tham số: chép từ tác giả, không đoán

Nguồn: `src/dire_modal/vendor/utils/config.py` (`DefaultConfigs`) và
`vendor/utils/trainer.py`. File cấu hình để **cạnh module**
(`src/dire_modal/demo_train.yaml`) chứ không ở `configs/` gốc repo, vì Modal chỉ
đóng gói thư mục module — cùng lý do `tiers.yaml` nằm ở đó.

| Tham số | Tác giả | Demo | Ghi chú |
|---|---|---|---|
| `seed` | 3407 | 3407 | |
| `arch` | resnet50, pretrained | như vậy | |
| lớp cuối | **1 logit** + `BCEWithLogitsLoss` | như vậy | `fc.weight` = `[1, 2048]` |
| `batch_size` | 64 | 64 | |
| `loadSize`/`cropSize` | 256 / 224 | như vậy | |
| `optim`/`lr`/`beta1` | adam / 1e-4 / 0.9 | như vậy | |
| aug | resize+crop+flip; blur/jpg/gray = 0.0 | như vậy | |
| `nepoch` | 400 + earlystop 5 | **5** | khác biệt có chủ ý |
| ảnh | toàn bộ split | **2 000** | khác biệt có chủ ý |

Test `test_config_matches_the_authors_source` đọc thẳng file vendor bằng regex,
nên chép sai một số là đỏ — không phụ thuộc việc ai đó nhớ đúng.

### Chi phí

`estimate()` tính tay được: 2 000 ảnh / 64 = 32 step/epoch × 5 = **160 step** ×
0.35 s = 56 s, cộng val (20 step, chỉ forward) ≈ 2,3 s ⇒ **≈ 58 s** ⇒ L4 @
$0.80/h ≈ **$0.013**. `sec_per_step` là **ước lượng,
chưa đo**; đo rồi thì truyền `--sec-per-step` để con số tiền có nghĩa.

## Hệ quả

**Được:**
- Có một đường loss **thật** để animate, kèm `loss_log.json` là nguồn số duy
  nhất cho scene manim (không có nhánh "số tự bịa").
- **Khép vòng với bản vá `fc.weight`**: checkpoint do module này sinh ra có
  `fc.weight [1, 2048]`, nên `eval.load_detector` **phải** nạp được nó và
  `prob_fake` phải dùng `sigmoid`. Nếu bản vá sai, test này đỏ.
- Dùng lại `eval.iter_images`, `eval.label_of`, `eval.preprocess`, `E.PREPROC`
  ⇒ train và eval không thể lệch nhau về nhãn/tiền xử lý. Val dùng đúng
  `eval.preprocess`, nên val ACC so được với eval ACC.

**Không được (và không có đường nào để làm nhầm):**
- Trích ACC/AP của `demo_train` vào paper/README/báo cáo. Checkpoint nằm ngoài
  `results/`, và nhãn `demo` đi theo cả checkpoint lẫn log.
- Gọi `demo_train` là "tái lập". Tái lập là `scripts/01_repro.sh` với checkpoint
  của tác giả (`docs/runbook-free.md` §8).

**Rủi ro còn lại:** ai đó chép `loss_log.json` vào một bài nói và bỏ phần nhãn.
Giảm thiểu: nhãn nằm trong **khoá `label` của chính file JSON**, nên bỏ nó đi là
phải sửa dữ liệu một cách có ý thức, không phải bỏ sót.

## Phương án đã loại

| Phương án | Vì sao loại |
|---|---|
| Gọi `vendor/trainer.py` | gắn cứng `exp_dir`/`logs_path`/400 epoch/visdom; kéo cả hệ thống file vào việc 5 phút |
| Train trên test split cho nhanh | leakage; số vô nghĩa kể cả để minh hoạ |
| Vẽ đường loss giả | đúng loại lỗi NFR-6 mà repo đã sửa nhiều lần |
| Đặt config ở `configs/demo_train.yaml` | Modal không đóng gói thư mục đó ⇒ container không thấy file |
| Dùng `np.random` cho thứ tự ảnh | phải seed cả `torch`/`random`/`numpy` mới tái lập được; seed 3407 như tác giả |
