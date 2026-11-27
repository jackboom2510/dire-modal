---
doc_id: EXP-008
title: Kaggle CelebA-HQ progress checkpoint
status: in-progress
snapshot_date: 2026-10-14
---

# Checkpoint tiến độ — Kaggle / CelebA-HQ-SDv2

> Tài liệu trạng thái tại snapshot, không phải ledger của một lần train. Số liệu Kaggle dưới đây do người dùng cung cấp từ output; không được đo lại trong checkout. Cập nhật checkpoint này khi đổi trạng thái/gate.

## Trạng thái hiện tại

- **Branch:** `arena/022ccec0-dire-modal` — active, từ base `f68450babe828886e93a0ccd6a89e650ca66cb51`.
- **Scope đã chọn:** research-extension CelebA-HQ-SDv2; chỉ dùng `dire/train/celebahq/{real,sdv2}` và test CelebA-HQ (real, bốn họ holdout, SDv2 control). Không trộn LSUN-Bedroom/ImageNet.
- **Dataset trên Kaggle:** input read-only tại `/kaggle/input/datasets/j4cknguyn/diffusionforensics/dire`; người dùng chọn giữ nguyên, không copy/re-upload/flatten.
- **Đã chạy trên Kaggle theo report người dùng:** audit cây toàn bộ DIRE, path-index CelebA-HQ, Stage 2 SHA-256/full-decode (`pass`, 72.600/72.600; không có SHA/decode failures). Agent chưa truy cập trực tiếp Kaggle hay artifact.
- **Stage 3 theo report người dùng:** v1 candidate tạo đúng counts nhưng `content_overlap_audit=review_required` với sáu real duplicate pairs. Người dùng đã duyệt protocol v2 hash-aware để tạo lại validation list; v2 code mới test cục bộ, Kaggle chưa rerun. Chưa chạy smoke train, train hoặc benchmark.

## Bằng chứng do người dùng cung cấp

### Audit cây toàn bộ DIRE

- Root: `/kaggle/input/datasets/j4cknguyn/diffusionforensics/dire`.
- **422.594** regular files, **10.330** directories, **40.877.686.851 byte** theo `stat`; suffixes `.jpeg`, `.jpg`, `.png`.
- Split/domain file counts: `test/celebahq` 4.600; `test/imagenet` 29.994; `test/lsun_bedroom` 24.000; `train/celebahq` 68.000; `train/imagenet` 80.000; `train/lsun_bedroom` 200.000; `val/imagenet` 11.000; `val/lsun_bedroom` 5.000.
- Có nhiều leaf lặp như `real/real` và `adm/adm`. Audit chỉ đếm metadata/suffix, không đọc ảnh, không hash, không xác minh provenance.
- Trước đó Kaggle UI từng hiển thị staging `313.297 files / 28,82 GiB`, khác với audit `422.594 files / 40.877.686.851 byte`. **Chưa giải thích/xác minh khác biệt phiên bản hoặc phạm vi.**

### Path index CelebA-HQ

Người dùng chạy notebook và báo `Index status: complete`, **72.600** file:

| Split | Generator | Label | Files |
|---|---|---:|---:|
| train | real | 0 | 28.000 |
| train | sdv2 | 1 | 40.000 |
| test | real | 0 | 1.000 |
| test | dalle2 | 1 | 500 |
| test | if | 1 | 1.000 |
| test | midjourney | 1 | 100 |
| test | sdv2 | 1 | 1.000 |
| test | stargan | 1 | 1.000 |

Notebook đã ghi `/kaggle/working/celebahq-path-index.json`; người dùng chỉ gửi console summary, chưa chia sẻ file JSON đầy đủ. Các counts khớp inventory mong đợi, nhưng không chứng minh ảnh giải mã được hay byte/provenance gốc.

### Integrity Stage 2 — kết quả do người dùng cung cấp

Người dùng báo:

- `Integrity status: pass`; records `72.600 / 72.600`; hashed this run `72.600`; reused checkpoint `0`.
- SHA failures `0`; decode failures `0`.
- SQLite: `/kaggle/working/celebahq-integrity-81dcda99aeb997b25b82.sqlite3`.
- JSONL manifest: `/kaggle/working/celebahq-integrity-81dcda99aeb997b25b82.jsonl`.
- Summary: `/kaggle/working/celebahq-integrity-81dcda99aeb997b25b82-summary.json`.

Đây là output người dùng dán; agent chưa đọc/kiểm tra trực tiếp manifest/summary trên Kaggle. Integrity pass xác nhận bytes hiện mount và decode theo verifier, không xác minh archive provenance hoặc source identity.

### Stage 3 candidate — kết quả do người dùng cung cấp

- Người dùng báo `candidate_created`; class 0: 28.000 tổng, 2.800 validation, 25.200 train; class 1: 40.000 tổng, 4.000 validation, 36.000 train.
- v1 `validation_id_list_sha256`: `1f550e2992e136873caf6df26cc46de3492794ebc5b8badc6eece9575f7047d6` (**sẽ được supersede nếu chạy protocol v2**).
- `content_overlap_audit.status`: `review_required`; người dùng sau đó báo **6 SHA trùng**, tất cả label 0 (`real`), label 1 không có overlap. Mỗi digest ghép một path validation với một path training:

| SHA-256 | Validation path | Training path |
|---|---|---|
| `00966d9f4063849acc52021f10e231ae2a3534e7faff56a5fc7b2419de6f631e` | `train/celebahq/real/real/021366.jpg` | `train/celebahq/real/real/160486.jpg` |
| `01659f1e75b8862d24f78371c5effcf8929feab9eb2bcdec7a171466299e171a` | `train/celebahq/real/real/083407.jpg` | `train/celebahq/real/real/067217.jpg` |
| `860572d7a700fac2e5f4b29b015e5b82d9474764d91dd79423e174c4dbe3a4ab` | `train/celebahq/real/real/021423.jpg` | `train/celebahq/real/real/019673.jpg` |
| `9507a308ff202aa9e4f72f10862eaa15b4cb04c798870d7f67848dedbf146df5` | `train/celebahq/real/real/074402.jpg` | `train/celebahq/real/real/051286.jpg` |
| `de6e6810283fa32808729048052d501d01e7fb15396cc8f5041210b200100c97` | `train/celebahq/real/real/020328.jpg` | `train/celebahq/real/real/186880.jpg` |
| `f52e23b0b52f5e1ac35db1f1c8f56922a5b82f3e92a55e706d0967e6c04fbcca` | `train/celebahq/real/real/074740.jpg` | `train/celebahq/real/real/069407.jpg` |

- Artifact: `/kaggle/working/celebahq-validation-split-v1-81dcda99aeb9.json`.

Chi tiết cặp/hash là do người dùng dán từ console, chưa được agent đọc trực tiếp file artifact. Sáu cặp xác nhận exact-byte duplication và gây overlap train/validation. Chưa chấp nhận split để train; cần chốt cách xử lý mà không âm thầm đổi ID hoặc bỏ samples.

## Quyết định người dùng đã chọn

- Giữ Kaggle input read-only; không tạo bản Dataset đã flatten để tránh copy/re-upload.
- Dùng reader riêng cho CelebA-HQ, chỉ xử lý các nhóm mục tiêu; không áp dụng preflight archive cũ cho cây đã giải nén.
- Dùng member-style locator suy ra bằng cách bỏ thư mục generator ngoài (ví dụ `real/real/x.png` → `real/x.png`); code phải đánh dấu `locator_inferred=true`. Locator này **chưa xác minh** với archive member gốc.
- Stage 2 SHA-256 + full image decode cho 72.600 file được người dùng báo pass, không lỗi, không resume. Implementation được thiết kế read-only/no-GPU; console excerpt không bao gồm các cờ `input_modified`/`gpu_work_started`.
- Người dùng đã **chốt** validation protocol: salt `dire-celebahq-val-v1`; SHA-256 trên salt + NUL + `member_locator_inferred`, sort digest tăng dần (tie-break locator), lấy `round-half-up(0,10 × N)` mỗi class.
- Người dùng đã duyệt protocol revision v2: loại khỏi validation candidate pool mọi train record có SHA-256 xuất hiện hơn một lần trên toàn bộ hai label; vẫn xếp các ứng viên còn lại theo salt/hash rank đã chốt, chọn đủ 10% dựa trên tổng records mỗi label. Duplicate groups giữ nguyên trong training. V2 giữ counts nhưng thay validation IDs; đây là protocol khác v1 và phải ghi version riêng.
- V1 Kaggle candidate có overlap nên không dùng để train. Chờ người dùng chạy lại standalone Stage 3 notebook v2 và xác nhận audit pass; không train trước kết quả đó.
- Chưa khóa seed/training budget; chưa train hoặc benchmark.

## Stage integrity và validation implementation

- `src/dire_modal/kaggle_celebahq_integrity.py` hash file theo chunk, sau đó dùng Pillow `load()` để full-decode từng ảnh; ghi SQLite checkpoint, JSONL per-file manifest và summary dưới `/kaggle/working`.
- Có thể resume khi cùng cây dữ liệu bất biến và fingerprint metadata (relative path, size, mtime) khớp. Hash và decode là hai lượt đọc riêng. Tổng `stat` bytes của các file CelebA-HQ trong audit là **5.775.377.067 byte cho một lượt**; lượng I/O thực tế của hai lượt và thời gian chưa đo.
- Hash xác định bytes hiện đang mount; không xác nhận archive provenance, source-image identity hay train/test independence.
- Stage 3 v2: `src/dire_modal/kaggle_celebahq_validation.py`, embed trong notebook tích hợp và standalone `platforms/kaggle/celebahq-validation-split.ipynb`. Đọc summary/JSONL read-only từ Kaggle Dataset, không rerun Stage 1/2; trước khi chọn 10% mỗi label, bỏ toàn bộ validation candidates có SHA-256 lặp trong combined train pool. Duplicate records còn lại được giữ trong training; count mục tiêu vẫn tính trên tổng class records. Xuất artifact v2 riêng, không ghi đè v1.
- Test cục bộ sau revision v2: **31 passed**; pyflakes, py_compile, notebook parse và `git diff --check` sạch. Không test nào truy cập Kaggle.

## Next gates

1. Chạy lại riêng Stage 3 notebook v2 trên cùng summary + JSONL input; không chạy lại Stage 1/2 và không dùng candidate v1 cũ.
2. Gửi output v2: counts, `candidate_audit_by_label`, `validation_id_list_sha256`, `content_overlap_audit.status`, và path artifact. Chỉ khi counts còn 2.800/4.000 và audit `pass` mới chấp nhận validation split.
3. Sau đó mới preflight dung lượng/GPU/khả năng lưu-resume; chốt seed, budget, checkpoint selection và xin phép compute.
4. Giữ bốn generator family CelebA-HQ (`dalle2`, `if`, `midjourney`, `stargan`) làm holdout riêng; không tune trên test. `sdv2` test là seen-generator control.

## Protocol validation đã chốt

Người dùng chấp thuận salt `dire-celebahq-val-v1` và exact-count rank rule: trong từng class, sắp xếp theo `SHA256(salt + NUL + member_locator_inferred)` tăng dần, tie-break bằng locator; lấy `round-half-up(10% × N)` IDs đầu tiên. Với inventory counts hiện có, mục tiêu là 2.800 real và 4.000 SDv2.

Sau khi Stage 3 v1 phát hiện sáu duplicate groups cắt qua train/validation, người dùng **đã duyệt revision v2**: tính tần suất SHA-256 trên combined train pool của cả hai label; mọi record có digest xuất hiện >1 lần không đủ điều kiện validation. Sắp xếp các eligible records theo cùng salt/hash/locator rank và lấy đủ `round-half-up(10% × N)` mỗi label, với N là tổng records của label trước lọc. Repeated groups giữ trong training. Đây đổi validation IDs so với v1 nhưng giữ exact 2.800/4.000 counts và loại exact-byte overlap giữa hai partition; protocol ID là `celebahq-content-disjoint-exact-count-hash-rank-v2`.

**V1 candidate không được dùng cho train** vì overlap audit `review_required`. V2 code mới test cục bộ; cần rerun riêng Stage 3 và xác nhận artifact v2 có audit `pass`. Locator vẫn suy ra từ cây thư mục (`locator_inferred=true`) và chưa xác minh với tar member gốc.
