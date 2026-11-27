---
doc_id: EXP-008
title: Kaggle CelebA-HQ progress checkpoint
status: in-progress
snapshot_date: 2026-10-14
---

# Checkpoint tiến độ — Kaggle / CelebA-HQ-SDv2

> Tài liệu trạng thái tại snapshot, không phải ledger của một lần train. Số liệu Kaggle dưới đây do người dùng cung cấp từ output; không được đo lại trong checkout. Cập nhật checkpoint này khi đổi trạng thái/gate.

## Trạng thái hiện tại

- **Branch:** `arena/3ff8bd36-dire-modal` — active, từ base `1a3e0a856142d17e133b78fa58c572739c782797`.
- **Scope đã chọn:** research-extension CelebA-HQ-SDv2; chỉ dùng `dire/train/celebahq/{real,sdv2}` và test CelebA-HQ (real, bốn họ holdout, SDv2 control). Không trộn LSUN-Bedroom/ImageNet.
- **Dataset trên Kaggle:** input read-only tại `/kaggle/input/datasets/j4cknguyn/diffusionforensics/dire`; người dùng chọn giữ nguyên, không copy/re-upload/flatten.
- **Đã chạy trên Kaggle theo report người dùng:** audit cây toàn bộ DIRE, path-index CelebA-HQ, Stage 2 SHA-256/full-decode (`pass`, 72.600/72.600; không có SHA/decode failures). Agent chưa truy cập trực tiếp Kaggle hay artifact.
- **Chưa chạy trên Kaggle:** validation split, smoke train, train hoặc benchmark. Stage 3 code tạo candidate deterministic split vừa được thêm vào notebook/module; hiện mới test cục bộ, chưa tạo split thực tế.

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

## Quyết định người dùng đã chọn

- Giữ Kaggle input read-only; không tạo bản Dataset đã flatten để tránh copy/re-upload.
- Dùng reader riêng cho CelebA-HQ, chỉ xử lý các nhóm mục tiêu; không áp dụng preflight archive cũ cho cây đã giải nén.
- Dùng member-style locator suy ra bằng cách bỏ thư mục generator ngoài (ví dụ `real/real/x.png` → `real/x.png`); code phải đánh dấu `locator_inferred=true`. Locator này **chưa xác minh** với archive member gốc.
- Stage 2 SHA-256 + full image decode cho 72.600 file được người dùng báo pass, không lỗi, không resume. Implementation được thiết kế read-only/no-GPU; console excerpt không bao gồm các cờ `input_modified`/`gpu_work_started`.
- Người dùng đã **chốt** validation protocol: salt `dire-celebahq-val-v1`; SHA-256 trên salt + NUL + `member_locator_inferred`, sort digest tăng dần (tie-break locator), lấy `round-half-up(0,10 × N)` mỗi class.
- Split **chưa được tạo**: Stage 3 code mới chỉ được test cục bộ; cần chạy cell Stage 3 trong Kaggle để sinh candidate manifest và kiểm tra duplicate-content audit.
- Chưa khóa seed/training budget; chưa train hoặc benchmark.

## Stage integrity và validation implementation

- `src/dire_modal/kaggle_celebahq_integrity.py` hash file theo chunk, sau đó dùng Pillow `load()` để full-decode từng ảnh; ghi SQLite checkpoint, JSONL per-file manifest và summary dưới `/kaggle/working`.
- Có thể resume khi cùng cây dữ liệu bất biến và fingerprint metadata (relative path, size, mtime) khớp. Hash và decode là hai lượt đọc riêng. Tổng `stat` bytes của các file CelebA-HQ trong audit là **5.775.377.067 byte cho một lượt**; lượng I/O thực tế của hai lượt và thời gian chưa đo.
- Hash xác định bytes hiện đang mount; không xác nhận archive provenance, source-image identity hay train/test independence.
- Stage 3 mới: `src/dire_modal/kaggle_celebahq_validation.py`, được embed trong `platforms/kaggle/celebahq-path-index.ipynb`; chỉ đọc summary/JSONL đã pass, chọn 10% train mỗi label theo protocol chốt, ghi candidate JSON dưới `/kaggle/working`. Đồng thời report SHA-256 giao nhau giữa train/validation; nếu có exact-byte duplicates thì `review_required`, không tự ý đổi rank selection.
- Test cục bộ cho integrity, validation split, notebook/tree adapters và docs: **28 passed**; pyflakes, py_compile, notebook JSON/code-cell parse và `git diff --check` sạch. Các test không truy cập Kaggle, không chứng minh Stage 3 đã chạy trên data thực.

## Next gates

1. Trong Kaggle runtime chứa output Stage 2, chạy cell Stage 3 mới nhất ở cuối `platforms/kaggle/celebahq-path-index.ipynb`. Nếu runtime mới, phải bảo đảm `/kaggle/working` còn các file integrity đã export; dùng summary và JSONL paths ở phần trên. Không chạy lại Stage 2 trừ khi artifacts không còn/không khớp.
2. Gửi lại output Stage 3: counts theo label, `validation_id_list_sha256`, content-overlap status và đường dẫn artifact. Dừng nếu integrity bundle không được xác nhận hoặc overlap audit là `review_required`; không tự điều chỉnh IDs.
3. Chỉ sau candidate split và review mới preflight dung lượng/GPU/khả năng lưu-resume; sau đó mới chốt seed, budget, checkpoint selection và xin phép compute.
4. Giữ bốn generator family CelebA-HQ (`dalle2`, `if`, `midjourney`, `stargan`) làm holdout riêng; không tune trên test. `sdv2` test là seen-generator control.

## Protocol validation đã chốt

Người dùng chấp thuận salt `dire-celebahq-val-v1` và exact-count rank rule: trong từng class, sắp xếp theo `SHA256(salt + NUL + member_locator_inferred)` tăng dần, tie-break bằng locator; lấy `round-half-up(10% × N)` IDs đầu tiên. Với inventory counts hiện có (28.000 real, 40.000 SDv2), mục tiêu dự kiến là 2.800 và 4.000 IDs; đây là suy ra từ report inventory, chưa phải realized counts.

**Chưa tạo validation split trên Kaggle.** Theo report, integrity gate đã pass; bước còn lại là chạy Stage 3 candidate cell. Locator hiện suy ra từ cây thư mục (`locator_inferred=true`) và chưa được xác minh với tar member gốc. Artifact sẽ ghi salt, protocol, realized counts và SHA-256 của danh sách validation IDs để tái lập.
