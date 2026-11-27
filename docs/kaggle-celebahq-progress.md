---
doc_id: EXP-008
title: Kaggle CelebA-HQ progress checkpoint
status: in-progress
snapshot_date: 2026-10-14
---

# Checkpoint tiến độ — Kaggle / CelebA-HQ-SDv2

> Tài liệu trạng thái tại snapshot, không phải ledger của một lần train. Số liệu Kaggle dưới đây do người dùng cung cấp từ output; không được đo lại trong checkout. Cập nhật checkpoint này khi đổi trạng thái/gate.

## Trạng thái hiện tại

- **Branch:** `arena/c4857c07-dire-modal` — active, từ base `df5361df42d060a2470eaab0acc7c2b8786b1681`.
- **Scope đã chọn:** research-extension CelebA-HQ-SDv2; chỉ dùng `dire/train/celebahq/{real,sdv2}` và test CelebA-HQ (real, bốn họ holdout, SDv2 control). Không trộn LSUN-Bedroom/ImageNet.
- **Dataset trên Kaggle:** input read-only tại `/kaggle/input/datasets/j4cknguyn/diffusionforensics/dire`; người dùng chọn giữ nguyên, không copy/re-upload/flatten.
- **Đã chạy trên Kaggle:** audit cây toàn bộ DIRE và path-index CelebA-HQ. Chưa chạy integrity hash/decode, validation split, smoke train, train hay benchmark.
- **Trong workspace hiện tại:** Stage 2 SHA-256/full-decode đã được chuẩn bị trong notebook, kèm checkpoint SQLite/JSONL. Đây là code đã test cục bộ; **chưa chạy trên Kaggle** và chưa có kết quả integrity thật.

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

## Quyết định người dùng đã chọn

- Giữ Kaggle input read-only; không tạo bản Dataset đã flatten để tránh copy/re-upload.
- Dùng reader riêng cho CelebA-HQ, chỉ xử lý các nhóm mục tiêu; không áp dụng preflight archive cũ cho cây đã giải nén.
- Dùng member-style locator suy ra bằng cách bỏ thư mục generator ngoài (ví dụ `real/real/x.png` → `real/x.png`); code phải đánh dấu `locator_inferred=true`. Locator này **chưa xác minh** với archive member gốc.
- Tiếp theo chạy SHA-256 và full image decode cho toàn bộ 72.600 file; không dùng GPU và không sửa input. Stage đã chuẩn bị checkpoint ngoài input.
- Người dùng đã **chốt** validation protocol: salt `dire-celebahq-val-v1`; SHA-256 trên salt + NUL + `member_locator_inferred`, sort digest tăng dần (tie-break locator), lấy `round-half-up(0,10 × N)` mỗi class. Split **chưa được tạo**; chỉ thực hiện sau khi integrity stage qua gate.
- Chưa khóa seed/training budget; chưa train hoặc benchmark.

## Stage integrity đã chuẩn bị

- `src/dire_modal/kaggle_celebahq_integrity.py` hash file theo chunk, sau đó dùng Pillow `load()` để full-decode từng ảnh; ghi SQLite checkpoint, JSONL per-file manifest và summary dưới `/kaggle/working`.
- Có thể resume khi cùng cây dữ liệu bất biến và fingerprint metadata (relative path, size, mtime) khớp. Hash và decode là hai lượt đọc riêng. Tổng `stat` bytes của các file CelebA-HQ trong audit là **5.775.377.067 byte cho một lượt**; lượng I/O thực tế của hai lượt và thời gian chưa đo.
- Hash xác định bytes hiện đang mount; không xác nhận archive provenance, source-image identity hay train/test independence.
- Test cục bộ: 22 passed; pyflakes, py_compile, notebook parsing và `git diff --check` sạch. Không có Kaggle run nào từ các test này.

## Next gates

1. Nếu Kaggle runtime cũ còn mở, thêm/chạy Stage 2 trong cùng phiên để tránh khởi động runtime khác; lưu SQLite/JSONL/summary ra khỏi `/kaggle/input` trước khi phiên kết thúc.
2. Đọc kết quả integrity: số hash/decode failures, per-group counts, kích thước/format; dừng và điều tra lỗi thay vì âm thầm bỏ ảnh.
3. Sau khi integrity qua gate, tạo validation IDs theo protocol đã chốt: salt `dire-celebahq-val-v1`; với mỗi class, tính `SHA256(salt + NUL + member_locator_inferred)`, sort digest tăng dần (tie-break locator), chọn `round-half-up(0,10 × N)` phần tử. Ghi realized counts và hashes/file list; chưa tạo split hiện tại.
4. Sau đó mới preflight dung lượng/GPU/khả năng lưu-resume; chỉ khi preflight xong mới chốt seed, budget, checkpoint selection và xin phép chạy compute.
5. Giữ bốn generator family CelebA-HQ (`dalle2`, `if`, `midjourney`, `stargan`) làm holdout riêng; không tune trên test. `sdv2` test là seen-generator control.

## Protocol validation đã chốt

Người dùng chấp thuận salt `dire-celebahq-val-v1` và exact-count rank rule: trong từng class, sắp xếp theo `SHA256(salt + NUL + member_locator_inferred)` tăng dần, tie-break bằng locator; lấy `round-half-up(10% × N)` IDs đầu tiên. Với inventory counts hiện có (28.000 real, 40.000 SDv2), mục tiêu dự kiến là 2.800 và 4.000 IDs; đây là suy ra từ report inventory, chưa phải realized counts.

**Chưa tạo validation split.** Thực hiện sau integrity gate, vì locator hiện suy ra từ cây thư mục (`locator_inferred=true`) và chưa được xác minh với tar member gốc. Khi tạo split, phải ghi salt, protocol, hash file list và realized counts để tái lập.
