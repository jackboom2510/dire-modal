---
doc_id: HIST-000
title: Chỉ mục lịch sử run
status: maintained
---

# Lịch sử chạy — chỉ mục và chuẩn đọc

Đây là **nhật ký provenance trong repo**, không phải lịch sử live từ Modal. Nó chỉ bao quát các run còn artifact/ledger được commit hoặc xuất vào checkout.

## Các tài liệu

| Mã | Tài liệu | Mục đích |
|---|---|---|
| HIST-001 | [`01-modal-dataset-runs.md`](01-modal-dataset-runs.md) | Snapshot các kích thước, split và phạm vi dataset ghi lại cho những nhóm run hiện có. |
| HIST-090 | [`90-run-record-template.md`](90-run-record-template.md) | Mẫu chuẩn cho một run hoặc campaign mới. |

## Nguyên tắc không nhập nhằng

1. **Inventory nguồn ≠ ảnh được xử lý ≠ ví dụ train/eval.** Ghi các đại lượng ở từng tầng riêng.
2. **Unique ≠ cumulative.** `n_distinct_maps` là số map duy nhất; `n_new`/`n_reused` có thể cộng dồn qua nhiều lượt, không tự động bằng số mẫu phân biệt.
3. **Run ID ≠ thời điểm đáng tin.** Chỉ dùng timestamp sau khi đối chiếu clock/manifest; không suy thứ tự chỉ từ tên thư mục.
4. **Một lần chạy ≠ protocol chung.** Init, split mode, generator, ảnh/nhãn, threshold, seed và phiên bản model đều phải khớp trước khi so metric.
5. **Không điền chỗ trống bằng phỏng đoán.** Ghi `not recorded`/`not verified`, rồi trỏ tới artifact gốc.
6. **Không sửa JSON metric để tạo báo cáo đẹp.** Ghi sai lệch/cảnh báo vào history; sửa output chỉ bằng quy trình đánh giá lại có thể truy nguyên.

## Khi thêm entry

- Dùng mã mới `HIST-###` và mẫu [`90-run-record-template.md`](90-run-record-template.md).
- Mỗi entry tóm tắt scope, protocol, counts, artifact path, hạn chế và hash/commit nếu có; không sao chép toàn bộ metrics.
- Ghi rõ `observed` (đếm được), `declared` (metadata), `derived` (tính từ output) và `unknown`.
- Cập nhật link ở bảng trên và route trong [`docs/README.md`](../README.md).
- Khi run gồm nhiều detector/campaign, tạo một index campaign + các entry con nếu protocol/scope khác; không ghép số từ các lượt khác nhau.
