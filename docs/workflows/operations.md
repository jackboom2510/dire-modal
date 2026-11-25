---
doc_id: WF-002
title: Execution and data operations workflow
status: maintained
---

# Workflow: thực thi, môi trường và dữ liệu

## 1. Chọn môi trường trước

- Modal/free tier: bắt đầu [`runbook-free.md`](../runbook-free.md); kiểm giới hạn, chi phí/preflight và các gates trước khi job có thể dùng compute.
- Local/manual: dùng [`runbook-local-manual.md`](../runbook-local-manual.md).
- Tải file thủ công: [`download-manual.md`](../download-manual.md).
- Dataset riêng: [`dataset-own.md`](../dataset-own.md).
- So sánh nền tảng/lưu trữ để đưa ra quyết định: [`iaas-options.md`](../iaas-options.md), [`storage-comparison.md`](../storage-comparison.md) và ADR tương ứng; các phân tích này không thay thế hướng dẫn chạy.

## 2. Thực hiện có kiểm soát

Trước khi bắt đầu, xác nhận nguồn dữ liệu/checkpoint, đường dẫn, split và loại run; kiểm tra volume/quota/gates theo runbook đã chọn. Không chuyển một thao tác local thành Modal job (hoặc ngược lại) nếu chưa xác nhận khác biệt môi trường. Không sửa artifact cũ để “làm khớp” một run mới.

## 3. Xác minh và lưu lại

Sau khi hoàn tất, kiểm tra output/ledger và ghi entry theo [`run record template`](../history/90-run-record-template.md). Snapshot ở [`history`](../history/00-index.md) chỉ ghi những gì repo có bằng chứng; Modal cloud live state phải được kiểm riêng nếu cần.
