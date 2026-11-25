---
doc_id: WF-001
title: Research and reproduction workflow
status: maintained
---

# Workflow: nghiên cứu và reproduction

## 1. Chốt câu hỏi trước khi chạy

- Thay đổi hành vi/tiêu chí: đọc [`spec.md`](../../spec.md), test liên quan và quyết định trong [`docs/adr/`](../adr/). Muốn đổi quyết định đã chốt thì cập nhật ADR; không suy ra quyết định mới từ một runbook cũ.
- Xác định mục tiêu reproduction theo [`ADR-0006`](../adr/adr-0006-reproduction-levels.md): R0 (chấm lại) không đồng nghĩa với tái lập; phân biệt nguồn ảnh/map/checkpoint.
- Đọc [kế hoạch tái hiện toàn bài](../plans/research/full-paper-reproduction-plan.md) để biết scope và phần còn thiếu; nếu tiếp tục từ artefact hiện có, mở [continuation runbook](../full-paper-continuation-runbook.md).

## 2. Chọn đúng protocol

- Table 4/5/Figure 4: [runbook chuyên biệt](../table4-table5-figure4-runbook.md); không nhập chung các protocol/model/checkpoint khác nhau.
- Thí nghiệm theo subset: [incremental subset plan](../plans/research/incremental-paper-subsets.md).
- Tái hiện từ đầu: [scratch reproduction](../scratch-reproduction.md).
- Tự triển khai không dùng code tác giả: [clean-room guide](../from-zero-clean-room.md).
- Figure/claim cần dẫn nguồn: [figure plan](../plans/research/figures-plan.md).

Plan nói về mục tiêu/trạng thái tại thời điểm cập nhật; runbook mô tả cách thực hiện; code/config và output mới nhất là bằng chứng thực thi. Không chuyển trạng thái “planned” thành “completed” nếu chưa xác minh.

## 3. Sau run

Ghi provenance, protocol, split, seed, checkpoint, counts theo stage và artifact bằng [`run record template`](../history/90-run-record-template.md); route theo [`history index`](../history/00-index.md). Chỉ dùng metric đã kiểm tra từ ledger/output gốc. Không trộn số giữa các run; ghi `not recorded` hoặc `not verified` khi thiếu dữ liệu.
