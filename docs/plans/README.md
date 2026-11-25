---
doc_id: DOC-020
title: Plans catalog
status: maintained
---

# Kế hoạch theo nhóm mục tiêu

Thư mục này tập trung các tài liệu **plan** dài hạn/trạng thái ở snapshot; đây không phải checklist thao tác, trạng thái run đã xác minh hay backlog cấp repo.

| Nhóm | Plan | Mục đích |
|---|---|---|
| Research | [`research/full-paper-reproduction-plan.md`](research/full-paper-reproduction-plan.md) | Phạm vi và tiêu chí cho chương trình tái hiện toàn bài. |
| Research | [`research/figures-plan.md`](research/figures-plan.md) | Registry figure, claims, dữ liệu và nguồn dựng hình. |
| Research | [`research/incremental-paper-subsets.md`](research/incremental-paper-subsets.md) | Kế hoạch subset, thứ tự stage và reuse cho Table 4/5/Figure 4. |
| Demo | [`demos/demo-web-plan.md`](demos/demo-web-plan.md) | Mục tiêu/guardrail sản phẩm cho demo web. |
| Demo | [`demos/demo-hf-spaces-plan.md`](demos/demo-hf-spaces-plan.md) | Kế hoạch triển khai giao diện trên HF Spaces. |
| Demo | [`demos/demo-modal-plan.md`](demos/demo-modal-plan.md) | Phương án backend GPU tùy chọn bằng Modal. |
| Presentation | [`presentations/manim-video-plan.md`](presentations/manim-video-plan.md) | Kịch bản nội dung video Manim; không phải báo cáo kết quả. |

## Chọn plan hay runbook?

- Dùng **plan** để hiểu mục tiêu, phạm vi, lựa chọn, trạng thái/việc dự kiến; trạng thái trong plan có thể cũ, phải đối chiếu source/config/artefact hiện hành.
- Dùng **runbook** để thực hiện quy trình từng bước và gates. Chọn runbook từ [workflow map](../workflows/README.md).
- Dùng **history/run record** cho bằng chứng về một lượt đã chạy; một kế hoạch không chứng minh rằng công việc đã được thực hiện.
- Root [`plan.md`](../../plan.md) là kế hoạch/backlog hiện hành cho agent ở cấp repo; **không** chuyển vào đây và không đồng nhất với các plan chủ đề trong `docs/plans/`.
