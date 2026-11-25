---
doc_id: WF-000
title: Workflow map
status: maintained
---

# Workflow map — chọn nhóm tài liệu theo mục tiêu

Mục đích là route theo **việc cần hoàn thành**, không nhân bản lệnh hoặc trạng thái ở các runbook/plan. Bắt đầu từ `AGENTS.md` → `AGENT.md` → [chỉ mục tài liệu](../README.md), rồi chỉ đọc group liên quan:

| Group | Mã | Chọn khi | Luồng |
|---|---|---|---|
| [Research & reproduction](reproduction.md) | WF-001 | Đổi tiêu chí/kiến trúc, tái hiện paper, chạy subset, hoặc làm clean-room | ADR/spec → chọn mức reproduction → plan/runbook → record run |
| [Execution & data operations](operations.md) | WF-002 | Chạy job, chuẩn bị/tải dữ liệu, chọn local hay Modal/free tier | Chọn môi trường/nguồn dữ liệu → preflight/gates → thực hiện → xác minh |
| [Demo & delivery](delivery.md) | WF-003 | Thay đổi/deploy demo, dựng figures/slides/video, cập nhật docs sau thay đổi | Kiểm bằng chứng → chọn artifact/workflow → kiểm tra → cập nhật nguồn chuẩn |

## Vai trò tài liệu — không tráo đổi

| Loại | Trả lời câu hỏi | Nguồn |
|---|---|---|
| Entry point / index | Tôi nên đọc gì cho task này? | `AGENTS.md`, `AGENT.md`, [`docs/README.md`](../README.md), trang này |
| Đặc tả | Hệ thống cần làm gì, tiêu chí hoàn tất là gì? | [`spec.md`](../../spec.md), tests liên quan |
| ADR | Vì sao quyết định kiến trúc được chọn, trạng thái quyết định ra sao? | [`docs/adr/`](../adr/) |
| Plan | Muốn làm gì / tình hình tại thời điểm ghi? | `*-plan.md`; kiểm tra code/config vì plan có thể lỗi thời |
| Runbook / workflow | Thực hiện task qua bước/gates nào? | `*-runbook.md`, `runbook-*.md`, hướng dẫn dữ liệu |
| Reference / tutorial | Khái niệm hoặc giải thích chi tiết ở đâu? | [`docs/book/`](../book/), [`dire-from-scratch.md`](../dire-from-scratch.md), [`paper-code-walkthrough.md`](../paper-code-walkthrough.md) |
| Skill | Khi task nào cần quy trình chuyên biệt có thể tái sử dụng? | `.claude/skills/*/SKILL.md` |
| Run history | Lần chạy cụ thể nào đã được xác minh, với protocol/artifact nào? | [`docs/history/`](../history/) và ledger/artifact gốc |
| Agent changelog | Coding agent đã thay đổi gì trong workspace? | [`changelog.md`](../../changelog.md); không phải release notes hay run history |

## Vì sao chỉ ADR được chuyển thư mục ở lượt này

ADR và plan chủ đề có vòng đời/ý nghĩa khác nhau nên được tập trung riêng ở `docs/adr/` và `docs/plans/`. Runbook/reference còn lại được source, tests, slide/deck và notes dùng trực tiếp; các group bên trên gom chúng theo workflow và giữ canonical path hiện có để tránh redirect giả hoặc hỏng anchor. Khi chuyển tài liệu sau này, cập nhật Markdown links, code/tests, citation/notes và kiểm tra toàn repo trong cùng thay đổi.
