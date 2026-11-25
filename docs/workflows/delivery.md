---
doc_id: WF-003
title: Demo, presentation, and documentation workflow
status: maintained
---

# Workflow: demo, figure, presentation và bảo trì docs

## Demo / deploy

Đọc [web demo plan](../plans/demos/demo-web-plan.md) để hiểu các mode/guardrail; tiếp đó chọn [HF Spaces plan](../plans/demos/demo-hf-spaces-plan.md) và chỉ dùng [Modal backend plan](../plans/demos/demo-modal-plan.md) nếu cần GPU. Đối chiếu các plan với code/config/deployment hiện hành trước khi triển khai vì trạng thái trong plan có thể cũ. Khi thay đổi hành vi, kiểm tra test và cập nhật hướng dẫn người dùng/runbook tương ứng.

## Figures / slide / video

- Registry figure và claims: [`figures-plan.md`](../plans/research/figures-plan.md).
- Slide học thuật: đọc [brief evidence/claim](../slide-agent-brief.md) và skill [dire-academic-slides](../../.claude/skills/dire-academic-slides/SKILL.md).
- Animation: [Manim video plan](../plans/presentations/manim-video-plan.md).

Giữ nguồn và provenance cho mọi claim; asset schematic không phải kết quả đo. Nêu rõ planned, implemented, executed và measured là trạng thái khác nhau.

## Thay đổi agent docs / skills

Đọc [DOC-010](../agent-docs-and-skills.md), chỉnh entrypoint ngắn và route tới nguồn chuẩn. Nạp/chỉnh một `SKILL.md` chỉ khi trigger của task phù hợp. Ghi thay đổi agent trong [`changelog.md`](../../changelog.md), không dùng changelog thay cho run record hoặc release notes.
