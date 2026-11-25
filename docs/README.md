# Chỉ mục tài liệu cho người và agent

> **DOC-000 · Điểm vào tài liệu.** Mục đích: giúp agent chọn đúng nguồn cần đọc cho một task, không nạp toàn bộ `docs/` vào ngữ cảnh. Đọc `AGENT.md` ở repo root trước; sau đó chỉ mở nhánh tài liệu phù hợp bên dưới.

## Quy tắc đọc nhanh

1. **Quy tắc repo / trạng thái công việc:** `AGENT.md` → `instructions.md` → `plan.md`.
2. **Task nhiều bước:** bắt đầu từ [workflow map](workflows/README.md), chọn một luồng và chỉ đọc tài liệu được route.
3. **Task cụ thể:** tra bảng route bên dưới; chỉ mở tài liệu chính và nguồn mà nó dẫn tới.
4. **Số liệu run:** đọc history ledger để biết phạm vi; kiểm artifact JSON mà ledger trỏ đến trước khi trích số. Ledger không thay thế output gốc.
5. **Skill:** chỉ nạp `SKILL.md` khi task khớp trigger; không đọc mọi skill từ đầu phiên.
6. **Mâu thuẫn:** code/config/output đã đo là nguồn thực thi; ADR/spec là quyết định/tiêu chí; plan/history mô tả trạng thái tại snapshot và có thể cũ.

## Nhóm tài liệu và mã số

Mã số giúp agent tham chiếu ổn định trong kế hoạch/ghi chú. Các ADR và plan chủ đề được gom vào thư mục riêng; tham chiếu hiện hành đã được cập nhật. Những đường dẫn legacy còn lại được giữ ổn định vì code, tests, slide notes và tài liệu khác đang trỏ trực tiếp tới chúng. History dùng mã `HIST-*`; không tạo bản sao của số liệu trong JSON.

### 00 — Định hướng và hướng dẫn agent

| Mã | Tài liệu | Đọc khi |
|---|---|---|
| DOC-000 | `docs/README.md` (tài liệu này) | Luôn tra cứu để chọn tài liệu theo task; không cần đọc lại toàn bộ nội dung mỗi lượt. |
| DOC-010 | [`docs/agent-docs-and-skills.md`](agent-docs-and-skills.md) | Tạo/sửa docs, hướng dẫn agent hoặc skills; xem quy ước phổ biến và nguồn tham khảo. |
| DOC-020 | [`docs/plans/README.md`](plans/README.md) | Chỉ mục plan theo nhóm research, demo, presentation; phân biệt plan với runbook/history. |
| WF-000 | [`docs/workflows/README.md`](workflows/README.md) | Chọn group workflow theo mục tiêu. |
| WF-001 | [`docs/workflows/reproduction.md`](workflows/reproduction.md) | Research, chọn reproduction level/protocol, rồi ghi run record. |
| WF-002 | [`docs/workflows/operations.md`](workflows/operations.md) | Chọn môi trường, chuẩn bị dữ liệu, chạy và xác minh. |
| WF-003 | [`docs/workflows/delivery.md`](workflows/delivery.md) | Demo, figures/slides/video và cập nhật docs/skills. |
| AGENT-ROOT | [`AGENTS.md`](../AGENTS.md) | Điểm vào chuẩn cho agent; chuyển tiếp tới quy tắc repo và chỉ mục này. |
| AGENT-LEGACY | [`AGENT.md`](../AGENT.md), [`instructions.md`](../instructions.md), [`skills.md`](../skills.md) | Quy tắc repo, cách làm việc và các quy trình/task đã được dự án dùng. |
| AGENT-SKILL | [`.claude/skills/dire-academic-slides/SKILL.md`](../.claude/skills/dire-academic-slides/SKILL.md) | Chỉ khi tạo/chỉnh slide học thuật về DIRE. Skill này yêu cầu đọc `docs/slide-agent-brief.md`. |

<a id="adr-index"></a>

### 10 — Quyết định kiến trúc (ADR)

Đọc ADR liên quan trước khi đổi quyết định đã chốt. ADR là hồ sơ quyết định, không phải runbook.

| Mã | Tài liệu | Chủ đề |
|---|---|---|
| ADR-0001 | [`adr/adr-0001-storage.md`](adr/adr-0001-storage.md) | Vai trò các lớp lưu trữ. |
| ADR-0002 | [`adr/adr-0002-iaas.md`](adr/adr-0002-iaas.md) | Chiến lược IaaS và ngưỡng rời Modal. |
| ADR-0003 | [`adr/adr-0003-demo-training.md`](adr/adr-0003-demo-training.md) | Huấn luyện cho demo. |
| ADR-0004 | [`adr/adr-0004-retrain-for-reconciliation.md`](adr/adr-0004-retrain-for-reconciliation.md) | Train lại chỉ để đối chiếu. |
| ADR-0005 | [`adr/adr-0005-local-first-modal-wrapper.md`](adr/adr-0005-local-first-modal-wrapper.md) | Lõi local-first và Modal wrapper. |
| ADR-0006 | [`adr/adr-0006-reproduction-levels.md`](adr/adr-0006-reproduction-levels.md) | Phân biệt R0/R1/R2/R3. |

### 20 — Nền tảng nghiên cứu và đọc paper

| Mã | Tài liệu | Đọc khi |
|---|---|---|
| SCI-001 | [`book/00-mo-dau.md`](book/00-mo-dau.md) | Cần lộ trình đọc tổng thể. |
| SCI-002 | [`book/00b-cong-cu-toan.md`](book/00b-cong-cu-toan.md) | Cần ký hiệu/toán nền. |
| SCI-003 | [`book/01-nen-tang.md`](book/01-nen-tang.md) | CNN, miền tần số, JPEG và nền tảng pháp y ảnh. |
| SCI-004 | [`book/02-khuech-tan.md`](book/02-khuech-tan.md) | Diffusion và DDIM. |
| SCI-005 | [`book/03-dire.md`](book/03-dire.md) | Định nghĩa, lựa chọn thiết kế và thí nghiệm DIRE. |
| SCI-006 | [`book/04-code.md`](book/04-code.md), [`book/04b-module-map.md`](book/04b-module-map.md) | Đối chiếu lý thuyết với code và bản đồ module. |
| SCI-007 | [`book/05-phu-luc.md`](book/05-phu-luc.md) | Tra cứu thuật ngữ, nguồn và phần phụ. |
| SCI-008 | [`paper-tung-trang.html`](paper-tung-trang.html), [`paper-cac-mo-hinh-sinh.html`](paper-cac-mo-hinh-sinh.html), [`paper-code-walkthrough.md`](paper-code-walkthrough.md), [`dire-tu-so-0.pdf`](dire-tu-so-0.pdf), [`dire-from-scratch.md`](dire-from-scratch.md) | Bài đọc sâu; mở đúng file/section liên quan, không nạp toàn bộ các tài liệu dài nếu chưa cần. |

### 30 — Thực nghiệm và kế hoạch nghiên cứu

| Mã | Tài liệu | Đọc khi |
|---|---|---|
| EXP-001 | [`plans/research/full-paper-reproduction-plan.md`](plans/research/full-paper-reproduction-plan.md) | Xác định phạm vi tái hiện toàn bài. |
| EXP-002 | [`full-paper-continuation-runbook.md`](full-paper-continuation-runbook.md) | Tiếp tục từ artefact/dữ liệu hiện có. |
| EXP-003 | [`plans/research/incremental-paper-subsets.md`](plans/research/incremental-paper-subsets.md) | Chọn thí nghiệm theo tập con/từng bước. |
| EXP-004 | [`table4-table5-figure4-runbook.md`](table4-table5-figure4-runbook.md) | Chạy hoặc đọc Table 4/5 và Figure 4. |
| EXP-005 | [`plans/research/figures-plan.md`](plans/research/figures-plan.md) | Tình trạng, nguồn và kế hoạch hình. |
| EXP-006 | [`scratch-reproduction.md`](scratch-reproduction.md), [`from-zero-clean-room.md`](from-zero-clean-room.md) | Tái hiện từ đầu hoặc kiểm chứng clean-room. |
| EXP-007 | [`slide-agent-brief.md`](slide-agent-brief.md) | Tạo deck; đây là brief bằng chứng/claim, không phải run ledger. |

### 40 — Chạy, dữ liệu và vận hành

| Mã | Tài liệu | Đọc khi |
|---|---|---|
| OPS-001 | [`runbook-free.md`](runbook-free.md) | Quy trình cloud/free-tier chính. |
| OPS-002 | [`runbook-local-manual.md`](runbook-local-manual.md) | Quy trình chạy local. |
| OPS-003 | [`download-manual.md`](download-manual.md) | Tải dữ liệu bằng tay. |
| OPS-004 | [`dataset-own.md`](dataset-own.md) | Dùng dataset riêng. |
| OPS-005 | [`history/00-index.md`](history/00-index.md), [`history/01-modal-dataset-runs.md`](history/01-modal-dataset-runs.md) | Kiểm kích thước/phạm vi các lần chạy đã được lưu. Đây là snapshot trong repo, không phải API/lịch sử live của Modal. |

### 50 — Nền tảng, demo và phân phối

| Mã | Tài liệu | Đọc khi |
|---|---|---|
| PLATFORM-001 | [`iaas-options.md`](iaas-options.md), [`storage-comparison.md`](storage-comparison.md) | Chọn IaaS hoặc lớp lưu trữ. |
| PLATFORM-002 | [`plans/demos/demo-hf-spaces-plan.md`](plans/demos/demo-hf-spaces-plan.md), [`plans/demos/demo-modal-plan.md`](plans/demos/demo-modal-plan.md), [`plans/demos/demo-web-plan.md`](plans/demos/demo-web-plan.md) | Làm demo/triển khai. Kiểm status trong code/config mới nhất vì plan có thể lịch sử. |
| PLATFORM-003 | [`plans/presentations/manim-video-plan.md`](plans/presentations/manim-video-plan.md) | Tạo video/animation giải thích. |

### 60 — Hình minh họa và asset

[`slide-assets/`](slide-assets/) chứa SVG nguồn, các asset đã sinh và manifest. Xem brief/skill slide trước khi sửa; không dùng asset schematic như kết quả đo.

### 80 — Lịch sử và mẫu ghi run

| Mã | Tài liệu | Dùng khi |
|---|---|---|
| HIST-000 | [`history/00-index.md`](history/00-index.md) | Quy ước và thứ tự tra run. |
| HIST-001 | [`history/01-modal-dataset-runs.md`](history/01-modal-dataset-runs.md) | Bằng chứng repo hiện có về inventory, số map, split và test scope. |
| HIST-090 | [`history/90-run-record-template.md`](history/90-run-record-template.md) | Tạo một entry mới sau khi run hoàn tất/được xác minh. |

## Chuẩn ghi tài liệu mới

- Trước tiên tìm trong chỉ mục; sửa **nguồn sự thật hiện có** thay vì tạo bản sao.
- File mới có mã số trong frontmatter/heading và nằm trong nhóm đúng; tên file ngắn, ổn định. Không đánh số lại ADR hoặc thư mục `book` đã có quy ước riêng.
- ADR canonical đặt tại `docs/adr/`; plan chủ đề đặt trong `docs/plans/<group>/`; chỉ mục/mã tài liệu giữ ổn định. Root `plan.md` là backlog cấp repo, không thuộc nhóm plan chủ đề.
- Run history chỉ tóm tắt provenance, protocol, counts, artifact paths và caveat; metric vẫn đọc từ ledger JSON. Ghi rõ `unknown/not recorded`, không điền bằng suy đoán.
- Tài liệu dài phải có mục lục/route, phần “đọc khi nào”, đường dẫn code/artifact và ngày xác minh. Agent chỉ mở section cần cho task.
- Các runbook/reference khác giữ path canonical vì còn được code, tests, notes và citations dùng trực tiếp; dùng `docs/workflows/README.md` để route. Chỉ chuyển file khi có thể cập nhật và kiểm tra toàn bộ consumer trong cùng thay đổi.
