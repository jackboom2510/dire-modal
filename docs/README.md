# Chỉ mục tài liệu cho người và agent

> **DOC-000 · Điểm vào tài liệu.** Mục đích: giúp agent chọn đúng nguồn cần đọc cho một task, không nạp toàn bộ `docs/` vào ngữ cảnh. Đọc `AGENT.md` ở repo root trước; sau đó chỉ mở nhánh tài liệu phù hợp bên dưới.

## Quy tắc đọc nhanh

1. **Quy tắc repo / trạng thái công việc:** `AGENT.md` → `instructions.md` → `plan.md`.
2. **Task cụ thể:** tra bảng route, chỉ đọc tài liệu chính được chỉ định và các nguồn mà nó dẫn tới.
3. **Số liệu run:** đọc history ledger để biết phạm vi; kiểm artifact JSON mà ledger trỏ đến trước khi trích số. Ledger không thay thế output gốc.
4. **Skill:** chỉ nạp `SKILL.md` khi task khớp trigger; không đọc mọi skill từ đầu phiên.
5. **Mâu thuẫn:** code/config/output đã đo là nguồn thực thi; ADR/spec là quyết định/tiêu chí; plan/history mô tả trạng thái tại snapshot và có thể cũ.

## Nhóm tài liệu và mã số

Mã số giúp agent tham chiếu ổn định trong kế hoạch/ghi chú. **Giữ nguyên các đường dẫn legacy** dưới đây vì code, test, slide notes và tài liệu khác đang trỏ trực tiếp tới chúng. Tài liệu mới thuộc history dùng mã `HIST-*`; không tạo bản sao của số liệu trong JSON.

### 00 — Định hướng và hướng dẫn agent

| Mã | Tài liệu | Đọc khi |
|---|---|---|
| DOC-000 | `docs/README.md` (tài liệu này) | Luôn tra cứu để chọn tài liệu theo task; không cần đọc lại toàn bộ nội dung mỗi lượt. |
| DOC-010 | [`docs/agent-docs-and-skills.md`](agent-docs-and-skills.md) | Tạo/sửa docs, hướng dẫn agent hoặc skills; xem quy ước phổ biến và nguồn tham khảo. |
| AGENT-ROOT | [`AGENTS.md`](../AGENTS.md) | Điểm vào chuẩn cho agent; chuyển tiếp tới quy tắc repo và chỉ mục này. |
| AGENT-LEGACY | [`AGENT.md`](../AGENT.md), [`instructions.md`](../instructions.md), [`skills.md`](../skills.md) | Quy tắc repo, cách làm việc và các quy trình/task đã được dự án dùng. |
| AGENT-SKILL | [`.claude/skills/dire-academic-slides/SKILL.md`](../.claude/skills/dire-academic-slides/SKILL.md) | Chỉ khi tạo/chỉnh slide học thuật về DIRE. Skill này yêu cầu đọc `docs/slide-agent-brief.md`. |

### 10 — Quyết định kiến trúc (ADR)

Đọc ADR liên quan trước khi đổi quyết định đã chốt. ADR là hồ sơ quyết định, không phải runbook.

| Mã | Tài liệu | Chủ đề |
|---|---|---|
| ADR-0001 | [`adr-0001-storage.md`](adr-0001-storage.md) | Vai trò các lớp lưu trữ. |
| ADR-0002 | [`adr-0002-iaas.md`](adr-0002-iaas.md) | Chiến lược IaaS và ngưỡng rời Modal. |
| ADR-0003 | [`adr-0003-demo-training.md`](adr-0003-demo-training.md) | Huấn luyện cho demo. |
| ADR-0004 | [`adr-0004-retrain-for-reconciliation.md`](adr-0004-retrain-for-reconciliation.md) | Train lại chỉ để đối chiếu. |
| ADR-0005 | [`adr-0005-local-first-modal-wrapper.md`](adr-0005-local-first-modal-wrapper.md) | Lõi local-first và Modal wrapper. |
| ADR-0006 | [`adr-0006-reproduction-levels.md`](adr-0006-reproduction-levels.md) | Phân biệt R0/R1/R2/R3. |

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
| EXP-001 | [`full-paper-reproduction-plan.md`](full-paper-reproduction-plan.md) | Xác định phạm vi tái hiện toàn bài. |
| EXP-002 | [`full-paper-continuation-runbook.md`](full-paper-continuation-runbook.md) | Tiếp tục từ artefact/dữ liệu hiện có. |
| EXP-003 | [`incremental-paper-subsets.md`](incremental-paper-subsets.md) | Chọn thí nghiệm theo tập con/từng bước. |
| EXP-004 | [`table4-table5-figure4-runbook.md`](table4-table5-figure4-runbook.md) | Chạy hoặc đọc Table 4/5 và Figure 4. |
| EXP-005 | [`figures-plan.md`](figures-plan.md) | Tình trạng, nguồn và kế hoạch hình. |
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
| PLATFORM-002 | [`demo-hf-spaces-plan.md`](demo-hf-spaces-plan.md), [`demo-modal-plan.md`](demo-modal-plan.md), [`demo-web-plan.md`](demo-web-plan.md) | Làm demo/triển khai. Kiểm status trong code/config mới nhất vì plan có thể lịch sử. |
| PLATFORM-003 | [`manim-video-plan.md`](manim-video-plan.md) | Tạo video/animation giải thích. |

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
- Run history chỉ tóm tắt provenance, protocol, counts, artifact paths và caveat; metric vẫn đọc từ ledger JSON. Ghi rõ `unknown/not recorded`, không điền bằng suy đoán.
- Tài liệu dài phải có mục lục/route, phần “đọc khi nào”, đường dẫn code/artifact và ngày xác minh. Agent chỉ mở section cần cho task.
- Không chuyển/đổi tên hàng loạt đường dẫn cũ nếu chưa có kế hoạch cập nhật mọi liên kết, source citation, tests và người dùng ngoài repo. Mã tài liệu ở chỉ mục là lớp điều hướng ổn định; các đường dẫn thực thi hiện có vẫn được giữ nguyên.
