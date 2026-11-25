# Agent changelog

> Nhật ký riêng cho các thay đổi do coding agent thực hiện trong repo này. **Không phải** changelog phát hành sản phẩm, lịch sử Modal cloud, hay nơi thay thế Git history. Chỉ ghi thay đổi đã làm trong workspace; không suy ra hay ghi công việc của người dùng/người khác.

## Quy ước ghi

- Mỗi mục ghi ngày, branch/PR nếu đã có, mục tiêu, files, kiểm chứng và phần chưa làm.
- Chỉ ghi điều đã xảy ra; ghi rõ khi test không chạy hoặc có lỗi sẵn có.
- Không sao chép toàn bộ metrics/run history vào đây; trỏ tới ledger phù hợp.
- Dùng dạng mới nhất ở đầu file.

## 2026-10-14 — Chuẩn hóa tài liệu agent và lịch sử dataset runs

- **Branch:** `arena/3e9b6e2a-dire-modal` (đang active khi ghi).
- **Yêu cầu:** tạo tài liệu lịch sử run riêng dưới `docs/history/`, đánh mã tài liệu và cải thiện cách agent tra cứu; khảo sát các tài liệu/skills phổ biến.
- **Đã thay đổi:**
  - Thêm `docs/history/00-index.md`, `01-modal-dataset-runs.md`, `90-run-record-template.md`.
  - Thêm `docs/README.md` làm chỉ mục có mã DOC/ADR/SCI/EXP/OPS/HIST; giữ các path cũ để không làm gãy citations và liên kết đang dùng.
  - Thêm `docs/agent-docs-and-skills.md` với các tham khảo Agent Skills, Anthropic Skills, OpenAI Plugins, Superpowers và Awesome Copilot.
  - Thêm `AGENTS.md`; cập nhật route tài liệu trong `AGENT.md` và `README.md`.
  - Thêm `tests/test_docs_index.py` để kiểm local links, mã tài liệu và một số count trong history khớp ledger.
- **Kiểm chứng:** `tests/test_docs_index.py` — 3 passed; kiểm tra link/tài liệu cơ bản trong `tests/test_agent_docs.py` — 2 passed.
- **Chưa đạt / ngoài phạm vi:** toàn bộ `tests/test_agent_docs.py` có 3 lỗi liên quan đến `plan.md`/đường dẫn cũ, không phát sinh từ thay đổi này; không sửa các lỗi đó trong task này. Không truy vấn Modal live history. Không di chuyển/đổi tên toàn bộ tài liệu legacy.

## 2026-10-14 — Tách ADR và tạo workflow map

- **Branch:** `arena/cd850592-dire-modal`.
- Chuyển sáu quyết định kiến trúc vào `docs/adr/`, giữ nguyên mã `ADR-0001`–`ADR-0006`; cập nhật các tham chiếu hiện hành trong docs, code, tests, scripts, README và tài liệu trình bày.
- Thêm `docs/workflows/README.md` (WF-000) cùng ba group guide WF-001–WF-003 cho reproduction, operations/data, và demo/delivery; route tới tài liệu nguồn thay vì nhân bản hướng dẫn.
- Sau góp ý về plan, gom sáu `docs/*-plan.md` cùng `incremental-paper-subsets.md` (kế hoạch theo nội dung) vào `docs/plans/{research,demos,presentations}/`, thêm `docs/plans/README.md` (DOC-020). Giữ root `plan.md` tại chỗ vì đó là backlog/agent plan cấp repo, không phải plan chủ đề.
- Cập nhật `docs/README.md`, `AGENT.md`, `CLAUDE.md` và DOC-010 theo cấu trúc mới. Các runbook/plan/reference ngoài ADR giữ nguyên canonical path để không làm gãy consumer; gom theo mục đích trong workflow map.
- Thêm test xác nhận ADR nằm đúng thư mục và kiểm tra link trong workflow map/index.
- **Kiểm chứng sau khi gom plans:** `tests/test_docs_index.py`, `test_n1_reconciliation.py`, `test_figures_tool.py`, `test_zero_demo_api.py` — 17 passed, 2 skipped; audit link Markdown nội bộ không phát hiện link hỏng; `git diff --check` sạch.
