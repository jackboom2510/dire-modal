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
