---
doc_id: DOC-010
title: Hệ tài liệu và skills cho coding agent
status: reference
---

# Hệ tài liệu và skills cho coding agent

> **DOC-010.** Tài liệu khảo sát để định hướng cách tổ chức repo DIRE, không phải tiêu chuẩn bắt buộc của mọi harness. Phân biệt chuẩn mở, định dạng của vendor và lựa chọn riêng của từng dự án.

## Tóm tắt cho repo này

- File chỉ dẫn ở **repo root** trả lời: “agent luôn phải biết gì về repo này?” Giữ ngắn, làm điểm vào, trỏ tới nguồn chi tiết.
- **Skill** trả lời: “khi gặp task loại X, quy trình chuyên biệt nào cần dùng?” Skill nên kích hoạt theo mô tả rõ, chứa workflow và trỏ tới reference/scripts/assets theo nhu cầu.
- **Run history** trả lời: “lần chạy cụ thể đã làm gì, trên tập nào, sinh artefact nào?” Đây là record có provenance, không phải skill hay hướng dẫn thao tác chung.
- **ADR/spec/runbook/plan** có vai trò riêng: quyết định, yêu cầu có tiêu chí, cách chạy có thể làm theo, và trạng thái/hướng đi. Không gộp chúng thành một “agent prompt” khổng lồ.
- Chỉ mục có mã DOC/ADR/SCI/EXP/OPS/HIST để agent định vị nhanh. Các path cũ được giữ ổn định vì nhiều chỗ trong repo trỏ tới chúng.

## Các loại tài liệu thường gặp

| Loại | Trả lời câu hỏi | Nơi đặt phổ biến | Quy tắc dùng |
|---|---|---|---|
| `AGENTS.md` | Quy tắc làm việc cho coding agent trên repo/cây thư mục | Root hoặc thư mục gần code | Điểm vào ngắn, chỉ dẫn agent tới nguồn cần thiết; quy tắc ở thư mục gần có thể chuyên biệt cho vùng đó. |
| `CLAUDE.md`, `GEMINI.md`, `copilot-instructions.md` | Adapter/hướng dẫn riêng một harness | Root hoặc thư mục config của harness | Tránh sao chép toàn bộ nội dung giữa vendor files; nếu cần nhiều harness, trỏ về nguồn dùng chung. Kiểm tra cơ chế kế thừa/áp dụng của từng client. |
| `README.md` | Người mới bắt đầu: repo làm gì, cách chạy và link tới chi tiết | Root và từng subsystem | README không thay thế runbook hay API reference. |
| `CONTRIBUTING.md` | Cách đóng góp, test, PR, review | Root | Dành cho người đóng góp và agent chuẩn bị thay đổi upstream. |
| `SKILL.md` | Workflow chuyên biệt được nạp theo task | Thư mục riêng cho từng skill | Trigger qua metadata; hướng dẫn quy trình; reference/script/assets chỉ đọc hoặc chạy khi cần. |
| ADR | Vì sao một quyết định được chọn và trạng thái quyết định | `docs/adr-*` hoặc `docs/decisions/` | Quyết định đã chốt; không dùng làm hướng dẫn chạy từng bước. |
| Spec / plan / runbook | Yêu cầu, tiến độ, thao tác lặp lại | `spec`, `plan`, `docs/` | Có tiêu chí hoàn tất; phân biệt trạng thái mong muốn với trạng thái đã đo. |
| History / experiment record | Một run thật đã thực hiện và bằng chứng của nó | `docs/history/` + machine-readable ledger | Không sửa số gốc bằng tay; ghi counts, protocol, commit/artifact và caveat. |

`AGENTS.md` và `SKILL.md` **không thay thế nhau**: file đầu là hướng dẫn nền theo repo/path; skill là năng lực/workflow có phạm vi hẹp, cần discover và nạp theo trigger.

## Các repo skills đáng tham khảo

### 1. Chuẩn mở Agent Skills

[agentskills/agentskills](https://github.com/agentskills/agentskills) mô tả format mở: một thư mục skill có `SKILL.md` với metadata `name`/`description`, và có thể kèm `scripts/`, `references/`, `assets/`. Mô hình nạp theo lớp là **discovery → activation → execution**: ban đầu chỉ cần metadata; khi task khớp mới đọc hướng dẫn; tài nguyên bổ sung chỉ lấy khi workflow cần.

**Bài học áp dụng:** skill nhỏ, có trigger và payload tách lớp; không đưa toàn bộ tài liệu repo vào mọi phiên agent.

### 2. Anthropic — `anthropics/skills`

[Repo chính thức](https://github.com/anthropics/skills) có nhiều ví dụ độc lập theo thư mục, gồm `SKILL.md` và một số skill lớn kèm scripts/references/assets. Đáng đọc:

- [`skills/skill-creator/SKILL.md`](https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md): bắt đầu từ ý định/trigger, phác thảo, tạo prompt đánh giá rồi lặp; nhấn mạnh progressive disclosure và file tham chiếu theo biến thể.
- [`skills/docx/SKILL.md`](https://github.com/anthropics/skills/blob/main/skills/docx/SKILL.md), cùng `pdf/`, `pptx/`, `xlsx/`: ví dụ skill thao tác tài liệu có hướng dẫn cụ thể, scripts và bước kiểm đầu ra.
- [`template/SKILL.md`](https://github.com/anthropics/skills/blob/main/template/SKILL.md): khung tối thiểu để bắt đầu skill.

**Bài học áp dụng:** `description` phải cho biết skill làm gì và khi nào bật; workflow chính nằm trong `SKILL.md`, kiến thức lớn hoặc chuyên biệt chuyển sang `references/`; file sinh ra phải có bước QA. Không cần sao chép toàn bộ style của một vendor.

### 3. OpenAI/Codex — lịch sử `openai/skills` và repo plugins hiện hành

[`openai/skills`](https://github.com/openai/skills) hiện tự đánh dấu là **deprecated** và chuyển người dùng sang hướng dẫn plugins. Không nên lấy repo đó làm đích phân phối Codex mới.

Repo [openai/plugins](https://github.com/openai/plugins) thể hiện mẫu plugin hiện tại: mỗi plugin có `.codex-plugin/plugin.json` và có thể thêm `skills/`, `agents/`, commands/hooks hoặc tài nguyên khác. Skill chỉ là một phần tùy chọn trong bundle; một skill mẫu nằm ở [`plugins/build-web-apps/skills/supabase-best-practices/SKILL.md`](https://github.com/openai/plugins/blob/main/plugins/build-web-apps/skills/supabase-best-practices/SKILL.md).

**Bài học áp dụng:** tách metadata phân phối plugin khỏi nội dung skill; không nhầm cơ chế cài/marketplace với format hướng dẫn bên trong skill.

### 4. `obra/superpowers`

[obra/superpowers](https://github.com/obra/superpowers) kết hợp `AGENTS.md` (quy tắc contributor) với skill bootstrap và các workflow như brainstorming, systematic debugging, TDD, verification. [`writing-skills/SKILL.md`](https://github.com/obra/superpowers/blob/main/skills/writing-skills/SKILL.md) mô tả viết skill như một vòng lặp test hành vi (scenario → quan sát baseline → viết → kiểm lại), khác với cách chỉ biên tập prose.

**Bài học áp dụng:** tài liệu thay đổi hành vi nên có test/eval phù hợp. Không sao chép các quy tắc “mọi task đều phải gọi skill” hay chính sách đóng góp của repo này sang DIRE; chúng là lựa chọn riêng của Superpowers.

### 5. GitHub — `github/awesome-copilot`

[github/awesome-copilot](https://github.com/github/awesome-copilot) tổ chức riêng **agents**, **instructions**, **skills**, **plugins** và cookbook; repo có các ví dụ `*.agent.md`, file instruction áp dụng theo phạm vi và skills dạng thư mục theo Agent Skills format.

**Bài học áp dụng:** một collection lớn cần catalog/filter theo resource type và trigger, không chỉ một danh sách dài. Với repo nhiều module, giữ instruction theo vùng nhỏ và rõ phạm vi; kiểm tra nội dung cộng đồng trước khi cài vì nguồn không đồng nhất.

## Quy chuẩn dùng cho repo DIRE

1. Root `AGENTS.md` là router chuẩn, giữ ngắn; `AGENT.md` hiện tại vẫn là nguồn project-specific chính cho tới khi nội dung được hợp nhất có chủ đích.
2. Đầu phiên agent chỉ đọc điểm vào + quy tắc chung; task-specific route được liệt kê trong [`docs/README.md`](README.md).
3. Tạo skill chỉ khi workflow lặp lại, có trigger phân biệt được và có giá trị dùng lại; một lần giải quyết run không tự động thành skill.
4. Skill mới dùng cấu trúc tối thiểu:

   ```text
   .agents/skills/<skill-name>/
   ├── SKILL.md
   ├── references/     # nội dung dài/ít khi cần, nếu có
   ├── scripts/        # thao tác tất định/lặp lại, nếu có
   └── assets/         # template/fixture, nếu có
   ```

   Hiện repo có `.claude/skills/dire-academic-slides/`; không chuyển runtime skill sang vị trí khác nếu chưa xác nhận client đang dùng discovery path nào.
5. Frontmatter `name` ngắn, ổn định; `description` nêu **khi nào dùng**, không chỉ tóm tắt nội dung. Trong body có: phạm vi, điều kiện trước, workflow, quyết định/stop conditions, kiểm chứng, link tới reference.
6. Đừng biến history thành skill: template là chuẩn định dạng; entry history là snapshot bằng chứng; JSON ledger vẫn là nguồn số.
7. Khi thay đổi skill, kiểm trigger + workflow bằng test/eval phù hợp và báo rõ phần nào được chạy. Không tuyên bố đã đánh giá hành vi agent nếu chỉ review văn bản.

## Nguồn đã khảo sát

- [Agent Skills specification repository](https://github.com/agentskills/agentskills)
- [Anthropic skills examples](https://github.com/anthropics/skills)
- [OpenAI skills archive (README ghi deprecated)](https://github.com/openai/skills)
- [OpenAI plugins](https://github.com/openai/plugins)
- [obra/superpowers](https://github.com/obra/superpowers)
- [GitHub Awesome Copilot](https://github.com/github/awesome-copilot)
