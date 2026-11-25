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

## 2026-10-14 — Ghi nhận khung nghiên cứu Kaggle

- **Branch:** `arena/8dfed113-dire-modal`.
- Từ câu trả lời đồng ý Q1–Q4 của người dùng, thêm ADR-0007 ghi hai track, tách LSUN/ImageNet, generator mới là holdout và Kaggle là môi trường thực thi mục tiêu với Modal làm fallback.
- Thêm root `GLOSSARY.md` cho các thuật ngữ đã chốt; route glossary/ADR trong `AGENTS.md`, `AGENT.md` và `docs/README.md`.
- Protocol chi tiết (split/sample IDs, generator list/family, residual abs/signed, ViT/fusion, seed/metrics, Kaggle quota/persistence) vẫn chưa chốt; không khởi chạy compute.
- **Kiểm chứng:** `tests/test_docs_index.py` — 5 passed; audit link Markdown nội bộ không thấy link hỏng; `git diff --check` sạch.
- Tiếp tục grill: người dùng xác nhận dùng `real+ADM` làm train chính, holdout cả họ generator, chạy Kaggle nhiều phiên; yêu cầu khảo sát DiT thay U-Net trong research-extension.
- Bổ sung glossary cho seen-generator, DiT và ranh giới DiT-vs-ViT; ADR ghi rõ DiT chỉ là ứng viên, không thay ADM/U-Net trong track paper.
- Khảo sát source-level official `facebookresearch/DiT`: latent, class-conditional ImageNet, VAE decode; repo archived và license CC-BY-NC. Không chạy code hoặc tải checkpoint; feasibility/conditioning/latency vẫn là quyết định mở.
- Theo lựa chọn “kiểm kê rồi đề xuất”, rà manifest: ImageNet train hiện `real+ADM`, `sdv1` chỉ được khai báo ở val/test; LSUN có map train iDDPM/PNDM/StyleGAN (tier `trainens`, tổng 14.60 GB) và val tương ứng. Đây là khai báo manifest, chưa xác minh dữ liệu trên Kaggle/cardinality; không chuyển sample val/test sang train.
- Người dùng chọn CelebA-HQ-SD-v2 thay vì sinh ảnh ImageNet độc lập; xác nhận tái dùng DIRE maps `real+sdv2` có sẵn và train detector mới (không dùng checkpoint `celebahq_sdv2.pth` thay thế). Đồng ý tách validation subset xác định theo sample ID từ train maps; tỷ lệ còn chờ inventory.
- Người dùng chốt holdout các họ test `dalle2`, `if`/Imagen, `midjourney`, `stargan`, báo cáo riêng từng họ; `sdv2` trên test IDs mới là seen-generator control riêng. Detector mới dùng ResNet-50 ImageNet-init; ViT/fusion để so sánh kiến trúc riêng.
- Metric chính được chốt: macro-AP trên bốn họ holdout; báo cáo AP/AUROC per-generator, class counts và CI, ACC@0.5 phụ; không tune threshold trên test. Người dùng muốn chốt số seed sau preflight; CI method, training budget và selection policy còn mở. Đây là track CelebA-HQ tách biệt; người dùng cho biết archive local có train maps `real+sdv2` và các test maps cho real, bốn họ holdout, SDv2 control; checkout không truy cập được nên cần xác minh file list/count/hash/provenance rồi tạo Kaggle Dataset snapshot. Counts/IDs, family aliases, checkpoint hash và khả năng mount/resume còn cần xác minh. Chưa xếp compute hoặc ước tính chi phí. Người dùng đồng ý đóng gói input thành Kaggle Dataset versioned riêng, mount read-only; artifacts/checkpoint/ledger được xuất riêng để resume qua phiên.
- Người dùng xác nhận các maps local là tar riêng theo split/generator; gồm train `real+sdv2`, test `real`, bốn họ holdout và SDv2 control. Không có archive nào được truy cập từ checkout.
- Thêm `tools/kaggle_maps_inventory.py` (read-only, không giải nén/upload) để ghi archive SHA-256, counts/member names, candidate IDs/overlaps; `--hash-members` kiểm bytes thành viên tùy chọn. Thêm 6 test mới; **6 passed**, cùng `tests/test_docs_index.py` **5 passed** (tổng **11 passed**). Pyflakes, py_compile và `git diff --check` sạch.
- Test toàn repo chưa thể collect trong môi trường hiện tại: thiếu các dependency ngoài (`modal`, `gradio`, `numpy`); không cài GPU stack/không chạy train. Bộ test mới fail 6/6 trên base commit trước khi thêm tool, pass 6/6 trên nhánh này.
