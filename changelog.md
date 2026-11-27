# Agent changelog

## 2026-10-14 — Tách Stage 3 thành notebook độc lập, đọc artifact input read-only

- **Branch:** `arena/eb2fe5cf-dire-modal` (active; base `e81ac48242a2b87697398f082b39bd16c8409f2a`).
- Thêm `platforms/kaggle/celebahq-validation-split.ipynb` để tiếp tục từ Stage 2 artifact đã upload, không quét lại ảnh/hash/decode. Chỉ cần summary + JSONL manifest; SQLite resume checkpoint không bắt buộc cho Stage 3.
- Thêm `integrity_manifest_path` override để summary/manifest có thể đọc trực tiếp từ Kaggle Dataset mount read-only; output candidate vẫn ghi riêng `/kaggle/working`.
- Cấu hình notebook theo mount path người dùng cung cấp: `/kaggle/input/datasets/j4cknguyn/dire-celeba-hq-stage-2-artifacts-v1`.
- Cập nhật EXP-008, test detached-manifest path và notebook standalone.
- **Kiểm chứng cục bộ:** 30 scoped tests passed; pyflakes, py_compile, `git diff --check` và JSON/code-cell parse của cả hai notebook sạch. Không có Kaggle access/run.

## 2026-10-14 — Chuẩn bị deterministic validation stage sau integrity pass do người dùng báo

- **Branch:** `arena/3ff8bd36-dire-modal` (active; base `1a3e0a856142d17e133b78fa58c572739c782797`).
- Người dùng báo Stage 2 trên Kaggle pass cho 72.600/72.600 file, 0 SHA failures và 0 decode failures; summary/JSONL/SQLite paths được ghi vào `docs/kaggle-celebahq-progress.md`. Agent chưa đọc trực tiếp artifacts Kaggle.
- Thêm `src/dire_modal/kaggle_celebahq_validation.py` và Stage 3 vào notebook: exact 10%/class theo salt/rank rule đã duyệt, chỉ train real/SDv2; xuất candidate JSON ngoài input và báo hash trùng nội dung qua ranh giới train/validation để review. Chưa tạo split trên Kaggle.
- **Kiểm chứng cục bộ:** 28 test scoped passed; pyflakes, py_compile, notebook JSON/code-cell parse và `git diff --check` sạch. Không có Kaggle access/run, không có train/GPU workload.

## 2026-10-14 — Lưu checkpoint tiến độ và chốt validation protocol

- **Branch:** `arena/c4857c07-dire-modal` (đang active).
- Thêm `docs/kaggle-celebahq-progress.md` (EXP-008) ghi trạng thái dữ liệu Kaggle, phân loại số liệu do người dùng cung cấp, quyết định/gate và deterministic validation protocol sau khi người dùng chấp thuận salt `dire-celebahq-val-v1` cùng exact-count SHA-256 rank rule.
- Chưa tạo validation split; chờ integrity stage trên Kaggle qua gate trước. Route checkpoint trong `docs/README.md`; bổ sung kiểm tra link/catalog.
- **Kiểm chứng:** 22 test (integrity adapter/notebook, path adapter/audit và docs index) passed; pyflakes, py_compile và `git diff --check` sạch. Không có thao tác hay compute nào trên Kaggle.

## 2026-10-14 — Thêm SHA-256/decode integrity stage cho CelebA-HQ

- **Branch:** `arena/c4857c07-dire-modal` (đang active khi ghi).
- Theo lựa chọn của người dùng, thêm `src/dire_modal/kaggle_celebahq_integrity.py` và Stage 2 vào `platforms/kaggle/celebahq-path-index.ipynb`: hash SHA-256 và full-decode từng ảnh, ghi checkpoint SQLite + JSONL/summary bên ngoài input, hỗ trợ tiếp tục cùng một tree fingerprint.
- Stage không sửa/copy input, không dùng GPU; hash bytes đang mount không xác nhận archive provenance. Hash và Pillow decode là hai lượt đọc riêng; thời gian thực chưa đo.
- **Kiểm chứng:** integrity module/notebook, tree adapter/audit và docs index — 22 passed; pyflakes, py_compile, notebook parse và `git diff --check` sạch. Chưa chạy trên Kaggle.

## 2026-10-14 — Tích hợp CelebA-HQ path index vào notebook Kaggle

- **Branch:** `arena/f2985266-dire-modal` (đang active khi ghi).
- Thêm `platforms/kaggle/celebahq-path-index.ipynb`, nhúng adapter đã test, dùng Kaggle root người dùng gửi, chỉ index tám nhóm CelebA-HQ và xuất counts + inferred locators sang `/kaggle/working/celebahq-path-index.json`.
- Notebook ghi rõ không sửa/copy input, không đọc/giải mã ảnh, không hash, không assign validation split và không train. Module và notebook source được test đồng bộ.
- **Kiểm chứng:** adapter, notebook, tree audit và docs index — 16 passed; pyflakes, py_compile, notebook cell parse và `git diff --check` sạch.
- **Chưa làm:** chưa chạy notebook này trên Kaggle; chưa xác minh archive-member provenance, không tạo validation split, không train.

## 2026-10-14 — Thêm path adapter read-only cho CelebA-HQ đã giải nén

- **Branch:** `arena/77ed3f4e-dire-modal` (đang active khi ghi).
- Theo lựa chọn của người dùng, thêm `src/dire_modal/kaggle_celebahq_tree.py` để index các ảnh trong 8 nhóm CelebA-HQ của cây DIRE đã giải nén, nhận dạng `generator/generator/<file>` hoặc flat layout, từ chối cấu trúc mơ hồ/thiếu nhóm, không sửa input và không đọc/giải mã ảnh.
- Adapter phát ra locator dạng member suy luận (ví dụ `real/<file>`), đánh dấu `locator_inferred=True`; đây chưa phải xác nhận provenance hay tar-member identity. Không tạo validation split, không tích hợp train loop và không chạy compute.
- Ghi lựa chọn giữ Dataset read-only/không copy và giới hạn locator trong addendum ADR-0007.
- Thêm 5 test adapter; adapter + tree audit — 8 passed. Pyflakes, py_compile và `git diff --check` sạch.
- **Chưa làm:** chưa chạy adapter trên Kaggle thật, chưa giải mã/đọc nội dung ảnh, chưa chốt salt/rounding/count validation hoặc train budget.

## 2026-10-14 — Tạo notebook audit cây DIRE trên Kaggle chỉ đọc

- **Branch:** `arena/33250b71-dire-modal` (đang active khi ghi).
- Theo lựa chọn của người dùng, thêm `platforms/kaggle/dire-tree-audit.ipynb` để đếm file/suffix/kích thước theo split-domain và phát hiện các cặp thư mục lồng cùng tên như `real/real`, không đọc nội dung, không theo symlink, không sửa `/kaggle/input`, không giải nén/copy/train; report ghi ở `/kaggle/working`.
- Thêm `tests/test_kaggle_dire_tree_audit.py` với fixture nested folders, kiểm tra count/report, tính bất biến input, xử lý symlink và root không tồn tại.
- **Kiểm chứng:** test audit — 3 passed; code cells notebook parse; pyflakes và `git diff --check` sạch.
- **Chưa làm:** chưa chạy notebook trên Kaggle; chưa có quan sát trực tiếp ngoài cây thư mục do người dùng cung cấp; không sửa/upload phiên bản dataset mới. Notebook này là audit tổng quát; preflight archive CelebA-HQ hiện hữu vẫn không thay thế audit này.

> Nhật ký riêng cho các thay đổi do coding agent thực hiện trong repo này. **Không phải** changelog phát hành sản phẩm, lịch sử Modal cloud, hay nơi thay thế Git history. Chỉ ghi thay đổi đã làm trong workspace; không suy ra hay ghi công việc của người dùng/người khác.

## Quy ước ghi

- Mỗi mục ghi ngày, branch/PR nếu đã có, mục tiêu, files, kiểm chứng và phần chưa làm.
- Chỉ ghi điều đã xảy ra; ghi rõ khi test không chạy hoặc có lỗi sẵn có.
- Không sao chép toàn bộ metrics/run history vào đây; trỏ tới ledger phù hợp.
- Dùng dạng mới nhất ở đầu file.

## 2026-10-14 — Thêm Kaggle read-only preflight trước smoke/train

- **Branch:** `arena/1abd31d0-dire-modal` (đang active khi ghi).
- **Đã thay đổi:** thêm `src/dire_modal/kaggle_preflight.py` để xác minh manifest và 8 archive CelebA-HQ (SHA-256, byte size, regular/image/suffix counts), kiểm tra dung lượng trống và khả năng ghi vào thư mục làm việc ngoài input; hỗ trợ truy vấn thông tin GPU bằng `nvidia-smi` mà không khởi tạo CUDA hay workload.
- Tích hợp cell preflight vào `platforms/kaggle/notebook.ipynb` sau bước chuẩn bị source nhưng trước mọi smoke/pack/train cell; bỏ CUDA query khỏi cell kiểm tra môi trường và bỏ nhánh tự tải mẫu trong bước tìm data. Notebook yêu cầu Accelerator Off khi preflight, dừng nếu fail và ghi rõ `/kaggle/working` không tự bảo đảm persistence qua phiên.
- **Kiểm chứng:** `tests/test_kaggle_preflight.py`, `tests/test_kaggle_maps_inventory.py`, `tests/test_docs_index.py` — 17 passed; notebook JSON và code cells parse; pyflakes, py_compile và `git diff --check` sạch.
- **Chưa làm:** chưa chạy trên Kaggle nên chưa có kết quả xác minh archive, GPU, quota hay persistence thật; không train/benchmark và không khóa seed/budget. Preflight không xác minh pixel/provenance hoặc việc export artifacts qua nhiều phiên.

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
- Người dùng gửi inventory local: train real 28.000 / SDv2 40.000; test real 1.000, DALL-E2 500, Imagen 1.000, Midjourney 100, StarGAN 1.000, SDv2 1.000; tổng archive 5.769.522.447 byte. Đây là user-provided archive counts/sizes, chưa độc lập kiểm chứng.
- Inventory nêu candidate basename SDv2 `0..999` xuất hiện ở cả train/test và train có basename lặp. Phát hiện v1 chỉ so sánh member đầu tiên khi candidate lặp; sửa overlap audit để xét toàn bộ members, in counts/member paths và số duplicate candidate IDs. Hai test mới fail trên base `97ab83f`, pass trên nhánh hiện tại.
- Kiểm chứng nhánh hiện tại: inventory 7 passed + docs index 5 passed; pyflakes, py_compile, `git diff --check` sạch. AP prevalence theo test generator không đồng nhất; người dùng chốt metric chính macro-AP chuẩn hóa prior 50:50 bằng sample weights, đồng thời báo cáo AP thô theo prevalence làm sensitivity.
- Người dùng xác nhận basename SDv2 `0..999` là chỉ số cục bộ, không phải ID ảnh nguồn; không diễn giải candidate basename overlap thành leakage. Validation locator sẽ dựa trên full member path, còn content/provenance overlap phải được kiểm riêng.
- Người dùng chọn class-weighted loss cho train real/SDv2: giữ toàn bộ maps và cân bằng tổng đóng góp loss theo class; weights chỉ tính lại từ member counts/IDs đã được kiểm tra, không khóa theo counts manifest chưa audit.
- Người dùng chốt validation 10% riêng cho train real và SDv2, tách theo deterministic hash trên full archive-relative member path (không dùng basename index). Tỷ lệ và locator đã khóa; hash salt/rounding, file hashes và realized counts vẫn cần ghi trong preflight manifest.
- Người dùng chốt CI là paired stratified bootstrap 95%: giữ cùng real test IDs qua các họ, bootstrap fake IDs riêng theo generator; seed variability sẽ tách riêng hoặc thêm tầng bootstrap sau khi chốt số seed.
- Người dùng hoãn chốt số seed và training budget tới khi hoàn tất preflight dữ liệu/tài nguyên trên Kaggle; không bắt đầu train hoặc benchmark trước đó.
