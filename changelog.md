# Agent changelog

## 2026-10-14 — Implement trainer CelebA-HQ để review, chưa chạy

- Thêm `src/dire_modal/kaggle_celebahq_training.py`: cấu hình đã duyệt, Stage5 bundle/count/hash checks, plan-only helper, ResNet50 ImageNet state-dict load có `weights_only=True`, head mới, BCE-with-logits pos_weight 0.7, FP32/Adam, microbatch accumulation đúng cả partial window, validation AP/ACC, early stopping, atomic resume/best checkpoints và output reserve gate.
- Train API chỉ chạy khi được gọi tường minh; dùng 1 GPU cuda:0, không dùng test-family datasets để chọn checkpoint. Import module không import torch/load model.
- Thêm CPU-only tests cho config, step arithmetic (957/epoch), AP, ID hashing và plan counts. CelebA-HQ suite: **47 passed**; pyflakes, py_compile và diff check sạch. Không chạy Kaggle, không đọc ảnh, không GPU/training.
- Cần review preprocessing (source-style crop/flip) và fixed LR trước smoke; publish code Dataset mới và preflight disk lại. Explicit training go-ahead vẫn chưa có.

## 2026-10-14 — Người dùng duyệt cấu hình kế hoạch; training vẫn gated

- User duyệt: 3 seed `3407, 3408, 3409` tuần tự trên một T4; microbatch 32 × accumulation 2; cap 20 epoch, patience 5; BCE-with-logits trên mọi train với `pos_weight=0,7`; best checkpoint theo validation AP; `last_resume` + `best_model`, reserve 2 GiB/run.
- User cũng duyệt Adam `lr=1e-4`, `betas=(0.9,0.999)`, `weight_decay=0`; FP32/no AMP; ImageNet ResNet-50 với fc mới `Linear(2048,1)`, weight Normal(0,0.02), bias 0. Đây là plan research-extension, không phải paper reproduction.
- Chỉ chốt cấu hình để lập protocol; chưa có explicit go-ahead GPU/training. Cần implementation/dry checks và disk check mới nhất trước khi xin phép smoke run.

## 2026-10-14 — Ghi nhận Stage 6 pass và chờ khóa training protocol

- Người dùng dán Stage 6 output `pass`: `resnet50-19c8e357.pth`, 102.502.400 bytes, SHA-256 `19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097`; prefix filename khớp và file ổn định khi đọc.
- Không deserialize, không đọc ảnh dataset, không input write, không GPU workload hoặc training. Full digest chưa được đối chiếu với một reference độc lập.
- Bước tiếp theo là review/khóa training protocol; chưa đủ phê duyệt để train. Output do người dùng cung cấp, chưa được kiểm chứng độc lập từ Kaggle.

## 2026-10-14 — Ghi nhận Stage 5 pass và thêm Stage 6 checkpoint audit

- Người dùng báo Stage 5 pass với inventory 72.600, counts/ID-list hash đã khóa; không image read, input write, GPU workload hoặc training. Đây là report người dùng dán, agent chưa đọc trực tiếp file report.
- Thêm cell Stage 6 vào notebook để tìm `resnet50-19c8e357.pth`, tính size/full SHA-256, so 8-hex filename prefix, lưu report dưới `/kaggle/working`. Checkpoint không được deserialise; không dùng GPU/train. Cần chạy trên Kaggle; chưa có output thực tế.
- Cục bộ: chạy lại 38 tests CelebA-HQ; chưa benchmark hoặc chạy Kaggle.

## 2026-10-14 — Bỏ metadata-fingerprint gate khỏi Stage 5 binding theo yêu cầu người dùng

- Binding không còn tính fingerprint trên cây mount hiện tại hay yêu cầu fingerprint trong split artifact; tránh từ chối cùng bộ dữ liệu chỉ vì metadata phiên mount khác.
- Giữ nguyên các kiểm tra protocol, content-overlap status, validation-ID-list SHA-256, root, path/locator/label membership, train/validation disjointness và counts. Báo cáo Stage 5 ghi rõ metadata fingerprint check không được thực hiện.
- Bỏ cell audit SHA-256 bổ sung khỏi notebook binding; Stage 5 không đọc image bytes, decode ảnh, dùng GPU hay ghi vào input. Đây là quyết định bỏ kiểm tra, không phải xác nhận bytes giữa các phiên giống hệt nhau.
- Chưa chạy Kaggle; cần publish/attach code Dataset mới và notebook đã cập nhật. Cục bộ sẽ chạy tests và kiểm tra notebook trước khi bàn giao.
- Thêm kiểm tra trong setup cell: in đường dẫn loader được import và từ chối package còn chứa fingerprint gate, kèm hướng dẫn publish code Dataset mới rồi restart kernel. Điều này phát hiện lỗi do Kaggle vẫn mount package cũ, dù chỉ có một package.

## 2026-10-14 — Sửa import setup cho attached Kaggle code Dataset

- Screenshot cho thấy `ModuleNotFoundError: No module named 'dire_modal'` ngay tại dòng import: Python chưa tìm package, nên call builder chưa bắt đầu quét dữ liệu.
- Cập nhật Stage 5 notebook để dò package root ở mount-level, thêm parent vào `sys.path`, rồi import; không recurse qua image tree.
- Không Kaggle/GPU/image access được chạy trong checkout.

## 2026-10-14 — Tối ưu metadata scan sau khi cell kéo dài trên Kaggle

- Phân tích code: lượt binder cũ quét tree, gọi `is_symlink`/`is_file` rồi fingerprint lại từng file qua `Path.resolve()` và `stat()`. Đây là nhiều metadata calls trên Kaggle mount; cell không có progress output nếu package cũ, nên không thể phân biệt chậm với treo. Chưa có Kaggle log để xác định latency chính xác.
- Refactor inventory sang `os.scandir`/`DirEntry.stat` và tái dùng `size/mtime` trong fingerprint, giữ nguyên recipe/fingerprint; thêm tiến độ định kỳ. Cần upload code Dataset version mới để Kaggle dùng tối ưu này.
- Verification: **37 tests** CelebA-HQ passed; pyflakes, py_compile, notebook parse và diff check sạch. Không truy cập Kaggle/image contents/GPU.

## 2026-10-14 — Triển khai Stage 5 metadata-only binding notebook

- Thêm `platforms/kaggle/celebahq-data-binding.ipynb` để bind loader với split v2/tree thật khi Accelerator=None; report chỉ ghi ở `/kaggle/working`.
- Thêm progress callback cho tree scan/fingerprint dài, đồng bộ module trong path-index notebook, và test notebook/fixture metadata-only.
- **Kiểm chứng cục bộ:** 36 `tests/test_kaggle_celebahq*.py` passed; pyflakes, py_compile, `git diff --check` sạch. Chưa chạy Kaggle thật, không ảnh/GPU/training.

## 2026-10-14 — Ghi nhận cell đọc ImageNet checkpoint thành công theo người dùng

- Người dùng xác nhận xem cell đọc `resnet50-19c8e357.pth` là pass sau khi cập nhật Kaggle Dataset version. Không có byte size/full SHA-256 output trong chat, nên chưa ghi giá trị hash cụ thể.
- Bước kế tiếp còn lại trước training: chạy loader/split binding metadata-only trên Kaggle tree và Stage 3 v2 artifact. Không GPU workload hoặc training.

## 2026-10-14 — Xác định được ImageNet init trong code tác giả vendored

- `vendor/utils/config.py` đặt `arch="resnet50"`, `pretrained=True`; `vendor/utils/utils.py::get_network` gọi `resnet(pretrained=True)`, rồi thay FC thành một logit và init lại head.
- `vendor/networks/resnet.py` trỏ pretrained ImageNet ResNet-50 tới `https://download.pytorch.org/models/resnet50-19c8e357.pth`. Sau đó người dùng báo cell đọc file trên Kaggle thành công; output full SHA-256/size chưa được cung cấp. Agent không tải hoặc đọc checkpoint.

## 2026-10-14 — Ghi nhận Stage 4 preflight pass theo report người dùng

- **Branch:** `arena/8f5bd2ba-dire-modal` (active; base `1b1b3d6287b6389908ea69e5aa57d6886e0b57ad`).
- Người dùng báo toàn bộ checks pass; GPU có 2×Tesla T4, `/kaggle/working` còn `20.924.903.424` bytes. Không workload GPU/training được chạy.
- Cập nhật EXP-008; thêm `kaggle_celebahq_data.py` nối Stage 3 v2 ID với extracted tree bằng metadata-only index, lazy decode, class/test-family partitions và strict guards.
- Fixture-only verification: 17 focused loader/tree/split tests passed; 26 tests with preflight/integrity suite. Không truy cập ảnh Kaggle, không GPU workload/training. Footprint và step budgets trong EXP-008 là tính toán/đề xuất, chưa benchmark.
- Kế tiếp: người dùng chạy path-only binding check trên Kaggle tree và review budget; cần explicit go-ahead trước bất kỳ GPU smoke/training nào.

## 2026-10-14 — Bổ sung GPU metadata fallback cho Stage 4

- **Branch:** `arena/cf863117-dire-modal` (active; base `024ffab52ef353abfebefc6c467519a569f38d13`).
- Sau khi Kaggle report thiếu `nvidia-smi`, bổ sung fallback query-only qua PyTorch device properties và memory info; không allocate tensors/model workload. Cập nhật preflight notebook, tests và EXP-008.
- **Kiểm chứng cục bộ:** 37 scoped tests passed; pyflakes, py_compile, notebook parse và `git diff --check` sạch. Chưa chạy trên Kaggle; cần upload notebook mới và rerun Stage 4 để biết GPU có được cấp hay không.

## 2026-10-14 — Ghi nhận Kaggle preflight needs_attention vì GPU query

- **Branch:** `arena/8f5bd2ba-dire-modal` (active; base `1b1b3d6287b6389908ea69e5aa57d6886e0b57ad`).
- Người dùng báo mọi input/split/write probes pass; `/kaggle/working` free `20.934.410.240` bytes. `nvidia-smi` không tồn tại nên GPU query unavailable và preflight status `needs_attention`.
- Report path `/kaggle/working/celebahq-training-preflight.json`; không image read, input write, GPU workload hay training. Ghi lại trong EXP-008; bước kế là bật/kiểm tra GPU accelerator và chỉ rerun Stage 4.

## 2026-10-14 — Triển khai Kaggle resource preflight trước training

- **Branch:** `arena/cf863117-dire-modal` (active; base `024ffab52ef353abfebefc6c467519a569f38d13`).
- Thêm `src/dire_modal/kaggle_celebahq_preflight.py` và Stage 4 vào standalone Stage 3 notebook: kiểm tra mount paths/split v2, disk usage, writable artifact/checkpoint paths và `nvidia-smi` metadata query-only. Báo cáo dưới `/kaggle/working`; không đọc ảnh, không ghi input, không CUDA workload/train.
- **Kiểm chứng cục bộ:** 35 scoped tests passed; pyflakes, py_compile, notebook parse và `git diff --check` sạch. Chưa chạy trên Kaggle; preflight status không quyết định budget sufficiency khi checkpoint/train budget chưa chốt.

## 2026-10-14 — Ghi nhận Stage 3 v2 pass theo report người dùng

- **Branch:** `arena/cf863117-dire-modal` (active; base `024ffab52ef353abfebefc6c467519a569f38d13`).
- Người dùng báo v2 candidate giữ counts 2.800/4.000, loại 50 real và 0 SDv2 records có repeated SHA khỏi validation candidates, `content_overlap_audit=pass`.
- Ghi `validation_id_list_sha256` mới và output path vào EXP-008; v1 candidate overlap bị supersede. Agent chưa đọc artifact trực tiếp; chưa train.

## 2026-10-14 — Duyệt protocol v2 content-disjoint để xử lý overlap validation

- **Branch:** `arena/022ccec0-dire-modal` (active; base `f68450babe828886e93a0ccd6a89e650ca66cb51`).
- Theo lựa chọn của người dùng, sửa Stage 3: globally exclude mọi repeated-SHA records khỏi validation candidates, giữ nguyên chúng trong training, và lấy tiếp theo thứ tự rank đến đủ exact-count mỗi label. Ghi schema/protocol output v2 để bảo toàn artifact v1.
- V1 candidate do người dùng chạy có 6 SHA overlap real; không train với v1. V2 đã qua 31 local tests, pyflakes, py_compile, notebook parse và `git diff --check`; Kaggle rerun riêng vẫn cần thiết, không rerun Stage 2.

## 2026-10-14 — Ghi nhận 6 exact-duplicate pairs giữa validation và training

- **Branch:** `arena/022ccec0-dire-modal` (active; base `f68450babe828886e93a0ccd6a89e650ca66cb51`).
- Người dùng cung cấp sáu cặp real có SHA-256 trùng, mỗi cặp một ảnh validation và một ảnh train; không có overlap ở SDv2. Paths/hashes được ghi vào EXP-008.
- Candidate hiện vẫn `review_required`; chưa train. Chờ người dùng chốt: hash-aware deterministic reselect giữ đủ records/validation count nhưng đổi 6 validation IDs, hay giữ IDs và loại 6 training duplicates.

## 2026-10-14 — Ghi nhận Stage 3 candidate và dừng ở overlap review

- **Branch:** `arena/022ccec0-dire-modal` (active; base `f68450babe828886e93a0ccd6a89e650ca66cb51`).
- Người dùng báo Stage 3 candidate tạo class 0: 2.800/28.000 validation, class 1: 4.000/40.000 validation; ID-list SHA-256 `1f550e2992e136873caf6df26cc46de3492794ebc5b8badc6eece9575f7047d6`.
- `content_overlap_audit` là `review_required`; artifact path và gate tiếp theo được ghi trong EXP-008. Chưa train; agent chưa đọc artifact trực tiếp.

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
