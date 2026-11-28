---
doc_id: EXP-008
title: Kaggle CelebA-HQ progress checkpoint
status: in-progress
snapshot_date: 2026-10-14
---

# Checkpoint tiến độ — Kaggle / CelebA-HQ-SDv2

> Tài liệu trạng thái tại snapshot, không phải ledger của một lần train. Số liệu Kaggle dưới đây do người dùng cung cấp từ output; không được đo lại trong checkout. Cập nhật checkpoint này khi đổi trạng thái/gate.

## Trạng thái hiện tại

- **Branch:** `arena/b7a68483-dire-modal` — active, từ base `2042ea1cff92d8bbd20ad7cc02ee755d9b3833da`.
- **Scope đã chọn:** research-extension CelebA-HQ-SDv2; chỉ dùng `dire/train/celebahq/{real,sdv2}` và test CelebA-HQ (real, bốn họ holdout, SDv2 control). Không trộn LSUN-Bedroom/ImageNet.
- **Dataset trên Kaggle:** input read-only tại `/kaggle/input/datasets/j4cknguyn/diffusionforensics/dire`; người dùng chọn giữ nguyên, không copy/re-upload/flatten.
- **Đã chạy trên Kaggle theo report người dùng:** audit cây toàn bộ DIRE, path-index CelebA-HQ, Stage 2 SHA-256/full-decode (`pass`, 72.600/72.600; không có SHA/decode failures). Agent chưa truy cập trực tiếp Kaggle hay artifact.
- **Stage 3 v2 theo report người dùng:** candidate tạo đúng counts 2.800 real / 4.000 SDv2, audit unique-content candidates ghi 50 real + 0 SDv2 bị loại khỏi validation pool; `content_overlap_audit=pass`.
- **Stage 4 preflight mới nhất theo report người dùng:** `pass`; tất cả input/split/write probes pass, GPU query thấy 2×Tesla T4 (15.360 MiB mỗi GPU). Không có image read, input write, GPU workload hoặc training.
- **Checkpoint ImageNet theo report người dùng:** cell đọc `resnet50-19c8e357.pth` được xem là thành công; output byte size/full SHA-256 chưa được cung cấp trong chat. Chưa có GPU workload/model training.
- **Stage 5 binding:** notebook `platforms/kaggle/celebahq-data-binding.ipynb` đã được thêm; chưa chạy trên Kaggle thật. Phiên bản scanner trước đó lặp nhiều metadata calls cho mỗi path (`is_symlink`/`is_file`, rồi `resolve`/`stat` khi fingerprint); trên mount chậm điều này có thể giải thích thời gian kéo dài, dù chưa có log Kaggle để xác định chính xác latency. Scanner mới dùng `os.scandir`/`DirEntry.stat` một lần và fingerprint từ stat đã lưu, kèm progress; cần cập nhật code Dataset Kaggle lên phiên bản này trước khi rerun với Accelerator=None. Chưa smoke train, train hay benchmark.

## Bằng chứng do người dùng cung cấp

### Audit cây toàn bộ DIRE

- Root: `/kaggle/input/datasets/j4cknguyn/diffusionforensics/dire`.
- **422.594** regular files, **10.330** directories, **40.877.686.851 byte** theo `stat`; suffixes `.jpeg`, `.jpg`, `.png`.
- Split/domain file counts: `test/celebahq` 4.600; `test/imagenet` 29.994; `test/lsun_bedroom` 24.000; `train/celebahq` 68.000; `train/imagenet` 80.000; `train/lsun_bedroom` 200.000; `val/imagenet` 11.000; `val/lsun_bedroom` 5.000.
- Có nhiều leaf lặp như `real/real` và `adm/adm`. Audit chỉ đếm metadata/suffix, không đọc ảnh, không hash, không xác minh provenance.
- Trước đó Kaggle UI từng hiển thị staging `313.297 files / 28,82 GiB`, khác với audit `422.594 files / 40.877.686.851 byte`. **Chưa giải thích/xác minh khác biệt phiên bản hoặc phạm vi.**

### Path index CelebA-HQ

Người dùng chạy notebook và báo `Index status: complete`, **72.600** file:

| Split | Generator | Label | Files |
|---|---|---:|---:|
| train | real | 0 | 28.000 |
| train | sdv2 | 1 | 40.000 |
| test | real | 0 | 1.000 |
| test | dalle2 | 1 | 500 |
| test | if | 1 | 1.000 |
| test | midjourney | 1 | 100 |
| test | sdv2 | 1 | 1.000 |
| test | stargan | 1 | 1.000 |

Notebook đã ghi `/kaggle/working/celebahq-path-index.json`; người dùng chỉ gửi console summary, chưa chia sẻ file JSON đầy đủ. Các counts khớp inventory mong đợi, nhưng không chứng minh ảnh giải mã được hay byte/provenance gốc.

### Integrity Stage 2 — kết quả do người dùng cung cấp

Người dùng báo:

- `Integrity status: pass`; records `72.600 / 72.600`; hashed this run `72.600`; reused checkpoint `0`.
- SHA failures `0`; decode failures `0`.
- SQLite: `/kaggle/working/celebahq-integrity-81dcda99aeb997b25b82.sqlite3`.
- JSONL manifest: `/kaggle/working/celebahq-integrity-81dcda99aeb997b25b82.jsonl`.
- Summary: `/kaggle/working/celebahq-integrity-81dcda99aeb997b25b82-summary.json`.

Đây là output người dùng dán; agent chưa đọc/kiểm tra trực tiếp manifest/summary trên Kaggle. Integrity pass xác nhận bytes hiện mount và decode theo verifier, không xác minh archive provenance hoặc source identity.

### Stage 3 candidate — kết quả do người dùng cung cấp

- Người dùng báo `candidate_created`; class 0: 28.000 tổng, 2.800 validation, 25.200 train; class 1: 40.000 tổng, 4.000 validation, 36.000 train.
- v1 `validation_id_list_sha256`: `1f550e2992e136873caf6df26cc46de3492794ebc5b8badc6eece9575f7047d6` (**đã được v2 supersede**).
- `content_overlap_audit.status`: `review_required`; người dùng sau đó báo **6 SHA trùng**, tất cả label 0 (`real`), label 1 không có overlap. Mỗi digest ghép một path validation với một path training:

| SHA-256 | Validation path | Training path |
|---|---|---|
| `00966d9f4063849acc52021f10e231ae2a3534e7faff56a5fc7b2419de6f631e` | `train/celebahq/real/real/021366.jpg` | `train/celebahq/real/real/160486.jpg` |
| `01659f1e75b8862d24f78371c5effcf8929feab9eb2bcdec7a171466299e171a` | `train/celebahq/real/real/083407.jpg` | `train/celebahq/real/real/067217.jpg` |
| `860572d7a700fac2e5f4b29b015e5b82d9474764d91dd79423e174c4dbe3a4ab` | `train/celebahq/real/real/021423.jpg` | `train/celebahq/real/real/019673.jpg` |
| `9507a308ff202aa9e4f72f10862eaa15b4cb04c798870d7f67848dedbf146df5` | `train/celebahq/real/real/074402.jpg` | `train/celebahq/real/real/051286.jpg` |
| `de6e6810283fa32808729048052d501d01e7fb15396cc8f5041210b200100c97` | `train/celebahq/real/real/020328.jpg` | `train/celebahq/real/real/186880.jpg` |
| `f52e23b0b52f5e1ac35db1f1c8f56922a5b82f3e92a55e706d0967e6c04fbcca` | `train/celebahq/real/real/074740.jpg` | `train/celebahq/real/real/069407.jpg` |

- Artifact: `/kaggle/working/celebahq-validation-split-v1-81dcda99aeb9.json`.

Chi tiết cặp/hash là do người dùng dán từ console, chưa được agent đọc trực tiếp file artifact. Sáu cặp xác nhận exact-byte duplication và gây overlap train/validation. Candidate v1 không dùng cho train.

### Stage 3 v2 — kết quả do người dùng cung cấp

- Inputs: summary và JSONL từ `/kaggle/input/datasets/j4cknguyn/dire-celeba-hq-stage-2-artifacts-v1/`.
- Counts: label 0 = 28.000 total, 2.800 validation, 25.200 training; label 1 = 40.000 total, 4.000 validation, 36.000 training.
- Candidate audit: label 0 có 27.950 content-unique validation candidates và loại 50 duplicate-hash records khỏi validation candidates; label 1 có 40.000 unique candidates, loại 0.
- `validation_id_list_sha256`: `c635c38a5888ef4e0196423dd8f342f412dc15ea633c17d835330c740d7dc9fd`.
- `content_overlap_audit.status`: `pass`.
- Output: `/kaggle/working/celebahq-validation-split-v2-81dcda99aeb9.json`.

Đây là console output người dùng cung cấp, chưa được agent đọc file artifact trên Kaggle. Theo report, v2 đạt exact counts và không còn SHA trùng qua train/validation; v1 ID-list hash đã superseded.

### Stage 4 resource preflight — kết quả do người dùng cung cấp

- Toàn bộ input checks pass: DIRE root/train real/train SDv2, Stage 2 artifact Dataset, summary và JSONL manifest đều tìm thấy.
- Stage 3 v2 check pass; ID-list SHA-256 `c635c38a5888ef4e0196423dd8f342f412dc15ea633c17d835330c740d7dc9fd`; counts 2.800/4.000; overlap pass.
- `/kaggle/working`: total `20.957.446.144` bytes, used `6.258.688`, free `20.934.410.240` bytes. Artifact/checkpoint write-delete probes đều pass.
- GPU query unavailable: `nvidia-smi` executable không tồn tại; overall status `needs_attention`.
- Execution flags: không đọc image contents, không thử ghi input, không chạy GPU workload, không training. Report `/kaggle/working/celebahq-training-preflight.json`.

Đây là report đầu do người dùng dán, chưa được agent truy cập trực tiếp. Lần đó không có `nvidia-smi`, nên GPU chưa được xác nhận; sau đó người dùng rerun Stage 4 và báo thành công như dưới đây.

### Stage 4 rerun — kết quả do người dùng cung cấp

- `Preflight status: pass`; toàn bộ input checks, Stage 3 v2 checks, disk query và write/delete probes pass.
- Working filesystem: total `20.957.446.144` bytes, used `15.765.504`, free `20.924.903.424` bytes.
- GPU: 2× Tesla T4; mỗi GPU total `15.360 MiB`, free `14.912 MiB`; driver `580.178.04`.
- Execution flags đều false: image contents không đọc, input không ghi, không GPU workload, không training.
- Report `/kaggle/working/celebahq-training-preflight.json`.

Đây là output người dùng dán, chưa được agent đọc file report. Pass chỉ xác nhận preflight checks; không chứng minh 20.9GB đủ cho budget/checkpoint chưa chốt.

## Quyết định người dùng đã chọn

- Giữ Kaggle input read-only; không tạo bản Dataset đã flatten để tránh copy/re-upload.
- Dùng reader riêng cho CelebA-HQ, chỉ xử lý các nhóm mục tiêu; không áp dụng preflight archive cũ cho cây đã giải nén.
- Dùng member-style locator suy ra bằng cách bỏ thư mục generator ngoài (ví dụ `real/real/x.png` → `real/x.png`); code phải đánh dấu `locator_inferred=true`. Locator này **chưa xác minh** với archive member gốc.
- Stage 2 SHA-256 + full image decode cho 72.600 file được người dùng báo pass, không lỗi, không resume. Implementation được thiết kế read-only/no-GPU; console excerpt không bao gồm các cờ `input_modified`/`gpu_work_started`.
- Người dùng đã **chốt** validation protocol: salt `dire-celebahq-val-v1`; SHA-256 trên salt + NUL + `member_locator_inferred`, sort digest tăng dần (tie-break locator), lấy `round-half-up(0,10 × N)` mỗi class.
- Người dùng đã duyệt protocol revision v2: loại khỏi validation candidate pool mọi train record có SHA-256 xuất hiện hơn một lần trên toàn bộ hai label; vẫn xếp các ứng viên còn lại theo salt/hash rank đã chốt, chọn đủ 10% dựa trên tổng records mỗi label. Duplicate groups giữ nguyên trong training. V2 giữ counts nhưng thay validation IDs; đây là protocol khác v1 và phải ghi version riêng.
- V1 Kaggle candidate có overlap nên không dùng để train. Stage 3 v2 và Stage 4 preflight pass theo report người dùng. Chưa train; seed/budget/checkpoint policy vẫn chưa khóa.
- Chưa khóa seed/training budget; chưa train hoặc benchmark.

## Stage integrity và validation implementation

- `src/dire_modal/kaggle_celebahq_integrity.py` hash file theo chunk, sau đó dùng Pillow `load()` để full-decode từng ảnh; ghi SQLite checkpoint, JSONL per-file manifest và summary dưới `/kaggle/working`.
- Có thể resume khi cùng cây dữ liệu bất biến và fingerprint metadata (relative path, size, mtime) khớp. Hash và decode là hai lượt đọc riêng. Tổng `stat` bytes của các file CelebA-HQ trong audit là **5.775.377.067 byte cho một lượt**; lượng I/O thực tế của hai lượt và thời gian chưa đo.
- Hash xác định bytes hiện đang mount; không xác nhận archive provenance, source-image identity hay train/test independence.
- Stage 3 v2: `src/dire_modal/kaggle_celebahq_validation.py`, embed trong notebook tích hợp và standalone `platforms/kaggle/celebahq-validation-split.ipynb`. Đọc summary/JSONL read-only từ Kaggle Dataset, không rerun Stage 1/2; trước khi chọn 10% mỗi label, bỏ toàn bộ validation candidates có SHA-256 lặp trong combined train pool. Duplicate records còn lại được giữ trong training; count mục tiêu vẫn tính trên tổng class records. Xuất artifact v2 riêng, không ghi đè v1.
- Stage 4 preflight mới: `src/dire_modal/kaggle_celebahq_preflight.py`, embed thành 2 cell cuối của standalone notebook. Kiểm tra input paths, Stage 3 v2 status/counts, disk free, write/delete probe cho artifact/checkpoint directories, GPU metadata qua `nvidia-smi` (PyTorch device-properties fallback nếu thiếu binary) và ghi report. Không đọc ảnh, cấp tensor/workload GPU, train hoặc ghi vào `/kaggle/input`; dung lượng chỉ được đo, chưa thể xác định đủ hay không cho tới khi budget/checkpoint sizes được chốt.
- `src/dire_modal/kaggle_celebahq_data.py` nối v2 artifact với extracted tree thành lazy datasets: check protocol/status/zero-overlap/id-list SHA/root/counts/locator, path membership và Stage 2 metadata fingerprint (path/size/mtime, không đọc bytes); train = toàn bộ train pool trừ validation IDs; test datasets giữ riêng theo từng generator. Xây index chỉ đọc metadata/path; `PIL.Image.open` chỉ nằm trong explicit `__getitem__`. Có progress callback cho path scan/fingerprint; scanner dùng `os.scandir` và stat cache từ `DirEntry`, fingerprint tái sử dụng metadata đã đọc thay vì resolve/stat lại từng file. Không tự tạo DataLoader, cấp GPU hoặc ghi vào input.
- `platforms/kaggle/celebahq-data-binding.ipynb` là notebook Stage 5 riêng: dùng code package attached làm input, Accelerator=None, kiểm tra counts/hash đã khóa, in tiến độ nếu package có callback, ghi report chỉ vào `/kaggle/working` và không truy cập item/dataloader.
- Fixture tests metadata-only: bộ đầy đủ `tests/test_kaggle_celebahq*.py` **37 passed**; pyflakes, py_compile và `git diff --check` sạch. Chưa chạy notebook trên Kaggle thật; không đọc ảnh Kaggle hay dùng GPU.
- Checkpoint footprint là ước tính giải tích, không phải file đã ghi: canonical ResNet-50 1-logit có khoảng 23.510.081 tham số; FP32 model-only ~94,0 MB; `state_dict + Adam moments` resumable ~282,1 MB. Giữ một `last_resume` + một `best_model` ~376 MB; trong lúc atomic replace của resumable, khoảng cực đại khoảng 658 MB (old + temp + best), chưa tính overhead/runtime/logs. Không lưu full optimizer checkpoint theo từng epoch lâu dài. 2 GiB output reserve/run là safety allowance đề xuất, không phải đo.
- Tập train sau split = 61.200 mẫu. Batch hiệu dụng 64 tương đương ceil(61.200/64)=957 optimizer steps/epoch; 20-epoch cap ~19.140 steps/run, 3 seeds ~57.420. Wall-clock chưa biết do không benchmark. Free-space snapshot user-reported ~20,9 GB cao hơn output reserve 2 GiB/run, nhưng cần kiểm tra lại trước khi run và không tính đây là xác nhận cuối.

## Next gates

1. Publish/attach a new private code-Dataset version containing optimized `kaggle_celebahq_data.py`, `kaggle_celebahq_tree.py`, and `kaggle_celebahq_integrity.py` (preserve checkpoint file); upload notebook `platforms/kaggle/celebahq-data-binding.ipynb` and attach split artifact if it is not in `/kaggle/working`. Notebook auto-adds the package parent to `sys.path`; run with Accelerator=None. Mong đợi train labels 25.200/36.000, val 2.800/4.000, validation hash đã khóa và test-family counts từ inventory; không mở/đọc nội dung ảnh.
2. Nguồn khởi tạo đã được xác định trong bản code tác giả được vendored: `DefaultConfigs.pretrained=True`; `get_network` gọi `resnet50(pretrained=True)` từ `https://download.pytorch.org/models/resnet50-19c8e357.pth`, rồi thay và khởi tạo lại `fc` 1-logit. Người dùng xem cell đọc file checkpoint là pass; cần ghi lại byte size/full SHA-256 từ output nếu muốn ledger có định danh đầy đủ. Review retention 2 GiB/run, seed/step cap và metric chọn checkpoint; kiểm tra free space lại trước compute.
3. Chỉ sau khi các gates trên pass và người dùng explicit go-ahead mới chạy GPU smoke train có giới hạn; hiện chưa train/full benchmark.
4. Báo cáo macro-AP theo từng unseen generator (`dalle2`, `if`, `midjourney`, `stargan`), AP raw prevalence, AUROC, counts/CI; giữ `sdv2` là seen-generator control và không tune test threshold.

## Ước tính tài nguyên và đề xuất chưa được khóa

- **Storage:** một model ResNet-50 1-logit FP32 (23.510.081 params, suy ra từ kiến trúc canonical, chưa được instantiate ở Kaggle) ≈94,0 MB. Resumable checkpoint với model + Adam m/v ≈282,1 MB; một best model-only ≈94,0 MB. Retain duy nhất `last_resume.pt` + `best_model.pt` ≈376 MB; khi ghi atomically `last_resume.tmp` còn tồn tại song song với bản cũ, đỉnh lý thuyết khoảng 658 MB. Đề xuất đặt 2 GiB/run cho checkpoint/temp/log/metadata/headroom; ba seed tuần tự và retention như trên ≈6 GiB, dưới free-space user-reported 20.924.903.424 bytes. Đây là capacity planning, chưa đo file thật; kiểm tra free space lại ngay trước compute. Không giữ optimizer checkpoint mọi epoch (20 epoch × full resume state đã ~5,64 GB/run trước temp/overhead).
- Input được mount read-only, nên loader không tạo bản dữ liệu: chỉ metadata/path index trong RAM, decode on demand. Không có image cache/output copy được tính; nếu implementation sau này cache ảnh thì cần budget mới. 72.600 records × path metadata memory chưa được đo.
- **Steps:** 61.200 training IDs sau v2. Batch hiệu dụng 64 tạo 957 optimizer updates/epoch; đề xuất cap 20 epochs = 19.140 updates/run, patience 5 theo validation-selected metric; 3 seeds tuần tự = tối đa 57.420 updates. Wall-clock/throughput cố ý chưa ước tính do user cấm benchmark.
- **Đề xuất để review, chưa khóa:** seed list `3407, 3408, 3409` (3407 cho run đầu; các seed bổ sung để báo seed variability riêng), một T4 tuần tự, microbatch 32 × accumulation 2 (effective 64), FP16 autocast/GradScaler nếu được xác nhận an toàn. Loss BCE-with-logits trên mọi train sample, `pos_weight = 25.200 / 36.000 = 0,7`; không resample/loại duplicate train records. Chọn best checkpoint chỉ bằng validation AP (val có 2.800 real / 4.000 SDv2; ghi prevalence), threshold 0,5 chỉ cho ACC phụ. Cần chốt chính thức hparams, metric chọn checkpoint và nguồn ImageNet weights trước khi chạy.

## Protocol validation đã chốt

Người dùng chấp thuận salt `dire-celebahq-val-v1` và exact-count rank rule: trong từng class, sắp xếp theo `SHA256(salt + NUL + member_locator_inferred)` tăng dần, tie-break bằng locator; lấy `round-half-up(10% × N)` IDs đầu tiên. Với inventory counts hiện có, mục tiêu là 2.800 real và 4.000 SDv2.

Sau khi Stage 3 v1 phát hiện sáu duplicate groups cắt qua train/validation, người dùng **đã duyệt revision v2**: tính tần suất SHA-256 trên combined train pool của cả hai label; mọi record có digest xuất hiện >1 lần không đủ điều kiện validation. Sắp xếp các eligible records theo cùng salt/hash/locator rank và lấy đủ `round-half-up(10% × N)` mỗi label, với N là tổng records của label trước lọc. Repeated groups giữ trong training. Đây đổi validation IDs so với v1 nhưng giữ exact 2.800/4.000 counts và loại exact-byte overlap giữa hai partition; protocol ID là `celebahq-content-disjoint-exact-count-hash-rank-v2`.

**V1 candidate không được dùng cho train** vì overlap audit `review_required`. V2 candidate được người dùng báo đã chạy với audit `pass` và đúng realized counts; artifact chưa được agent đọc trực tiếp. Locator vẫn suy ra từ cây thư mục (`locator_inferred=true`) và chưa xác minh với tar member gốc.
