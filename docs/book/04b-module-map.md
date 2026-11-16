# Phần 13 — Bản đồ module của kho này

Mục đích của phần này: với **mỗi** module trong `src/`, nói trong **ba dòng** nó
làm gì, ai gọi nó, và nếu bỏ nó đi thì hỏng cái gì. Phần 10–11 đã đi sâu vào những
hàm quyết định kết quả; ở đây phủ **phần còn lại**, kể cả những file chưa từng được
nhắc: `labels`, `provenance`, `fetchcore`, `pack`, `archive`, `bench`, `api`,
`demo_train`, `manifest`, `download`, `local_fetch`, `upload`.

## 13.1 Sơ đồ phụ thuộc

```
         ┌───────────────────────── dire_core (Python thuần, không cần torch) ─┐
         │ labels · preprocess · detector · score · batch · provenance          │
         └───────▲──────────────────────▲───────────────────────▲───────────────┘
                 │                      │                       │
     ┌───────────┴──────┐   ┌───────────┴────────┐   ┌──────────┴───────────┐
     │ dire_local (CLI) │   │ dire_modal.eval    │   │ dire_modal.api       │
     └──────────────────┘   └────────┬───────────┘   └──────────┬───────────┘
                                     │                          │
                    ┌────────────────┴──────────┬───────────────┴─────┐
                    │  dữ liệu: fetchcore        │  hạ tầng: common    │
                    │  manifest · download       │  volumes · images   │
                    │  local_fetch · upload      │  secrets            │
                    └────────────────┬───────────┴─────────────────────┘
                                     │
                    ┌────────────────┴───────────────┬──────────────────┐
                    │ pack (tar → shards)           │ archive (R2/HF/  │
                    │ bench (đo thông lượng)        │ Kaggle mirror)   │
                    └───────────────────────────────┴──────────────────┘
```

Quy tắc một chiều: `dire_core` **không** import `dire_modal`; `dire_modal` không
import `platforms/`. `tests/test_core_parity.py` kiểm điều này bằng AST.

## 13.2 `dire_core` — lõi thuần Python

### `labels.py` — một hàm, một quyết định

`label_of(member_name, archive_name="") -> int` suy nhãn từ **đường dẫn**:
`0_real → 0`, `1_fake → 1`. Vì sao cần hàm riêng thay vì đọc thuộc tính của
dataset: dữ liệu ở đây là **file tar** (không phải thư mục), nên không có
`ImageFolder` để suy nhãn; và nhãn phải suy **cùng một cách** ở eval, ở CLI và ở
API, nếu không ba đường cho ba con số khác nhau. Nếu hàm này sai (ví dụ nhận
`0_real` là giả), ACC thành `1 − ACC` và **không có exception nào**.

### `provenance.py` — nhãn trung thực, có cấu trúc

Định nghĩa ba thành phần `("dire_maps", "detector", "images")` và ba nhãn
`R1/R2/R3` (ADR-0006). `class Claim` gói ba cờ đó lại, có:

- `as_dict()` — để ghi vào `run.json` (dataclass `asdict` trả tuple nên phải tự ép
  sang list, nếu không JSON round-trip sẽ khác);
- `is_full` — chỉ `True` khi cả ba thành phần đều tự làm;
- `data_source` — chuỗi mô tả, phải dùng `self.self_made` (viết trần `self_made` là
  `NameError`, lỗi đã từng có);
- `banner(c)` — in khung cảnh báo ở đầu mỗi báo cáo, ví dụ "R0 — chấm lại trên
  artefact của tác giả".

Giá trị của module này không nằm ở code (khoảng 120 dòng) mà ở chỗ nó **buộc**
mọi báo cáo phải khai mức. Bỏ nó đi thì bảng số vẫn đúng nhưng người đọc không còn
cách phân biệt "tôi đo" với "tôi chép".

### `preprocess.py` — hợp đồng bốn bước

Đã nói ở Phần 11.2. Thêm một chi tiết: module **không** import torch ở mức module,
`torch`/`transforms` được **truyền vào**. Nhờ vậy container chỉ đọc CSV không phải
cài torch, và test chạy được bằng Python thuần.

### `detector.py` — ba hàm, một sự cố thật

- `unwrap_state(state)`: bóc `model`/`state_dict`/`net`/`model_state_dict`, rồi bỏ
  tiền tố `module.` do `DataParallel`. Vì sao cần: file checkpoint của tác giả là
  **dict lồng** (269,4 MB, gồm cả optimizer), không phải state_dict thuần.
- `fc_out_features(state)`: trả `fc.weight.shape[0]` (1 hoặc 2), hoặc `0` nếu không
  nhận ra — để chỗ gọi quyết định báo lỗi thay vì đoán.
- `load_detector(ckpt, torch, resnet50, device)`: dựng ResNet-50 với `num_classes`
  đọc từ checkpoint rồi nạp. Docstring ghi lại sự cố 2026-10-28: chọn
  `num_classes=2` cho checkpoint 1 logit làm `load_state_dict` nổ **sau khi** đã
  trả tiền tải 3,25 GB. Đây là ví dụ của nguyên tắc: *kiểm rẻ trước, tải đắt sau*.

### `score.py` — hai hàm, một quy ước

`sigmoid(z)` (chia hai nhánh, không tràn) và `prob_fake(rows, n_out)` (sigmoid cho
1 logit, softmax cho ≥2). Cả hai thuần `math`, không numpy/torch — nên test được
bằng giá trị đã biết mà không cần cài gì. Xem Phần 0.5 để hiểu vì sao sigmoid đi
cặp với BCE.

### `batch.py` — một vòng lặp cho ba đường

`batch_logits(model, tensors, torch, device, batch_size=64)` chia lô, chạy trong
`inference_mode` (không dựng đồ thị gradient ⇒ tiết kiệm bộ nhớ), trả **logit thô**
chứ không trả xác suất. Vì sao tách hai bước: nơi gọi biết `n_out`, và `prob_fake`
test được độc lập với model.

`class Detector` bọc `(model, n_out, torch, transforms, device)` với
`load()` / `logits_of()` / `score_pils()` / `info()`. Đây là API mà CLI và HF Space
dùng; nhờ nó, "nạp checkpoint → tiền xử lý → logit → xác suất" chỉ có **một** bản.

## 13.3 `dire_modal` — lớp Modal

### `common.py` — mọi thứ dùng chung, khai một lần

- `APP_NAME = "dire"`, `app = modal.App(APP_NAME, image=cpu_image)`.
- `cpu_image`: Debian slim + Python 3.11, cài `pyyaml`, `boto3`, `tqdm`,
  `webdataset`, `pillow`, `huggingface_hub`, `kaggle`, và **BaiduPCS-Go** (tải từ
  GitHub release). Đây là image cho mọi job không cần GPU.
- `gpu_image`: image riêng có torch + torchvision + fastapi (dùng cho eval/API).
- `vol_raw` (`/raw`) · `vol_shards` (`/shards`) · `vol_ckpt` (`/ckpt`): ba volume,
  tách theo vòng đời — dữ liệu gốc / shard cho train / checkpoint + kết quả.
- `baidu_secret`, `r2_secret`: `modal.Secret.from_name`, **không** hard-code.
- `mount_python_sources(img)`: mount `src/` vào image để container chạy đúng code
  của lần commit đó, thay vì bản trong image.

Vì sao gom vào một file: mỗi `modal.App` cần image; nếu ba module tự dựng image
riêng, chúng sẽ lệch phiên bản thư viện và kết quả không so sánh được.

### `eval.py` — nơi con số ra đời (đã nói ở Phần 11.4)

Bổ sung phần chưa nhắc:

- `resolve_ckpt(explicit, name)`: thử theo thứ tự `explicit` → `/raw/checkpoints` →
  `/ckpt/pretrained`. Nhờ vậy cùng một lệnh chạy được trên máy sạch và trên máy đã
  có checkpoint.
- `iter_images(archive)`: đọc **trực tiếp từ tar** (`tarfile` streaming) — không giải
  nén ra đĩa, vì 3,25 GB giải nén sẽ vượt dung lượng volume.
- `pairs_from_raw(root, split)`: dò cặp `(ảnh gốc, DIRE map)` theo cấu trúc thư mục;
  `filter_pairs` + `tier_paths` giới hạn theo tier; `expected_n_pairs` cho biết
  **trước khi chạy** sẽ chấm bao nhiêu cặp (dùng để bắt lỗi đường dẫn sớm).
- `_roc_points`, `_tpr_at_fpr`: cài ROC bằng tay để báo `TPR@FPR=1%`, thứ mà
  sklearn không cho trực tiếp.
- `collect_results` / `collect_reproduction` / `compare_to_paper` /
  `render_report`: gom CSV từ nhiều lần chạy, in bảng Δ so với `paper_table2.tsv`.
  Đây là chỗ **duy nhất** trộn số của mình với số của bài báo, và nó in Δ chứ không
  in lại số bài báo như thể của mình.

### `api.py` — mặt tiền HTTP, có khoá

- `api_image(weights)`: `gpu_image` + fastapi/uvicorn, có thể **nhúng** checkpoint
  vào image (`BAKED_WEIGHTS`) hoặc đọc từ volume (`VOLUME_WEIGHTS`).
- `_endpoint(...)`: decorator gói `@modal.fastapi_endpoint` để mọi endpoint có cùng
  `max_containers=2`, `scaledown_window=120` — tức **trần chi phí** nằm trong code,
  không trong tài liệu.
- `check_key(given, expected)`: so khoá bằng `hmac.compare_digest` (so sánh thời
  gian hằng, chống timing attack). Trả `(ok, status)`.
- `_cached_detector()`: nạp một lần mỗi container, giữ trong bộ nhớ — cold start
  ~1 s với checkpoint nhúng.
- `detect(payload)`: nhận base64, chạy `Detector.score_pils`, trả
  `{p_fake, label, preproc, n_out, threshold}`. Trả kèm `preproc` để client biết
  mình đang gọi phiên bản pipeline nào.
- `healthz()`: trả `auth` bật/tắt và tên checkpoint — dùng để kiểm triển khai mà
  không cần gửi ảnh.

### `compute_dire.py` — sinh DIRE (đã nói kỹ ở Phần 11.3)

Bổ sung phần chưa nhắc: `_clamp_timeout` (60–3600 s), `_clamp_containers` (≤16),
`estimate_cost`, `preflight_cost` (chặn **cả** chi phí kỳ vọng **và** chi phí xấu
nhất `containers × timeout`), `list_images` (chạy trên CPU, chỉ liệt kê),
`make_gallery` (ghép ảnh gốc / tái tạo / DIRE).

### `train.py` — chưa cài đặt, và điều đó được ghi rõ

`main(exp, resume)` hiện `raise NotImplementedError("FR-8: …")` kèm 5 bước TODO cụ
thể. Vì sao để vậy thay vì viết vội: một vòng train **gần đúng** sẽ tạo ra checkpoint
mà không ai biết nó lệch chỗ nào, rồi mọi con số sau đó đều nhiễm. `NotImplementedError`
là trạng thái trung thực.

### `demo_train.py` — train bản demo, có trần chi phí

Huấn luyện **thu nhỏ** để chứng minh đường ống chạy được: 2 000 ảnh, 5 epoch, L4,
`MAX_MINUTES = 15`, và nhãn `DEMO_LABEL = "demo — KHÔNG phải số của paper"`. Các
hằng số lấy từ `DefaultConfigs` của tác giả (`seed=3407`, `arch=resnet50`,
`N_OUT=1`, `batch=64`, `lr=1e-4`) để ít nhất **siêu tham số** khớp. Hàm `_eval()`
tính ACC trên tập val nhỏ. Điểm quan trọng về liêm chính: kết quả của module này
**không được** dùng để tuyên bố tái lập.

### `manifest.py` — nguồn sự thật về "có những file gì"

- `class File` (remote path, note, local name) và `class DirSpec` (một thư mục phải
  liệt kê động).
- `_LSUN_TEST_GENS`, `_LSUN_IMAGENET_TEST_GENS`, `_CELEBAHQ_TEST_GENS`,
  `_AEROBLADE_GENS`, `GALLERY_DIRS`: danh sách generator cho từng nhánh. Viết thành
  hằng số vì ghép chuỗi `{gen}.tar.gz` bằng vòng lặp **đã sai** nhiều lần
  (`midjourney→midjouney`, `stylegan→stylegan_offical`, `sdv1→sdv1_new`).
- `class Policy`: đọc `tiers.yaml`, có `_POLICY_FALLBACK` khi thiếu pyyaml.
- `is_never(remote, allow=False)`: chặn 8 tar `*ddim*`, `lsun_bedroom.pt` (2,5 GB),
  `celebahq/`, `rawx*`. Cờ mở chỉ qua `File.allow_never`.
- `resolve/expand/entries`: tier → danh sách file, kèm dung lượng **ước lượng**.
  Lưu ý: `manifest.File` **không** có `size`, nên đừng tính tổng dung lượng từ nó.

### `fetchcore.py` — kiểm tra tính toàn vẹn, không tải gì

Đây là module "phòng thủ" cho dữ liệu tải từ Baidu:

- `_sha256`, `_md5`: băm theo luồng 1 MiB (file 2,5 GB không nạp vào RAM).
- `_parse_ls_estimates`, `_parse_ls_long`: đọc đầu ra `ls` của BaiduPCS-Go để biết
  dung lượng **trước khi** tải.
- `_demangle`, `_demangle_list`: sửa lỗi Git Bash (MSYS2) đổi `/raw` thành
  `C:/Program Files/Git/raw`.
- `_validate_ex`, `_validate`, `_tar_ok`: mở thử tar, đọc tới `cap=200_000` entry;
  `Verdict` phân biệt "ok" / "zip rỗng" / "rar" / "hỏng". Nhờ vậy một file `.tar.gz`
  thực chất là `.zip` (Baidu hay đóng gói sai) bị phát hiện **trước** khi tốn 12 giờ.

### `download.py` — tải trên Modal, có ngân sách thời gian

Ổ đĩa và mạng của container Modal là nơi duy nhất tải đủ 61 GB. Module này:

- `_baidu`, `_baidu_login`: gọi BaiduPCS-Go bằng `subprocess`.
- `ORCHESTRATOR_TIMEOUT_S = 24h`, `ORCHESTRATOR_BUDGET_S = 20h`, `_MAX_WAVES = 40`,
  `_MIN_WAVE_S = 600`: chia việc thành nhiều "đợt", mỗi đợt có trần thời gian, để
  job không bị kill giữa đường mà mất tiến độ.
- `_Stalled`, `_POLL_S = 10`, `_request_stop`: dừng một attempt khi 15 phút không
  thêm byte, rồi thử lại (resume), thay vì treo tới hết giờ.
- `_attempt_timeout(started, now)`: tính thời gian còn lại cho attempt hiện tại.
- `_tree_lines`, `_flat_lines`, `AUDIT_DIRS`: in cây thư mục để **audit** sau khi
  tải — so số file với manifest.

### `local_fetch.py` — tải trên máy người dùng

Cùng mục đích nhưng chạy ở máy nhà (nhanh hơn Baidu nhiều, theo yêu cầu của dự án):

- `DEFAULT_DEST = os.environ.get("DIRE_LOCAL_ROOT", "data/raw")` — **không**
  hard-code ổ đĩa (yêu cầu của người dùng).
- `_download_with_watchdog`, `POLL_S`, `DEFAULT_STALL_S = 900`, `DEFAULT_ATTEMPTS = 5`:
  cùng cơ chế "không tiến bộ thì bỏ attempt, resume".
- `_load_state` / `_save_state` / `_append_manifest`: ghi `.state/*.json` và
  `_manifest.tsv` để lần chạy sau biết đã có gì.
- `ALLOW_NEVER_AREAS = ("recons/",)`: vùng được phép mở khoá có kiểm soát.
- `_untouched(before, staging, got)`: kiểm file tải ra **không** bị Baidu cắt cụt.

### `upload.py` — đẩy lên Volume, có kế hoạch trước

- `plan_upload(files, dest)` / `check_ready(files, remote_files)`: so file cần đẩy
  với file đã có ⇒ in kế hoạch, **không** đẩy lại cái đã có (tôn trọng "không xoá
  byte đã tải").
- `put_command`, `put_supports_force(modal_bin)`: dò xem bản `modal` có cờ `--force`
  hay không (đã đổi giữa các phiên bản) thay vì giả định.
- `internal_area_of`, `INTERNAL_AREAS = (".staging", ".bench", ".state")`:
  `clean_tree` chỉ xoá được trong các vùng nội bộ này — hàng rào chống xoá nhầm dữ liệu.
- `remote_index(root)`: đọc chỉ mục Volume một lần, dùng lại cho nhiều quyết định.

### `pack.py` — tar → shard WebDataset

`SHARD_SIZE = 2000` ảnh/shard. `pack_one(src_rel, split)` giải nén một tar, đóng
gói lại thành shard `.tar` nhỏ (đọc tuần tự, tốt cho train), `main(tiers)` chạy hàng
loạt, `stats()` in số shard. Vì sao cần: train đọc 40k ảnh từ **một** tar 200 MB sẽ
ngẫu nhiên hoá tệ; đọc từ nhiều shard nhỏ cho phép shuffle theo shard.

### `archive.py` — mirror công khai

- `push_hf` / `verify_hf` / `push_kaggle` / `push_r2` / `pull_r2`: bốn đích, một
  giao ước (attribution + sha256 + takedown). `verify_hf` so **sha256 từng file**
  giữa local và Hub — không tin `commit` thành công là đủ.
- `make_manifest(out)`: sinh `_manifest_sha256.tsv`, nguồn sự thật để kiểm mirror.
- `_optional_secret`, `_r2_client`: R2/S3 client chỉ dựng khi có secret, để module
  import được trong môi trường không có quyền.

### `bench.py` — đo, không đoán

Bốn nhóm: đọc Volume tuần tự (`volume_seq_read`), đọc nhiều file nhỏ
(`volume_small_read`), ghi (`volume_write`), R2 qua API và qua mount
(`r2_api_io`, `r2_mount_seq_read`), và **thông lượng DIRE**
(`dire_throughput(n, steps, batch, params_m=552)`) dùng một U-Net giả cùng cỡ tham
số để đo `sec/step` — chính là giả định `sec_per_step = 0,35` mà mọi ước lượng chi
phí dựa vào, và nó **chưa được đo**. Module này là công cụ để biến giả định đó thành
số đo.

## 13.4 `vendor/` — mã của tác giả, chép nguyên

`vendor/` không phải code của dự án; nó là bản sao có kiểm soát của mã tác giả, để
kho này **không** phụ thuộc `third_party/DIRE` (thư mục đó rỗng).

| File | Nội dung | Điểm cần biết |
|---|---|---|
| `utils/config.py` | `class DefaultConfigs(ABC)`: toàn bộ siêu tham số | `warmup=False`, `blur_prob=0`, `jpg_prob=0`, `nepoch=400`, `lr=1e-4`, `batch_size=64`, `earlystop_epoch=5`, `seed=3407`. Có `str2bool` và `args_list2dict` cho kiểu gọi `key value` của `train.sh` |
| `utils/utils.py` | `str2bool`, `to_cuda`, `HiddenPrints`, `Logger`, `get_network`, `pad_img_to_square` | `get_network` mới là chỗ tạo `fc=Linear(2048,1)`; `pad_img_to_square` đệm ảnh không vuông **trước** khi cắt |
| `utils/warmup.py` | `class GradualWarmupScheduler(_LRScheduler)` | Bọc một scheduler khác, tăng LR tuyến tính trong `total_epoch`. **Không được dùng** vì `warmup=False` |
| `utils/trainer.py` | `class BaseModel`, `class Trainer` | `BaseModel` giữ `save_networks`/`load_networks`/`eval`/`test`; `Trainer` thêm `set_input`, `forward`, `get_loss`, `optimize_parameters`, `adjust_learning_rate(min_lr=1e-6)`. `init_weights` **không có chỗ gọi** (mã chết) |
| `utils/eval.py` | `get_val_cfg(cfg, split, copy=True)`, `validate(model, cfg)` | `copy=True` là bản `deepcopy` cứu khỏi lỗi "validate trên tập train"; `validate` trả `r_acc`, `f_acc`, `ACC`, `AP` |
| `networks/resnet.py` | `conv3x3`, `conv1x1`, `BasicBlock`, `Bottleneck`, `ResNet`, `resnet18…resnet152` | `model_zoo` URL cũ (không phải torchvision mới); `ResNet.forward(x, *args)` nhận thêm `meta`; `zero_init_residual=False` nên nhánh zero-init **không** chạy |

## 13.5 Những file "nhỏ" nhưng có vai trò

| File | Vai trò | Nếu hỏng thì sao |
|---|---|---|
| `src/dire_modal/__init__.py` | export công khai của package | import sai tên ⇒ `modal run -m` không tìm thấy hàm |
| `src/dire_modal/paper_table2.tsv` | 32 ô Table 2 + 4 hàng avg | `compare_to_paper` không có gì để so; test `test_paper_table.py` kiểm avg khớp 8 ô |
| `src/dire_modal/tiers.yaml` | định nghĩa tier (`e2min`, `e2`, `ckpts`…) và policy `NEVER` | `manifest.load_policy` rơi về `_POLICY_FALLBACK`, có thể **cho phép** file lẽ ra bị chặn |
| `src/dire_modal/demo_train.yaml` | cấu hình cho `demo_train` | demo chạy với mặc định, không rõ đã đổi gì |
| `platforms/hf-spaces/app.py` | UI Gradio: `detect(image)`, `visualize(name)`, `_gallery_items()` | mặt tiền demo công khai mất |
| `platforms/site/build.py` | đọc repo → `data.json` + `index.html` | trang web tự bịa số (module này ghi số **ở đây**, không ở HTML, để trang không thể bịa) |

# Phần 14 — Hạ tầng: script, nền tảng, công cụ

## 14.1 `scripts/` — đúng hai quy trình

Theo yêu cầu của dự án, `scripts/` chỉ có **hai** quy trình thực nghiệm; mọi thứ
khác là tiện ích.

| Script | Là quy trình? | Làm gì |
|---|---|---|
| `01_repro.sh` | **có** — tái lập thực nghiệm của tác giả | `--stage data/eval/report/cross`, `--quick` (tier `e2min`, 1 ô, cổng S2), `DETACH=1` để app sống độc lập. Không train gì |
| `02_improve.sh` | **có** — cải thiện **training-free** | Theo ràng buộc C3 của `spec.md`: mọi cải thiện nằm **ngoài** trọng số (hiệu chỉnh, kết hợp điểm, định tuyến, phân tích lỗi). Hai nguồn ý tưởng được ghi rõ nguồn (AEROBLADE [[20]] và các phương pháp training-free khác) |
| `check.sh` | không | Kiểm môi trường: có `modal`, đăng nhập chưa, volume có không |
| `lib.sh` | không | Hàm dùng chung cho hai script trên (`set -euo pipefail`, tắt path conversion của MSYS2) |
| `setup.sh` | không | Tạo secret trên Modal từ `.env`, idempotent |

Vì sao **không** đặt tái lập trong script riêng: đã có giai đoạn có 5–6 script
"repro*" và chúng trôi khỏi nhau. Hai quy trình, hai file, một `lib.sh`.

## 14.2 `platforms/` — nơi code chạy, tách khỏi code tính

| Thư mục | Dùng cho | Ghi chú |
|---|---|---|
| `hf-spaces/` | demo công khai (CPU free) | `app.py` Gradio: tải ảnh → `Detector.score_pils` → xác suất; `visualize(name)` hiện bộ ba ảnh/DIRE có sẵn; `requirements.txt` ghim phiên bản |
| `site/` | trang tĩnh | `build.py` đọc repo (manifest, demo, status) rồi sinh `data.json`; `deck.js` + `style.css` hiển thị; số liệu khai **trong Python**, không trong HTML |
| `kaggle/` | notebook mirror dữ liệu | `notebook.ipynb` — chạy trên Kaggle để tải/mirror, egress $0 |
| `runpod/` | GPU rẻ hơn Modal cho job dài | `run.sh`: clone code, lấy dữ liệu từ R2/HF (**không** copy dữ liệu vào pod vì pod có thể bị thu hồi), chạy job |
| `vast/` | GPU marketplace, có thể bị preempt | `bootstrap.sh`: như trên + nhắc **bắt buộc** checkpoint thường xuyên |

Nguyên tắc chung của cả năm: **dữ liệu không nằm trên máy chạy**. Nó nằm ở R2/HF,
máy chạy chỉ tải phần cần. Nhờ vậy một pod bị thu hồi không làm mất gì.

## 14.3 `tools/` — tiện ích đo lường và xem

| File | Việc |
|---|---|
| `triplet_view.py` | ghép `ảnh gốc / ảnh tái tạo / DIRE` thành một ảnh để nhìn; không import Pillow ở mức module (để test chạy được khi thiếu Pillow) |
| `tree.sh` | in cây thư mục dữ liệu (có xử lý lỗi đường dẫn Windows) |
| `bench_storage.sh`, `bench_baidu_mode.sh` | đo thông lượng đọc/ghi và chế độ tải Baidu |
| `measure_dire.sh` | đo `sec/step` thật cho DIRE (biến giả định 0,35 s thành số đo) |
| `pack_shards.sh` | gọi `dire_modal.pack` từ shell |
| `local_download.sh` | gọi `dire_modal.local_fetch` |
| `free_quota.py` | kiểm hạn mức free tier còn lại |
| `storage_cost.py` | tính chi phí lưu trữ theo dung lượng × thời gian |
| `manim/` | scene Manim minh hoạ pipeline (**không** cài manim trong sandbox; render ở máy người dùng hoặc image riêng) |
| `pdf/` | bộ dựng PDF của tài liệu này (`md2pdf.py` + README) |

## 14.4 Bản đồ "muốn làm X thì đọc file nào"

| Muốn… | Đọc / chạy |
|---|---|
| hiểu một con số ACC/AP đến từ đâu | `dire_core/{preprocess,detector,score,batch}.py` → `dire_modal/eval.py::score_pair` → `metrics` |
| chấm một ảnh của mình | `dire detect anh.jpg --weights …` (CLI) hoặc `Detector.score_pils` |
| biết mình đang ở mức tái lập nào | `dire_core/provenance.py` + `run.json` |
| tải dữ liệu | `manifest.py` (danh sách) → `local_fetch.py` (máy nhà) hoặc `download.py` (Modal) → `fetchcore.py` (kiểm) |
| đẩy lên Volume | `upload.py::plan_upload` rồi `put` |
| mirror công khai | `archive.py::push_hf` / `push_kaggle` / `push_r2`, sau đó `verify_hf` |
| đo thông lượng / `sec/step` | `bench.py` hoặc `tools/measure_dire.sh` |
| sửa một tham số train | `vendor/utils/config.py` (nguồn) → `demo_train.py` (bản demo) |
| dựng lại tài liệu này | `tools/pdf/README.md` |
