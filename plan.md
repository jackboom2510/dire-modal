# plan.md — đang ở đâu, làm gì tiếp

> Cập nhật lần cuối: 2026-10-25. Cập nhật file này mỗi khi merge xong một việc.
> Bối cảnh và quy tắc: [AGENT.md](AGENT.md) · [instructions.md](instructions.md).

## Mục tiêu chi phối

Có kết quả DIRE trên dữ liệu thật với **0đ compute** (free tier), trên máy nhẹ:
local chỉ điều khiển và verify, Modal làm mọi việc nặng. Ưu tiên số 1 là **có
kết quả nhanh nhất**, không phải pipeline đầy đủ.

Phạm vi đã chốt: **Tier A** — eval trên checkpoint có sẵn, **không train lại**.
47 GB (77%) dữ liệu trong cây Baidu là train, nằm ngoài phạm vi.

## Đường chạy hiện tại (chốt 2026-10-25)

Không train ResNet50. Nạp `lsun_adm.pth` của tác giả rồi eval:

`scripts/` giờ chỉ có **đúng 2 quy trình** (xem `scripts/README.md`):

```bash
# QUY TRÌNH 1 — tái lập Table 2 (4 detector × 8 generator = 32 ô)
bash scripts/01_repro.sh --quick   # e2min: 3 file ~0.40 GB ~1.5 h → 1 ô + cổng S2
bash scripts/01_repro.sh           # e2,ckpts: 22 file ~3.25 GB ~12.4 h → cả 32 ô
bash scripts/01_repro.sh --stage report   # Δ từng ô so với src/dire_modal/paper_table2.tsv

# QUY TRÌNH 2 — cải thiện (training-free; AEROBLADE + đề xuất của dự án)
bash scripts/02_improve.sh --stage plan       # lộ trình FR-2…FR-6
bash scripts/02_improve.sh --stage baseline   # điểm DIRE trên val (e5) + test (e2)
```

`download::main` **tự lặp vòng** trong một lần gọi (ngân sách `ORCHESTRATOR_BUDGET_S = 20 h`,
tối đa `_MAX_WAVES = 40`): sau mỗi vòng nó tính lại file còn thiếu **từ đĩa**, và
dừng nếu một vòng không xong thêm file nào (chốt chống đốt 20 h khi BDUSS hết hạn).
Vì thế không còn phải gõ lại lệnh ~9 lần cho `e2,e4`.

`eval::main` tự tìm checkpoint: `/raw/checkpoints/lsun_adm.pth` trước, rồi
`/ckpt/pretrained/lsun_adm.pth`; `--ckpt` để chỉ định tay. Không tìm thấy thì nó
in đúng lệnh cần chạy.

## Đã xong

| # | Việc | Kết quả đo được |
|---|---|---|
| PR #1 | So sánh Volume/R2/Local + IaaS, đường 0đ, mirror HF/Kaggle | `tools/free_quota.py`, `tools/storage_cost.py` |
| PR #2 | Vá 6 lỗi tải Baidu + tái cấu trúc tier theo mục tiêu | 21/154 file; mặc định `e2,e4` = 22 file / 3.39 GB |
| PR #3 | Khớp inventory với audit thật | 4 lỗi đường dẫn + 39 file mồ côi |
| PR #4 | `TypeError: 'Function' object is not callable` | `_fetch_one_impl` tách khỏi decorator |
| PR #5 | Chẩn đoán định dạng + dừng khi lỗi tất định | hết 19 phút đốt vô ích |
| PR #6 | tar không nén + so từng ảnh thay vì sha256 container | phát hiện dữ liệu thật là tar không nén |
| PR #7 | Guard chi phí GPU | trần $2 400 → **$5** |
| PR #8 | PEP 563 làm `compute_dire` không import được | 51 test; cổng 0 bằng modal thật |
| PR #9 *(đang mở)* | Staging trong Volume + rename thay vì copy + cache kiểm tra | resume xuyên container; bỏ 6.8 GB đọc lại/lần |
| PR #9 *(đang mở)* | Bộ tài liệu agent: `AGENT.md`, `instructions.md`, `plan.md`, `skills.md`, `CLAUDE.md` | 10 test canh link/entrypoint/đường dẫn trong tài liệu |
| PR #9 *(đã merge)* | `spec.md` — SPEC-001 theo quy trình SDD: 7 yêu cầu `FR-*`, 6 mốc `M0–M5` | 10 test (thêm 2 test canh `PLANNED_ENTRYPOINTS`) |
| PR #10 | Ngân sách thời gian + watchdog 0B/s + manifest bền từng file | 5 test mới; tái hiện đúng `FunctionTimeoutError` 18 000s từ log thật |
| PR #11 | Lọc file đã có trước `.map()` · `subprocess_timeout` 5400→1800 · `bench_concurrency` · **port `eval.py` (M1 của SPEC-001)** · sửa `pack` làm mất nhánh | 117 test; AP/TPR khớp sklearn trên 500 ca ngẫu nhiên |
| PR #12 | Tái lập pre-trained: tier `e2min` · `--paths` · tự lặp vòng 20 h · `--mode` BaiduPCS-Go · `eval::main` tự tìm ckpt + in CỔNG S2 · `download::from_hf` · gộp vào `scripts/01_repro.sh` + `tools/bench_baidu_mode.sh` | 117 test vẫn xanh; đo song song: k=1 76.6 KiB/s vs k=2 75.0 KiB/s ⇒ `max_containers: 1` |
| PR #13 | `download::tree` (cây Volume) + `tools/tree.sh` + sửa test đỏ ngẫu nhiên | 119 test |
| PR này | Gom `scripts/` về **ĐÚNG 2 quy trình** (`01_repro.sh` / `02_improve.sh`) · bảng tham chiếu `paper_table2.tsv` + `eval::report` in Δ · sửa `~32k` → `~17k` · làm rõ `lsun_adm.pth` = detector ≠ weight ADM · tier↔paper | 138 test |

Kết luận dữ liệu quan trọng (chốt bằng `probe_pair` trên dữ liệu thật):

> `dire/test/lsun_bedroom/imagenet/` và `.../lsun_bedroom/` dùng **cùng 1000 ảnh
> real** ⇒ nhánh này là **cross-generator**, KHÔNG phải cross-domain. Không cộng
> nó vào bảng eval như một test domain riêng (sẽ đếm trùng 1000 ảnh).

## Việc tiếp theo (theo thứ tự đề xuất)

### 1. `tools/iaas_cost.py` — file được nhắc nhưng không tồn tại

`docs/iaas-options.md:242` và `:284` hướng dẫn "sửa giá ở `tools/iaas_cost.py`,
một chỗ duy nhất". File đó không có trong `tools/` (chỉ có `free_quota.py`,
`storage_cost.py`).

**Vì sao quan trọng:** tài liệu đang hứa một công cụ không có. Người đọc làm
theo sẽ không tìm thấy chỗ để sửa giá.

**Cách làm (chọn một, ghi lý do vào PR):**
- (a) Viết `iaas_cost.py` — bảng giá GPU theo nhà cung cấp, đọc từ **một**
  `PRICES` dict như `storage_cost.py`, tính chi phí job theo giờ × giá.
- (b) Sửa docs để trỏ về `tools/storage_cost.py` cho phần lưu trữ và nói rõ
  phần GPU nằm trong `docs/iaas-options.md`.

**Xong khi:** không còn tham chiếu chết; `python3 tools/iaas_cost.py --help`
chạy được (nếu chọn a), hoặc docs không còn nhắc file không tồn tại (nếu chọn b).

### 2. `configs/train_lsun_adm.yaml` rỗng 0 byte

Stage train sẽ fail khi đọc. Cần hoặc điền cấu hình thật (khớp
`train.py` + checkpoint tác giả), hoặc xoá file và sửa chỗ tham chiếu.

**Lưu ý:** dự án **không train lại**, và `scripts/` nay không còn script train
(đã xoá `05_train_repro.sh`). Nếu chọn xoá file thì chỉ cần kiểm `README` +
`train.py` để không còn trỏ vào file rỗng.

### 3. Submodule `third_party/DIRE` chưa init

`common.py::gpu_image` clone `ZhendongWang6/DIRE` thẳng trong image nên container
không cần submodule. Nhưng ai clone repo về đọc code gốc sẽ thấy thư mục rỗng.

**Cách làm:** hoặc `git submodule update --init` rồi ghi vào README, hoặc bỏ
`.gitmodules` và ghi rõ "code gốc được clone trong image, không vendored".
Đừng để trạng thái nửa vời.

### 4. Đo `dire_sec_per_image` trên máy thật

`--concurrency` **đã đo xong** (xem mục 8): `k=1` 76.6 KiB/s vs `k=2` 75.0 KiB/s
⇒ `max_containers: 1`. Còn lại `dire_sec_per_image: 5.0` vẫn là giả định:

- chạy `dire_batch --n 20` rồi lấy thời gian thật (sau khi có `e2min`).

**Xong khi:** `dire_sec_per_image` trong `tiers.yaml` là số đo, không phải ước
lượng, và `docs/runbook-free.md` ghi ngày đo.

### 5. Tab AEROBLADE trong HF Spaces

Detector trước (đã có), DIRE-viz sau. Cần VAE từ HF Hub và gallery đã sinh.

### 6. Dọn staging theo tuổi (chưa cấp bách)

`.staging` không tự dọn. Hiện `prune` chỉ **báo** dung lượng và nói rõ đó là
tiến độ, không phải rác. Nếu thêm tính năng xoá tự động thì phải có ngưỡng tuổi
và mặc định là dry-run.

### 7. Thực thi SPEC-001 (`spec.md`)

Spec đã viết xong, **chưa có dòng code nào** của nó. Thứ tự mốc trong spec §12
là ràng buộc, không phải đề xuất: `M2` (FR-2) chặn `M4` (FR-4).

Mỗi mốc ứng với **một stage** của `scripts/02_improve.sh` — stage đã có gọi code
thật, stage chưa có **dừng** và in đúng thiếu gì (không giả vờ đã chạy).

| Mốc | Việc | Test phải viết kèm | Entrypoint phải tạo |
|---|---|---|---|
| M1 ✅ | Port `eval.py`, thêm TPR@5%FPR, ghi CSV + `run.json` | `tests/test_run_json.py`, `tests/test_eval_contract.py` | — (`dire_modal.eval::main` đã có) |
| M2 | `compute_dire` khớp DIRE map của tác giả (tương quan ≥ 0.90) | `tests/test_dire_equivalence.py` | `dire_modal.compute_dire::equivalence` |
| M3 | Bọc AEROBLADE | `tests/test_aero_sign.py` | `dire_modal.aero::main` |
| M4 | Format bias + fusion (chỉ fit trên val) | `tests/test_fusion_no_leak.py` | `dire_modal.fusion::fit` |
| M5 | Robustness + báo cáo tổng | — | — |

**Vì sao 5 test trên chưa tồn tại:** spec viết trước code, và chúng được khai
báo trong `KNOWN_MISSING` của `tests/test_agent_docs.py` để không bị quên. Khi
viết xong một cái thì **xoá khỏi danh sách** — test sẽ đỏ nếu để lại.

**Xong khi:** `pytest tests/ -q` xanh với test mới, và mỗi dòng `results/*.csv`
truy được về một `run.json`.

### 7b. Tái lập Table 2 — quy trình đã xong, SỐ thì chưa

`bash scripts/01_repro.sh` (mặc định `e2,ckpts` = 22 file / 3.25 GB / ~12.4 h)
chạy 4 detector × 8 generator rồi `eval::report` in Δ so với
`src/dire_modal/paper_table2.tsv`. Cần chạy thật để có cột Δ — đó là cổng **S7**.

**Xong khi:** S7 đạt (|ΔACC| ≤ 0.5 ở mọi ô) và số được ghi vào báo cáo.

### 8. ✅ Đã đo băng thông tải Baidu (2026-10-25) — chốt `max_containers: 1`

```bash
modal run -m dire_modal.download::bench_concurrency --tiers e2 --limit 2 --ks 1,2
```

| k (container) | thời gian | tổng | mỗi file |
|---|---|---|---|
| 1 | 5183.8 s | **76.6 KiB/s** | 37.5 KiB/s |
| 2 | 5293.3 s | **75.0 KiB/s** | 37.5 KiB/s |

⇒ `k=2` **KHÔNG nhanh hơn** (chậm hơn 2.1 %, trong sai số). Trần ~77 KiB/s là
của **tài khoản**, không phải của cấu hình. Nên `max_containers: 1`,
`baidu_parallel: 1`; `tiers.yaml` đã ghi kèm bảng này.

Hệ quả về thời gian (ở 76.6 KiB/s): `e2min` 0.40 GB → **~1.5 h** · `e2` 1.94 GB →
**7.4 h** · `e2,e4` 3.39 GB → **12.9 h** · cả cây 61.40 GB → 233.5 h.

**Còn treo:** `--mode locate|pcs|stream` chưa đo. Ba mode là ba API khác nhau của
Baidu, có thể có lịch sử throttle khác nhau. `bash tools/bench_baidu_mode.sh` đo
trên một file 26.57 MB (~6 phút/mode). Đo trên tài khoản **đang bình thường**,
nếu không cả ba mode sẽ cùng cho ≈0 và kết luận sẽ sai.

**Đường thoát thật sự khỏi 77 KiB/s không phải tham số, mà là đổi nguồn:**
mirror một lần (`archive::push_hf`) rồi `download::from_hf` — HF Hub thường cho
vài MB/s, tức nhanh hơn 1–2 bậc.

### 9. M2 của SPEC-001 — chứng minh `compute_dire` khớp map tác giả

M1 (port `eval.py`) đã xong. Việc tiếp theo **không đổi**: `compute_dire.py` vẫn
chưa từng chạy, và nó là điều kiện CHẶN cho FR-4 (format bias) + FR-6
(robustness). Phải có `compute_dire::equivalence` trước khi công bố bất kỳ số nào
từ DIRE map tự tính.

Đường rẻ nhất để có kết quả thật trước M2: `download --tiers e2min` rồi
`eval::main --tiers e2min` trên `lsun_adm.pth` — chỉ ResNet50 inference, không cần
ADM, không cần GPU mạnh, không train gì. `eval::main` đã tự tìm ckpt và in
**CỔNG S2** (`ACC(adm) ≥ 0.99`), nên đây là con số đầu tiên chứng minh đường eval
đúng.

## Việc đã quyết là KHÔNG làm

- **Không train lại** detector (trừ khi có lý do mới được ghi thành ADR).
- **Không dùng Azure** $200 credit (đã bỏ, xem `docs/adr-0002-iaas.md`).
- **Không mirror `recons/`** (44 tar) — bị `max_gb` chặn và không cần cho eval.
- **Không đặt `max_containers` cao** cho tải Baidu: 26 phiên ⇒ `0B/s` hàng loạt.

## Cách biết một việc đã thật sự xong

1. `pytest tests/ -q` xanh, và test mới **fail trên `main` cũ**.
2. `bash scripts/check.sh` xanh (nếu chạm Modal).
3. Docs/`plan.md` đã cập nhật.
4. Trong PR có ghi phần **chưa** làm và số đo thật (nếu có).
