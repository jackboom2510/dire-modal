# plan.md — đang ở đâu, làm gì tiếp

> Cập nhật lần cuối: 2026-10-24. Cập nhật file này mỗi khi merge xong một việc.
> Bối cảnh và quy tắc: [AGENT.md](AGENT.md) · [instructions.md](instructions.md).

## Mục tiêu chi phối

Có kết quả DIRE trên dữ liệu thật với **0đ compute** (free tier), trên máy nhẹ:
local chỉ điều khiển và verify, Modal làm mọi việc nặng. Ưu tiên số 1 là **có
kết quả nhanh nhất**, không phải pipeline đầy đủ.

Phạm vi đã chốt: **Tier A** — eval trên checkpoint có sẵn, **không train lại**.
47 GB (77%) dữ liệu trong cây Baidu là train, nằm ngoài phạm vi.

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
| PR #9 *(đang mở)* | `spec.md` — SPEC-001 theo quy trình SDD: 7 yêu cầu `FR-*`, 6 mốc `M0–M5` | 10 test (thêm 2 test canh `PLANNED_ENTRYPOINTS`) |

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

**Lưu ý:** dự án **không train lại**. Nếu chọn xoá, kiểm `scripts/05_train_repro.sh`
và `README` để không còn trỏ vào file rỗng.

### 3. Submodule `third_party/DIRE` chưa init

`common.py::gpu_image` clone `ZhendongWang6/DIRE` thẳng trong image nên container
không cần submodule. Nhưng ai clone repo về đọc code gốc sẽ thấy thư mục rỗng.

**Cách làm:** hoặc `git submodule update --init` rồi ghi vào README, hoặc bỏ
`.gitmodules` và ghi rõ "code gốc được clone trong image, không vendored".
Đừng để trạng thái nửa vời.

### 4. Đo `dire_sec_per_image` và `--concurrency` trên máy thật

Hai con số đang là **giả định**, và mọi ước tính tiền scale theo chúng:

- `dire_sec_per_image: 5.0` → chạy `dire_batch --n 20` rồi lấy thời gian thật.
- `--concurrency 2` vs `4` → đo trên cùng một tier. Baidu throttle theo **số
  phiên**, nên tăng song song có thể làm chậm đi.

**Xong khi:** hai con số trong `tiers.yaml` là số đo, không phải ước lượng, và
`docs/runbook-free.md` ghi ngày đo.

### 5. Tab AEROBLADE trong HF Spaces

Detector trước (đã có), DIRE-viz sau. Cần VAE từ HF Hub và gallery đã sinh.

### 6. Dọn staging theo tuổi (chưa cấp bách)

`.staging` không tự dọn. Hiện `prune` chỉ **báo** dung lượng và nói rõ đó là
tiến độ, không phải rác. Nếu thêm tính năng xoá tự động thì phải có ngưỡng tuổi
và mặc định là dry-run.

### 7. Thực thi SPEC-001 (`spec.md`)

Spec đã viết xong, **chưa có dòng code nào** của nó. Thứ tự mốc trong spec §12
là ràng buộc, không phải đề xuất: `M2` (FR-2) chặn `M4` (FR-4).

| Mốc | Việc | Test phải viết kèm | Entrypoint phải tạo |
|---|---|---|---|
| M1 | Port `eval.py`, thêm TPR@5%FPR, ghi CSV + `run.json` | `tests/test_run_json.py`, `tests/test_eval_contract.py` | — (`dire_modal.eval::main` đã có) |
| M2 | `compute_dire` khớp DIRE map của tác giả (tương quan ≥ 0.90) | `tests/test_dire_equivalence.py` | `dire_modal.compute_dire::equivalence` |
| M3 | Bọc AEROBLADE | `tests/test_aero_sign.py` | `dire_modal.aero::main` |
| M4 | Format bias + fusion (chỉ fit trên val) | `tests/test_fusion_no_leak.py` | `dire_modal.fusion::fit` |
| M5 | Robustness + báo cáo tổng | — | — |

**Vì sao 5 test trên chưa tồn tại:** spec viết trước code, và chúng được khai
báo trong `KNOWN_MISSING` của `tests/test_agent_docs.py` để không bị quên. Khi
viết xong một cái thì **xoá khỏi danh sách** — test sẽ đỏ nếu để lại.

**Xong khi:** `pytest tests/ -q` xanh với test mới, và mỗi dòng `results/*.csv`
truy được về một `run.json`.

## Việc đã quyết là KHÔNG làm

- **Không train lại** detector (trừ khi có lý do mới được ghi thành ADR).
- **Không dùng Azure** $200 credit (đã bỏ, xem `docs/adr-0002-iaas.md`).
- **Không mirror `recons/`** (44 tar) — bị `max_gb` chặn và không cần cho eval.
- **Không đặt `max_containers` cao** cho tải Baidu: 26 phiên ⇒ `0B/s` hàng loạt.

## Cách biết một việc đã thật sự xong

1. `pytest tests/ -q` xanh, và test mới **fail trên `main` cũ**.
2. `./scripts/07_check_imports.sh` xanh (nếu chạm Modal).
3. Docs/`plan.md` đã cập nhật.
4. Trong PR có ghi phần **chưa** làm và số đo thật (nếu có).
