---
doc_id: HIST-001
title: Snapshot kích thước và phạm vi các run Modal đã lưu trong repo
status: evidence-snapshot
---

# Snapshot kích thước/phạm vi dataset trong các run đã lưu

> **Phạm vi:** tổng hợp các ledger hiện có trong checkout, không phải truy vấn trực tiếp Modal. Số ở đây là định hướng để tìm artifact; JSON bên dưới mới là nguồn chi tiết.
>
> **Không so sánh điểm giữa các hàng như cùng protocol.** Các nhóm run khác nhau về init, split, mẫu, generator scope, số seed và quan hệ giữa checkpoint với evaluation.

## 1. Inventory nguồn được khai báo

`modal-runs/runs/report.json` và `modal-runs/runs/rebuild/rebuild-2026-10-14/report.json` ghi:

- Dataset: `lsun_bedroom`; split names: `train`, `val`, `test`.
- Inventory được khai báo: **42.000 ảnh thật, 170.492 ảnh giả**, với 15 nhãn generator trong metadata.
- `fake_source: author`; đây là ảnh/nguồn theo metadata tác giả, không tự động đồng nghĩa ảnh do dự án tự sinh.

Các số inventory ở trên **không phải** số ảnh đã biến thành map, cũng không phải số ví dụ mà một detector cụ thể đã thấy. Một số generator chỉ có ở test hoặc thiếu split ở inventory; đọc `data.warnings` trong report trước khi mô tả phạm vi.

## 2. Nhóm `runs/` ban đầu — nhiều train/eval ledger, protocol không đồng nhất

Nguồn chính:

- Summary: `modal-runs/runs/report.json`, `train.json`, `eval.json`.
- Chi tiết từng detector: `modal-runs/runs/train/gen-*/train.json`.
- Eval ADM riêng: `modal-runs/runs/eval/gen-adm/eval.json`.

### 2.1 Map và train/eval đã ghi

`report.json` ghi `dire.n = 5.900`, `steps = 20`. Cùng report ghi `n_new = 5.869` và `n_reused = 2.431`; đó là bộ đếm hoạt động cộng dồn, không phải hai tập disjoint để cộng ra số map duy nhất. Không suy rằng inventory 212.492 ảnh đã được xử lý hết.

Các run `gen-adm`, `gen-iddpm`, `gen-dalle2`, `gen-ddpm`, `gen-diff-projectedgan`, `gen-diff-stylegan` và `gen-if` ghi `init=scratch`, `split_mode=hash`, seed 3407. `gen-fake` mang nhãn/provenance và split riêng. Số lượng theo generator khác nhau:

| Detector run | Train / val / test trong split | Test thật / giả | Ghi chú |
|---|---:|---:|---|
| `gen-adm` | 1.360 / 240 / 400 | 202 / 198 | Chi tiết: `runs/train/gen-adm/train.json`. |
| `gen-iddpm` | 1.360 / 240 / 400 | 209 / 191 | Run riêng; không gộp với ADM. |
| `gen-dalle2` | 1.020 / 180 / 300 | 197 / 103 | Test lệch lớp. |
| `gen-ddpm` | 1.020 / 180 / 300 | 199 / 101 | Test lệch lớp. |
| `gen-diff-projectedgan` | 1.020 / 180 / 300 | 207 / 93 | Test lệch lớp. |
| `gen-diff-stylegan` | 1.020 / 180 / 300 | 197 / 103 | Test lệch lớp. |
| `gen-if` | 1.020 / 180 / 300 | 197 / 103 | Test lệch lớp. |
| `gen-fake` | 595 / 105 / 700 | 500 / 200 | Ledger mang nhãn `R1+R2+R3`; provenance khác nhóm author-source. |

Đây là số trong các sidecar train hiện có; không suy số split cho generator chưa có sidecar hoặc coi từng run là dùng chung mẫu.

### 2.2 Eval ADM riêng

`modal-runs/runs/eval/gen-adm/eval.json` ghi test **1.000 ảnh (500 thật + 500 giả)** và validation `n_val=150`. Đây là artifact eval riêng; đừng mặc định nó là evaluation cùng checkpoint/split với `runs/train/gen-adm/train.json` chỉ vì cùng tên generator.

## 3. Rebuild riêng `rebuild-2026-10-14`

Nguồn: `modal-runs/runs/rebuild/rebuild-2026-10-14/`.

- `report.json`: **23.868 map phân biệt**, `steps=20`, `n_runs=6`.
- Có bốn detector ledger (`gen-adm`, `gen-iddpm`, `gen-pndm`, `gen-stylegan`); mỗi cái ghi **2.000 train + 2.000 val + 2.000 test**.
- Mỗi test của các detector trên là **1.000 thật + 1.000 giả**; split mode `author`, init `scratch`, seed 3407.
- Các trường `n_real=3.000`, `n_fake=3.000` ở cấp train ledger là tổng ví dụ train/val/test của mỗi model run; **không** phải 6.000 test examples.
- `report.json` có cảnh báo eval ADM và iDDPM được tạo từ checkpoint hash khác checkpoint `.pth` hiện lưu. Không gọi các điểm đó là kết quả của checkpoint hiện tại nếu chưa evaluate lại.

Run rebuild lớn hơn một số pilot/legacy split nhưng vẫn chỉ xử lý một phần inventory nguồn; không gọi đây là toàn bộ dataset.

## 4. Pilot Table 4 — subset riêng

Nguồn: `modal-runs/experiments/experiments/table4/table4-lsun-pilot/subsets/per-cell-000200/table4.json` và các `S*/_protocol.json`.

- Các mức `S=5/10/20`: mỗi split có **400 ảnh (200 thật + 200 giả)** cho train, validation và test.
- ImageNet init; author split; seed 3407; từng S là một detector riêng, một seed.
- Cùng định danh pilot/subset không biến nó thành full-data run. Không kết luận tác động nhân quả của S chỉ từ ba model này.

## 5. Mốc so sánh và giới hạn snapshot

| Nếu muốn so… | Cần kiểm tra tối thiểu |
|---|---|
| Subset với full-size run | Tập mẫu/manifest, tỷ lệ lớp, split mode, generator, init, seed, số epoch/early stopping, threshold policy; thay đổi đồng thời nhiều mục thì không cô lập được ảnh hưởng kích thước. |
| Hai lần eval | Checkpoint hash, score file và label/ID theo ảnh, test IDs, threshold; cùng tên generator không chứng minh cùng model hoặc cùng test set. |
| Số map | Phân biệt file unique, file mới trong lượt, file reuse, retries và tổng lượt job; chỉ dùng manifest/hash để kết luận unique. |
| “Run trên Modal” | JSON repository chứng minh artifact đã được lưu trong repo; nó không nhất thiết bao quát các job bị xóa, chưa tải về hoặc chưa commit. Muốn complete cloud history cần Modal dashboard/API/logs của tài khoản. |

### Timestamp note

`report.json` embeds `data.created = 2026-11-17T03:17:09Z`, trong khi rebuild directory mang hậu tố `2026-10-14`. Không dùng hai giá trị đó để xếp thứ tự lịch sử khi chưa xác minh clock/export metadata. Run ID được dùng ở đây làm nhãn folder, không được xem là timestamp đã kiểm toán.

## 6. Artifact paths

- `modal-runs/runs/report.json`
- `modal-runs/runs/train.json` và `modal-runs/runs/train/gen-*/train.json`
- `modal-runs/runs/eval.json` và `modal-runs/runs/eval/gen-adm/eval.json`
- `modal-runs/runs/rebuild/rebuild-2026-10-14/{report,train,eval}.json`
- `modal-runs/experiments/experiments/table4/table4-lsun-pilot/subsets/per-cell-000200/table4.json`

Đây là snapshot theo checkout. Khi dùng ở một commit khác, đối chiếu `git rev-parse HEAD` và mở lại các file nguồn; không dựa riêng vào bản tóm tắt này.
