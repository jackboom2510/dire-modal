---
doc_id: HIST-090
title: Mẫu ghi lịch sử một run/campaign
status: template
---

# Mẫu chuẩn ghi lịch sử run

> Sao chép file này thành `HIST-###-<slug>.md` trong `docs/history/` sau khi run có artifact được xác minh. Xóa hướng dẫn trong ngoặc vuông, nhưng **giữ nguyên `not recorded` nếu thiếu bằng chứng**. Không dùng template để tự tạo lại hoặc sửa metric.

```yaml
---
doc_id: HIST-###
title: "[Tên ngắn và mục tiêu run]"
status: verified | partial | blocked | superseded
run_id: "[run_id chính xác từ artifact; not recorded nếu không có]"
run_group: "[campaign nếu có]"
source_artifact: "[đường dẫn report/ledger gốc trong repo]"
code_commit: "[SHA code; not recorded nếu không có]"
verified_on: "YYYY-MM-DD"
---
```

## Tóm tắt

- **Câu hỏi run trả lời:** [một câu]
- **Trạng thái:** [verified/partial/blocked/superseded; lý do]
- **Platform:** [Modal app/function/volume; region/GPU nếu artifact xác nhận]
- **Nguồn số:** [`report.json`, `train.json`, `eval.json`, score file, dashboard export…]

## 1. Nguồn và phạm vi dữ liệu

| Trường | Giá trị | Phân loại bằng chứng / nguồn |
|---|---|---|
| Dataset/source/version | | declared / observed / unknown |
| Generator/real classes in scope | | |
| Split names/source split | | |
| Total inventory — real | | Không nhầm với số đã map hoặc đã train. |
| Total inventory — fake | | Nêu nguồn tạo/nguồn tải và provenance. |
| Subset selection rule / manifest | | Path, manifest ID/hash, seed. |
| Image IDs overlap/exclusion | | Nêu cách ngăn leakage hoặc `not verified`. |
| Resolution / crop / format | | Ghi theo protocol. |

## 2. Số lượng theo stage

Ghi từng stage riêng; đếm mẫu thành công và loại bỏ/reject, không chỉ giới hạn `--n`.

| Stage | Unique real | Unique fake | Total unique | Attempted / reused / failed | Artifact chứng minh |
|---|---:|---:|---:|---|---|
| Inventory eligible | | | | | |
| Map generation | | | | | |
| Train | | | | | |
| Validation | | | | | |
| Test | | | | | |
| Evaluation by generator/condition | | | | | |

> Chú thích counters: định nghĩa rõ `n_new`, `n_reused`, retries và phạm vi cộng dồn. Không cộng counters lặp qua nhiều run để gọi là distinct count.

## 3. Protocol/model

- **Representation / map source:** [author-provided, recomputed, cache path/hash]
- **Model/checkpoint:** [name + hash]; init: [scratch/ImageNet/other]
- **Train/eval split:** [mode, source, seed, IDs/manifest]
- **Per-model train / val / test counts:** [kèm class counts]
- **Generator/condition coverage:** [đã chạy / chưa chạy]
- **Steps / resolution / preprocessing:**
- **Hyperparameters / stopping rule:**
- **Threshold selection:** [val source; frozen before test?]
- **Seeds / repeats:**
- **Changes relative to comparison run:**

## 4. Outputs và kiểm chứng

| Artifact | Path / URI | Hash | Đã xác minh bằng |
|---|---|---|---|
| Map manifest | | | |
| Checkpoint | | | |
| Scores / labels | | | |
| Summary ledger | | | |
| Logs / Modal job reference | | | |

- **Validation command(s):**
- **Result:** [pass/fail; exact evidence]
- **Warnings/stale artifacts:**
- **Known caveats / claims not allowed:**

## 5. Quan hệ với run khác

- **Có thể so trực tiếp với:** [run IDs, chỉ khi protocol, sample IDs và checkpoint phù hợp]
- **Không thể gộp với:** [run IDs + lý do: init, split, size, source, seed, threshold, model…]
- **Tiếp theo:** [action có thể kiểm chứng hoặc `none`]

## Checklist trước khi đánh dấu `verified`

- [ ] Run ID, commit, source dataset và artifact path có bằng chứng.
- [ ] Inventory, unique processed maps và train/val/test counts được ghi tách biệt.
- [ ] Class counts, generator coverage, split mode/seed/IDs có ghi nhận.
- [ ] Checkpoint hash khớp với evaluation score ledger.
- [ ] Threshold chỉ chọn trên validation; evaluation/test không dùng để tune.
- [ ] Run claims không vượt quá nguồn dữ liệu và provenance.
- [ ] Các lần chạy riêng vẫn tách record; chỉ tổng hợp khi protocol so sánh được.
- [ ] Không đưa bí mật/token hoặc đường dẫn riêng tư nhạy cảm vào repo.
