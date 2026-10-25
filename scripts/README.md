# scripts/ — đúng HAI quy trình thực nghiệm

Thư mục này cố ý chỉ có **hai** quy trình. Trước đây nó là dãy `00…10` rời rạc:
mỗi lần thêm một bước lại thêm một file, người đọc phải tự ghép thứ tự, và không
ai nói được "repo này tái lập cái gì". Nay mỗi file ứng với **một câu hỏi**:

| Quy trình | Câu hỏi | File |
|---|---|---|
| **1 — Tái lập** | "Chúng ta có tái tạo được số của paper không?" | `01_repro.sh` |
| **2 — Cải thiện** | "Training-free, ta làm tốt hơn DIRE được không?" | `02_improve.sh` |

Hai file còn lại **không** phải quy trình thực nghiệm:

| File | Vai trò |
|---|---|
| `setup.sh` | điều kiện tiên quyết, một lần: tạo secret từ `.env` |
| `check.sh` | **CỔNG 0**: import bằng `modal` THẬT + toàn bộ test + pyflakes, 0 credit |
| `lib.sh` | hàm dùng chung (`say`, `want`, `run_modal`, `missing_stage`…), không chạy trực tiếp |

Tiện ích đo/xem (không sinh ra kết quả khoa học) nằm ở `tools/`:
`tools/tree.sh`, `tools/bench_storage.sh`, `tools/bench_baidu_mode.sh`,
`tools/measure_dire.sh`, `tools/pack_shards.sh`.

---

## Quy trình 1 — `01_repro.sh`

Tái lập khối **DIRE (ours)** của **Table 2**, ICCV 2023 (arXiv:2303.09295v1):
4 detector × 8 generator = **32 ô**, headline `Avg 99.9 ACC / 100 AP`.

Không train gì — nhánh `dire/` đã chứa DIRE map tác giả tính sẵn (ADM-LSUN,
DDIM S=20), nên chỉ cần ResNet-50 forward trên CPU.

```bash
bash scripts/01_repro.sh                  # data → eval → report
bash scripts/01_repro.sh --stage data     # chỉ plan → fetch → verify
bash scripts/01_repro.sh --stage eval     # dữ liệu đã có, chỉ eval 4 detector
bash scripts/01_repro.sh --stage report   # chỉ in Δ so với paper
bash scripts/01_repro.sh --stage cross    # Table 3 (cross-dataset ImageNet)
bash scripts/01_repro.sh --quick          # e2min: 3 file, 1 ô, cổng S2
DETACH=1 bash scripts/01_repro.sh         # app sống độc lập với máy local
```

| tiers | file / GB | ~giờ @76.6 KiB/s | phủ được |
|---|---|---|---|
| `e2min` | 3 / 0.40 | 1.5 | 1 ô (cổng S2) |
| `e2` | 17 / 1.94 | 7.4 | hàng ADM (8 cột) |
| **`e2,ckpts`** | **22 / 3.25** | **12.4** | **cả 32 ô** (mặc định của script) |

Số tham chiếu của paper nằm ở `src/dire_modal/paper_table2.tsv` (có ghi nguồn,
trang, và mapping `ADM → lsun_adm.pth`). `eval::report` in Δ từng ô.

## Quy trình 2 — `02_improve.sh`

Ràng buộc N1/C3: **không train lại**. Nên cải thiện nằm ngoài trọng số: hiệu
chỉnh, kết hợp điểm, định tuyến, phân tích lỗi.

Hai nguồn ý tưởng, ghi rõ nguồn:

- **(A) AEROBLADE** (Ricker et al., CVPR 2024) — training-free, đo lỗi tái tạo
  qua VAE của LDM, lấy **điểm âm**. Mạnh trên ảnh LDM/SD, yếu trên ADM (ADM
  không dùng VAE cùng họ) ⇒ dùng làm **tín hiệu thứ hai**, không thay DIRE.
- **(B) Ý tưởng của dự án** — rank-fusion DIRE ⊕ AEROBLADE với trọng số fit trên
  **val** (FR-5), định lượng format bias JPEG/PNG (FR-4), đường cong robustness
  (FR-6), định tuyến theo họ generator.

```bash
bash scripts/02_improve.sh                   # baseline val + in lộ trình
bash scripts/02_improve.sh --stage baseline  # điểm DIRE trên e5 (val) + e2 (test)
bash scripts/02_improve.sh --stage plan      # chỉ in lộ trình FR-2…FR-6
FETCH=1 bash scripts/02_improve.sh --stage baseline   # tải e5 trước rồi eval
```

Hiện **chỉ stage `baseline` có code**. Các stage khác (`equivalence`, `aero`,
`bias`, `fusion`, `robustness`) **dừng** và nói rõ thiếu gì, trỏ về mốc tương
ứng trong `spec.md` §12 — không có stage nào giả vờ đã chạy.

---

## Quy ước

- `set -euo pipefail` mọi file; `bash -n scripts/*.sh` phải sạch.
- Không in "xong" khi chưa đo gì (repo này đã gặp lỗi đó vài lần).
- Mọi con số về thời gian/dung lượng phải là **số đo**, hoặc ghi rõ "ước lượng".
- Thêm một bước vào quy trình ⇒ thêm **stage**, không thêm file thứ ba.
