# Notebook — chạy figure theo thứ tự, không viết lại logic vẽ

Bốn notebook, chia theo **thứ tự phụ thuộc dữ liệu** (xem `docs/figures-plan.md`):

| Notebook | Đợt | Cần gì | Chạy được khi |
|---|---|---|---|
| `00_moi_truong.ipynb` | — | repo | luôn (kiểm môi trường, in ra thiếu gì) |
| `01_hinh_schematic.ipynb` | 0 | chỉ code | **luôn** — F3, F2, A1 |
| `02_hinh_R0_tu_dire.ipynb` | 1 | map đã giải nén (+ ảnh đã ingest) | có `dire/` ⇒ **0 đồng GPU** |
| `03_hinh_do_luong.ipynb` | 2 | `scores.npy`/`labels.npy` + manifest | sau một lần eval thật |

## Nguyên tắc

1. **Notebook không chứa logic vẽ.** Mọi ô gọi `tools/figures.py`. Nhờ vậy hình
   trong notebook và hình trong báo cáo **luôn là cùng một hình** — nếu notebook
   tự vẽ lại, hai bản sẽ lệch nhau và không ai biết bản nào đúng.
2. **Ô nào thiếu dữ liệu thì in ra thiếu gì rồi dừng**, không vẽ hình rỗng, không
   điền số mẫu.
3. **Nhãn on the figure**: `R0` map/detector của tác giả · `R1+R2` map của mình ·
   `R1+R2+R3` tất cả của mình · `schematic` hình vẽ, không phải số đo.
4. **Đường dẫn qua biến môi trường**, không sửa trong code:
   `DIRE_FIG_MAPS`, `DIRE_FIG_IMAGES`, `DIRE_FIG_MANIFEST`, `DIRE_FIG_SPLIT`,
   `DIRE_FIG_GENERATOR`, `DIRE_FIG_SCORES`, `DIRE_FIG_LABELS`, `DIRE_FIG_GENS`.
5. **Chạy từ thư mục gốc repo**, kernel Python 3.11+, `pip install matplotlib`
   (matplotlib **không** là phụ thuộc của `dire_zero`).

## Cài

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[zero]" matplotlib jupyter
jupyter lab notebooks/
```

## Đầu ra

`figs/` (không commit `.png` lớn) và `figs/data/<tên>.json` (sidecar: nguồn,
`n`, seed, nhãn, lệnh tái lập — **commit** file này). `figs/README.md` do tool
sinh từ chính các sidecar.
