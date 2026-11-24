# Bộ notebook của DIRE

Các notebook được tách theo mục đích: một nhóm tạo hình cho bài viết, một nhóm trình bày pipeline và so sánh kết quả. Notebook không chạy GPU tự động; các lệnh Modal chỉ được in ra để người vận hành chủ động chạy.

## Hình cho bài viết

| Notebook | Đợt | Cần gì | Chạy được khi |
|---|---|---|---|
| `00_moi_truong.ipynb` | — | repo | luôn (kiểm môi trường, in ra thiếu gì) |
| `01_hinh_schematic.ipynb` | 0 | chỉ code | **luôn** — F3, F2, A1 |
| `02_hinh_R0_tu_dire.ipynb` | 1 | map đã giải nén (+ ảnh đã ingest) | có `dire/` ⇒ **0 đồng GPU** |
| `03_hinh_do_luong.ipynb` | 2 | `scores.npy`/`labels.npy` + manifest | sau một lần eval thật |

## Pipeline và phân tích kết quả

| Notebook | Mục tiêu | Dữ liệu cần |
|---|---|---|
| `04_dire_pipeline.ipynb` | Giải thích flow ảnh → inversion/reconstruction → DIRE map → detector; trình bày snapshot inventory và lệnh tiếp theo | Không cần tải map; inventory trong notebook được ghi nhãn là snapshot do người dùng cung cấp |
| `05_so_sanh_ket_qua.ipynb` | Đọc các ledger train/eval/report của nhiều run, dựng bảng so sánh ACC/AP/AUC và cảnh báo metadata | JSON dưới `runs/rebuild/<run_id>/` |

Notebook 05 nhận `DIRE_RUNS_ROOT` (thư mục chứa các run ID) và `DIRE_RUN_IDS` (ID cách nhau bằng dấu phẩy). Chạy local thì tải JSON từ Volume trước; chạy trong Jupyter có mount Volume thì đặt root tới `/zero/runs/rebuild`.

## Chạy

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\\Scripts\\activate
pip install -e ".[zero]" matplotlib jupyter
jupyter lab notebooks/
```

**Modal:** Modal đã ngừng trình soạn thảo Modal Notebooks; các file `.ipynb` vẫn dùng được trong Jupyter/JupyterLab. Nếu muốn Jupyter chạy trên Modal, tự host Jupyter server qua Modal Tunnel; xem [Modal Jupyter integration](https://modal.com/docs/guide/jupyter-notebooks) và [Tunnel guide](https://modal.com/docs/guide/tunnels). Notebook 04/05 không gắn Volume hay chạy GPU mặc định.

## Nguyên tắc

1. Notebook về hình chỉ gọi `tools/figures.py`, không sao chép logic vẽ.
2. Notebook phân tích chỉ đọc artifact; không cập nhật ledger hoặc sửa Volume.
3. Thiếu artifact thì báo rõ file cần có, không điền metric giả.
4. Đường dẫn dữ liệu/results đặt qua biến môi trường hoặc biến cấu hình đầu notebook, không hardcode ổ đĩa cá nhân.
5. Đầu ra đã lưu trong `.ipynb` phải để trống; không commit kết quả cũ khó truy nguyên.

## Đầu ra hình

`figs/` (không commit `.png` lớn) và `figs/data/<tên>.json` (sidecar: nguồn,
`n`, seed, nhãn, lệnh tái lập — **commit** file này). `figs/README.md` do tool
sinh từ chính các sidecar.
