# AGENT.md — điểm vào cho mọi agent

> Đọc file này trước, rồi mới mở code. Cập nhật lần cuối: 2026-10-25.
> Các file còn lại: [instructions.md](instructions.md) (cách làm việc) ·
> [plan.md](plan.md) (đang ở đâu, làm gì tiếp) · [skills.md](skills.md) (công thức
> cho từng loại việc) · [CLAUDE.md](CLAUDE.md) (ghi chú riêng cho Claude) ·
> [spec.md](spec.md) (đặc tả SPEC-001: tái tạo DIRE + cải thiện theo AEROBLADE).

## Repo này là gì

Pipeline tính **DIRE** (Diffusion Reconstruction Error) để phát hiện ảnh do
diffusion model sinh ra, chạy trên **Modal**, dữ liệu tải từ **Baidu Netdisk**.

Đường chạy đã chốt: **eval checkpoint `lsun_adm.pth` của tác giả, KHÔNG train
lại ResNet50** (`modal run -m dire_modal.download::main --tiers e2min --mode fetch`
→ `modal run -m dire_modal.eval::main --tiers e2min`). Mọi hướng dẫn trong repo
phải bám theo đường này; `train.py` + `scripts/05_train_repro.sh` là tham chiếu,
không phải đường chính.

Ràng buộc chi phối mọi quyết định: **ngân sách compute 0đ, chỉ free tier**.
Máy local là máy nhẹ — chỉ điều khiển và verify, **không** train, **không** giữ
bản dữ liệu gốc. Vì vậy mọi thứ nặng chạy trên Modal, và mọi thiết kế phải trả
lời được "cái này tốn bao nhiêu credit".

## Bản đồ repo

| Đường dẫn | Nội dung |
|---|---|
| `src/dire_modal/download.py` | Tải Baidu → Volume `/raw`. Phần lớn nhất, nhiều cạm bẫy nhất. Có `--paths` (tải đúng danh sách file), **tự lặp vòng 20 h**, `from_hf` (kéo từ mirror HF) |
| `src/dire_modal/manifest.py` | Inventory file + `Policy` + `load_policy()` |
| `src/dire_modal/tiers.yaml` | **Nguồn sự thật của policy.** Không phải `configs/tiers.yaml` |
| `src/dire_modal/compute_dire.py` | DIRE bằng DDIM inversion + guard chi phí GPU |
| `src/dire_modal/pack.py` | `/raw` → WebDataset shards |
| `src/dire_modal/eval.py` | **Đã port (M1)**: eval checkpoint trên mọi cặp (real, generator), ghi CSV + `run.json`, in **CỔNG S2** (`ACC(adm) ≥ 0.99`). `main(ckpt="")` tự tìm `/raw/checkpoints/<name>` rồi `/ckpt/pretrained/<name>`. Đọc **tar gốc trong `/raw`**, không đọc `/shards` (xem `pack.shard_group`) |
| `src/dire_modal/train.py`, `bench.py`, `archive.py` | Stage 3–6 |
| `src/dire_modal/common.py` | `app`, image, Volume, Secret — import từ đây |
| `tools/` | Script thuần stdlib: `free_quota.py`, `storage_cost.py` |
| `scripts/00..09_*.sh` | Cổng tuần tự; `07_check_imports.sh` là cổng 0. `08_repro_minimal.sh` = chuỗi tái lập pre-trained; `09_bench_mode.sh` = đo `--mode` BaiduPCS-Go |
| `tests/` | 117 test; 2 test trong `test_real_modal_import.py` cần `modal` thật (skip nếu không có) |
| `spec.md` | **SPEC-001** — đặc tả tái tạo DIRE + cải thiện theo AEROBLADE. Yêu cầu đánh số `FR-*`, mỗi cái có tiêu chí chấp nhận. Đọc trước khi viết `eval.py`, `aero.py`, `fusion.py` |
| `docs/runbook-free.md` | Runbook đầy đủ, có 10 cổng |
| `docs/adr-000*.md` | Quyết định kiến trúc đã chốt — **đọc trước khi đổi thiết kế** |
| `platforms/hf-spaces/` | Demo Gradio trên HF Spaces (CPU free) |

## Ba sự thật về tải Baidu (đã đo, đừng suy lại)

1. **Trần ~77 KiB/s là của TÀI KHOẢN, không phải của cấu hình.** Đo 2026-10-25:
   `k=1` → 5183.8 s / 76.6 KiB/s; `k=2` → 5293.3 s / 75.0 KiB/s. Thêm container
   **không** nhanh hơn. `max_containers: 1`, `baidu_parallel: 1`.
2. **Tăng `-p`/thread là phản tác dụng.** Upstream BaiduPCS-Go: tài khoản miễn phí
   phải `max_parallel=1`; tăng sẽ bị throttle về ~0 tốc độ trong nhiều giờ tới
   nhiều ngày. Đừng "tối ưu" bằng cách này.
3. **Đường thoát duy nhất là đổi NGUỒN, không phải đổi tham số:** mirror một lần
   (`archive::push_hf`) rồi `download::from_hf` (HF Hub thường vài MB/s).

Hệ quả về thời gian ở 76.6 KiB/s: `e2min` ~1.5 h · `e2` 7.4 h · `e2,e4` 12.9 h ·
cả cây 233.5 h.

## Sáu điều dễ làm hỏng nhất

Đây là các lỗi đã xảy ra thật trong repo này, không phải lý thuyết. Chi tiết và
cách phòng ở [instructions.md](instructions.md).

1. **`from __future__ import annotations` + `modal.parameter` = chết lúc import.**
   PEP 563 biến annotation thành chuỗi; Modal đọc `__annotations__` thô. Triệu
   chứng: `KeyError: 'str'` → `AttributeError: 'str' object has no attribute
   '__name__'`. File nào có `@app.cls` thì **không** được có future import.
2. **Dữ liệu thật là tar KHÔNG nén**, dù tên file là `.tar.gz`. Luôn mở bằng
   `tarfile.open(p, "r:*")`, không bao giờ `"r:gz"`.
3. **Đừng tin `sha256` của container.** Tar chứa `mtime`/owner/thứ tự. Muốn biết
   hai tập ảnh có giống nhau thì so **từng member**.
4. **Stub quá dễ dãi sẽ che chính lỗi đang tìm.** Stub phải mô phỏng hành vi gây
   lỗi (raise đúng chỗ), không được bỏ qua nó.
5. **Không raise ra khỏi `.map()`.** Một task raise là huỷ cả đợt và mất file
   đang tải dở. Trả dict `{"status": ...}` và tự retry bên trong.
6. **Mọi thứ chạm GPU phải đi qua preflight chi phí.** Xem `preflight_cost()`.

## Xác minh trước khi nói "xong"

```bash
./scripts/07_check_imports.sh     # cổng 0: modal THẬT + toàn bộ test, 0 credit
```

Chạy tay từng phần:

```bash
python3 -m pytest tests/ -q                     # 117 test (2 skip nếu thiếu modal)
python3 -m pytest tests/test_real_modal_import.py -q   # cần `pip install modal`
python3 -m pyflakes src/dire_modal/download.py tests/
bash -n scripts/*.sh
```

Nếu môi trường chưa có venv, tạo trước (system pip bị PEP 668 chặn):

```bash
python3 -m venv /tmp/venv && /tmp/venv/bin/pip install -r requirements-dev.txt
```

## Nguồn sự thật (đừng tạo bản sao thứ hai)

| Thứ | Ở đâu |
|---|---|
| Policy tải + guard GPU | `src/dire_modal/tiers.yaml` (code đọc file **trong package**) |
| Danh sách file của từng tier | `src/dire_modal/manifest.py` (`TIERS`) |
| Quy ước bí mật / env | `.env.example` |
| Quyết định kiến trúc | `docs/adr-000*.md` |
| Chi phí, hạn mức free tier | `docs/iaas-options.md` |

`configs/tiers.yaml` **chỉ là bảng tra cứu cho người đọc** — code không đọc nó.
Modal chỉ đóng gói thư mục package nên `/root/configs/` không tồn tại trong
container. Sửa policy mà chỉ sửa `configs/` là sửa vào chỗ vô tác dụng.

## Quy ước trình bày

- Tài liệu và comment viết **tiếng Việt**; định danh code, log, commit subject
  dùng tiếng Anh hoặc tiếng Việt đều được, miễn nhất quán trong một file.
- Comment giải thích **vì sao**, đặc biệt là vì sao một cách làm hiển nhiên lại
  sai. Ví dụ trong `download.py`: vì sao `shutil.move` chứ không `rename`.
- Con số trong tài liệu phải là **số đo được**, không phải ước lượng — hoặc ghi
  rõ là ước lượng. Nhiều lỗi tài liệu trong repo này đến từ chỗ đó.

## Bắt đầu từ đâu

- Muốn biết **sẽ xây cái gì và tại sao** → [spec.md](spec.md), rồi [plan.md](plan.md)
- Muốn biết đang làm gì tiếp → [plan.md](plan.md)
- Sắp sửa code → [instructions.md](instructions.md)
- Gặp một việc cụ thể (tải hỏng, thêm tier, guard GPU, viết test) →
  [skills.md](skills.md)
