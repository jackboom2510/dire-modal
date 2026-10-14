# dire-modal

Tái lập **DIRE — Diffusion Reconstruction Error** (Wang et al., ICCV 2023) theo hướng
*serverless-first, chi phí gần 0*, chạy được trên nhiều nền tảng IaaS.

> Mục tiêu: từ máy tính cá nhân + **0 ₫** (chỉ free tier) → có detector, có bảng eval, có demo công khai,
> và có mirror dữ liệu để người khác tái lập.

## Bắt đầu nhanh

```bash
cp .env.example .env      # điền token (Modal, R2, HF, Kaggle)
pip install -e .
./scripts/00_setup_secrets.sh
python3 tools/free_quota.py     # xem còn bao nhiêu quota miễn phí
```

Đường đi đầy đủ từ đầu tới demo: **[`docs/runbook-free.md`](docs/runbook-free.md)** (~$13–30 trong $30 credit/tháng).

## Kiến trúc

```
Baidu/Rec ──► Volume /raw        (Modal, 1 TiB free — staging + verify sha256)
              │
              ├──► Volume /shards   (WebDataset shards — HOT PATH train, POSIX, $0)
              ├──► Volume /ckpt     (checkpoint mới nhất + log)
              │        └──► R2      (checkpoint đã chốt + manifest sha256; 10 GB free, egress $0)
              └──► HF Hub + Kaggle  (MIRROR CÔNG KHAI — chia sẻ & chống link rot, free)

Compute:  Modal (chính)  ·  Kaggle/Colab (dev, $0)  ·  Thunder/Vast/RunPod (khi vượt credit)
Demo:     HF Spaces (CPU free)
Local:    git + modal CLI + xem mẫu nhỏ (máy mỏng — không giữ dataset)
```

Chi tiết quyết định: [`docs/adr-0001-storage.md`](docs/adr-0001-storage.md) · [`docs/adr-0002-iaas.md`](docs/adr-0002-iaas.md).

## Tài liệu

| Tài liệu | Nội dung |
|---|---|
| [`docs/runbook-free.md`](docs/runbook-free.md) | Runbook 10 bước: 0 ₫ → model + demo, kèm cổng kiểm tra từng bước |
| [`docs/iaas-options.md`](docs/iaas-options.md) | So sánh ~16 nền tảng IaaS × 28 tiêu chí, ánh xạ workload, decision tree, 10 cái bẫy |
| [`docs/storage-comparison.md`](docs/storage-comparison.md) | Modal Volume vs R2 vs Local vs Kaggle/HF — 18 tiêu chí, điểm hoà vốn, quy tắc ngưỡng |
| [`docs/adr-0001-storage.md`](docs/adr-0001-storage.md) | ADR: vai trò từng lớp lưu trữ |
| [`docs/adr-0002-iaas.md`](docs/adr-0002-iaas.md) | ADR: chiến lược đa nền tảng + ngưỡng rời Modal |

## Công cụ

```bash
python3 tools/free_quota.py --used modal=8 kaggle=6      # sổ quota free tier
python3 tools/storage_cost.py --size-gb 150              # chi phí + điểm hoà vốn
python3 tools/storage_cost.py --from-manifest manifests/raw.tsv
```

## Lệnh chính

| Lệnh | Việc |
|---|---|
| `./scripts/01_fetch_baidu.sh` | Tải Tier A + checkpoints từ Baidu về Volume |
| `./scripts/02_verify.sh` | `gzip -t` toàn bộ + (tuỳ chọn) đẩy R2 |
| `./scripts/03_pack_shards.sh` | Pack WebDataset + benchmark storage/compute |
| `./scripts/04_eval_pretrained.sh` | Eval checkpoint tác giả (cổng: ACC ≈ 99% trên ADM-LSUN) |
| `./scripts/05_train_repro.sh` | Train reproduce (`--detach`) |
| `./scripts/06_measure_dire.sh` | Đo ảnh/giây của DIRE → số giờ GPU thật |
| `./scripts/bench_storage.sh` | Đo Volume vs R2 (mount + S3 API) |
| `modal run -m dire_modal.archive::push_hf --repo-id <user>/<repo>` | Mirror lên HF |
| `modal run -m dire_modal.archive::verify_hf --repo-id <user>/<repo>` | Đối chiếu sha256 mirror |
| `modal run -m dire_modal.compute_dire::make_gallery --n 12` | Sinh gallery |x−recon| cho demo |

## Cấu trúc

```
src/dire_modal/
  common.py         # Modal app, images, volumes, secrets
  manifest.py       # định nghĩa tier dữ liệu (A/B/C/D + ckpt)
  download.py       # tải Baidu → /raw (resumable, verify gzip)
  pack.py           # /raw → WebDataset shards
  bench.py          # benchmark storage (Volume/R2) + throughput DIRE
  archive.py        # đẩy/kéo: R2, HF Hub, Kaggle + verify sha256
  train.py          # (TODO) training loop
  eval.py           # (TODO) eval 16 generator
  compute_dire.py   # tính DIRE bằng ADM DDIM inversion + gallery
  vendor/           # module DIRE gốc đã vendorize
tools/              # free_quota.py, storage_cost.py (thuần stdlib)
platforms/          # bootstrap cho kaggle / runpod / vast / hf-spaces
docs/               # runbook, so sánh, ADR
manifests/          # manifest sha256 (commit); *.tsv size thật (gitignore)
```

## Trạng thái

| Hạng mục | Trạng thái |
|---|---|
| Tải + verify dữ liệu | ✅ code xong (chưa chạy trong CI) |
| Pack shards | ✅ |
| Benchmark storage | ✅ code xong — **số đo cần chạy** `scripts/bench_storage.sh` |
| Mirror HF/Kaggle + verify sha256 | ✅ |
| Train / eval | ⏳ TODO — port từ `vendor/` và repo DIRE gốc |
| Stage-2 (compute DIRE toàn dataset) | ⏳ code viết theo API guided-diffusion, **chưa chạy thử**; cần ~30–60 A100-h |
| Demo HF Spaces | ✅ code + hướng dẫn; ⏳ chờ checkpoint & gallery |

## Cảnh báo

- **License**: dataset DiffusionForensics là dataset nghiên cứu; repo gốc đã archive (2025-07-08).
  Mirror kèm ghi nguồn và sẵn sàng gỡ theo yêu cầu — xem `docs/runbook-free.md` §5.
- **Không lạm dụng free tier**: không đa tài khoản, không dùng Colab free làm worker (vi phạm ToS).
- **Con số ước lượng**: mọi chỗ ghi "ước tính" cần đo lại bằng `scripts/06_measure_dire.sh` và `scripts/bench_storage.sh`.
