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
| `./scripts/07_check_imports.sh` | **CỔNG 0**: import bằng `modal` THẬT + toàn bộ test. 0 credit, chạy trước mọi lệnh khác |
| `modal run -m dire_modal.download::local_status` | In tier + policy — chạy **local**, 0 credit |
| `modal run -m dire_modal.download::audit` | Đối chiếu cây Baidu thật ↔ manifest (không tải). `--dirs a,b` để quét thư mục riêng |
| `modal run -m dire_modal.download::plan --tiers e2,e4` | Kế hoạch + **size thật** (không tải) |
| `modal run -m dire_modal.download::probe_pair` | Chốt ngữ nghĩa `dire/test/<A>/<B>/` bằng **so từng ảnh** trong tar (~53 MB) |
| `./scripts/01_fetch_baidu.sh` | audit → plan → hỏi xác nhận → tải 22 file / 3.39 GB → verify |
| `modal run -m dire_modal.download::main --tiers e2,e4 --mode fetch` | Tải thật (mặc định là dry-run) |
| `modal run -m dire_modal.download::prune --tiers e2,e4` | Liệt kê file đã tải thừa (`--mode apply` mới xoá) |
| `./scripts/02_verify.sh` | Kiểm định dạng thật từng file (gzip **hoặc** tar không nén) + (tuỳ chọn) đẩy R2 |
| `./scripts/03_pack_shards.sh` | Pack WebDataset + benchmark storage/compute |
| `./scripts/04_eval_pretrained.sh` | Eval checkpoint tác giả (cổng: ACC ≈ 99% trên ADM-LSUN) |
| `./scripts/05_train_repro.sh` | Train reproduce (`--detach`) |
| `./scripts/06_measure_dire.sh` | Đo ảnh/giây của DIRE → số giờ GPU thật |
| `./scripts/bench_storage.sh` | Đo Volume vs R2 (mount + S3 API) |
| `modal run -m dire_modal.archive::push_hf --repo-id <user>/<repo>` | Mirror lên HF |
| `modal run -m dire_modal.archive::verify_hf --repo-id <user>/<repo>` | Đối chiếu sha256 mirror |
| `modal run -m dire_modal.compute_dire::dire_batch --n 200` | Tính DIRE trên GPU — **có preflight chặn chi phí** trước khi bật container |
| `modal run -m dire_modal.compute_dire::dire_batch --n 200 --dry-run` | Chỉ in ước tính tiền, không bật GPU nào |
| `modal run -m dire_modal.compute_dire::make_gallery --n 12` | Sinh gallery |x−recon| cho demo |

## Chọn dữ liệu theo mục tiêu

Detector DIRE (`vendor/networks/resnet.py`: `conv1=Conv2d(3,64)`, `num_classes=2`)
đọc dữ liệu qua `torchvision.ImageFolder` trên nhánh `dire/` với hai lớp
`0_real`/`1_fake`. Nghĩa là `recons/` và phần lớn `images/` **không nằm trên
đường train/eval** của detector.

**Mọi con số dưới đây là SỐ ĐO THẬT** từ `download::audit` ngày 2026-10-21
(15 thư mục, 104 file), không phải ước lượng.

| tier | file | size | mục đích |
|---|---|---|---|
| `e0` | 0 | — | code / tooling / docs |
| `e1` | 0 | — | demo AEROBLADE zero-shot (VAE ~1 GB kéo thẳng từ HF Hub) |
| **`e2`** | **17** | **1.94 GB** | **eval headline DIRE trên LSUN-Bedroom, 16 generator** |
| `e3` | 24 | 4.85 GB | +10 tar `lsun_bedroom/imagenet` (**cross-generator**, xem dưới), ImageNet, CelebA-HQ |
| **`e4`** | **5** | **1.45 GB** | **eval AEROBLADE trên ảnh gốc (nhánh `images/`)** |
| `e4x` | 17 | 3.47 GB | AEROBLADE trên MỌI generator (tự loại 8 file `*ddim*`) |
| `e5` | 11 | 2.11 GB | ensemble + hiệu chỉnh ngưỡng (val có đủ 5 tar) |
| `train` | 4 | 5.74 GB | train lại detector headline từ đầu |
| `trainx` | 7 | 11.24 GB | train lại detector cross-domain |
| `trainens` | 3 | 14.60 GB | train 3 detector ensemble |
| `rawx` | 4 | 4.48 GB | ảnh gốc để TỰ tính lại DIRE (cần GPU + ADM 2.5 GB) |
| `rawxens` | 6 | 12.10 GB | ảnh gốc cho ensemble |
| `ckpts` | 6 | 1.58 GB | toàn bộ checkpoint detector (mỗi file 269.4 MB) |

| tổ hợp | file | size |
|---|---|---|
| **`e2,e4` (mặc định)** | **22** | **3.39 GB** |
| `e2,e3,e4,e5` — eval đầy đủ, không train | 57 | 10.35 GB |
| `e2,e3,e4,e4x,e5` | 69 | 12.37 GB |
| Tier A **cũ** (2 train + 2 val + 16 test) | 20 | **7.42 GB** |
| Toàn bộ cây đã audit | 104 | **61.40 GB** |
| ↳ trong đó dữ liệu **train** | — | **47.00 GB (77%)** |

Docs từng ghi "Tier A ≈ 40–70 GB" — đó là số của **cả cây** gán nhầm cho Tier A;
sai 5.4–9.4 lần. Vì 77% cây là dữ liệu train mà ràng buộc dự án là *chỉ cải
thiện, không train lại*, phần bỏ được rất lớn: **eval đầy đủ chỉ 10.35 GB**.

Alias cũ vẫn chạy được: `ckpt`→`ckpts` · `A`→`e2,train` · `B`→`e3` · `C`→`rawx` ·
`D`→ rỗng (toàn bộ nội dung cũ nằm trong `NEVER`).

`NEVER` chặn cứng, không tier nào với tới: `recons/` · `*ddim*` (8 tar ablation
số bước DDIM, không có bản `dire/` tương ứng — 2 trong số đó chỉ **130 B**, tức
rỗng ngay trong dataset gốc) · `lsun_bedroom.pt` (weight ADM 2.5 GB không có
trong cây Baidu). Audit đo được **1.6 GB** bị chặn trong phạm vi đã quét.

`expand()` khử trùng lặp theo **đường dẫn remote**, không theo object: `e4 ⊂ e4x`
nên `--tiers e4,e4x` cho 17 file chứ không phải 22.

Policy (tier mặc định, trần dung lượng, số container song song) nằm ở
[`src/dire_modal/tiers.yaml`](src/dire_modal/tiers.yaml) — cố ý đặt **trong
package** chứ không phải `configs/`, vì Modal chỉ đóng gói thư mục của module
được chạy nên function không thấy `configs/` ở repo root.

## Cấu trúc

```
src/dire_modal/
  common.py         # Modal app, images, volumes, secrets
  manifest.py       # INVENTORY dữ liệu theo MỤC TIÊU (e0–e5, train, never)
  tiers.yaml        # POLICY: tier mặc định, trần GB, số container song song
  download.py       # tải Baidu → /raw (dry-run, guard dung lượng, resumable)
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
