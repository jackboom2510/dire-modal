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

Đường đi đầy đủ từ đầu tới demo: **[`docs/runbook-free.md`](docs/runbook-free.md)** (~$3–6: **không train lại**, bỏ hẳn khoản $8–20 của A100).

## Tái lập thử nghiệm bằng checkpoint có sẵn (KHÔNG train lại)

Ràng buộc dự án: *chỉ cải thiện, không train lại*. Nên "tái lập" ở đây nghĩa là
**nạp `lsun_adm.pth` của tác giả rồi eval** — không đụng tới ResNet50 training.

```bash
# 1) tải 3 file tối thiểu (ckpt + real + adm của test LSUN-Bedroom) ≈ 400 MB
modal run -m dire_modal.download::main --tiers e2min --mode fetch

# 2) eval — eval::main tự tìm /raw/checkpoints/lsun_adm.pth, in CỔNG S2
modal run -m dire_modal.eval::main --tiers e2
```

Hoặc một lệnh gói cả hai: `bash scripts/08_repro_minimal.sh`.

| bước | lệnh | thời gian | cổng |
|---|---|---|---|
| tải `e2min` | `download::main --tiers e2min --mode fetch` | ~1.5 h @ 76.6 KiB/s | `download::verify` OK |
| eval | `eval::main --tiers e2min` | < 2 phút CPU | **S2: `ACC(adm) ≥ 0.99`** |
| eval đầy đủ | `eval::main --tiers e2` | ~10 phút CPU | bảng 16 generator |

> ⚠ **Trần tốc độ là của TÀI KHOẢN Baidu, không phải của số container.** Đo
> 2026-10-25 (`bench_concurrency --tiers e2 --limit 2 --ks 1,2`, 387.7 MB/lượt):
> `k=1` → **5183.8 s / 76.6 KiB/s**; `k=2` → **5293.3 s / 75.0 KiB/s**. Tức thêm
> container **không** nhanh hơn. Vì thế `max_containers: 1` và `baidu_parallel: 1`
> trong `tiers.yaml` — và đó cũng là khuyến cáo của upstream BaiduPCS-Go cho tài
> khoản miễn phí (tăng thread ⇒ bị throttle 0 tốc độ nhiều giờ tới nhiều ngày).
>
> `download::main` **tự lặp nhiều vòng** trong một lần gọi (ngân sách 20 h), tính
> lại file còn thiếu từ đĩa sau mỗi vòng, nên không phải gõ lại lệnh ~9 lần.
> Muốn nhanh hơn hẳn thì phải đổi **nguồn**, không phải đổi tham số: mirror lên
> HF rồi `modal run -m dire_modal.download::from_hf --repo-id <user>/<repo> --tiers e2`.

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
| [`spec.md`](spec.md) | **SPEC-001** (SDD): tái tạo experiment DIRE + cải thiện theo AEROBLADE. 7 yêu cầu `FR-*` có tiêu chí nghiệm thu, 6 mốc, bảng rủi ro |
| [`AGENT.md`](AGENT.md) | Điểm vào cho agent: bản đồ repo, 6 lỗi dễ làm hỏng nhất, nguồn sự thật |
| [`instructions.md`](instructions.md) | Quy tắc làm việc: test phải "cắn", stub đúng ngữ nghĩa, tiền, dữ liệu, git |
| [`plan.md`](plan.md) | Đang ở đâu, việc tiếp theo, việc đã quyết là KHÔNG làm |
| [`skills.md`](skills.md) | Công thức 8 việc thường gặp: tải hỏng, thêm tier, guard GPU, viết test… |
| [`CLAUDE.md`](CLAUDE.md) | Bản ngắn cho Claude: 5 điểm dễ sai + lệnh kiểm |

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
| `modal run -m dire_modal.download::main --tiers e2,e4 --mode fetch` | Tải thật (mặc định là dry-run), **tự lặp vòng tới khi đủ hoặc hết 20 h**. `--concurrency N` chỉnh số container (clamp 1–8, nhưng đo được là >1 không nhanh hơn) |
| `modal run -m dire_modal.download::main --paths a/b.tar.gz,c/d.tar.gz --mode fetch` | Tải **đúng danh sách file**, không cần tier — vẫn qua `NEVER` + guard dung lượng |
| `modal run -m dire_modal.download::from_hf --repo-id <user>/<repo> --tiers e2` | Kéo từ mirror HF Hub (**nhanh hơn Baidu 1–2 bậc**) |
| `bash scripts/08_repro_minimal.sh` | Chuỗi tái lập gọn: status → plan → fetch → verify → eval pre-trained |
| `modal run -m dire_modal.download::prune --tiers e2,e4` | Liệt kê file đã tải thừa (`--mode apply` mới xoá) |
| `./scripts/02_verify.sh` | Kiểm định dạng thật từng file (gzip **hoặc** tar không nén) + (tuỳ chọn) đẩy R2 |
| `./scripts/03_pack_shards.sh` | Pack WebDataset + benchmark storage/compute |
| `bash scripts/08_repro_minimal.sh` | **Đường chính hiện tại**: tải `e2min` → eval checkpoint tác giả (cổng S2: ACC ≥ 0.99 trên ADM-LSUN) |
| `bash scripts/09_bench_mode.sh` | Đo `--mode locate\|pcs\|stream` của BaiduPCS-Go trên 1 file nhỏ |
| `./scripts/04_eval_pretrained.sh` | Eval checkpoint tác giả; `TIERS=e2min` để chỉ chạy cổng S2 |
| `./scripts/05_train_repro.sh` | Train reproduce — **ngoài phạm vi dự án** (chỉ để tham chiếu) |
| `./scripts/06_measure_dire.sh` | Đo ảnh/giây của DIRE → số giờ GPU thật |
| `./scripts/bench_storage.sh` | Đo Volume vs R2 (mount + S3 API) |
| `modal run -m dire_modal.archive::push_hf --repo-id <user>/<repo>` | Mirror lên HF |
| `modal run -m dire_modal.archive::verify_hf --repo-id <user>/<repo>` | Đối chiếu sha256 mirror |
| `modal run -m dire_modal.compute_dire::dire_batch --n 200` | Tính DIRE trên GPU — **có preflight chặn chi phí** trước khi bật container |
| `modal run -m dire_modal.compute_dire::dire_batch --n 200 --dry-run` | Chỉ in ước tính tiền, không bật GPU nào |
| `modal run -m dire_modal.eval::main --tiers e2` | **Eval checkpoint pre-trained trên mọi cặp (real, generator)** → CSV + `run.json`. Tự tìm ckpt trong `/raw/checkpoints` rồi `/ckpt/pretrained`; `--ckpt` để chỉ định tay |
| `modal run -m dire_modal.download::bench_concurrency --tiers e2 --limit 2 --ks 1,2` | Đo băng thông thật ở các mức song song (quyết định `max_containers`) |
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
| **`e2min`** | **3** | **~0.40 GB** | **đường ngắn nhất tới con số đầu tiên**: `lsun_adm.pth` + test `real` + test `adm` |
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
  train.py          # (TODO) training loop — ngoài phạm vi, xem spec.md N1
  eval.py           # eval 16 generator bằng checkpoint có sẵn + CỔNG S2
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
| Eval (pre-trained) | ✅ code xong — `eval::main` tự tìm ckpt, in CỔNG S2; **số đo cần chạy** `scripts/08_repro_minimal.sh` |
| Train lại | ⛔ ngoài phạm vi (spec.md N1: chỉ cải thiện, không train lại) |
| Stage-2 (compute DIRE toàn dataset) | ⏳ code viết theo API guided-diffusion, **chưa chạy thử**; cần ~30–60 A100-h |
| Demo HF Spaces | ✅ code + hướng dẫn; ⏳ chờ checkpoint & gallery |

## Cảnh báo

- **License**: dataset DiffusionForensics là dataset nghiên cứu; repo gốc đã archive (2025-07-08).
  Mirror kèm ghi nguồn và sẵn sàng gỡ theo yêu cầu — xem `docs/runbook-free.md` §5.
- **Không lạm dụng free tier**: không đa tài khoản, không dùng Colab free làm worker (vi phạm ToS).
- **Con số ước lượng**: mọi chỗ ghi "ước tính" cần đo lại bằng `scripts/06_measure_dire.sh` và `scripts/bench_storage.sh`.
