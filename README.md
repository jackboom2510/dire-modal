# dire-modal

Tái lập **DIRE — Diffusion Reconstruction Error** (Wang et al., ICCV 2023) theo hướng
*serverless-first, chi phí gần 0*, chạy được trên nhiều nền tảng IaaS.

> Mục tiêu: từ máy tính cá nhân + **0 ₫** (chỉ free tier) → có detector, có bảng eval, có demo công khai,
> và có mirror dữ liệu để người khác tái lập.

## Bắt đầu nhanh

```bash
cp .env.example .env      # điền token (Modal, R2, HF, Kaggle)
pip install -e .
bash scripts/setup.sh           # tạo secret trên Modal từ .env
bash scripts/check.sh           # CỔNG 0: modal thật + test + pyflakes (0 credit)
python3 tools/free_quota.py     # xem còn bao nhiêu quota miễn phí
```

Đường đi đầy đủ từ đầu tới demo: **[`docs/runbook-free.md`](docs/runbook-free.md)** (~$3–6: **không train lại**, bỏ hẳn khoản $8–20 của A100).

> **Windows / Git Bash**: MSYS2 đổi mọi tham số bắt đầu bằng `/` thành đường dẫn
> Windows (`--path /raw` → `C:/Program Files/Git/raw`) và container Linux không
> thấy gì. Mọi script trong `scripts/` + `tools/` đã `export MSYS_NO_PATHCONV=1`.
> Nếu gõ `modal run … --path /raw/…` trực tiếp thì thêm biến đó vào trước lệnh;
> code cũng tự sửa đường dẫn dạng `C:/…/Git/raw` (`_demangle` trong `download.py`).

## Hai quy trình thực nghiệm

`scripts/` cố ý chỉ có **hai** quy trình — mỗi file là một câu hỏi, không phải
một bước trong dãy 00…10 như trước. Chi tiết: [`scripts/README.md`](scripts/README.md).

| Quy trình | Câu hỏi | File | Chi phí |
|---|---|---|---|
| **1 — Tái lập** | "Có tái tạo được số của paper không?" | `scripts/01_repro.sh` | ~$0.2 (CPU) |
| **2 — Cải thiện** | "Training-free, làm tốt hơn DIRE được không?" | `scripts/02_improve.sh` | ~$0–20 theo FR |

### Quy trình 1 — tái lập Table 2 của paper

Khối **DIRE (ours)** trong Table 2 (ICCV 2023): **4 detector × 8 generator =
32 ô**, headline `Avg 99.9 ACC / 100 AP`. Không train gì — nhánh `dire/` đã có
DIRE map tác giả tính sẵn (ADM-LSUN, DDIM S=20), nên chỉ cần ResNet-50 forward
trên CPU.

```bash
bash scripts/01_repro.sh --quick        # e2min: 3 file ≈ 1.5 h → 1 ô + cổng S2
bash scripts/01_repro.sh                # e2,ckpts: 22 file ≈ 12.4 h → cả 32 ô
bash scripts/01_repro.sh --stage report # chỉ in Δ so với paper
DETACH=1 bash scripts/01_repro.sh       # app sống độc lập với máy local
```

Số tham chiếu nằm ở `src/dire_modal/paper_table2.tsv` (ghi rõ nguồn: arXiv
2303.09295v1, Table 2, trang 7) — `eval::report` in Δ từng ô và đếm số ô khớp.

### Tier ↔ bảng trong paper

| tiers | file / GB | ~giờ @76.6 KiB/s | phủ được | không phủ |
|---|---|---|---|---|
| `e2min` | 3 / 0.40 | 1.5 | 1 ô (cổng S2: `ACC(adm) ≥ 0.99`) | mọi ô còn lại |
| `e2` | 17 / 1.94 | 7.4 | **hàng ADM** — 8 cột của Table 2 | 3 hàng detector còn lại |
| **`e2,ckpts`** | **22 / 3.25** | **12.4** | **cả 32 ô của Table 2** | Table 3/4/5/6 |
| `+ 3 file `--paths`` | +0.14 | +0.5 | **Table 3** (cross-dataset ImageNet) | — |
| Table 4/5/6 + hàng baseline (CNNDet\*, Patchfor\*, F3Net\*) | — | — | ❌ **ngoài phạm vi** | cần **train lại** + weight ADM 2.5 GB + 30–60 A100-h |

> ⚠ `checkpoints/lsun_adm.pth` (269.4 MB) là **detector ResNet-50**, **KHÔNG**
> phải weight diffusion ADM (`lsun_bedroom.pt`, 2.5 GB — nằm trong `NEVER` vì
> không có trong cây Baidu). Nhầm hai thứ này từng là lỗi trong
> `compute_dire.py`.

### Quy trình 2 — cải thiện (training-free)

Nguồn ý tưởng: **AEROBLADE** (Ricker et al., CVPR 2024 — training-free, điểm âm
của lỗi tái tạo qua VAE của LDM) + các đề xuất của dự án (rank-fusion trên val,
format bias, robustness, định tuyến theo họ generator). Hiện chỉ stage
`baseline` có code; các stage còn lại **dừng** và trỏ về `spec.md` §12.

```bash
bash scripts/02_improve.sh --stage plan       # lộ trình FR-2…FR-6
bash scripts/02_improve.sh --stage baseline   # điểm DIRE trên val (e5) + test (e2)
```

> ⚠ **Trần tốc độ là của TÀI KHOẢN Baidu, không phải của số container.** Đo
> 2026-10-25 (`bench_concurrency --tiers e2 --limit 2 --ks 1,2`, 387.7 MB/lượt):
> `k=1` → **5183.8 s / 76.6 KiB/s**; `k=2` → **5293.3 s / 75.0 KiB/s**. Thêm
> container **không** nhanh hơn ⇒ `max_containers: 1`, `baidu_parallel: 1`.
>
> `download::main` **tự lặp nhiều vòng** trong một lần gọi (ngân sách 20 h), tính
> lại file còn thiếu từ đĩa sau mỗi vòng ⇒ không phải gõ lại lệnh ~9 lần. Muốn
> nhanh hơn hẳn thì phải đổi **nguồn**: mirror lên HF rồi
> `download::from_hf --repo-id <user>/<repo> --tiers e2`.

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
| [`scripts/README.md`](scripts/README.md) | **Đúng hai quy trình**: tái lập Table 2 · cải thiện (AEROBLADE + đề xuất dự án) |
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

**Quy trình** (xem [`scripts/README.md`](scripts/README.md)):

| Lệnh | Việc |
|---|---|
| `bash scripts/check.sh` | **CỔNG 0**: import bằng `modal` THẬT + toàn bộ test + pyflakes. 0 credit, chạy trước mọi lệnh khác |
| `bash scripts/setup.sh` | Điều kiện tiên quyết, một lần: tạo secret trên Modal từ `.env` |
| `bash scripts/01_repro.sh` | **QUY TRÌNH 1** — tái lập Table 2: data → eval 4 detector → báo cáo Δ so với paper |
| `bash scripts/01_repro.sh --quick` | Bản nhanh: `e2min` (3 file) → 1 ô + cổng S2 |
| `bash scripts/01_repro.sh --stage report` | Chỉ in bảng so sánh với paper (`eval::report`) |
| `bash scripts/02_improve.sh --stage plan` | **QUY TRÌNH 2** — lộ trình cải thiện FR-2…FR-6 (AEROBLADE + đề xuất của dự án) |
| `bash scripts/02_improve.sh --stage baseline` | Baseline DIRE trên val (`e5`) + test (`e2`) |
| `DETACH=1 bash scripts/01_repro.sh` | Thêm `--detach`: app sống độc lập với máy local (mất mạng không giết job) |

**Tiện ích** (không phải quy trình thực nghiệm, ở `tools/`):

| Lệnh | Việc |
|---|---|
| `bash tools/local_download.sh` | Tải dữ liệu về **máy local** (không Modal, không Volume) — cần `BaiduPCS-Go` đã login; xem runbook §2b |
| `modal run -m dire_modal.upload::main --tiers e2min --mode push` | Đẩy dữ liệu **đã tải ở máy** lên Volume (chỉ file thiếu), rồi `--mode clean --yes` để dọn `.staging` cũ; xem runbook §2c |
| `bash tools/tree.sh` | Cây toàn bộ Volume (`modal volume ls` chỉ đọc 1 tầng nên không ra cây được) |
| `bash tools/bench_storage.sh` | Đo Volume vs R2 (mount + S3 API) |
| `bash tools/bench_baidu_mode.sh` | Đo `--mode locate\|pcs\|stream` của BaiduPCS-Go trên 1 file nhỏ |
| `bash tools/measure_dire.sh` | Đo ảnh/giây của DIRE → số giờ GPU thật |
| `bash tools/pack_shards.sh` | Pack WebDataset shards (chỉ cần nếu TRAIN — ngoài phạm vi) |

**Lệnh Modal trực tiếp:**

| Lệnh | Việc |
|---|---|
| `modal run -m dire_modal.download::local_status` | In tier + policy — chạy **local**, 0 credit |
| `modal run -m dire_modal.download::audit` | Đối chiếu cây Baidu thật ↔ manifest (không tải). `--dirs a,b` để quét thư mục riêng |
| `modal run -m dire_modal.download::plan --tiers e2,ckpts` | Kế hoạch + **size thật** (không tải) |
| `modal run -m dire_modal.download::probe_pair` | Chốt ngữ nghĩa `dire/test/<A>/<B>/` bằng **so từng ảnh** trong tar (~53 MB) |
| `modal run -m dire_modal.download::main --tiers e2,ckpts --mode fetch` | Tải thật (mặc định là dry-run), **tự lặp vòng tới khi đủ hoặc hết 20 h** |
| `modal run -m dire_modal.download::main --paths a/b.tar.gz,c/d.tar.gz --mode fetch` | Tải **đúng danh sách file**, không cần tier — vẫn qua `NEVER` + guard dung lượng |
| `modal run -m dire_modal.download::from_hf --repo-id <user>/<repo> --tiers e2` | Kéo từ mirror HF Hub (**nhanh hơn Baidu 1–2 bậc**) |
| `modal run -m dire_modal.download::verify` | Kiểm định dạng thật từng file (gzip **hoặc** tar không nén) |
| `modal run -m dire_modal.download::prune --tiers e2` | Liệt kê file đã tải thừa (`--mode apply` mới xoá) |
| `modal run -m dire_modal.download::tree --path /raw` | Cây **toàn bộ** Volume. `--flat` · `--depth N` · `--out` |
| `modal run -m dire_modal.eval::main --tiers e2 --ckpt /raw/checkpoints/lsun_adm.pth` | Eval 1 checkpoint → CSV + `run.json`; in **CỔNG S2** |
| `modal run -m dire_modal.eval::report` | Gộp mọi lần eval, in Δ so với Table 2 của paper |
| `modal run -m dire_modal.archive::push_hf --repo-id <user>/<repo>` | Mirror lên HF |
| `modal run -m dire_modal.archive::verify_hf --repo-id <user>/<repo>` | Đối chiếu sha256 mirror |
| `modal run -m dire_modal.compute_dire::dire_batch --n 200 --dry-run` | Chỉ in ước tính tiền, không bật GPU nào |
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
| **`e2`** | **17** | **1.94 GB** | **eval headline DIRE trên LSUN-Bedroom, 15 generator** |
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
  eval.py           # eval 15 generator bằng checkpoint có sẵn + CỔNG S2
  compute_dire.py   # tính DIRE bằng ADM DDIM inversion + gallery
  vendor/           # module DIRE gốc đã vendorize
tools/              # free_quota.py, storage_cost.py (thuần stdlib) + tiện ích .sh
                    #   local_download.sh = tải ở MÁY LOCAL (không cần Modal)
scripts/            # ĐÚNG 2 quy trình: 01_repro.sh · 02_improve.sh (+ setup/check/lib)
platforms/          # bootstrap cho kaggle / runpod / vast / hf-spaces
docs/               # runbook, so sánh, ADR
manifests/          # manifest sha256 (commit); *.tsv size thật (gitignore)
```

## Trạng thái

| Hạng mục | Trạng thái |
|---|---|
| Tải + verify dữ liệu | ✅ code xong (chưa chạy trong CI) |
| Pack shards | ✅ |
| Benchmark storage | ✅ code xong — **số đo cần chạy** `tools/bench_storage.sh` |
| Mirror HF/Kaggle + verify sha256 | ✅ |
| Eval (pre-trained) | ✅ code xong — `eval::main` tự tìm ckpt, in CỔNG S2; `eval::report` in Δ so với paper |
| Tái lập Table 2 | ✅ code + quy trình xong (`scripts/01_repro.sh`) — **số của mình chưa có**, cần chạy thật |
| Train lại | ⛔ ngoài phạm vi (spec.md N1: chỉ cải thiện, không train lại) |
| Stage-2 (compute DIRE toàn dataset) | ⏳ code viết theo API guided-diffusion, **chưa chạy thử**; cần ~30–60 A100-h |
| Demo HF Spaces | ✅ code + hướng dẫn; ⏳ chờ checkpoint & gallery |

## Cảnh báo

- **License**: dataset DiffusionForensics là dataset nghiên cứu; repo gốc đã archive (2025-07-08).
  Mirror kèm ghi nguồn và sẵn sàng gỡ theo yêu cầu — xem `docs/runbook-free.md` §5.
- **Không lạm dụng free tier**: không đa tài khoản, không dùng Colab free làm worker (vi phạm ToS).
- **Con số ước lượng**: mọi chỗ ghi "ước tính" cần đo lại bằng `tools/measure_dire.sh` và `tools/bench_storage.sh`.
