# dire-modal

Chạy lại **DIRE — Diffusion Reconstruction Error** (Wang et al., ICCV 2023) theo
hướng *serverless-first, chi phí gần 0*, trên nhiều nền tảng IaaS.

> Mục tiêu: từ máy tính cá nhân + **0 ₫** (chỉ free tier) → có detector, có bảng
> eval, có demo công khai, và có mirror dữ liệu để người khác kiểm lại.

**Nói đúng mức bằng chứng.** Đường eval mặc định dùng lại **cả hai** thứ quyết
định phép đo: DIRE map do tác giả tính (`dire/`) và detector do tác giả train
(`checkpoints/`). Vì thế nó là **chấm lại (R0)** — kiểm đường đo của mình khớp
đường đo của họ — **không phải** tái lập thí nghiệm. Tự tính map (R1), tự train
lại detector (R2), tự sinh ảnh gốc (R3) là ba trục **độc lập**, hiện **chưa
làm**. Nhãn này được ghi tự động vào `run.json` + cột `data_source` của CSV, và
in ở đầu mọi báo cáo — xem `src/dire_core/provenance.py`.

## Chạy trên máy nhà (không cần Modal)

```bash
pip install -e .                                  # chỉ cần Python để chạy `dire info`
python -m pip install torch torchvision pillow    # cho detect/batch/serve
python -m dire_modal.local_fetch --tiers e2min --mode fetch   # 3 file ≈ 400 MB
dire info                                         # in cấu hình tiền xử lý
dire detect anh.jpg --weights checkpoints/lsun_adm.pth
dire batch ./thu_muc --out scores.csv
dire serve --port 8000                            # FastAPI + uvicorn
```

Ba lớp, **một chiều phụ thuộc** — xem [`docs/adr-0005-local-first-modal-wrapper.md`](docs/adr-0005-local-first-modal-wrapper.md):

```
dire_core    ← lõi thuần (tiền xử lý, đọc checkpoint, logit → P(fake))
  ↑     ↑
dire_local   dire_modal            ← CLI/API tại máy · wrapper @app.function
```

Vì sao phải tách: trước đó **mọi** module đều `from .common import ...`, mà
`common.py` import `modal` ngay dòng đầu ⇒ không cài `modal` thì **không import
được cả phần logic**. Và nếu bản local được viết mới thì nó sẽ có pipeline tiền
xử lý thứ hai — đúng lỗi `demo.py` của tác giả (thiếu `Normalize`).

## Bắt đầu nhanh (đường cloud)

```bash
cp .env.example .env      # điền token (Modal, R2, HF, Kaggle)
pip install -e .
bash scripts/setup.sh           # tạo secret trên Modal từ .env
bash scripts/check.sh           # CỔNG 0: modal thật + test + pyflakes (0 credit)
python3 tools/free_quota.py     # xem còn bao nhiêu quota miễn phí
```

### SaaS: bọc Modal quanh cùng lõi

```bash
modal serve -m dire_modal.api      # URL tạm để thử
modal deploy -m dire_modal.api     # deploy thật
curl -s -X POST "$URL/v1/detect" -H 'content-type: application/json' \
  -d "{\"image_b64\": \"$(base64 -w0 anh.jpg)\"}"
```

Wrapper **không chứa logic**: nó nạp model một lần/container, giới hạn
`max_containers=2`, `scaledown_window=120` (scale-to-zero), nhúng weight vào image
để cold start ~1 s, và gọi `dire_local.server.score_payload` — cùng đường decode
với bản local. **Auth**: tạo secret `modal secret create dire-api-key DIRE_API_KEY=...`;
không có secret thì `/healthz` báo `auth: "disabled"` và endpoint mở.

Đường đi đầy đủ từ đầu tới demo: **[`docs/runbook-free.md`](docs/runbook-free.md)** (~$3–6: **eval checkpoint tác giả**, bỏ hẳn khoản $8–20 của A100).

> **Windows / Git Bash**: MSYS2 đổi mọi tham số bắt đầu bằng `/` thành đường dẫn
> Windows (`--path /raw` → `C:/Program Files/Git/raw`) và container Linux không
> thấy gì. Mọi script trong `scripts/` + `tools/` đã `export MSYS_NO_PATHCONV=1`.
> Chiều ngược lại — `/d/…` → `D:/…` cho binary Windows (manim, ffmpeg, `PYTHONPATH`) —
> đi qua `tools/lib/native_path.sh`; xem runbook §Windows.
> Nếu gõ `modal run … --path /raw/…` trực tiếp thì thêm biến đó vào trước lệnh;
> code cũng tự sửa đường dẫn dạng `C:/…/Git/raw` (`_demangle` trong `download.py`).

## Hai quy trình thực nghiệm

`scripts/` cố ý chỉ có **hai** quy trình — mỗi file là một câu hỏi, không phải
một bước trong dãy 00…10 như trước. Chi tiết: [`scripts/README.md`](scripts/README.md).

| Quy trình | Câu hỏi | File | Chi phí |
|---|---|---|---|
| **1 — Chấm lại** | "Đường đo của mình có ra đúng số paper trên cùng đặc trưng không?" (R0) | `scripts/01_repro.sh` | ~$0.2 (CPU) |
| **2 — Cải thiện** | "Training-free, làm tốt hơn DIRE được không?" | `scripts/02_improve.sh` | ~$0–20 theo FR |
| **3 — Từ đầu** | "Không dùng map/checkpoint tác giả, dựng lại được không?" (R1+R2+R3) | `tools/scratch_repro.sh` | < 1 GPU-giờ / 2 000 ảnh |
| **4 — Từ số 0** | "Không dùng **code** tác giả, còn dựng lại được không?" (R1+R2+R3, clean-room) | `python -m dire_zero · dire_zero.modal_app` | như trên (cùng GPU, cùng NFE) |

### Quy trình 1 — chấm lại Table 2 của paper (R0)

Khối **DIRE (ours)** trong Table 2 (ICCV 2023): **4 detector × 8 generator =
32 ô**, headline `Avg 99.9 ACC / 100 AP`. Không train gì — nhánh `dire/` đã có
DIRE map tác giả tính sẵn (ADM-LSUN, DDIM S=20), nên chỉ cần ResNet-50 forward
trên CPU.

⚠ Vì cả map lẫn trọng số đều của tác giả, bảng khớp `32/32` chỉ chứng minh
**harness eval** của mình đúng (tiền xử lý, cách đọc `fc.weight`, logit →
`P(fake)`, ACC/AP). Nó **không** chứng minh DIRE map đúng, **không** chứng minh
detector train lại được. Muốn nói "tái lập" thì phải làm R1/R2 — xem
`src/dire_core/provenance.py`.

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

## Bản clean-room — `dire_zero`

`src/dire_zero/` là bản viết lại **độc lập**: U-Net 256 (566 tensor khớp từng
shape, 552 814 086 tham số), DDIM + DIRE, ResNet-50, ACC/AP đều viết từ công
thức. Image Modal của nó **không** clone repo tác giả, và `selfcheck` assert
điều đó.

```bash
pip install -e ".[zero]"
python -m dire_zero info        # lịch, mốc, NFE, bảng tiền
python -m dire_zero keys        # đối chiếu 566 key với fixture (không cần checkpoint)
python -m dire_zero selftest    # gen → dire → train trên U-Net 32×32, ~20 giây
```

Dùng **dataset riêng trên ổ** (cây `DiffusionForensics`, tar.gz) thay vì tải LSUN:

```bash
python -m dire_zero peek    --tar F:/du-lieu/.../dire/test/lsun_bedroom/adm.tar.gz  # bên trong tar
python -m dire_zero dataset --root F:/du-lieu --license research-only   # kiểm, 0 đồng
python -m dire_zero ingest  --root F:/du-lieu                        # PNG 256² + manifest
#   → ./data/dire-zero-norm; tự sang F:/dire-zero-norm nếu ổ chính không đủ chỗ
#   (+ --strict: archive hỏng ⇒ dừng, mặc định giữ phần đọc được và ghi chú)
```

Trên Modal: `modal run -m dire_zero.modal_app::checkdata|selfcheck|bench|gen|dire|train|report`.
Ghi chú đầy đủ: [`docs/from-zero-clean-room.md`](docs/from-zero-clean-room.md).

## Tài liệu

| Tài liệu | Nội dung |
|---|---|
| [`docs/runbook-free.md`](docs/runbook-free.md) | Runbook 10 bước: 0 ₫ → model + demo, kèm cổng kiểm tra từng bước |
| [`docs/scratch-reproduction.md`](docs/scratch-reproduction.md) | Tái lập **từ đầu** (R1+R2+R3): 9 cổng vàng, kịch bản tải dữ liệu, chi phí theo GPU-giờ, hạn chế |
| [`docs/dataset-own.md`](docs/dataset-own.md) | Dùng **dataset riêng trên ổ** (`DiffusionForensics`): đọc tar.gz, nhãn theo generator, chặn rò rỉ, `dire/` bị từ chối, nhãn `R1+R2` vs `R1+R2+R3` |
| [`docs/demo-web-plan.md`](docs/demo-web-plan.md) | **Kế hoạch demo web**: 3 chế độ (gallery R0 / upload CPU / GPU), cổng kiểm trước khi công khai, lỗi "chấm ảnh RGB bằng model học DIRE" phải sửa, chữ bắt buộc trên UI, chi phí, chia PR D1–D5 |
| [`docs/figures-plan.md`](docs/figures-plan.md) | **Kế hoạch figure**: dựng lại 10 hình của paper (F1–F10) theo 4 đợt phụ thuộc dữ liệu, 8 hình paper **không có** (phân bố điểm, ROC/PR + CI, chất lượng–NFE, audit dữ liệu, calibration, ca sai), quy ước + định nghĩa "xong" |
| [`docs/from-zero-clean-room.md`](docs/from-zero-clean-room.md) | **Clean-room**: viết lại U-Net/DDIM/DIRE/ResNet-50 từ công thức, không dùng một dòng code tác giả — 4 lớp chứng minh, 3 lỗi tìm được trong code cũ, cách chạy và đọc số |
| [`docs/download-manual.md`](docs/download-manual.md) | Tải dữ liệu **bằng tay**: danh sách file, vị trí dưới `data/raw/`, tên dễ sai, kiểm offline, đẩy lên Volume |
| [`docs/runbook-local-manual.md`](docs/runbook-local-manual.md) | Runbook 11 bước trên **máy cá nhân**: kiểm máy → tải tay → ingest → bench → gen → dire → train, kèm tiền và cách sửa lỗi |
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
| `bash scripts/01_repro.sh` | **QUY TRÌNH 1** — chấm lại Table 2 (R0): data → eval 4 detector → báo cáo Δ so với paper |
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
| `bash tools/manim/render.sh --check` | Liệt kê 5 scene video giải thích DIRE (không render) |
| `bash tools/manim/render.sh --quality l` | Render video giải thích (nháp 480p15) — xem `tools/manim/README.md` |

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
| `train` | 4 | 5.74 GB | train lại **để đối chiếu** trọng số tác giả (FR-8) — không phải đường báo cáo |
| `trainx` | 7 | 11.24 GB | train lại detector cross-domain |
| `trainens` | 3 | 14.60 GB | train 3 detector ensemble |
| `rawx` | 4 | 4.48 GB | ảnh gốc để TỰ tính lại DIRE (cần GPU + ADM 2.5 GB) |
| `rawxens` | 6 | 12.10 GB | ảnh gốc cho ensemble |
| `ckpts` | 6 | 1.58 GB | toàn bộ checkpoint detector (mỗi file 269.4 MB) |
| **`gallery`** | **9** | **~0.67 GB** | **bộ ba ảnh gốc + recons + DIRE map** cho báo cáo (real/pndm/ldm) — KHÔNG cho eval |

| tổ hợp | file | size |
|---|---|---|
| **`e2,e4` (mặc định)** | **22** | **3.39 GB** |
| `e2,e3,e4,e5` — eval đầy đủ, không train | 57 | 10.35 GB |
| `e2,e3,e4,e4x,e5` | 69 | 12.37 GB |
| Tier A **cũ** (2 train + 2 val + 16 test) | 20 | **7.42 GB** |
| Toàn bộ cây đã audit | 104 | **61.40 GB** |
| ↳ trong đó dữ liệu **train** | — | **47.00 GB (77%)** |

Docs từng ghi "Tier A ≈ 40–70 GB" — đó là số của **cả cây** gán nhầm cho Tier A;
sai 5.4–9.4 lần. Vì 77% cây là dữ liệu train mà đường báo cáo chính chỉ cần
*eval checkpoint tác giả*, phần bỏ được rất lớn: **eval đầy đủ chỉ 10.35 GB**.
Chỉ khi làm **FR-8** (train lại để đối chiếu trọng số — xem `docs/adr-0004-retrain-for-reconciliation.md`)
mới cần thêm tier `train` (4 file / 5.74 GB).

Alias cũ vẫn chạy được: `ckpt`→`ckpts` · `A`→`e2,train` · `B`→`e3` · `C`→`rawx` ·
`D`→ rỗng (toàn bộ nội dung cũ nằm trong `NEVER`).

`NEVER` chặn theo mặc định: `recons/` · `*ddim*` (8 tar ablation số bước DDIM,
không có bản `dire/` tương ứng — 2 trong số đó chỉ **130 B**, tức rỗng ngay
trong dataset gốc) · `lsun_bedroom.pt` (weight ADM 2.5 GB không có trong cây
Baidu). Audit đo được **~25 GB** bị chặn (recons 24 GB + ddim 0.87 GB).

Một ngoại lệ **hẹp và có tên**: tier **`gallery`** được mang `recons/` cho báo
cáo (ảnh tái tạo `x̂` để đối chiếu với DIRE map). Nó khai báo **từng file** bằng
`File(allow_never=True)`, nên `*ddim*` và weight ADM **vẫn bị chặn** — kể cả khi
gõ `--paths ... --allow-never`. Lý do tồn tại: `NEVER` nghĩa là "không tier nào
tự động kéo về", không phải "file này vô dụng". Xem `tools/triplet_view.py`.

`expand()` khử trùng lặp theo **đường dẫn remote**, không theo object: `e4 ⊂ e4x`
nên `--tiers e4,e4x` cho 17 file chứ không phải 22.

Policy (tier mặc định, trần dung lượng, số container song song) nằm ở
[`src/dire_modal/tiers.yaml`](src/dire_modal/tiers.yaml) — cố ý đặt **trong
package** chứ không phải `configs/`, vì Modal chỉ đóng gói thư mục của module
được chạy nên function không thấy `configs/` ở repo root.

## Cấu trúc

```
src/dire_core/      # LÕI THUẦN — không import modal, không import torch ở tầng module
  preprocess.py     #   Resize 256 → CenterCrop 224 → Normalize  (một chỗ duy nhất)
  detector.py       #   đọc `fc.weight` từ chính checkpoint (1 logit vs 2 lớp)
  score.py          #   sigmoid / softmax → P(fake)
  labels.py         #   0_real / 1_fake
  batch.py          #   vòng lặp batch + `Detector` (dùng chung eval/CLI/API)
src/dire_local/     # CHẠY Ở MÁY NHÀ
  cli.py            #   `dire info|detect|batch|serve`
  server.py         #   FastAPI thuần (uỷ quyền bởi dire_modal.api)
src/dire_modal/     # CONTROL PLANE trên Modal
  api.py            #   wrapper @modal.fastapi_endpoint → /v1/detect, /healthz
  common.py         # Modal app, images, volumes, secrets
  manifest.py       # INVENTORY dữ liệu theo MỤC TIÊU (e0–e5, train, never)
  tiers.yaml        # POLICY: tier mặc định, trần GB, số container song song
  download.py       # tải Baidu → /raw (dry-run, guard dung lượng, resumable)
  pack.py           # /raw → WebDataset shards
  bench.py          # benchmark storage (Volume/R2) + throughput DIRE
  archive.py        # đẩy/kéo: R2, HF Hub, Kaggle + verify sha256
  train.py          # (TODO) FR-8: train lại để ĐỐI CHIẾU trọng số tác giả (spec.md §7)
  eval.py           # eval 15 generator bằng checkpoint có sẵn + CỔNG S2
  compute_dire.py   # tính DIRE bằng ADM DDIM inversion + gallery
  vendor/           # module DIRE gốc đã vendorize
tools/              # free_quota.py, storage_cost.py (thuần stdlib) + tiện ích .sh
                    #   local_download.sh = tải ở MÁY LOCAL (không cần Modal)
                    #   manim/ = video giải thích (5 scene, số đọc từ repo)
scripts/            # ĐÚNG 2 quy trình: 01_repro.sh · 02_improve.sh (+ setup/check/lib)
platforms/          # bootstrap cho kaggle / runpod / vast / hf-spaces
                    #   site/ = trang pitch deck tĩnh (12 slide, số do build.py sinh)
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
| Video giải thích (manim) | ✅ code 5 scene xong — **chưa render** (máy cần cairo/pango; hoặc image Modal riêng, xem `tools/manim/README.md`) |
| `demo_train` (train demo để quay video) | ✅ code + 29 test — **chưa chạy thật** (cần tier `train`) |
| Train lại (FR-8) | ⚠ **được phép nhưng chỉ để đối chiếu** — không dùng làm số của paper; xem `docs/adr-0004-retrain-for-reconciliation.md` |
| Stage-2 (compute DIRE toàn dataset) | ⏳ code viết theo API guided-diffusion, **chưa chạy thử**; cần ~30–60 A100-h |
| Demo HF Spaces | ✅ code + hướng dẫn; ⏳ chờ checkpoint & gallery |

## Trang giới thiệu (pitch deck)

`platforms/site/` — trang tĩnh 12 slide để thuyết trình: vấn đề → ý tưởng DIRE →
bằng chứng của paper → **trạng thái thật** → việc còn thiếu → giới hạn. Không CDN,
không build step, chạy offline.

```bash
python3 platforms/site/build.py        # sinh data.json từ TSV/manifest/demo_train
python3 -m http.server 8000 --bind 0.0.0.0 --directory platforms/site
```

HTML **không chứa con số nào** — mỗi số là một `[data-bind]` trỏ vào `data.json`,
mà `data.json` lại đọc từ `paper_table2.tsv`, manifest tier `e2` và
`demo_train.yaml`. Sửa dữ liệu mà quên build ⇒ `tests/test_site.py` đỏ. Ba tầng số
(*của paper* · *đo được* · *ước lượng demo*) luôn được in nhãn ngay cạnh số.
Chi tiết: [`platforms/site/README.md`](platforms/site/README.md).

## Cảnh báo

- **License**: dataset DiffusionForensics là dataset nghiên cứu; repo gốc đã archive (2025-07-08).
  Mirror kèm ghi nguồn và sẵn sàng gỡ theo yêu cầu — xem `docs/runbook-free.md` §5.
- **Không lạm dụng free tier**: không đa tài khoản, không dùng Colab free làm worker (vi phạm ToS).
- **Con số ước lượng**: mọi chỗ ghi "ước tính" cần đo lại bằng `tools/measure_dire.sh` và `tools/bench_storage.sh`.
