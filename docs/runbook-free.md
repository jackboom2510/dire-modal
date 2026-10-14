# Runbook: từ 0 ₫ → có model + demo công khai

> Kèm theo `docs/iaas-options.md` (chọn nền tảng) và `docs/storage-comparison.md` (chọn chỗ lưu).
> Mọi bước đều ghi **chi phí dự kiến** và **cổng kiểm tra** (không được bỏ qua cổng).

## Bảng tổng — nên tốn bao nhiêu

| Bước | Nền tảng | Chi phí | Cổng kiểm tra (pass mới đi tiếp) |
|---|---|---|---|
| 1. Chuẩn bị | — | $0 | `modal secret list` thấy 4 secret |
| 2. Tải dữ liệu | Modal CPU | ~$1–2 credit | `download::verify` in "All OK." |
| 3. Manifest sha256 | Modal CPU | ~$0.5 | Có `_manifest_sha256.tsv`, số file đúng |
| 4. Pack shards | Modal CPU | ~$1–2 | `pack::stats` đếm đủ ảnh mỗi generator |
| 5. Mirror công khai | Modal CPU + egress | ~$0–1 | `verify_hf` → `mismatch=0, missing=0` |
| 6. Smoke test | **Kaggle** | **$0** | Notebook chạy 3 step train, loss giảm |
| 7. Train thật | Modal A100 | ~$8–20 (3–8 h) | ACC/AP trên val ≥ ngưỡng chọn trước |
| 8. Eval 16 generator | Modal L4/A100 | ~$2–4 | Bảng ACC 16 dòng |
| 9. Gallery + demo | Modal A100 + HF Spaces | ~$0.5 + $0 | Space public trả kết quả thật |
| **Tổng** | | **≈ $13–30** (vừa $30/tháng) | |

> **Đo trước khi cam kết:** `./scripts/06_measure_dire.sh` cho biết số giờ GPU thật của stage-2.
> Nếu ước lượng > 30 A100-giờ → **không làm stage-2 toàn bộ**; chỉ làm 2–5k ảnh.

---

## 1. Chuẩn bị (một lần, ~30 phút, $0)

```bash
git clone <repo> && cd dire-modal
cp .env.example .env        # rồi điền các token bên dưới
pip install -e .

# Đăng nhập Modal (mở trình duyệt)
modal token new

# Điền token vào .env:
#   BAIDU_BDUSS/BAIDU_STOKEN  — cookie Baidu Netdisk (xem hướng dẫn dưới)
#   R2_*                      — Cloudflare R2 → API Tokens
#   HF_TOKEN                  — https://huggingface.co/settings/tokens (write)
#   KAGGLE_USERNAME/KAGGLE_KEY— https://www.kaggle.com/settings → Create New Token

./scripts/00_setup_secrets.sh
modal secret list                     # ✅ CỔNG 1: thấy baidu-cred, r2-cred, hf-cred, kaggle-cred
```

**Lấy BDUSS/STOKEN:** đăng nhập pan.baidu.com → DevTools → Application → Cookies → sao chép `BDUSS` và `STOKEN`.

---

## 2. Tải dữ liệu về Volume (Modal CPU, ~$1–2)

```bash
# (a) Dò cây remote 1 lần để biết đường dẫn thật
modal run -m dire_modal.download::ls_remote --path /
# → chép kết quả vào src/dire_modal/manifest.py::BAIDU_ROOT

# (b) Tải Tier A + checkpoints
./scripts/01_fetch_baidu.sh

# (c) Verify toàn bộ
./scripts/02_verify.sh                 # ✅ CỔNG 2: "All OK."
```

Chi phí thực tế chủ yếu là **thời gian chờ Baidu** (server chậm); CPU 0.25 core ≈ $0.012/giờ → rẻ.
`fetch_one` là resumable: chạy lại sẽ skip file đã có và `gzip -t` pass.

---

## 3. Manifest sha256 — nguồn sự thật cho mọi mirror (~$0.5)

```bash
modal run -m dire_modal.archive::make_manifest
# → /raw/_manifest_sha256.tsv  (source, path, size_bytes, sha256)

modal volume get dire-raw /_manifest_sha256.tsv manifests/   # ✅ CỔNG 3
```

Giữ file này trong git. Mọi bản sao (HF, Kaggle, R2) sẽ được đối chiếu với nó.

---

## 4. Pack WebDataset shards (~$1–2)

```bash
./scripts/03_pack_shards.sh
# ✅ CỔNG 4: pack::stats in số ảnh từng generator, tổng khớp với manifest
```

Thiết kế đã tính tới giới hạn Modal Volume v1 (≤50k file khuyến nghị): shard 2000 ảnh → 100k ảnh chỉ tốn ~50 inode.

---

## 5. Mirror công khai (HF + Kaggle) — phần "chia sẻ cho người khác"

### 5a. Hugging Face (mirror chính, ~300 GB/repo)

```bash
# Tạo repo dataset (public)
huggingface-cli repo create diffusionforensics-dire --type dataset --public

modal run -m dire_modal.archive::push_hf --repo-id <user>/diffusionforensics-dire

# Đối chiếu sha256 Volume ↔ Hub
modal run -m dire_modal.archive::verify_hf --repo-id <user>/diffusionforensics-dire
# ✅ CỔNG 5a: mismatch=0, missing=0
```

### 5b. Kaggle (mirror phụ, 200 GB/dataset, **≤50 file top-level**)

Đẩy theo từng thư mục nhỏ (mỗi thư mục có ít file `.tar.gz`):

```bash
modal run -m dire_modal.archive::push_kaggle \
  --dataset-id <user>/dire-tier-a-train-lsun \
  --src-subdir dire/train/lsun_bedroom
```

### 5c. R2 (chỉ checkpoint + manifest — 10 GB free)

```bash
modal run -m dire_modal.archive::push_r2 --prefix diffusion-forensics/v1
# mặc định KHÔNG đẩy .tar.gz (vì R2 free chỉ 10 GB); muốn đẩy thêm: --include-tar
```

### ⚠ License — đọc trước khi mirror

Dataset DiffusionForensics là **dataset nghiên cứu**; repo gốc **đã archive 2025-07-08** nên khó xin xác nhận.
Khi mirror toàn bộ, bạn tự chịu trách nhiệm. Bắt buộc làm:

1. Ghi rõ nguồn + bài báo + link Baidu/Rec gốc trong mô tả dataset.
2. Ghi điều kiện: **nghiên cứu / phi thương mại**, không bán, không dùng cho sản phẩm.
3. Nêu rõ dataset mirror **để chống link rot**, và bạn sẽ gỡ nếu tác giả yêu cầu.
4. Không mirror bất cứ thứ gì bạn không có quyền phân phối (ảnh gốc LSUN/ImageNet/CelebA-HQ thuộc bên thứ ba).

---

## 6. Smoke test trên Kaggle ($0, không tốn credit)

1. Upload `platforms/kaggle/notebook.ipynb` lên Kaggle → *Settings → Accelerator → GPU P100*.
2. Nếu muốn notebook đọc dữ liệu trực tiếp: tạo Kaggle Dataset từ mirror ở bước 5b rồi *Add Data* vào notebook.
3. Chạy hết notebook.
   ✅ **CỔNG 6**: cell cuối in `[ok] đường đi dữ liệu hoạt động — 3 bước trên cuda`

Kiểm tra quota trước khi chạy:

```bash
python3 tools/free_quota.py --used kaggle=0
```

---

## 7. Train thật trên Modal (~$8–20)

```bash
# Xem còn bao nhiêu credit TRƯỚC khi chạy
python3 tools/free_quota.py --used modal=0

# Train (chạy nền, không cần giữ terminal)
modal run --detach -m dire_modal.train::main --exp lsun_adm_repro

# Theo dõi
modal app logs dire
```

Cơ chế bảo vệ đã có trong thiết kế: checkpoint mỗi epoch + `vol_ckpt.commit()` → **nếu hết credit giữa chừng,
vẫn resume được** ở lần chạy sau (kể cả trên nền tảng khác, miễn là shards truy cập được).

> Nếu `train.py` chưa port xong: đây là việc còn TODO trong repo. Xem `references` của DIRE gốc
> (`train.py`, `utils/trainer.py` đã vendorize trong `src/dire_modal/vendor/`).

✅ **CỔNG 7**: ACC/AP val đạt ngưỡng bạn đặt trước khi chạy (ghi ngưỡng vào PR description).

---

## 8. Eval 16 generator (~$2–4)

```bash
./scripts/04_eval_pretrained.sh
# ✅ CỔNG 8: có bảng ACC cho 16 generator; so với paper (~99% trên ADM-LSUN)
```

Chạy song song bằng `.map()` — mỗi generator 1 container, scale-to-zero khi xong.

---

## 9. Gallery + demo công khai (~$0.5 + $0)

```bash
# 9a. Sinh gallery trên GPU (cần checkpoint ADM)
modal run -m dire_modal.compute_dire::make_gallery --n 12 --steps 20
modal volume get dire-shards /_gallery ./platforms/hf-spaces/gallery

# 9b. Đẩy checkpoint lên HF để Space tải được
modal volume get dire-ckpt /lsun_adm_repro/last.pt ./
huggingface-cli upload <user>/dire-detector ./last.pt dire_resnet50.pth

# 9c. Tạo Space (xem platforms/hf-spaces/README.md)
```

✅ **CỔNG 9**: Space public mở được, upload 1 ảnh → ra P(real)/P(fake); tab viz hiện ảnh với chú thích.

---

## 10. Vòng lặp hàng tháng

```bash
python3 tools/free_quota.py --used modal=12 kaggle=20 lightning=0     # còn gì?
python3 tools/storage_cost.py --from-manifest manifests/raw.tsv       # lưu trữ tốn gì?
```

---

## Khi hết credit: phao cứu sinh

| Tình huống | Làm gì |
|---|---|
| Hết $30 giữa lúc train | Resume trên **Lightning free** (A100, restart mỗi 4 h — checkpoint mỗi epoch đã lo được) |
| Cần chạy nhiều giờ liên tục | **Thunder $1.09/h** → Vast ~$0.8–1.5/h → RunPod $1.79/h (A100 80GB), kèm mount R2/HF để lấy dữ liệu |
| Không có thẻ quốc tế | VNSO A100 40GB 29.000₫/h (~$1.12) |
| Cần TPU miễn phí | **TRC**: apply, phải open-source nghiên cứu, hàng đợi có thể dài |
| Chỉ có thời gian, không có tiền | Kaggle 30 h/tuần + Colab (tương tác) + Lightning → đủ để dev, **không** đủ cho stage-2 |

## Những việc **không** làm (để không mất tài khoản / mất dữ liệu)

1. ❌ Dùng nhiều tài khoản Kaggle/Colab để lấy thêm quota (vi phạm ToS → ban).
2. ❌ Dùng Colab free làm worker SSH/remote desktop (ToS cấm rõ ràng).
3. ❌ Dựng pipeline phụ thuộc Kaggle (không SSH, mất máy khi hết phiên).
4. ❌ Coi Modal Volume là backup (không versioning; xoá là mất; vẫn tính phí 4 ngày).
5. ❌ Đẩy >10 GB lên R2 mà không kiểm tra (`push_r2` sẽ cảnh báo).
6. ❌ Cam kết stage-2 khi chưa chạy `06_measure_dire.sh` (con số 30–60 A100-h mới là ước lượng).
