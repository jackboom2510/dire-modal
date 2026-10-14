# Lưu trữ dataset / pretrained model: Modal Volume vs R2 vs Local (và Kaggle/HF)

> **Ngày:** 2026-10-14 · **Ngữ cảnh:** `dire-modal`, Tier A ≈ 40–70 GB (raw `.tar.gz`), tổng raw+shards ≈ 100–150 GB.
> **Kết luận đã thay đổi so với bản nháp đầu tiên** — xem §5 "Kiến trúc chốt".

---

## TL;DR

1. **Modal Volume free tier (1 TiB) đã bao trùm toàn bộ nhu cầu hiện tại** → chi phí lưu trữ hot/warm = **$0/tháng**.
   Điểm hoà vốn Volume vs R2 Standard = **1.227 GB ≈ 1.20 TiB** (do `tools/storage_cost.py` tính); trên mốc đó R2 rẻ hơn và
   khoảng cách tăng nhanh — ở 3 TB thì R2 rẻ hơn ~4× ($45.9 vs $184.3/tháng).
2. **R2 không còn là "cold archive của dataset"** — vai trò của nó co lại thành **kho nhỏ chứa checkpoint + manifest sha256**
   (10 GB free, egress $0). Lý do: dataset **không tái tạo được miễn phí** (tải lại từ Baidu), còn checkpoint thì nhỏ và quý.
3. **Kaggle Datasets + HF Hub là mặt phẳng chia sẻ miễn phí tốt hơn R2**: 200 GB/dataset (Kaggle) và ~300 GB/repo (HF Hub),
   public, mount trực tiếp trong notebook Kaggle. R2 10 GB free không đủ chứa dataset.
4. **Local** (máy mỏng của bạn): chỉ control plane. Không giữ dataset, không giữ ảnh gốc — upload 60 GB từ nhà mất 1.3–4.5 h.
5. **Sai lầm đắt nhất không phải tiền storage mà là GPU chờ I/O**: một run 20 h trên A100 mà để mất 15% thời gian vì I/O
   ≈ **$7.5** — lớn hơn tiền lưu trữ cả tháng.

---

## 1. Số liệu nền tảng (đã kiểm chứng 2026-10-14)

| | **Modal Volume (Starter)** | **Cloudflare R2** | **Local (ổ/NAS)** | **Kaggle Datasets** | **HF Hub** |
|---|---|---|---|---|---|
| Giá lưu | $0.09/GiB/tháng, **1 TiB/tháng đầu miễn phí** | $0.015/GB/tháng (Standard); IA $0.01/GB + $0.01/GB retrieval, tối thiểu 30 ngày | $0 marginal (~$6–7/tháng khấu hao ổ 4 TB NVMe) | **free** (200 GB/dataset, 200 GB tổng private) | **free** (public, ~300 GB/repo, file ≤50 GB) |
| Free tier | 1 TiB | 10 GB | — | 200 GB | ~300 GB/repo |
| Phí thao tác | Không tính request (chỉ overhead commit/reload) | Class A $4.50/triệu · Class B $0.36/triệu | 0 | 0 | 0 |
| Egress | $0.04/GiB (1 TiB/tháng free) | **$0** | 0 | 0 (public) | 0 |
| Giới hạn kỹ thuật | v1: khuyến nghị ≤50k file, **hard 500k inode**; v2 (beta): không giới hạn file, ≤262.144 file/thư mục, file <1 TiB | object ≤5 TiB; single-part ≤5 GiB; 1 writer/giây/key | dung lượng ổ, 1 máy, 1 điểm chết | 50 file top-level; upload ≤200 GB | <100k file/repo khuyến nghị |
| Semantics | POSIX thật + commit/reload | Object store (S3 API); qua `CloudBucketMount` = mountpoint-s3 | POSIX đầy đủ | Read-only khi mount trong notebook | Git/LFS |
| Tốc độ | tới 2.5 GB/s | tốt khi **tuần tự**; latency cao với random/small-file | NVMe vài GB/s | đọc local trong Kaggle (nhanh) | qua mạng |
| Rủi ro | Không versioning/snapshot; **bị tính phí tới 4 ngày sau khi xoá**; v2 beta "data loss may be possible" | object store, có versioning/lifecycle | mất ổ = mất hết | quota/ToS; dataset phải public nếu vượt 200 GB private | public mặc định; private cần PRO $9/th |
| Tuyên bố chính thức | [Modal khuyến nghị **lưu model weights trong Volume**](https://modal.com/docs/guide/model-weights) | — | — | — | — |

### Bảng chi phí theo dung lượng (raw + shards)

| Kịch bản | Modal Volume | R2 Standard | R2 IA | Kaggle + HF |
|---|---|---|---|---|
| **S — 60 GB** (Tier A raw) | **$0** | $0.75/th | $0.60/th | **$0** |
| **M — 150 GB** (raw + shards) | **$0** | $2.10/th | $1.50/th | **$0** |
| L — 1.2 TB (Tier A+B+C+D) | $15.84/th | $17.85/th | $11.90/th | cần nhiều repo/dataset |
| XL — 3 TB | $184.32/th | **$45.89/th** | $30.60/th | cần nhiều repo/dataset |

→ **Điểm hoà vốn Volume vs R2 Standard = 1.227 GB ≈ 1.20 TiB** (dưới mốc đó Volume rẻ hơn — thực tế là miễn phí tới 1 TiB).
Đến ~1.2 TB hai bên gần như bằng nhau; vượt xa thì R2 thắng rõ rệt.
Sinh số bằng: `python3 tools/storage_cost.py --size-gb 150` (hoặc `--size-gb 3000`).

---

## 2. Bộ tiêu chí (18 tiêu chí, 6 nhóm)

| Nhóm | Tiêu chí | Modal Volume | R2 | Local | Kaggle/HF |
|---|---|---|---|---|---|
| **Chi phí** | Giá/GB/tháng | $0 (≤1 TiB) | $0.015 | ~$0.0018 | $0 |
| | Free tier | **1 TiB** | 10 GB | — | 200 GB / 300 GB |
| | Phí ops | 0 | Class A/B | 0 | 0 |
| | Egress | $0.04/GiB (1 TiB free) | **$0** | 0 | 0 |
| | Chi phí ẩn | tính phí 4 ngày sau xoá | min 30 ngày (IA) | điện/ổ | quota, ToS |
| **Hiệu năng** | Tuần tự | 2.5 GB/s | tốt | NVMe | local-read trong Kaggle |
| | Random/small-file | tốt (v2 tốt hơn) | **kém** | tốt nhất | — |
| | Ảnh hưởng GPU | thấp | trung bình (mount) | thấp | thấp (mount sẵn) |
| | Cold start/attach | attach nhanh (v1 ≤50k file) | mount chậm hơn | 0 | 0 |
| | Writer đồng thời | v1 ≤5 khuyến nghị; **v2 hàng trăm** | 1 writer/giây/key | 1 máy | 1 uploader |
| **Kỹ thuật** | POSIX | ✅ | ⚠️ mountpoint-s3 (cấm append/seek, phải `truncate`) | ✅ | read-only |
| | webdataset/DataLoader | ✅ trực tiếp | ⚠️ được (đọc tuần tự) nhưng chậm hơn | ✅ | ✅ |
| | Giới hạn file | 500k inode (v1) | không | inode ổ | 50 file top-level |
| **Bền & an toàn** | Versioning/rollback | ❌ | ✅ (+lifecycle) | ❌ | ✅ git history |
| | Rủi ro mất | xoá = mất | rất thấp | 1 điểm chết | thấp |
| | Verify | tự làm (sha256) | ETag + sha256 | tự làm | LFS hash |
| | DR | phải tự sao | object store | — | public = có bản sao |
| **Vận hành** | Setup | cần modal CLI | S3 API, presigned URL | trực tiếp | tài khoản free |
| | Chia sẻ ngoài | ❌ (không URL công khai) | ✅ presigned / custom domain (+ [r2.dev chỉ để test, bị rate-limit](https://developers.cloudflare.com/r2/platform/limits/)) | ❌ | ✅ URL công khai |
| | Multi-user | workspace | S3 keys | 1 máy | public |
| **Chiến lược** | Lock-in | cao (API Modal) | thấp (S3 chuẩn) | — | trung bình |
| | Mở rộng >1 TiB | $0.09/GiB | $0.015/GB | mua ổ | xin tăng quota |

---

## 3. Đo lường (cần chạy để thay số ước lượng)

```bash
# Volume vs R2 qua CloudBucketMount: throughput tuần tự + latency small-file
./scripts/bench_storage.sh

# Chi phí + điểm hoà vốn, tự đọc manifests/*.tsv nếu có size thật
python3 tools/storage_cost.py --size-gb 150 --months 12
python3 tools/storage_cost.py --from-manifest manifests/raw.tsv
```

Kết quả đổ vào `manifests/bench_storage_<ngày>.json`. **Chưa có số đo trong phiên soạn tài liệu**
(sandbox không có modal CLI) → mọi con số "2.5 GB/s" là số nhà cung cấp công bố, không phải số đo tại chỗ.

---

## 4. Quy tắc ngưỡng (đề xuất, ghi vào ADR-0001)

| Tình huống | Hành động |
|---|---|
| Tổng Volume < 900 GiB | Giữ nguyên, tất cả trên Volume — **$0** |
| Sắp vượt 900 GiB | Đẩy `raw/` sang R2/HF rồi xoá khỏi Volume; giữ `shards/` (hot path) trên Volume |
| Cần chia sẻ công khai | HF Hub + Kaggle Datasets (free) — **không** dùng R2 làm mirror chính |
| Checkpoint | Volume giữ bản mới nhất; **bản đã chốt đẩy R2** (nhỏ, quý, egress $0) |
| Số file > 50k | Chuyển Volume sang `version=2` (beta) **hoặc** đảm bảo shard hóa (2000 ảnh/shard như `pack.py`) |
| Commit từ nhiều container | Gom commit (v1 ≤5 đồng thời) hoặc dùng v2 |

---

## 5. Kiến trúc chốt (sau khi biết repo cần multi-IaaS + chia sẻ công khai)

```
Baidu/Rec ──► Volume /raw  (Modal, 1 TiB free, staging + verify sha256)
                 │
                 ├──► Volume /shards   ← hot path train (WebDataset, POSIX, $0)
                 ├──► HF Hub + Kaggle  ← MIRROR CÔNG KHAI free (chia sẻ + tái lập)
                 └──► R2 (10 GB free)  ← CHỈ checkpoint đã chốt + manifest sha256
Local = git + modal CLI + mẫu nhỏ. Không giữ dataset.
```

**Thay đổi so với nháp đầu:** R2 từ "cold archive toàn bộ dataset" → "vault nhỏ cho checkpoint/manifest";
Kaggle/HF từ "không có trong bảng" → "mặt phẳng chia sẻ chính". Lý do:
- Dataset ở R2 tốn tiền khi >10 GB, trong khi Kaggle/HF cho **miễn phí và có sẵn URL công khai**.
- Checkpoint là dữ liệu **nhỏ + không tái tạo được** → đúng chỗ để trả tiền (và 10 GB free là đủ).
- Volume là chỗ **nhanh nhất trong lúc chạy job Modal** → giữ làm cache.

---

## 6. Nguồn

- [Modal pricing](https://modal.com/pricing) — Volume $0.09/GiB-mo, 1 TiB free; egress $0.04/GiB, 1 TiB free (Starter)
- [Modal Volumes docs](https://modal.com/docs/guide/volumes) — v1 ≤50k file khuyến nghị / 500k inode; v2 beta không giới hạn file; ≤5 writer đồng thời (v1)
- [Modal model weights guide](https://modal.com/docs/guide/model-weights) — khuyến nghị chính thức
- [Modal Cloud bucket mounts](https://modal.com/docs/guide/cloud-bucket-mounts) — mountpoint-s3: cấm append, cấm seek+write, phải `truncate`
- [Cloudflare R2 pricing](https://developers.cloudflare.com/r2/pricing/) — $0.015/GB, 10 GB free, egress $0
- [Cloudflare R2 limits](https://developers.cloudflare.com/r2/platform/limits/) — 5 TiB/object, 5 GiB single-part, r2.dev bị rate-limit
- [Kaggle datasets docs](https://www.kaggle.com/docs/datasets) — 200 GB/dataset, 200 GB private, 50 file top-level
- [HF Hub size limits](https://discuss.huggingface.co/t/is-there-a-size-limit-for-dataset-hosting/14861) — ~300 GB/repo, file ≤50 GB
