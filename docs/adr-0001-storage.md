# ADR-0001 — Vai trò của từng lớp lưu trữ

- **Trạng thái:** Accepted
- **Ngày:** 2026-10-14
- **Người quyết định:** chủ repo
- **Liên quan:** `docs/storage-comparison.md`, `docs/iaas-options.md`, ADR-0002

## Bối cảnh

Repo cần giữ: dữ liệu gốc tải từ Baidu (`.tar.gz`), WebDataset shards dùng cho
train (hot path), checkpoint + log, và bản mirror cho người khác tái lập.

**Dung lượng thật, đo bằng `download::audit` ngày 2026-10-21** (thay cho ước
lượng "Tier A ≈ 40–70 GB" trong bản đầu — ước lượng đó là của *cả cây*, sai
5.4–9.4 lần so với Tier A):

| | file | size |
|---|---|---|
| Tập eval mặc định `e2,e4` | 22 | **3.39 GB** |
| Eval đầy đủ không train `e2..e5` | 57 | **10.35 GB** |
| Toàn bộ cây đã audit | 104 | **61.40 GB** (cận dưới) |
| ↳ dữ liệu train | — | **47.00 GB (77%)** |

Vì 77% cây là dữ liệu train mà ràng buộc dự án là *chỉ cải thiện, không train
lại*, nhu cầu thật chỉ **3–10 GB** — dưới cả free tier của Volume (1 TiB) lẫn
R2 (10 GB). Hệ quả: quyết định lưu trữ **không còn là bài toán giá** mà là bài
toán *chia sẻ được hay không*.
Các phương án khả dụng: Modal Volume (1 TiB free), Cloudflare R2 (10 GB free, egress $0),
Kaggle Datasets (200 GB/dataset), HF Hub (~300 GB/repo), ổ local (máy mỏng).

Nháp đầu tiên (trước khi khảo sát) giả định: *Volume = nhà, R2 = cold archive toàn bộ dataset, Local = bản sao*.

## Quyết định

**Đảo vai trò R2. Phân tầng theo tính chất dữ liệu, không theo dung lượng:**

| Lớp | Chứa gì | Vì sao |
|---|---|---|
| **Modal Volume `/raw`** | bản gốc `.tar.gz` + manifest | staging nhanh, 1 TiB free, POSIX thật, pipeline đang chạy sẵn |
| **Modal Volume `/shards`** | WebDataset shards | hot path train — cần POSIX + throughput, `$0` |
| **Modal Volume `/ckpt`** | checkpoint mới nhất, log | cùng Volume = latency thấp khi train |
| **HF Hub (public)** | mirror tập eval + manifest | free, có URL công khai, tải được từ mọi nền tảng, chống link rot. 3.39 GB dư sức trong ~300 GB/repo |
| **Kaggle Datasets (public)** | mirror phụ theo từng thư mục con | 200 GB/dataset, mount trực tiếp trong notebook Kaggle |
| **Cloudflare R2** | **chỉ** checkpoint đã chốt + manifest sha256 | nhỏ, không tái tạo được, egress $0 — đúng chỗ để trả tiền (10 GB free là đủ). Sau số đo: 10 GB free cũng đủ chứa *tập eval mặc định* 3.39 GB, nhưng R2 không cho URL công khai free ⇒ vẫn xếp sau HF cho mục đích mirror |
| **Local** | không giữ dataset | máy mỏng; upload 60 GB từ nhà tốn 1.3–4.5 h |

**Quy tắc ngưỡng (bắt buộc theo):**

1. Tổng Volume < **900 GiB** → giữ nguyên, `$0`.
2. Sắp vượt 900 GiB → đẩy `raw/` lên HF rồi **xoá khỏi Volume**; giữ `shards/` trên Volume.
3. Vượt **1.227 GB** (điểm hoà vốn) → mọi khối lượng lớn thuộc về R2/HF, không phải Volume.
4. Số file trong 1 Volume > **50k** → chuyển Volume sang `version=2` (beta) hoặc shard hoá thêm.
5. Commit từ **>5 container đồng thời** → gom commit, hoặc dùng Volume v2.
6. Checkpoint: Volume giữ bản mới nhất; **bản đã chốt đẩy R2**; trước khi xoá Volume, chạy `verify_hf`.

## Hệ quả

**Tích cực**
- Chi phí lưu trữ hiện tại = **$0/tháng** (Volume 1 TiB free + HF/Kaggle free + R2 10 GB free).
- Mirror công khai miễn phí thay vì trả R2 cho 150 GB dữ liệu.
- Dữ liệu nằm trên S3 API/HTTP chuẩn → đọc được từ RunPod/Vast/Kaggle/Lightning (phục vụ ADR-0002).

**Tiêu cực / chấp nhận**
- Phải giữ 2 bản (Volume + mirror) và tốn thời gian upload lần đầu (~150 GB).
- Phụ thuộc ToS của HF/Kaggle: dataset public, có thể bị yêu cầu gỡ → **giữ bản gốc trên Volume**.
- R2 không còn là "bản sao toàn bộ" → nếu Volume bị xoá **và** mirror bị gỡ, phải tải lại từ Baidu (chậm).
- Volume v1 không versioning → không có rollback cho dữ liệu trên Volume.

**Việc phải làm kèm theo**
- `_manifest_sha256.tsv` được commit vào git (nguồn sự thật để verify mọi bản sao).
- `archive.py` có `verify_hf` (đối chiếu sha256) — chạy sau mỗi lần mirror.
- `tools/storage_cost.py` chạy lại khi vượt 500 GiB (giá sẽ khác sau mốc free).

## Phương án đã cân nhắc & loại

| Phương án | Vì sao loại |
|---|---|
| Toàn bộ dữ liệu trên R2 (nháp đầu) | 150 GB > 10 GB free → ~$2.1/tháng, trong khi HF/Kaggle cho cùng thứ miễn phí; và R2 không có URL công khai "đẹp" (r2.dev chỉ để test, bị rate-limit) |
| Toàn bộ trên Volume | Không chia sẻ được (Volume không có URL công khai); không phải backup |
| Local làm bản chính | Máy mỏng, 1 điểm chết, upload từ nhà tốn nhiều giờ |
| Git LFS trong repo | File ≤50 GB, git không hợp với 100+ GB nhị phân |
