# IaaS cho DIRE (diffusion-generated image detection) — so sánh & chọn nền tảng

> **Ngày lập / ngày truy cập nguồn:** 2026-10-14
> **Ngữ cảnh:** repo `dire-modal` — tải DiffusionForensics, pack WebDataset,
> fine-tune ResNet-50 detector (DIRE trên LSUN-Bedroom ADM), eval 16 generator, demo public.
> **Dung lượng thật (audit 2026-10-21):** tập eval mặc định `e2,e4` = **3.39 GB** / 22 file;
> eval đầy đủ `e2..e5` = **10.35 GB**; toàn cây = **61.40 GB** (77% là dữ liệu train).
> Con số "Tier A ≈ 40–70 GB" trong bản đầu là của *cả cây*, sai 5.4–9.4 lần.
> **Ngân sách mục tiêu của chủ repo: 0 ₫ (chỉ free tier).** Ưu tiên: có kết quả nhanh. Có thẻ quốc tế. Cần demo public.

---

## TL;DR

1. **Modal là engine chính** cho repo này: `$30/tháng` credit + **1 TiB Volume miễn phí** + scale-to-zero, đúng với kiến trúc `@app.function` đang có. Nhưng **cần cảnh giác**: `$30` ≈ **12 giờ A100 80GB** — đủ cho train Tier A, **không đủ** cho stage-2 (tự tính DIRE cho ~200k ảnh).
2. **Kaggle là nền tảng dev rẻ nhất**: 30 h GPU/tuần, khởi động tức thì, mount dataset 200 GB miễn phí. Nhược điểm chí mạng: **không có SSH, không chạy được worker dài hạn**, và không có storage bền gắn với job.
3. **Nếu phải chọn 1 chỗ để *mua* GPU-giờ**: Thunder Compute `$1.09/h` (A100 80GB), Vast `~$0.80–1.50/h`, RunPod `$1.79/h` — rẻ hơn Modal `$2.50/h` từ **30–55%**. Đổi lại: bạn tự lo Docker/ops, không có Volume, không có `.map()`.
4. **Trả phí hyperscaler là sai**: Azure A100 `$3.67–4.10/h`, AWS p4d `$4.10/GPU-h`, GCP `~$3.28/GPU-h` — đắt hơn 2–4× so với chuyên gia, và **credit trial GCP không gắn được GPU vào VM**.
5. **Với 0 ₫, làm được:** train detector trên ảnh DIRE *có sẵn* + eval + demo public + mirror dữ liệu. **Không làm được:** tự tính lại DIRE (DDIM inversion) cho toàn bộ dataset — phải thu nhỏ còn ~2–5k ảnh để kiểm chứng code.

---

## 1. Số liệu đã kiểm chứng (2026-10-14)

Quy ước: `$` = USD. VNĐ quy đổi theo tỷ giá giả định **26.000 ₫/USD** (giá VN hay niêm yết bằng VNĐ).
Mọi con số dưới đây lấy từ trang chính thức của nhà cung cấp trong ngày 2026-10-14, trừ khi ghi rõ "nguồn so sánh".

### 1.1 GPU theo giờ (chuẩn hoá $/giờ, GPU mạnh nhất–nhỏ nhất)

| Nền tảng | A100 80GB | H100 80GB | A100 40GB | L40S | L4 | T4 | Entry (3090/4090) | Ghi chú |
|---|---|---|---|---|---|---|---|---|
| **Modal** (serverless) | **$2.50** | **$3.95** | $2.10 | $1.95 | $0.80 | $0.59 | — | per-second, tính cả khi container "warm" 60s |
| **HF Jobs** (serverless) | **$2.50** | — (có H200 $5.00) | — | $1.80 | $0.80 | $0.40 | — | per-minute, **build không tính tiền** |
| **RunPod Pods** | **$1.79** | $2.89 (PCIe) / $3.99 (SXM) | — | $1.09 | $0.59 | — | 3090 $0.50 · 4090 $0.89 | per-second, 30+ region |
| **RunPod Serverless** | $2.72 | $4.79 | — | $1.75 | $0.69 | — | 4090 $1.10 | FlashBoot, sub-200 ms claim |
| **Vast.ai** (marketplace) | ~$0.80–1.50\* | từ $1.73 (median $2.93) | — | từ $0.43 | từ $0.20 | — | 3090 từ $0.12 · 4090 từ $0.36 | giá do host đặt, biến động |
| **Lambda** | $2.79 | $3.99 | $1.99 | — | — | — | A6000 $1.09 · A10 $1.29 | **không phí egress**, SSD lớn kèm theo |
| **Thunder Compute** | **$1.09** | $3.20 (PCIe) | — | — | — | — | — | per-minute, **không cần xin quota** |
| **Nebius** | — | $4.50 (từ 2026-10-01) | — | — | — | — | preemptible $2.15 | tăng giá 2 lần trong 3 tháng |
| **CoreWeave** | — | $6.16 (node 8 GPU) | — | — | — | — | spot ~$2.46 | **chỉ bán node 8 GPU** |
| **Baseten** (serverless) | — | ~$6.50 | — | — | — | ~$0.63 | — | per-minute; SOC2/HIPAA |
| **Replicate** | ~$5.04 | — | — | — | — | ~$0.81 | — | model public: chỉ tính giờ "active" |
| **Azure** | $3.67–4.10 | $6.98 | — | — | — | $0.53 | — | cần **xin quota N-series**; spot A100 ~$1.20 |
| **AWS** | — | $12.29 (p5) | $4.10 (p4d) | — | — | — | — | GPU chỉ bán theo node 8 GPU |
| **GCP** | — | — | ~$3.28 (a2-highgpu)\* | — | — | — | — | trial account **không gắn được GPU vào VM** |
| **VNSO** (VN) | 49.000₫ ≈ $1.88 | — | 29.000₫ ≈ **$1.12** | — | — | — | — | kèm 1–2 TB NVMe, thanh toán VNĐ |
| **Sunteco** (VN) | ~70.000₫ ≈ $2.69 | ~80.000₫ ≈ $3.08 | ~50.000₫ ≈ $1.92 | — | — | ~15.000₫ ≈ $0.58 | 4090 ~40.000₫ ≈ $1.54 | VAT/tháng có thể rẻ hơn |

\* = ước lượng từ nguồn so sánh, **chưa xác minh trên trang chính thức** — xem §8.

### 1.2 Free tier & quota (điểm mấu chốt với ngân sách 0 ₫)

| Nền tảng | Free cho ai | Quy mô | Giới hạn phiên | Storage bền kèm free |
|---|---|---|---|---|
| **Kaggle** | mọi tài khoản | **~30 h GPU/tuần** (P100 16 GB), reset hàng tuần; có TPU v3-8 quota riêng | ~9–12 h/phiên, idle 60 phút, **không SSH** | **200 GB/dataset**, 200 GB tổng dataset private |
| **Colab** | mọi tài khoản | T4 free; Pro $9.99/th = 100 CU; Pro+ $49.99 = 600 CU; PAYG $9.99/100 CU (CU hết hạn sau 90 ngày) | 12 h, idle 90 phút; **free tier cấm SSH/remote desktop/distributed worker** | Google Drive (15 GB free) |
| **Modal** | Starter $0/th | **$30 credit/tháng** (tự reset) ≈ 12 h A100-80GB hoặc ~50 h T4 | không có giới hạn phiên (job tới 24 h, `--detach`) | **Volume 1 TiB/tháng free** |
| **Lightning AI** | Free | tới 30 credit/tháng (nhiều người chỉ nhận ~15) ≈ ~22 h T4 | **Studio restart mỗi 4 h** — nhưng có checkpoint/resume | 200 GB persistent (mọi tier) |
| **HF Jobs** | user có credit dương (PRO $9/th bao gồm credit) | tính theo phút; **build free** | mặc định timeout **30 phút** (phải tự đặt) | eph. storage lớn (1000 GB với A100) nhưng **không bền** |
| **HF Hub** | public repo | ~**300 GB/repo**, file ≤50 GB (LFS) | — | private repo **cần PRO $9/th** |
| **HF Spaces** | CPU free | 2 vCPU/16 GB, public | — | — |
| **Azure** | tài khoản mới | **$200 / 30 ngày** (~50 h A100) + 65 dịch vụ always-free | — | — |
| **GCP** | tài khoản mới | **$300 / 90 ngày** — ⚠️ **không gắn được GPU vào Compute Engine VM khi còn trial** | — | 5 GB GCS (US) always-free |
| **TRC (TPU)** | nghiên cứu sinh, ai cũng apply được | TPU free, **30 ngày + gia hạn**, phải công khai nghiên cứu | — | VM + GCS **tính tiền** (nhỏ) |
| ~~SageMaker Studio Lab~~ | — | **đã đóng với người mới từ 2026-07-30** | — | — |

### 1.3 Storage & egress (nối tiếp `docs/storage-comparison.md`)

| Nền tảng | Giá storage | Egress |
|---|---|---|
| Modal Volume | $0.09/GiB/tháng, **1 TiB free** | $0.04/GiB (1 TiB/tháng free) |
| Cloudflare R2 | $0.015/GB/tháng (**10 GB free**) | **$0** |
| Kaggle Datasets | free (200 GB/dataset) | free (public) |
| HF Hub | free (public ~300 GB/repo) | free |
| RunPod | Network storage **$0.07/GB** (<1 TB) · Volume `$0.10` chạy / **$0.20 idle** | free |
| Lightning | 200 GB kèm tier | — |
| Lambda | kèm SSD (tới 22 TiB/node) | **$0** |
| Azure/AWS/GCP | ~$0.08–0.12/GB/tháng | **có phí** (AWS $0.09/GB đi Internet) |

---

## 2. Bộ tiêu chí so sánh (28 tiêu chí, 7 nhóm)

Dùng để đánh giá bất kỳ nền tảng nào — không chỉ các nền tảng ở trên.

### A. Chi phí
| # | Tiêu chí | Câu hỏi kiểm tra |
|---|---|---|
| A1 | Giá GPU/giờ | Giá niêm yết vs giá thực tế (có phụ phí region/spot?) |
| A2 | Độ hạt billing | theo giây (Modal) > theo phút (HF/Baseten/Thunder) > theo giờ (Lambda/Nebius) |
| A3 | Phụ phí ngoài GPU | storage, egress, IP, region multiplier (Modal: region ×1.15–1.75) |
| A4 | Chi phí khi idle | Container warm 60 s của Modal **có tính tiền**; RunPod Volume idle **$0.20/GB** (đắt gấp đôi khi chạy!) |

### B. Free tier & quota
| # | Tiêu chí | Ghi chú cho DIRE |
|---|---|---|
| B1 | Quy mô free | Kaggle 30 h/tuần ≈ 120 h/tháng → **nhiều GPU-giờ hơn $30 Modal**, nhưng GPU yếu (P100) |
| B2 | Cơ chế reset | Tuần (Kaggle) vs tháng (Modal) → chia nhỏ job theo tuần |
| B3 | **Điều khoản (ToS)** | Colab free **cấm** SSH, remote desktop, distributed worker, đa tài khoản → không được dùng làm compute farm |
| B4 | Nguồn credit khác | Azure $200 · GCP $300 (giới hạn GPU) · TRC (TPU) · AWS Activate |

### C. Hiệu năng & phần cứng
| # | Tiêu chí | Ghi chú |
|---|---|---|
| C1 | Độ đa dạng GPU | Vast/RunPod: 15+ loại; Colab/Kaggle: chỉ 1–2 loại cố định |
| C2 | Cold start / provisioning | Modal ~1 s; RunPod FlashBoot sub-200 ms; Vast phụ thuộc host; Azure cần xin quota (ngày) |
| C3 | Giới hạn phiên | Kaggle 12 h · Lightning free 4 h · Modal 24 h · VM: không giới hạn |
| C4 | Preemption | Vast/RunPod Community: có thể bị thu hồi; Azure Spot: <30 s báo trước → **phải checkpoint ra R2** |

### D. Kiểm soát
| # | Tiêu chí | Ghi chú |
|---|---|---|
| D1 | Root/SSH | VM (Vast/RunPod/Lambda/Azure): có · Modal: không SSH truyền thống nhưng `modal shell` + Sandbox · Kaggle/Colab free: **không** |
| D2 | Custom Docker | Vast/RunPod/Lambda: có · Modal: có (Image DSL) · Kaggle/Colab: pip tại chỗ, mất khi hết phiên |
| D3 | Job nền / detach | Modal `--detach` + timeout 24 h · Lightning "unlimited background execution" · Kaggle **không có** |
| D4 | Port / endpoint | Modal web endpoint · HF Spaces public URL · VM: mở port thủ công |

### E. Dữ liệu
| # | Tiêu chí | Ghi chú |
|---|---|---|
| E1 | Storage bền | Modal Volume 1 TiB free (v1: ≤50k file khuyến nghị, v2 beta: không giới hạn file) |
| E2 | Nguồn dữ liệu lúc chạy | Kaggle notebook **mount dataset trực tiếp** (200 GB, free) = điểm mạnh rất lớn |
| E3 | Ingress/Egress | Lambda/R2 egress $0; Modal 1 TiB/tháng free; Azure/AWS tính tiền |
| E4 | Checkpoint / resume | **Bắt buộc** với Lightning free (4 h) & preemptible; thiết kế `train.py` đã có checkpoint mỗi epoch |

### F. Vận hành & cộng tác
| # | Tiêu chí | Ghi chú |
|---|---|---|
| F1 | DX | Modal: Python decorator, không YAML · RunPod/Vast: Docker + SSH, thô hơn |
| F2 | Chia sẻ / URL công khai | HF Spaces (URL công khai free) > Kaggle notebook public > Modal endpoint |
| F3 | Region & latency từ Hà Nội | Modal region selection (×1.15–1.75 giá) · VN provider: latency thấp nhất · Vast/RunPod: 30–40+ region |
| F4 | Log / theo dõi | Modal log retention **1 ngày** (Starter) — đẩy metric sang R2/W&B nếu cần |

### G. Rủi ro & chiến lược
| # | Tiêu chí | Ghi chú |
|---|---|---|
| G1 | Lock-in | Code `modal.Image`/`.map()` khoá vào Modal → tách lớp `dataplane` (R2) giảm rủi ro |
| G2 | Thanh toán & KYC VN | Thẻ quốc tế: Modal/RunPod/Vast/Lambda OK · VN provider: VNĐ + VAT |
| G3 | Chính sách & ổn định | Kaggle/Colab/Lightning đổi hạn mức không báo trước; Studio Lab đã đóng; Nebius tăng giá 2 lần/3 tháng |
| G4 | Giờ công | VM tự lo = **3–8 h setup** lần đầu; quy ra ≈ $30–150 theo giá thời gian của bạn |

---

## 3. Ánh xạ workload DIRE → nền tảng

| # | Workload | Tài nguyên | Khuyến nghị chính | Phương án 2 | Ghi chú |
|---|---|---|---|---|---|
| 1 | Tải ~200 GB từ Baidu + verify sha256 | CPU + mạng, chạy nhiều giờ | **Modal CPU** (đang có code) | VPS VN | Tốn gần như 0 credit (CPU rẻ: $0.047/core-h) |
| 2 | Pack WebDataset shards | CPU + I/O | **Modal CPU** | VM rẻ | Volume v1 giới hạn ~50k file → shard 2000 ảnh là đúng thiết kế |
| 3 | Smoke test trên 1–5k ảnh | GPU nhỏ | **Kaggle** (330 h/tháng) | Colab/Lightning | Nhanh nhất, không tốn credit Modal |
| 4 | **Fine-tune detector trên DIRE có sẵn** | A100/H100 vài giờ | **Modal** ($30 credit ≈ 12 A100-h) | Lightning free 4 h/phiên + resume | Fine-tune ResNet-50 hội tụ trong vài epoch (theo paper) |
| 5 | Eval 16 generator song song | 16 job nhỏ | **Modal `.map()`** | VM + tự schedule | Scale-to-zero thắng ở đây |
| 6 | **Stage-2: tự tính DIRE (~200k ảnh, DDIM-20)** | **~30–60 A100-h** | ❌ **không free tier nào đủ** | Thu nhỏ còn 2–5k ảnh (≈1–3 GPU-h) | Xem §4 |
| 7 | Demo public | inference CPU | **HF Spaces free** | Modal web endpoint | ResNet-50 CPU ~0.2–1 s/ảnh |
| 8 | Mirror dữ liệu công khai | storage | **HF Hub (~300 GB/repo)** + **Kaggle Datasets (200 GB/dataset)** | R2 (10 GB free) | Xem rủi ro license §6 |

---

## 4. Đường đi 0 đồng (chi tiết, có tính toán)

### 4.1 Ngân sách giờ GPU

| Nguồn | Giờ GPU/tháng | Quy đổi tương đương | Loại GPU |
|---|---|---|---|
| Modal $30 credit | ~12 h | 12 × $2.50 | A100 80GB |
| | ~50 h | 50 × $0.59 | T4 |
| | ~15 h | 15 × $1.95 | L40S |
| Kaggle | ~120 h | 30 h/tuần × 4 | P100 16GB |
| Lightning free | ~15–22 h | 15–30 credit | T4/L40S; A100 cũng có |
| Colab free | phiên 12 h, T4 | — | phải tương tác, không detach |
| Azure $200 (tuỳ chọn, 30 ngày) | ~50 h A100 | $3.67–4.10/h | A100 80GB, **cần xin quota** |

**Tổng khả dụng thực tế ~150–180 h GPU/tháng với 0 ₫** — nghe nhiều, nhưng phần lớn là GPU yếu (P100/T4) và **bị cắt phiên**. Đủ để phát triển + train Tier A, không đủ cho stage-2.

### 4.2 Phân bổ đề xuất (theo tuần)

```
Tuần 1  Kaggle/Colab: viết & sửa code trên 1–2k ảnh mẫu        (0 credit)
        Modal CPU:    tải dữ liệu + verify sha256               (~$1–2 credit)
Tuần 2  Modal A100:   fine-tune ResNet-50 (3–8 h)               (~$8–20 credit)
        Kaggle:       tái lập kết quả nhỏ để so sánh             (0 credit)
Tuần 3  Modal A100:   eval 16 generator song song               (~$2–4 credit)
        HF Spaces:    deploy demo (CPU)                          (0)
Tuần 4  HF/Kaggle:    mirror dữ liệu + manifest                   (0)
        Dự phòng:     2–5 h A100 còn lại cho bug/retrain          (~$5–12 credit)
```

### 4.3 Ranh giới cứng — nói thẳng

| Việc | Chi phí | 0 ₫? |
|---|---|---|
| Train detector trên ảnh DIRE **có sẵn** trong dataset gốc | 3–8 A100-h | ✅ |
| Eval + demo public + mirror | ~1–3 A100-h + 0 | ✅ |
| **Tự tính DIRE (DDIM-20) cho 200k ảnh** | **30–60 A100-h ≈ $75–150** | ❌ → làm mẫu 2–5k ảnh |
| Train lại ADM từ đầu | hàng nghìn GPU-h | ❌ (không cần: dùng checkpoint tác giả) |

> **Con số 30–60 h cần được ĐO, không đoán.** Chạy `scripts/06_measure_dire.sh` (xem §7)
> để lấy ảnh/giây thật trên A100, rồi nhân với 200k ảnh.

### 4.4 Phao cứu sinh khi hết credit

Theo thứ tự ưu tiên:
1. **Lightning free + checkpoint resume** — A100/H100, restart 4 h/phiên, nhưng vì `train.py` lưu `last.pt` mỗi epoch nên có thể trải training qua nhiều phiên.
2. **TRC (TPU Research Cloud)** — miễn phí nhưng phải apply, phải open-source nghiên cứu, và hàng đợi được báo là dài (2 tháng).
3. **Trả phí rẻ nhất khi cần**: Thunder $1.09/h · Vast ~$0.80–1.50/h · RunPod $1.79/h · VN VNSO $1.12/h (A100 40GB).

---

## 5. Decision tree

```
Cần chạy GPU?
├─ Chỉ để THỬ CODE / smoke test?
│   └─► Kaggle (30h/tuần, không SSH) — nếu cần SSH: Lightning free
├─ Job NGẮN (<4h), cần song song nhiều instance, không muốn lo hạ tầng?
│   └─► Modal  ($30 credit; scale-to-zero; Volume 1 TiB free)
├─ Job DÀI (>10 GPU-h) hoặc hết credit Modal?
│   └─► Có thẻ quốc tế?
│        ├─ Có → Thunder $1.09/h › Vast ~$1.0/h › RunPod $1.79/h   (kèm checkpoint lên R2)
│        └─ Không → VNSO/Sunteco (VNĐ, 29–50k₫/h)
├─ Cần DEMO public cho người khác dùng?
│   └─► HF Spaces (CPU free; ZeroGPU nếu cần GPU)
└─ Cần LƯU dữ liệu để người khác tải lại?
    └─► HF Hub (~300 GB/repo, public) + Kaggle Datasets (200 GB/dataset)
        └─ riêng cho repo này: nếu muốn máy khác tự tải dữ liệu → xem §7.
```

---

## 6. Cái bẫy (đã kiểm chứng, không phải phỏng đoán)

| # | Bẫy | Chi tiết | Cách tránh |
|---|---|---|---|
| 1 | **GCP trial không gắn được GPU** | Tài khoản free-trial **không được thêm GPU vào VM**; $300 dùng được cho dịch vụ khác/Vertex nhưng **không** cho Compute Engine GPU. Không thể xin tăng quota khi còn trial | Đừng tính $300 GCP vào ngân sách GPU |
| 2 | **Colab free cấm SSH & worker** | ToS cấm: SSH/remote desktop, chạy distributed worker, "dùng containerization để lách chính sách", đa tài khoản. Vi phạm → chấm dứt runtime | Chỉ dùng Colab để code tương tác |
| 3 | **Kaggle không phải compute farm** | 30 h/tuần, P100 yếu, **không SSH, không chạy nền**, mất máy khi hết phiên | Dùng cho dev; đừng thiết kế pipeline phụ thuộc Kaggle |
| 4 | **SageMaker Studio Lab đã đóng** | Ngừng nhận người mới từ **2026-07-30**; tài khoản cũ vẫn dùng được | Đừng đưa vào kế hoạch |
| 5 | **Modal Volume v1 giới hạn inode** | Khuyến nghị ≤50k file, hard limit 500k inode; commit đồng thời nên ≤5 container | Dùng WebDataset shards (đã có trong `pack.py`); cân nhắc `version=2` (beta) |
| 6 | **Modal Volume không phải backup** | Không versioning, không snapshot; xoá là mất; **vẫn bị tính phí tới 4 ngày sau khi xoá** | Giữ bản gốc ở HF/Kaggle; ckpt quan trọng đẩy R2/kiểm tra sha256 |
| 7 | **RunPod Volume idle đắt hơn chạy** | `$0.20/GB/tháng` khi idle vs `$0.10` khi chạy | Nếu dùng RunPod, dùng Network Storage ($0.07/GB) làm kho, Volume chỉ làm cache tạm |
| 8 | **HF private repo cần PRO** | Free chỉ public; file ≤50 GB; ~300 GB/repo | Mirror public (đúng ý định của repo) hoặc nén/chunk hợp lý |
| 9 | **Nebius tăng giá liên tục** | H100 $3.85 → $4.50 (2026-10-01), lần tăng thứ 2 trong 3 tháng | Không hard-code giá vào script; dùng `tools/iaas_cost.py` |
| 10 | **License DiffusionForensics chưa rõ** | Dataset nghiên cứu; repo gốc **đã archive 2025-07-08**; mirror toàn bộ ảnh có thể vi phạm điều khoản | Giữ attribution + link gốc; mirror kèm `LICENSE-NOTE.md`; sẵn sàng takedown |

---

## 7. Kiểm chứng bằng đo lường (chưa chạy trong phiên này)

Sandbox soạn tài liệu **không có modal CLI/đăng nhập** → các số "3–8 A100-h" và "30–60 A100-h" ở §4 là **ước lượng cần đo**. Chạy:

```bash
# 1) Đo throughput DIRE (ảnh/giây) trên A100 → suy ra tổng giờ cho stage-2
modal run -m dire_modal.bench::dire_throughput --n 100 --steps 20

# 2) Đo storage (Volume vs R2 mount) — xem scripts/bench_storage.sh
./scripts/bench_storage.sh

# 3) Sổ quota free — nhập giờ đã dùng, in còn lại
python3 tools/free_quota.py --used modal=8.0 --used kaggle=6.0
```

Kết quả JSON sẽ nằm trong `manifests/` (đã `.gitignore` các file `.tsv`; JSON kết quả nên commit khi đã chốt).

---

## 8. Nguồn & độ tin cậy

**Đã kiểm chứng trực tiếp trên trang chính thức (2026-10-14):**
Modal pricing · Modal docs (Volumes, Cloud bucket mounts, limits, model weights) · Cloudflare R2 pricing & limits ·
Kaggle docs (efficient GPU usage, datasets) · Google Colab FAQ & signup · Colab paid services ·
Lightning AI pricing · HF Jobs pricing · HF Hub dataset size discussions · RunPod pricing · Vast.ai pricing ·
Lambda instances · Paperspace pricing · AWS SageMaker Studio Lab availability change · VNSO · Sunteco · Thunder Compute (Azure comparison) · Nebius/CoreWeave price hikes.

**Chưa kiểm chứng trực tiếp (đánh dấu trong bảng):**
| Mục | Vấn đề |
|---|---|
| Vast.ai A100 | Không có trên trang chủ; lấy từ nguồn so sánh ($0.80–1.50/h) |
| GCP a2-highgpu ~$3.28/GPU-h | Từ bảng so sánh, chưa xác minh trên GCP pricing |
| DataCrunch, Hyperstack, Oracle always-free GPU | **Không đưa vào khuyến nghị** cho tới khi xác minh |
| VNG Cloud / Viettel IDC / FPT Smart Cloud | Có dịch vụ GPU nhưng giá không công khai trên trang chủ |
| Kaggle TPU quota | Có TPU v3-8 free nhưng quota riêng — chưa xác minh số giờ |

**Nguyên tắc cập nhật:** giá GPU biến động nhanh (Nebius +17% trong 1 tháng). Mọi con số dùng cho tính toán nằm trong
`tools/iaas_cost.py` (một chỗ duy nhất) — sửa ở đó, không sửa rải rác trong tài liệu.

---

## 9. Kết luận cho repo `dire-modal`

| Vai trò | Chọn | Vì sao |
|---|---|---|
| **Orchestration + train + eval** | Modal Starter | $30 credit, Volume 1 TiB free, `.map()` cho 16 generator, code đã viết sẵn |
| **Dev loop & smoke test** | Kaggle (+ Colab tương tác) | 330 h/tháng, không tốn credit |
| **Dữ liệu (nguồn công khai)** | HF Hub + Kaggle Datasets | Free, public, có thể mount trực tiếp trong notebook |
| **Dữ liệu (kho riêng, nhỏ)** | R2 (10 GB free) | Chỉ chứa checkpoint + manifest sha256, không chứa dataset |
| **Demo** | HF Spaces CPU | Public URL, ResNet-50 chạy được trên CPU |
| **Khi vượt ngân sách** | Thunder/Vast/RunPod | $0.80–1.79/h thay vì $2.50/h; kèm checkpoint lên R2 |
