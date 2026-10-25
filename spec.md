# SPEC-001 — Tái tạo DIRE (ICCV 2023) và cải thiện theo AEROBLADE (CVPR 2024)

| | |
|---|---|
| **ID** | SPEC-001 |
| **Trạng thái** | `draft` — chờ duyệt trước khi code |
| **Phiên bản** | 1.0 |
| **Ngày** | 2026-10-24 |
| **Người viết** | Agent (Arena) |
| **Tài liệu liên quan** | [AGENT.md](AGENT.md) · [instructions.md](instructions.md) · [plan.md](plan.md) · [skills.md](skills.md) · [docs/runbook-free.md](docs/runbook-free.md) · [docs/adr-0002-iaas.md](docs/adr-0002-iaas.md) |

**Spec này dùng để làm gì.** Đây là bản đặc tả *cái gì và tại sao*, viết **trước**
khi code. Mỗi yêu cầu có ID và tiêu chí chấp nhận kiểm được. Khi thực thi, mỗi
mục trong §12 trở thành một PR; trạng thái thực thi theo dõi ở [plan.md](plan.md).
Spec không phải tài liệu mô tả code hiện có — code hiện có chỉ được nhắc khi nó
là điểm xuất phát hoặc là chỗ đang thiếu.

---

## 1. Vấn đề

Repo hiện có pipeline tải dữ liệu, guard chi phí và một bản `compute_dire.py`
chưa từng chạy. Nhưng **chưa có một dòng kết quả nào** trên dữ liệu thật:

- `eval.py` — `raise NotImplementedError` (chưa port).
- `train.py` — `raise NotImplementedError` (chưa port).
- `compute_dire.py` — viết theo API guided-diffusion nhưng docstring tự ghi
  "**CHƯA được chạy thử**"; ba điểm chưa xác nhận: tên tham số model, kiểu trả
  về của `ddim_reverse_sample_loop`, bộ nhớ ở batch > 1.

Nghĩa là: muốn nói "DIRE đạt X%" thì phải dựng đường đo trước. Spec này định
nghĩa đường đo đó, và định nghĩa luôn phần **cải thiện** dựa trên AEROBLADE.

## 2. Mục tiêu

| # | Mục tiêu |
|---|---|
| G1 | Tái tạo được bảng eval của DIRE trên checkpoint tác giả, **không train lại** |
| G2 | Tự tính được DIRE map và chứng minh nó **khớp** với map tác giả phát hành |
| G3 | Đo được AEROBLADE trên **cùng tập ảnh** để so sánh công bằng |
| G4 | Định lượng **format bias** — yếu tố làm DIRE sụp trong thực tế |
| G5 | Một detector kết hợp (training-free) tốt hơn từng cái ở chỗ yếu của nó |
| G6 | Mọi con số báo cáo đều tái lập được: cùng lệnh, cùng seed, có manifest |

## 3. Không phải mục tiêu (non-goals)

| # | Không làm | Lý do |
|---|---|---|
| N1 | Train lại detector DIRE | Ràng buộc dự án: **chỉ cải thiện, không train lại**. Nếu sau này cần → ADR riêng |
| N2 | Tái lập benchmark AEROBLADE trên LAION | Ảnh real LAION không phân phối được (bản quyền); xem §5 R4 |
| N3 | Toàn bộ 200k ảnh rawx/rawxens | 12.1 GB + hàng chục giờ GPU, không cần cho mục tiêu G1–G5 |
| N4 | Train từ đầu trên GenImage / benchmark ngoài DIRE | Ngoài Tier A |
| N5 | Tối ưu SOTA | Mục tiêu là **đo đúng và cải thiện có kiểm chứng**, không phải vượt paper |

## 4. Thuật ngữ

| Thuật ngữ | Nghĩa trong spec này |
|---|---|
| **DIRE map** | Ảnh `\|x − x̂\|` (uint8) do DDIM inversion 20 bước trên ADM 256×256 rồi tái tạo |
| **DIRE detector** | ResNet50 nhị phân (`mode=binary`), ăn **DIRE map** đã resize 256 → CenterCrop 224 → chuẩn hoá ImageNet |
| **AEROBLADE** | Điểm bất thường = khoảng cách LPIPS giữa `x` và `D(E(x))` qua VAE của LDMs; **training-free** |
| **AP** | Average Precision (`average_precision_score`), lớp dương = ảnh giả |
| **TPR@5%FPR** | TPR tại ngưỡng mà FPR = 5%. Đây là metric AEROBLADE dùng; DIRE paper không dùng |
| **Format bias** | Detector học *định dạng nén* thay vì dấu vết sinh ảnh: real=JPEG, generated=PNG |
| **Protocol-matched** | Hai detector đo trên **cùng ảnh nguồn**, cùng tiền xử lý, cùng cách chia real/fake |

## 5. Sự thật đã kiểm chứng

Bảng này là **nền của spec**. Cột "Xác minh" nói rõ cái gì đã đọc từ nguồn gốc
và cái gì phải chép lại từ PDF trước khi trích dẫn.

| ID | Sự thật | Nguồn | Xác minh |
|---|---|---|---|
| F1 | DIRE detector là ResNet50 nhị phân, `batch_size=64`, `loadSize=256`, `cropSize=224`, `nepoch=400`, `lr=1e-4`, adam, earlystop patience 5, `pretrained=True`, **không** bật jpg/blur augmentation (`jpg_prob=0`, `blur_prob=0`) | `utils/config.py` của ZhendongWang6/DIRE (đọc trực tiếp) | ✅ đọc code |
| F2 | `demo.py` nhận **DIRE map** làm đầu vào (resize 256 → CenterCrop 224 → normalize), **không** nhận ảnh gốc | `demo.py` (đọc trực tiếp) | ✅ đọc code |
| F3 | Dataset tổ chức `images/recons/dire` × `train/val/test` × `lsun_bedroom/imagenet/celebahq` × `{real,adm,…}`; train nối real→`0_real`, sinh→`1_fake` | `README.md` (đọc trực tiếp) | ✅ đọc code |
| F4 | Eval chính thức: `sh test.sh` → `test.py --ckpt <lsun_adm.pth> datasets_test release` | `test.sh` (đọc trực tiếp) | ✅ đọc code |
| F5 | AEROBLADE mặc định dùng AE của SD1 + SD2 + Kandinsky 2.1, khoảng cách = LPIPS layer 2; lưu **khoảng cách âm** nên AE tốt nhất = `max` | `README.md` của jonasricker/aeroblade | ✅ đọc code |
| F6 | AEROBLADE hỗ trợ `--experiment-id robustness` với `jpeg_90..50`, `blur_1..5`, `crop_0.9..0.5`, `noise_0.05..0.25` | README (lệnh mẫu) | ✅ đọc code |
| F7 | Điểm AEROBLADE báo cáo (mean AP ≈ 0.99) và điểm DIRE mà AEROBLADE đo lại được (AP ~0.46–0.51, TPR@5%FPR ≈ 0) | AEROBLADE paper (arXiv 2401.17879) | ⚠️ **cần chép lại từ Table khi làm M0** |
| F8 | DIRE paper báo cáo ACC/AP rất cao trên benchmark của họ (~99% với ADM-LSUN) | DIRE paper (arXiv 2303.09295) | ⚠️ **cần chép lại số chính xác khi làm M0** |
| F9 | Dữ liệu DIRE trên Baidu là **tar không nén** dù tên `.tar.gz` | `probe_pair` chạy thật 2026-10-21 | ✅ đo trong repo |
| F10 | `dire/test/lsun_bedroom/imagenet/real` và `dire/test/lsun_bedroom/lsun_bedroom/real` là **cùng 1000 ảnh** ⇒ nhánh thứ hai là cross-**generator** | `probe_pair` 2026-10-21 | ✅ đo trong repo |
| F11 | `dire/test/lsun_bedroom/lsun_bedroom/real.tar.gz` = 1000 ảnh DIRE map, 25.8 MB dữ liệu ảnh | `probe_pair` 2026-10-21 | ✅ đo trong repo |

**Quy tắc rút ra từ bảng này:** F7 và F8 chưa xác minh thì **không được** xuất
hiện trong báo cáo như số của repo. Nhiệm vụ M0 bắt buộc chép chúng từ PDF, kèm
số bảng/số trang.

## 6. Ràng buộc

| ID | Ràng buộc | Hệ quả lên thiết kế |
|---|---|---|
| C1 | Compute **0đ**, chỉ free tier | Mọi bước phải có `preflight_cost`; không chạy 200k ảnh |
| C2 | Máy local nhẹ, không train, không giữ dữ liệu gốc | Mọi thứ nặng trên Modal; local chỉ đọc CSV/JSON |
| C3 | Không train lại detector | "Cải thiện" phải **training-free** (fusion, calibration, routing) |
| C4 | Không phân phối lại ảnh LAION | Dùng ảnh real của bộ DIRE, không dùng LAION |
| C5 | ~$30 credit Modal | Trần $5/lần chạy đã có; tổng spec phải < $20 |
| C6 | Tái lập được | Mọi báo cáo kèm `git_sha`, seed, phiên bản thư viện |

## 7. Yêu cầu chức năng

Ký hiệu tiêu chí: **Given / When / Then**.

### FR-1 — Đo DIRE trên checkpoint tác giả

Hệ thống phải eval được `lsun_adm.pth` trên 16 generator của
`dire/test/lsun_bedroom/lsun_bedroom/` và xuất bảng ACC/AP từng generator.

- **Given** `/raw` đã có tier `e2` và `/ckpt/pretrained/lsun_adm.pth`
  **When** chạy lệnh eval
  **Then** in bảng 16 dòng + dòng tổng, ghi CSV theo §10.
- **Ràng buộc tiền xử lý:** resize 256 → CenterCrop 224 → ToTensor → chuẩn hoá
  `mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]`. Sai một bước là lệch số.
- **AC-1.1** ACC trên generator `adm` ≥ **0.99** (khớp CỔNG 8 trong runbook).
- **AC-1.2** Không có generator nào có `n_real` hoặc `n_fake` = 0 (bắt lỗi
  nhãn/đường dẫn im lặng).
- **AC-1.3** Tổng ảnh đọc được khớp `meta.json` sinh bởi `pack.py` (sai lệch 0).
- **AC-1.4** Chạy lại 2 lần cho cùng số (trong sai số ±0.0 vì eval tất định).

### FR-2 — Chứng minh DIRE tự tính khớp DIRE tác giả phát hành

Trước khi dùng `compute_dire.py` cho bất kỳ kết luận nào, phải chứng minh nó
tái tạo được map của tác giả.

- **Given** 50 ảnh real + 50 ảnh `adm` từ `images/test/lsun_bedroom/`
  **When** tự tính DIRE map và so với map tương ứng trong
  `dire/test/lsun_bedroom/lsun_bedroom/`
  **Then** báo cáo MAE giữa hai map và tương quan của `mean|diff|` từng ảnh.
- **AC-2.1** Tương quan Pearson của `mean|diff|` từng ảnh ≥ **0.90**.
- **AC-2.2** Dấu của hiệu `mean|diff|(real) − mean|diff|(fake)` **giống** bản
  tác giả (đây mới là thứ detector dùng; MAE tuyệt đối không quan trọng bằng).
- **AC-2.3** Nếu AC-2.1 hoặc AC-2.2 fail → **dừng**, ghi vào spec là "DIRE tự
  tính không dùng được", và mọi kết luận sau đó chỉ dựa trên map tác giả.
- **Ghi chú:** đây là điều kiện *chặn* cho FR-4. Không có FR-2 thì không được
  công bố bất cứ số nào từ map tự tính.

### FR-3 — Đo AEROBLADE trên cùng tập ảnh

- **Given** tier `e4` (`images/test/lsun_bedroom/*.tar.gz`)
  **When** chạy AEROBLADE với cấu hình mặc định (SD1+SD2+KD2.1, LPIPS layer 2)
  **Then** xuất AP + TPR@5%FPR từng generator, cùng schema §10.
- **AC-3.1** `n_real` và `n_fake` **bằng đúng** FR-1 (cùng ảnh nguồn, cùng số).
- **AC-3.2** Điểm dùng để tính metric là **giá trị âm** như tác giả lưu
  (best AE = `max`); nếu đảo dấu thì AP sẽ ~0 và test phải bắt được.
- **AC-3.3** Với generator LDM (sdv1*, sdv2*, ldm, …) AP ≥ **0.90**. Nếu thấp
  hơn nhiều, nghi thứ tự dấu hoặc sai tiền xử lý — phải kiểm trước khi báo cáo.
- **AC-3.4** Chạy trên `adm` được phép **kém** (AEROBLADE là training-free cho
  LDM; ADM không dùng VAE tương tự) — ghi rõ đây là kỳ vọng, không phải lỗi.

### FR-4 — Định lượng format bias

- **Given** mẫu 200 real + 200 fake của một generator
  **When** recompute DIRE map ở 3 điều kiện: bản gốc, tất cả re-encode JPEG q90,
  tất cả re-encode PNG
  **Then** báo cáo AP của cùng checkpoint ở cả 3 điều kiện.
- **AC-4.1** Có bảng `AP(clean)`, `AP(jpeg90)`, `AP(png)` và hiệu số.
- **AC-4.2** Chạy tiền đề bắt buộc: FR-2 phải đạt trước.
- **AC-4.3** Chi phí mỗi điều kiện ≤ **$3** (400 ảnh × 20 step), có preflight.
- **Ghi chú thiết kế:** điều kiện "tất cả JPEG" và "tất cả PNG" là cách duy nhất
  tách *dấu vết sinh ảnh* khỏi *định dạng nén*. Chỉ re-encode DIRE map **không**
  có tác dụng — bias xảy ra ở ảnh nguồn.

### FR-5 — Detector kết hợp, training-free

- **Given** điểm DIRE (từ FR-1) và điểm AEROBLADE (từ FR-3) trên `val` (tier `e5`)
  và `test`
  **When** khớp hai điểm bằng chuẩn hoá rank trên **val**, chọn trọng số trên
  **val**
  **Then** báo cáo AP của điểm kết hợp trên **test**.
- **AC-5.1** Trọng số và ngưỡng **chỉ** fit trên `val`; test không được chạm
  trong lúc chọn.
- **AC-5.2** Báo cáo cả 3 dòng: DIRE, AEROBLADE, kết hợp — cùng bảng, cùng tập.
- **AC-5.3** Kết hợp chỉ được gọi là "cải thiện" nếu AP trên test **cao hơn cả
  hai** thành phần ở **≥ 12/16** generator. Nếu không, phải báo cáo là "không
  cải thiện" — đây là kết quả hợp lệ.
- **AC-5.4** Không có tham số nào của detector bị sửa (ràng buộc C3). Chỉ một
  trọng số vô hướng + một ngưỡng.

### FR-6 — Đường cong robustness

- **Given** bộ biến đổi của AEROBLADE (`jpeg_90..50`, `blur_1..5`,
  `crop_0.9..0.5`, `noise_0.05..0.25`)
  **When** áp lên **ảnh nguồn** rồi recompute DIRE (FR-2 đã đạt) và chạy AEROBLADE
  **Then** xuất AP theo từng mức biến đổi cho cả hai detector.
- **AC-6.1** 20 điều kiện × 2 detector, mỗi ô có AP và số ảnh.
- **AC-6.2** Tổng chi phí ≤ **$8**; nếu vượt, giảm số ảnh xuống 100/nhóm và ghi
  rõ trong báo cáo.
- **AC-6.3** Điều kiện `clean` trong FR-6 phải trùng số với FR-1/FR-3 (kiểm chéo).

### FR-7 — Báo cáo tự chứa

- **Given** một lần chạy
  **When** đọc thư mục kết quả
  **Then** có đủ để người khác tái lập mà không hỏi.
- **AC-7.1** `run.json` có: `git_sha`, `modal_image_id`, phiên bản
  torch/torchvision/sklearn, seed, danh sách tier đã dùng + sha256 của
  `_manifest_sha256.tsv`, `n_images` mỗi nhóm, ước tính và thực chi (nếu có).
- **AC-7.2** CSV đúng schema §10, không có ô rỗng ở cột metric.
- **AC-7.3** Mỗi con số trong báo cáo trỏ được về một dòng CSV.

## 8. Yêu cầu phi chức năng

| ID | Yêu cầu | Ngưỡng |
|---|---|---|
| NFR-1 | Chi phí toàn spec | ≤ **$20** credit (C5), mỗi lần chạy có preflight |
| NFR-2 | Không lần chạy nào vượt | $5 (guard hiện có) |
| NFR-3 | Thời gian một eval DIRE đầy đủ | ≤ 30 phút trên 1 GPU |
| NFR-4 | Tất định | eval 2 lần cho cùng kết quả; nếu có randomness phải seed |
| NFR-5 | Không đọc lại dữ liệu thừa | dùng cache `.state/validated.json` đã có |
| NFR-6 | Không bịa số | số chưa đo phải ghi `null` + lý do, không điền 0 |

## 9. Thiết kế

### 9.1 Luồng dữ liệu

```
Baidu ──► /raw ──► pack ──► /shards
                              │
      ┌───────────────────────┼────────────────────────┐
      │                       │                        │
  dire/test/…            images/test/…            val/…
  (DIRE map tác giả)     (ảnh gốc)                (hiệu chuẩn)
      │                       │                        │
   FR-1 ResNet50         FR-3 AEROBLADE           FR-5 trọng số
      │                       │                        │
      └──────────► CSV ◄──────┘                        │
                     │                                 │
                FR-7 run.json ◄────────────────────────┘
```

Hai nhánh FR-1 và FR-3 **độc lập** về dữ liệu (một bên ăn map, một bên ăn ảnh
gốc) nhưng phải **cùng ảnh nguồn** — đó là AC-3.1.

### 9.2 Vì sao FR-1 rẻ

`dire/` chứa **DIRE map đã tính sẵn** bởi tác giả. Nên tái tạo bảng của họ chỉ
cần ResNet50 inference: 16 generator × ~2000 ảnh = ~32k ảnh ở 224² → **< 10 phút
L4/L40S**, gần như miễn phí. Không cần ADM, không cần diffusion.

Chỉ FR-2, FR-4, FR-6 cần `compute_dire.py` (ADM inversion) — và đó là lý do
FR-2 là điều kiện chặn.

### 9.3 Vì sao điều kiện chặn là FR-2

`compute_dire.py` chưa từng chạy (ba điểm chưa xác nhận, xem §1). Nếu bỏ qua
FR-2 mà chạy FR-4, ta sẽ báo cáo "format bias = Δ" trong khi Δ có thể chỉ là
sai số của việc tự cài đặt. Đây đúng loại lỗi repo đã gặp: kết luận đứng trên
một phép so chưa được kiểm.

### 9.4 Thành phần code

| Thành phần | Trạng thái | Việc phải làm |
|---|---|---|
| `eval.py::eval_one` | **xong (M1)** | Nạp ResNet50 từ `vendor/networks/resnet.py`, đọc **tar gốc `/raw`**, trả ACC/AP/TPR@5%FPR |
| `eval.py::main` | **xong (M1)** | Gom theo `branch`+`generator`, ghi CSV + `run.json`, in bảng ra log |
| `compute_dire.py` | chưa chạy | FR-2 sửa tới khi AC-2.1/2.2 đạt |
| `aero.py` | **chưa có** | Bọc AEROBLADE: nạp AE, tính khoảng cách, trả điểm thô |
| `fusion.py` | **chưa có** | Rank-normalise + chọn trọng số trên val |
| `train.py` | `NotImplementedError` | **Ngoài phạm vi** (N1) — chỉ ghi chú |
| `vendor/utils/datasets.py` | chưa vendor | **Không cần nữa**: `eval.py` tự dựng pipeline 256→224→Normalize và đọc tar trực tiếp |

> Ghi chú: `vendor/utils/eval.py` tính ACC/AP/R_ACC/F_ACC nhưng **không** có
> TPR@5%FPR. `eval.py` cài đặt metric bằng Python thuần và **đã đối chiếu 500 ca
> ngẫu nhiên** (kể cả score trùng) với `sklearn`: khớp tuyệt đối cả
> `average_precision_score` và `roc_curve(drop_intermediate=False)`. Dùng
> `drop_intermediate=False` là cố ý — mặc định `True` của sklearn bỏ các điểm
> thẳng hàng, và tại một ngân sách FPR cố định điểm bị bỏ có thể là điểm tốt nhất.
>
> Ghi chú 2: eval đọc **tar gốc `/raw`**, không đọc `/shards`, vì `pack_one` bản
> cũ đặt shard vào `/shards/{split}/{tên tar}` nên ba nhánh khác nhau có
> `real.tar.gz` ghi vào cùng một thư mục (đã sửa bằng `pack.shard_group`, nhưng
> eval vẫn không phụ thuộc `pack` để ít bước hơn).

## 10. Hợp đồng dữ liệu

### 10.1 Kết quả — `results/<experiment_id>/<detector>.csv`

```
experiment_id,detector,generator,split,transform,n_real,n_fake,acc,ap,tpr_at_5fpr,threshold,git_sha,created_at
```

| Cột | Kiểu | Ghi chú |
|---|---|---|
| `detector` | str | `dire` \| `aeroblade` \| `fusion` |
| `generator` | str | tên như trên Baidu, **giữ nguyên typo** (`midjouney`) |
| `split` | str | `test` \| `val` |
| `transform` | str | `clean` \| `jpeg_90` \| … \| `noise_0.25` |
| `acc,ap,tpr_at_5fpr` | float | 6 chữ số thập phân; `null` nếu không tính được |
| `threshold` | float | ngưỡng dùng cho ACC/TPR (lấy từ val, hoặc 0.5 với ACC) |

### 10.2 `results/<experiment_id>/run.json`

```json
{
  "spec": "SPEC-001", "experiment_id": "...", "git_sha": "...",
  "modal_image_id": "...", "created_at": "...",
  "versions": {"torch": "...", "torchvision": "...", "sklearn": "..."},
  "data": {"tiers": ["e2"], "manifest_sha256": "...", "n_images": {"real": 1000, "adm": 1000}},
  "codec": {"transform": "clean"},
  "cost": {"preflight_usd": 0.0, "actual_usd": null},
  "seed": 3407
}
```

### 10.3 Lệnh

```bash
# FR-1
modal run -m dire_modal.eval::main --tiers e2 --exp-id spec001_fr1

# FR-2 (chặn)
modal run -m dire_modal.compute_dire::equivalence --n 100 --exp-id spec001_fr2

# FR-3
modal run -m dire_modal.aero::main --tiers e4 --exp-id spec001_fr3

# FR-5
modal run -m dire_modal.fusion::fit --val-tier e5 --exp-id spec001_fr5
```

> Các entrypoint `equivalence`, `aero::main`, `fusion::*` **chưa tồn tại** — đó
> là phần phải implement. Spec đặt tên trước để test hợp đồng CLI viết được
> trước code (xem §11).

## 11. Kế hoạch kiểm chứng

| Gate | Kiểm gì | Cách kiểm | Ngưỡng |
|---|---|---|---|
| **S0** | Spec nhất quán | `pytest tests/test_agent_docs.py tests/test_spec.py -q` | pass |
| **S1** | Hợp đồng CLI tồn tại | test gọi từng entrypoint với `--dry-run`/stub | không `AttributeError` |
| **S2** | FR-1 tái tạo | so ACC `adm` với ngưỡng runbook | ≥ 0.99 |
| **S3** | FR-2 khớp map tác giả | tương quan `mean|diff|` | ≥ 0.90 |
| **S4** | FR-3 dấu điểm đúng | AP trên sdv1* | ≥ 0.90 |
| **S5** | FR-5 không rò test | test: trọng số chỉ phụ thuộc val | pass |
| **S6** | FR-7 tái lập | chạy lại 1 eval → cùng CSV | diff = 0 |

Test code phải viết **trước** phần lớn nhất của spec: schema CSV (§10.1) và hợp
đồng CLI (§10.3) kiểm được mà chưa cần GPU.

## 12. Mốc thực thi

| Mốc | Nội dung | Chi phí ước tính | Cổng |
|---|---|---|---|
| **M0** | Chép F7/F8 từ 2 PDF vào §5 (số bảng + trang); chốt lại tiêu chí S2/S4 | $0 | — |
| **M1** ✅ | Port `eval.py` + TPR@5%FPR + ghi CSV/run.json → **FR-1, FR-7** | < $1 | S2, S6 — **code xong**, số chưa có (cần chạy `eval::main` trên Modal) |
| **M2** | `compute_dire` sửa tới khi khớp map tác giả → **FR-2** | ~$1 | S3 |
| **M3** | Bọc AEROBLADE → **FR-3** | ~$2 | S4 |
| **M4** | Format bias (FR-4) + fusion (FR-5) | ~$5 | S5 |
| **M5** | Robustness (FR-6) + báo cáo tổng | ~$8 | — |
| **Tổng** | | **≈ $17** | < C5 ($30) |

Thứ tự **không đổi được**: M1 trước M2 (cần eval để đo), M2 trước M4 (FR-2 chặn
FR-4), M1+M3 trước M4 (fusion cần hai điểm).

## 13. Rủi ro

| ID | Rủi ro | Khả năng | Ảnh hưởng | Giảm thiểu |
|---|---|---|---|---|
| R1 | `compute_dire` không khớp map tác giả (API guided-diffusion khác phiên bản) | **cao** | chặn FR-4/FR-6 | FR-2 là điều kiện chặn; nếu fail thì báo cáo chỉ dùng map tác giả và ghi rõ giới hạn |
| R2 | Số DIRE của ta không đạt 0.99 vì khác metric (AP vs ACC) | trung bình | hiểu nhầm "tái tạo thất bại" | M0 chốt metric trước; báo cáo cả ACC và AP |
| R3 | AEROBLADE cần tải nhiều AE (~vài GB) qua HF Hub | trung bình | tốn thời gian/credit | tải một lần vào `/ckpt`, commit; preflight riêng |
| R4 | Ảnh real của DIRE ≠ ảnh real AEROBLADE dùng trong paper | **chắc chắn** | không so được với paper | chỉ so **nội bộ** (protocol-matched); ghi rõ trong báo cáo |
| R5 | Generator `adm` khiến AEROBLADE sụp (không phải LDM) | **chắc chắn** | người đọc hiểu nhầm | AC-3.4: ghi rõ đây là kỳ vọng, không phải lỗi |
| R6 | Credit cạn giữa M4 | trung bình | dừng giữa chừng | mỗi mốc là một lần chạy độc lập, có CSV riêng |
| R7 | Dữ liệu test có generator chỉ 130 B (rỗng trong dataset gốc) | đã biết | eval sai | `NEVER` đã loại; AC-1.2 bắt `n_fake=0` |

## 14. Câu hỏi mở

| # | Câu hỏi | Cần trả lời trước |
|---|---|---|
| Q1 | Có nên thêm TPR@1%FPR nữa không (DIRE paper hay dùng ngưỡng chặt)? | M1 |
| Q2 | `dire/` và `images/` của cùng generator có thật sự cùng ảnh nguồn không? (chưa kiểm được vì map đã biến đổi) | M2 — có thể kiểm gián tiếp qua tương quan |
| Q3 | Fusion nên gộp ở mức **điểm** hay mức **quyết định** (OR/AND)? | M4 |
| Q4 | Có cần eval trên `dire/test/imagenet/*` không, hay chỉ LSUN là đủ cho kết luận? | M1 |
| Q5 | `midjouney` (typo gốc) và `midjourney` là hai generator khác nhau hay cùng một? | M1 |

## 15. Truy vết

| Yêu cầu | Code | Test | Cổng |
|---|---|---|---|
| FR-1 | `eval.py` | `tests/test_eval_contract.py` (mới) | S2 |
| FR-2 | `compute_dire.py::equivalence` | `tests/test_dire_equivalence.py` (mới) | S3 |
| FR-3 | `aero.py` | `tests/test_aero_sign.py` (mới) | S4 |
| FR-4 | `compute_dire.py` + `eval.py` | — | S3 |
| FR-5 | `fusion.py` | `tests/test_fusion_no_leak.py` (mới) | S5 |
| FR-6 | `compute_dire.py` + `aero.py` | — | — |
| FR-7 | `eval.py::_write_run_json` | `tests/test_run_json.py` (mới) | S6 |

## 16. Lịch sử thay đổi

| Ngày | Phiên bản | Thay đổi |
|---|---|---|
| 2026-10-24 | 1.0 | Bản đầu: 7 yêu cầu chức năng, 6 phi chức năng, 6 mốc, 7 rủi ro |
