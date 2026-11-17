# Kế hoạch demo web — cái gì chạy được thật, cái gì phải nói "chưa"

- **Trạng thái:** kế hoạch (đã sửa app — xem `docs/demo-hf-spaces-plan.md`) · **Ngày:** 2026-11-16
- **Liên quan:** `platforms/hf-spaces/` (app hiện có), `platforms/site/` (trang
  tĩnh), `docs/figures-plan.md` (dùng chung gallery), `docs/dataset-own.md` (nhãn
  R0/R1+R2/R3), `docs/runbook-free.md` (§ demo), ADR-0003
- **Đã thực hiện:** app Gradio → app **Streamlit** 5 trang; Tab 1 (upload ảnh →
  `P(fake)`) đã bị bỏ, không phải sửa. Kế hoạch triển khai, số đo cold start và
  quy trình deploy: **`docs/demo-hf-spaces-plan.md`**.
- **Mục đích:** chốt *demo làm được gì trên CPU free* trước khi viết thêm UI.
  Repo đã có sẵn một app Gradio; vấn đề không phải "chưa có demo", mà là **app
  hiện tại đang trả một con số vô nghĩa** (mục 2).

---

## 1. Hiện trạng, đọc trước khi thiết kế

| Thứ | Đang có | Thực tế |
|---|---|---|
| `platforms/hf-spaces/app.py` | 2 tab: Detector, DIRE viz | chạy được, nhưng mục 2 |
| `gallery/` | chỉ có `.gitkeep` | **rỗng** ⇒ tab 2 hiện chỉ hiện hướng dẫn |
| checkpoint cho demo | cần `dire_resnet50.pth` trên HF Hub | **chưa tồn tại** |
| 6 `checkpoints/*.pth` của tác giả | 269 MB/file, đang nằm trên ổ của bạn | chưa ai mở thử ⇒ chưa biết bên trong |
| `recons/` | chỉ `real`, `ldm`, `pndm` | **không có `adm`** ⇒ panel "tái tạo" của dòng headline không vẽ được |
| số đo CPU | chưa có | mọi con số thời gian dưới đây là **giả định có nhãn** |

Hai thứ này quyết định toàn bộ kế hoạch, nên tách bạch:

- `platforms/site/` là **trang tĩnh** (deck HTML, `data.json` do `build.py` sinh).
  Không tương tác, không cần model, không có rủi ro "số giả" — chỉ cần đúng nhãn.
- `platforms/hf-spaces/` là **demo tương tác**. Mọi rủi ro nằm ở đây.

## 2. Lỗi phải sửa trước khi công khai: Tab 1 đang chấm ảnh RGB bằng model học DIRE

`detect()` nhận ảnh upload, chuẩn hoá ImageNet, đưa thẳng vào ResNet-50 — trong
khi model đó được train trên **DIRE map**, không phải ảnh RGB. Ghi chú trong app
nói "độ chính xác thấp hơn paper", nhưng cách nói đó **nhẹ hơn sự thật**:

- đây không phải "kém chính xác hơn", mà là **chấm một phân bố đầu vào chưa từng
  được train**. Không có cơ sở nào để nói con số đó có nghĩa;
- Table 5 của paper cho thấy đổi đầu vào là đổi hẳn kết quả: trên SD-v1, `RGB`
  được 77,4/85,8 còn `DIRE` được 99,7/100 — và đó là model **được train cho RGB**
  trong thí nghiệm của họ. Model của mình thì không.

Một demo trả `P(fake) = 0.93` trông rất thuyết phục và không ai kiểm lại được.
Đây đúng loại lỗi repo này đã sửa nhiều lần (ADR-0003: "một hình trông như dữ
liệu nhưng không phải dữ liệu").

**Quyết định:** Tab 1 chỉ được trả xác suất khi đầu vào **là DIRE map thật**. Ba
chế độ dưới đây là ba cách có DIRE map, xếp theo chi phí.

## 3. Ba chế độ, chọn theo số đo chứ không theo cảm giác

| | A — Gallery | B — Upload tự tính (CPU) | C — GPU backend (Modal) |
|---|---|---|---|
| Đầu vào | mẫu có sẵn | ảnh bạn upload | ảnh bạn upload |
| DIRE map | tác giả (có sẵn) | mình tính trên CPU | mình tính trên A10G |
| Thời gian (`S=5`) | < 1 s | **101 s** (đã đo) | **0,92 s** |
| Chi phí | 0 ₫ | 0 ₫ | ~0,0003 USD/ảnh |
| Nhãn | `R0` | `R1+R2` | `R1` (map tự tính) |
| Trạng thái | **làm ngay** | có, **mặc định tắt** (quá ngưỡng) | **đã dựng**, cần secret mới bật |

### A. Gallery — chế độ chính, mặc định, làm trước

Người xem chọn một mẫu trong bộ **thật + 15 generator** của `images/test/lsun_bedroom/`,
nhìn ảnh gốc, nhìn DIRE map, và thấy detector chấm **chính map đó**. Mọi thứ đã
có trên ổ: `dire/test/lsun_bedroom/` (map tác giả) + `checkpoints/` (detector tác
giả) ⇒ đây là **R0**, và là chế độ duy nhất không cần GPU.

Điều chế độ A **không** làm được: chấm ảnh của người dùng. Phải nói thẳng trên
UI, không để người xem tự phát hiện.

### B. Upload tự tính — chỉ bật khi có số đo

Trên CPU free (2 vCPU), mỗi ảnh cần `2·S` lượt forward của U-Net 552 M tham số ở
256². Với `S = 20` là 40 lượt. **Chưa ai đo**, nên chưa được hứa gì.

Trình tự bắt buộc:

```bash
# trên máy, CÙNG cấu hình CPU (không phải GPU rồi suy ra)
python -m dire_zero bench --device cpu --steps 5 10 20   # khi có cờ này
```

Ngưỡng quyết định, chốt trước khi đo để không tự thương lượng với chính mình:

| p95 thời gian/ảnh | Quyết định |
|---|---|
| < 15 s | bật mặc định, `S = 5` |
| 15–45 s | bật nhưng **sau nút bấm**, hiện đồng hồ đếm và ước lượng thời gian |
| > 45 s | **tắt** chế độ B; chỉ còn A, ghi rõ "cần GPU để chấm ảnh của bạn" |

**Đã đo (2026-11-17), 2 vCPU Xeon 2,6 GHz, ADM 256² thật:** 10,1 s/forward fp32
⇒ `S = 5` = **101 s**, `S = 20` = **404 s**. Rơi vào hàng cuối ⇒ chế độ B **tắt**
theo đúng ngưỡng đã chốt ở trên. Vẫn có trang "Tính DIRE thật" nhưng **mặc định
tắt**, phải tick xác nhận thời gian chờ. Chi tiết + ba cách thấy phép tính:
`docs/demo-hf-spaces-plan.md` §13.

Nếu bật: `S = 5` mặc định (paper Table 4: `S=5` vẫn 100/100 trên ADM, 87,5/99,8
trên SD-v1), và `S` hiện **trên** kết quả. Không được im lặng dùng `S=20` rồi để
người xem chờ 3 phút.

### C. GPU backend — đã dựng, bật bằng secret

Đã viết xong và kiểm bằng test, chưa deploy (cần tài khoản Modal): endpoint
`POST /dire` trên A10G (`src/dire_zero/demo_api.py` + `demo_service.py`), Space gọi
qua `demolib/remote.py`. Ba ràng buộc đã thành code, không còn là lời khuyên:

- **Trình duyệt không gọi Modal.** Nó gọi server của Space (cùng origin, URL tương
  đối); Space giữ token trong `st.secrets` rồi gọi Modal. Nếu để trình duyệt cầm
  token thì token không còn là bí mật — chỉ là chuỗi ai cũng đọc được trong DevTools;
- **Endpoint có token** (`X-Demo-Token`); thiếu cấu hình ⇒ **503**, không mở;
  CORS chỉ bật khi đặt `DEMO_ORIGINS` (mặc định đóng);
- **giới hạn 10 MB + 6 ảnh/60 s** + `S ≤ 20`, nếu không một người có thể đốt hết
  credit trong một buổi chiều.

Chi phí, cold start, và cách chạy buổi nói: `docs/demo-modal-plan.md`.

## 4. Màn hình: chữ nào bắt buộc phải có

Thứ tự trên trang, và lý do từng dòng:

1. **Một câu nói demo làm được gì** — "chọn một ảnh trong bộ dữ liệu, xem DIRE và
   điểm của detector", không phải "upload ảnh để phát hiện AI".
2. **Ba panel** `ảnh gốc | DIRE | điểm`. Panel giữa là thứ giải thích *vì sao* có
   điểm; bỏ nó đi thì demo còn lại một hộp đen.
3. **Dòng nguồn gốc** dưới mỗi kết quả:
   `DIRE: tác giả · detector: tác giả (lsun_adm.pth) · nhãn: R0 · S=20`.
   Đây là câu trả lời cho "con số này của ai", và nó phải nằm cạnh số, không nằm
   trong tab "Giới thiệu".
4. **Ngưỡng + nó từ đâu**: `ngưỡng 0,5`, chọn trên **val**, không bao giờ trên test.
5. **Cảnh báo khi mẫu thuộc generator yếu** (nếu AUC của generator đó < 0,9, in
   "generator này khó — xem hình A2").
6. **Không có phần "độ chính xác 99 %"** ở bất kỳ đâu trên trang. Số của paper, nếu
   nhắc, phải kèm "số của paper (ICCV 2023), không phải của demo này".

Trạng thái lỗi cũng là nội dung, và phải nói được *cách sửa*:

| Tình huống | Hiện gì |
|---|---|
| chưa có checkpoint | tên repo/file đang tìm + lệnh tạo + "đừng tin số nào ở đây" |
| checkpoint có nhưng `fc.weight` sai shape | **dừng**, in shape đọc được (app hiện đã làm đúng chỗ này) |
| gallery rỗng | lệnh sinh gallery; không hiện panel trống giả |
| ảnh upload quá lớn / không đọc được | nói rõ giới hạn, không trả xác suất |

## 5. Cổng kiểm trước khi công khai (bắt buộc)

Demo chỉ được bật chế độ A khi qua **cả ba** cổng, chạy trước khi public:

1. **Cổng checkpoint**: mở `checkpoints/lsun_adm.pth` bằng `torch.load`, in
   `key → shape` của `fc.weight`. Nếu không phải `[1, 2048]` hoặc `[2, 2048]` thì
   *không* dùng được với app hiện tại — và 269 MB/file là dấu hiệu bên trong có
   thể không chỉ là trọng số thuần (ResNet-50 fp32 ≈ 98 MB), nên **phải xem**, không đoán.
2. **Cổng tái lập**: chấm một tập con `dire/val/lsun_bedroom/` (không phải test)
   và so với Table 2. Nếu ACC lệch quá xa (ví dụ < 80 % trên ADM) thì chuỗi
   `checkpoint ↔ map ↔ chuẩn hoá` đang sai ⇒ **không hiện xác suất**, chỉ hiện
   map và ghi "chưa đạt cổng".
3. **Cổng chuẩn hoá**: DIRE map tác giả lưu dạng gì (`uint8`? `float16 [0,2]`?),
   và app chuẩn hoá thế nào? Sai một bước là lệch số **âm thầm** — đúng lỗi
   `Normalize` đã gặp ở `demo.py` của tác giả (README dòng 40). Ghi kết quả vào
   `platforms/hf-spaces/README.md`.

Chưa qua cổng 2 thì demo vẫn deploy được, nhưng ở trạng thái "chỉ xem", và điều
đó phải hiện trên trang — không phải trong log.

## 6. Kiến trúc & vận hành

```
platforms/hf-spaces/
├── app.py            # UI; KHÔNG tự tính DIRE ở chế độ A
├── gallery/          # <tên>_orig.png  <tên>_recon.png  <tên>_diff.png  <tên>.json
│                     # .json: generator, split, nhãn R, nguồn map, sha256, S
├── requirements.txt  # gim phiên bản (đã gim)
└── README.md         # cổng kiểm, cách thay checkpoint, cách thay gallery
```

| Vấn đề | Cách xử lý | Vì sao |
|---|---|---|
| cold start | nạp model trong `_load_model()` lúc khởi động, có cờ "đang nạp" trên UI | người xem bấm Detect trong 30 s đầu sẽ tưởng app hỏng |
| RAM | ResNet-50 fp32 ≈ 100 MB ⇒ thoải mái; **không** nạp ADM ở chế độ A | 552 M × 4 B = 2,2 GB chỉ để vẽ map đã có sẵn |
| upload lớn | giới hạn 10 MB, resize 256 theo đúng giao thức | ảnh 4000² làm chậm mọi thứ mà không thêm thông tin |
| ảnh động / ảnh lạ | chỉ nhận RGB; ảnh có alpha thì bỏ alpha, nói rõ | nhất quán với lúc train |
| tải lại nhiều lần | `gr.Queue`, cache điểm theo sha256 của map | người xem bấm 5 lần không nên tốn 5 lần tính |
| log | **không** log ảnh, không log IP; chỉ log `sha256`, thời gian, lỗi | ảnh người dùng là dữ liệu của họ |
| lưu ảnh upload | chỉ trong bộ nhớ / `tempfile`, xoá sau request | không có lý do giữ |
| license | caption gallery ghi "DiffusionForensics, Wang et al. ICCV 2023, research-only" + takedown | dữ liệu không phải của mình |

## 7. Chi phí

| Thành phần | Chi phí |
|---|---|
| HF Space CPU basic (2 vCPU, 16 GB) | **0 ₫** — đủ cho chế độ A |
| HF Space GPU | > 0 ₫/giờ — **không bật** trước khi có số đo chế độ B |
| Modal A10G cho chế độ C | > 0 ₫/ảnh — chỉ khi bạn chọn |

Nguyên tắc giữ nguyên từ trước: **chưa đo `sec/step`/`sec/ảnh` thì không có bảng
tiền**. Bảng trên chỉ ghi "0 ₫" cho free tier vì đó là điều kiện của gói, không
phải ước lượng.

## 8. Việc triển khai, chia theo PR

| PR | Nội dung | Phụ thuộc |
|---|---|---|
| D1 | Cổng kiểm: script đọc checkpoint + chuẩn hoá map + chấm val, in bảng | cần 1 checkpoint tác giả |
| D2 | Chế độ A: gallery có `.json` nguồn gốc, UI 3 panel, nhãn R trên khung | D1 |
| D3 | Sửa Tab 1: bỏ đường "RGB → xác suất", thay bằng "chọn map trong gallery" | D2 |
| D4 | Đo CPU `sec/ảnh` cho `S = 5/10/20`, quyết định bật/tắt chế độ B | máy thật |
| D5 (tuỳ chọn) | Chế độ C: Modal endpoint + token + rate limit | bạn đồng ý chi tiền |

D1–D3 **không cần GPU**, không cần train gì, và đó là toàn bộ demo dùng được.
D4 cần bạn chạy trên máy thật. D5 để sau, và có thể không bao giờ cần.

## 9. Định nghĩa "xong" cho demo

- [ ] không có đường nào trên UI trả xác suất từ ảnh RGB thô;
- [ ] mọi kết quả có dòng `DIRE: … · detector: … · nhãn: R0/R1+R2 · S=…`;
- [ ] gallery có `.json` cho từng mẫu (generator, split, nguồn, sha256);
- [ ] trạng thái "chưa có checkpoint / chưa đạt cổng" hiện **trên trang**, kèm lệnh sửa;
- [ ] `tests/` có test đọc `app.py` và chặn tái phát: không được có nhánh nào
      `detect()` trả xác suất mà không đi qua DIRE map;
- [ ] README của Space ghi cổng kiểm, phiên bản, cách thay checkpoint/gallery;
- [ ] không log ảnh người dùng; upload giới hạn 10 MB;
- [ ] nhãn "số của paper" tách khỏi "số của demo" ở mọi chỗ có số.

## 10. Cái sẽ không làm, và nói trước

- **Không** làm demo "upload ảnh → phát hiện AI" như một sản phẩm. Nó cần chế độ
  B/C, và cả hai đều chưa có số đo. Hứa trước rồi trượt là kiểu thất bại tệ nhất
  với một demo công khai.
- **Không** dùng ảnh do người khác sinh trong gallery mà không ghi nguồn.
- **Không** hiện `P(fake)` khi chuỗi `checkpoint ↔ map ↔ chuẩn hoá` chưa qua cổng 2.
- **Không** vẽ panel "ảnh tái tạo" cho generator không có `recons/` (thiếu `adm`,
  `ddpm`, `iddpm`, `sdv2`, `vqdiffusion`, …) — xem `docs/figures-plan.md` §2.
