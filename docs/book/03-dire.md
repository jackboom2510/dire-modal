# Phần 7 — DIRE: định nghĩa, các lựa chọn thiết kế, chi phí

## 7.1 Định nghĩa đầy đủ

$$[17] \mathrm{DIRE}(x_0) = \left\vert\, x_0 - R\big(I(x_0)\big) \,\right\vert $$

- `I`: **phép đảo ngược** (DDIM inversion), `S` bước, `t` **tăng** từ 0 tới mốc cao
  nhất của tập mốc.
- `R`: **phép tái tạo** (DDIM sampling), `S` bước, `t` **giảm** từ mốc cao nhất về 0.
- `|·|`: trị tuyệt đối **theo từng điểm ảnh**, giữ **3 kênh**.
- Kết quả: ảnh $256\times256\times3$ cùng kích thước đầu vào.

Ba điều kiện để định nghĩa này có nghĩa:

1. `I` và `R` phải là **cùng một** mô hình, **cùng một** lịch, **cùng một** tập mốc.
   Nếu khác, `x_T` do `I` tạo ra không phải điểm khởi đầu mà `R` "mong đợi", và
   phép trừ đo một đại lượng vô nghĩa. Đây là lỗi F2 của kho này.
2. `η = 0` (tất định). Nếu có nhiễu ngẫu nhiên, DIRE không tái lập được.
3. Thứ tự bước phải khớp: `I` dùng `ᾱ_{t+1}`, `R` dùng `ᾱ_{t−1}`.

## 7.2 Vì sao lấy trị tuyệt đối — có số liệu

Xét residual `r = x₀ − x̂₀` trước khi lấy abs. `r` có **dấu**: chỗ tái tạo sáng hơn
ảnh gốc thì dương, tối hơn thì âm. Vấn đề: dấu phụ thuộc chiều lệch, mà chiều lệch
**không ổn định** giữa các ảnh. Hai vùng cùng "khó tái tạo" có thể lệch ngược dấu
và triệt tiêu nhau khi ResNet tổng hợp.

`|r|` chỉ giữ **độ lớn** — đại lượng có tính hệ thống: ảnh thật khó tái tạo nên
`|r|` lớn, ảnh sinh dễ tái tạo nên `|r|` nhỏ.

Ablation của bài báo (Table 6, train trên ADM, định dạng ACC/AP %):

| | ADM | DDPM | iDDPM | PNDM | SD-v1 |
|---|---|---|---|---|---|
| **không** abs | 100/100 | 99,4/100 | 100/100 | 98,2/100 | **87,0/93,0** |
| **có** abs | 100/100 | 100/100 | 100/100 | 99,7/100 | **99,7/100** |

Chênh lệch 12,7 điểm ACC ở SD-v1. Đây là lý do bài báo gọi phép abs là thành phần
"critical", không phải chi tiết phụ.

**Một lưu ý về abs và nhiễu.** Abs là phép phi tuyến, nó biến phân phối Gauss đối
xứng thành phân phối nửa chuẩn (half-normal). Với ảnh sinh, `r` gần 0 nên `|r|`
tập trung quanh 0; với ảnh thật, `|r|` phân tán rộng. Đây chính là cấu trúc mà
ResNet học: không phải "giá trị trung bình" mà "**hình dạng phân phối** của sai số".

## 7.3 Vì sao 256×256, vì sao crop 224

- ADM dùng trong bài báo là `256x256_diffusion_uncond`: kiến trúc **cố định** 256.
  Đưa ảnh khác kích thước vào sẽ sai shape ở tầng attention.
- Ảnh SD là 512×512, bị **bicubic xuống 256** (`reshape_image`). Đây là mất mát
  thông tin có thật; nó cũng là lý do mọi con số trong bài báo ở 256.
- Crop 224 cho ResNet-50 vì đó là kích thước chuẩn ImageNet. Train: crop ngẫu
  nhiên từ 256 (32 pixel dư để dịch, tạo augmentation); test: crop giữa.

Hai lựa chọn này có hệ quả cần nhớ khi so sánh với phương pháp khác: nếu đối thủ
của bạn làm việc ở 512, họ có nhiều thông tin hơn. So sánh "DIRE 99,7 vs phương
pháp X 98" ở hai độ phân giải khác nhau là so sánh chưa công bằng, và bài báo
không thảo luận điểm này.

## 7.4 Vì sao giữ 3 kênh, không chuyển xám

Kết quả DIRE là ảnh 3 kênh. Chuyển xám sẽ trộn thông tin của ba kênh và mất phần
sai số khác nhau theo kênh — vì mô hình khuếch tán xử lý ba kênh như ba kênh độc
lập (không có ràng buộc vật lý giữa R, G, B), nên mức khó tái tạo khác nhau theo
kênh.

Bài báo **không** thảo luận điểm này và **không** làm ablation 3 kênh vs 1 kênh.
Đọc mã thì thấy 3 kênh được giữ nguyên. Đây là lựa chọn ngầm định, và là một thí
nghiệm nhỏ còn bỏ ngỏ (Phần 12.2).

## 7.5 Số bước `S`: đánh đổi độ chính xác ↔ chi phí

Table 4 của bài báo (train trên ADM, test 5 bộ sinh, ACC/AP %):

| `S` | ADM | DDPM | iDDPM | PNDM | SD-v1 |
|---|---|---|---|---|---|
| 5 | 100/100 | 100/100 | 100/100 | 97,5/100 | 87,5/99,8 |
| 10 | 100/100 | 100/100 | 100/100 | 99,4/100 | 98,2/100 |
| **20** | 100/100 | 100/100 | 100/100 | 99,7/100 | **99,7/100** |
| 50 | 100/100 | 100/100 | 100/100 | 100/100 | 99,9/100 |

Đọc bảng này đúng cách:

- `S` càng lớn, xấp xỉ Euler càng chính xác, DIRE càng "sạch" ⇒ hiệu năng tăng.
- Chi phí tăng **tuyến tính** theo `S`, và **gấp đôi** vì có cả đảo ngược.
- Điểm khuỷu là 20: từ 20 lên 50 chỉ thêm 0,2 điểm ở SD-v1 mà tốn 2,5× chi phí.

Đây là ví dụ mẫu của đánh đổi độ chính xác ↔ chi phí, và cách một bài báo tốt chọn
điểm khuỷu thay vì chọn cực trị. Nhưng cần nói thêm: con số 99,7 ở `S = 20` và
99,9 ở `S = 50` nằm trong khoảng tin cậy của nhau (1 000 ảnh, ±0,6 điểm), nên bảng
này **không** chứng minh 50 tốt hơn 20 về mặt thống kê; nó chỉ cho thấy không có
hại.

## 7.6 Vì sao DIRE tốt hơn RGB, REC, và RGB&DIRE

Table 5 của bài báo (train trên subset ADM, ACC/AP %):

| Đầu vào | DDPM | iDDPM | PNDM | SD-v1 |
|---|---|---|---|---|
| REC (ảnh tái tạo) | 57,1/57,7 | 49,7/92,6 | 87,1/98,7 | 46,9/57,0 |
| RGB (ảnh gốc) | 87,3/99,6 | 100/100 | 77,8/99,1 | 77,4/85,8 |
| RGB & DIRE | 99,8/100 | 99,9/100 | 99,2/100 | **62,4/92,4** |
| **DIRE** | **100/100** | **100/100** | **99,7/100** | **99,7/100** |

Ba kết luận, mỗi kết luận là một bài học thiết kế:

1. **REC tệ nhất.** Ảnh tái tạo đã bị "kéo" về phía phân phối mô hình, nên mất
   chính thông tin phân biệt. Nghĩa là thông tin nằm ở **hiệu**, không ở **ảnh tái
   tạo**. Đây là bằng chứng trực tiếp cho định nghĩa DIRE.
2. **RGB khá nhưng không bền.** Nó học dấu vết; dấu vết đó chuyển được giữa các
   bộ sinh *cùng họ* (iDDPM 100%) nhưng vỡ khi sang SD-v1 (77,4%).
3. **RGB & DIRE tệ hơn DIRE.** Ghép thêm RGB làm mô hình có thể "lười": nhánh RGB
   dễ học hơn trong tập train nên mô hình dựa vào nó và bỏ qua nhánh DIRE, mất tính
   bất biến. Đây là **học đường tắt**, và là lời cảnh báo cho mọi thiết kế "ghép
   thêm đặc trưng cho chắc".

## 7.7 Chi phí thật của DIRE

Một ảnh = `2S` lượt chạy ADM + 1 lượt ResNet-50.

| Đại lượng | Công thức | Với `S = 20` |
|---|---|---|
| NFE mỗi ảnh | $2S$ | 40 |
| NFE cho 1 000 ảnh | `2S·1000` | 40 000 |
| NFE cho toàn bộ bộ dữ liệu (~336k ảnh) | `2S·N` | ≈ 13,4 triệu |

Đây là hoạt động tốn kém nhất của bài báo, và là lý do nhóm tác giả **phát hành
sẵn** DIRE map: người tái lập không phải trả chi phí đó. Đổi lại, người tái lập
không thể tuyên bố "tự tính DIRE" — chỉ có thể **chấm lại** trên artefact của họ
(nhãn `R0`, Phần 11.0).

Với ước lượng thời gian (giả định **chưa đo**, phải ghi rõ là giả định):

| Giả định `sec/ảnh` trên A100 | Giờ cho 200k ảnh | 2,0 USD mỗi giờ |
|---|---|---|
| 0,004 | 0,22 | ≈ 0,4 |
| 0,004 × 25 (nếu chạy sai 1000 bước) | 5,6 | ≈ 11 |
| 0,027 | 1,5 | ≈ 3 |

Con số "×25" là để thấy hệ quả của lỗi F2: chạy 1020 thay vì 40 NFE không phải
"sai một chút", nó là **sai 25 lần hoá đơn** và đồng thời làm biểu diễn vô nghĩa.

# Phần 8 — Thực nghiệm của bài báo

## 8.1 Bộ dữ liệu (Table 1)

| Nguồn | Bộ sinh | Điều kiện | Số ảnh thật/giả | Chia |
|---|---|---|---|---|
| LSUN-Bedroom [[18]] | ADM [[4]] | không điều kiện | 42k/42k | 40k/1k/1k |
| | DDPM [[3]] | không điều kiện | — /1k | **chỉ test** |
| | iDDPM [[8]] | không điều kiện | 42k/42k | 40k/1k/1k |
| | PNDM [[7]] | không điều kiện | 42k/42k | 40k/1k/1k |
| | LDM [[6]] | text→ảnh | 1k/1k | chỉ test |
| | Stable Diffusion v1 [[9]] | text→ảnh | 1k/1k | chỉ test |
| | Stable Diffusion v2 [[9]] | text→ảnh | 1k/1k | chỉ test |
| | VQ-Diffusion [[10]] | text→ảnh | 1k/1k | chỉ test |
| ImageNet | ADM [[4]] | có điều kiện lớp | 50k/50k | 40k/5k/5k |
| | Stable Diffusion v1 [[9]] | text→ảnh | 10k/10k | chỉ test |

Bốn điều cần chú ý:

- **Chỉ 4 bộ sinh có tập train** (ADM, iDDPM, PNDM ở LSUN; ADM ở ImageNet). Bảy
  bộ sinh còn lại **chỉ test** — đó là thiết kế để đo tổng quát hoá, không phải
  thiếu dữ liệu.
- Ảnh thật được chia theo **đúng split của ảnh giả tương ứng**. Nếu không, ảnh thật
  trong train sẽ trùng ảnh thật trong test.
- Mọi ảnh phát hành dưới dạng **bộ ba** (ảnh nguồn, ảnh tái tạo, ảnh DIRE). Quyết
  định này rất tốt cho cộng đồng: người sau thử biểu diễn mới mà không phải chạy
  40 NFE/ảnh.
- ADM (LSUN) được train trên **42k** ảnh sinh nhưng chỉ dùng 40k cho train, 1k cho
  val, 1k cho test. Với `n = 1000` mỗi lớp ở test, sai số chuẩn của ACC ở mức 99%
  là khoảng 0,3% (Phần 2.5).

## 8.2 Prompt: chi tiết nhỏ, hệ quả lớn

- LSUN text→ảnh: prompt cố định `"A photo of bedroom"`.
- ImageNet SD-v1: `"A photo of {class}"` cho 1000 lớp.

Vì prompt cố định, "tổng quát hoá" trong Table 2 **không** bao gồm biến thiên
prompt. Đây là hạn chế cần biết: ảnh SD ngoài đời có prompt đa dạng, và phân phối
đó rộng hơn nhiều so với một prompt duy nhất. Một detector đạt 99,7% trên "một
prompt" chưa chắc đạt 99,7% trên "mọi prompt".

## 8.3 Kết quả chính (Table 2, khối DIRE)

Bốn hàng dưới đây là bốn **detector** khác nhau, mỗi cái train trên ảnh sinh bởi
một bộ sinh (cột "Train"), rồi test trên 8 bộ sinh:

| Train | ADM | DDPM | iDDPM | PNDM | SD-v1 | SD-v2 | LDM | VQ | TB |
|---|---|---|---|---|---|---|---|---|---|
| **ADM** | 100 | 100 | 100 | 99,7 | 99,7 | 100 | 100 | 100 | **99,9** |
| **PNDM** | 100 | 100 | 100 | 100 | 89,4 | 100 | 100 | 100 | 98,7 |
| **iDDPM** | 99,6 | 100 | 100 | 89,7 | 99,7 | 100 | 99,9 | 99,9 | 98,6 |
| **StyleGAN** | 98,8 | 99,8 | 99,9 | 89,6 | 95,2 | 100 | 100 | 100 | 97,9 |

(AP của mọi ô trong khối này là 100,0 trừ ba ô 99,8–99,9; xem
`src/dire_modal/paper_table2.tsv` để có đủ 32 ô.)

Đọc bảng này đúng cách:

- **Đường chéo gần 100**: điều kiện tiên quyết — cùng bộ sinh thì phải gần hoàn
  hảo, nếu không có gì để bàn tiếp.
- **Các ô lệch** (89,4 · 89,7 · 89,6) đều ở cột **PNDM** hoặc **SD-v1** khi detector
  train trên bộ sinh khác. Vậy hai bộ sinh này "khó" nhất, và bài báo không phân
  tích vì sao.
- **Trung bình 97,9–99,9** rất cao, nhưng như đã nói, chênh lệch vài phần mười
  không có ý nghĩa thống kê ở `n = 1000`.

Baseline để so (cùng bảng, các cột đã train lại trên ảnh ADM):

| Phương pháp | ACC | AP |
|---|---|---|
| CNNDet [[11]] (bản gốc, 2019) | 51,1–58,9 | — |
| CNNDet\* (train lại) | 86,5 | 97,3 |
| Patchfor [[12]]\* | 90,9 | 96,4 |
| F3-Net [[13]]\* | 91,6 | 97,7 |
| **DIRE** | **99,9** | **100,0** |

Chênh lệch lớn nhất không phải 99,9 so với 91,6 (ACC), mà là **AP 100 so với 97,7**:
AP đo khả năng xếp hạng trên toàn bộ dải ngưỡng, và ở đây DIRE gần như hoàn hảo.

## 8.4 Tổng quát hoá chéo (Table 3)

| Train trên | ADM (ImageNet) | SD-v1 (ImageNet) | StyleGAN (LSUN) |
|---|---|---|---|
| LSUN ADM | 90,2/97,9 | 97,2/99,8 | 99,9/100 |
| LSUN iDDPM | 90,2/97,9 | 93,7/99,3 | 99,9/100 |
| LSUN StyleGAN | **76,9**/94,4 | 89,7/99,0 | 100/100 |

Bài học: đổi **miền** (LSUN phòng ngủ → ImageNet vật thể) khó hơn nhiều đổi **bộ
sinh** trong cùng miền. Ô 76,9% là mức tụt lớn nhất trong toàn bộ bài báo, và bài
báo **không phân tích** nó. Đây là chỗ hạn chế quan trọng: nếu người dùng thật
muốn phát hiện ảnh sinh trong một miền ít phổ biến (y tế, ảnh vệ tinh, tài liệu),
con số 76,9% gần với kỳ vọng hơn 99,9%.

## 8.5 Robustness trước làm mờ và nén (Hình 4)

- Làm mờ Gauss `σ = 1, 2, 3`.
- Nén JPEG `quality = 65` và `30`.
- Bài báo mô tả: DIRE "gần như không giảm", các baseline dựa trên RGB sụp mạnh.

Ba điều cần nói thêm để đọc đúng:

1. Đây là **biểu đồ**, không phải bảng số. Không thể trích một con số chính xác, và
   bài báo không công bố bảng tương ứng.
2. Về lý thuyết điều này hợp lý: DIRE là hiệu của hai ảnh đi qua **cùng một** phép
   biến đổi, nên phần bị phá huỷ triệt tiêu một phần.
3. Nhưng lý thuyết đó cũng dự đoán DIRE sẽ **giảm** khi làm mờ mạnh, vì làm mờ xoá
   tần số cao — mà theo Mục 4.5 của bài báo, tần số cao chính là nơi ảnh thật khác
   ảnh sinh rõ nhất. Bài báo không đo mức giảm đó.

## 8.6 Quan sát định tính (Mục 4.5)

Bài báo quan sát: DIRE của ảnh **thật** có vân **có quy luật**, phổ **giàu tần số**;
DIRE của ảnh **sinh** trông lộn xộn và nghèo phổ. Quan sát này khớp với trực giác
"ảnh thật khó tái tạo ở những chỗ có cấu trúc" (biên, texture) — sai số tập trung
ở đó tạo ra vân có quy luật.

Nhưng đây là **quan sát**, không phải phép đo. Không có phân tích phổ định lượng,
không có hệ số tương quan. Nó hữu ích cho trực giác, không dùng được làm tiêu chí.

# Phần 9 — Phản biện: bài báo mạnh và yếu ở đâu

Phần này viết dưới dạng một bản nhận xét phản biện (peer review) có cấu trúc. Mục
9.1–9.3 là bản chất ý tưởng và những gì tác giả tái dùng; 9.4–9.5 là điểm mạnh và
điểm yếu về phương pháp; 9.6–9.7 là khoảng trống nghiên cứu và ba câu hỏi tiếp.

## 9.1 Ý tưởng, phát biểu đầy đủ

1. **Tiền đề đo được.** Các detector cũ (CNNDet, Patchfor, F3-Net) đạt 86–91% ACC
   khi được **train lại** trên đúng ảnh ADM, so với 51–59% của bản train trên
   ProGAN. Nói cách khác: không phải kiến trúc kém, mà là **dấu vết không chuyển
   được**. Đây là tiền đề định lượng, không phải giả định.
2. **Quan sát cơ chế.** Ảnh sinh bởi khuếch tán nằm gần quỹ đạo mà mô hình khuếch
   tán định nghĩa; ảnh thật thì không. Do đó sai số tái tạo khác nhau một cách hệ
   thống.
3. **Biểu diễn.** `DIRE = |x − R(I(x))|`, tất định, không cần train thêm, không cần
   dữ liệu ảnh giả lúc suy luận.
4. **Bộ phân loại.** ResNet-50 nhỏ, 1 logit, BCE, train 400 epoch trên ảnh DIRE.
5. **Kiểm chứng.** 8 bộ sinh LSUN + 2 bộ sinh ImageNet, 4 detector khác nhau, kèm
   ablation cho `S`, cho abs, cho loại đầu vào, và kiểm robustness.

## 9.2 Pipeline huấn luyện, tài nguyên, và những gì tác giả tái dùng

| Giai đoạn | Làm gì | Tài nguyên | Nguồn |
|---|---|---|---|
| Sinh ảnh giả | chạy 8 bộ sinh với prompt/seed cố định | GPU (nhiều giờ) | Phụ lục A |
| Sinh ảnh thật | lấy từ LSUN/ImageNet **đúng split** | — | dữ liệu công khai |
| Tính DIRE | ADM 256², `ddim20`, `η=0` | `mpiexec -n 4`, GPU | mã tác giả |
| Train detector | ResNet-50 ImageNet, Adam 1e-4, batch 64, 400 epoch, patience 5 | 1 GPU | mã tác giả |
| Đánh giá | ACC@0,5 + AP | — | mã tác giả |

Tái dùng (đọc Phụ lục A và dòng Acknowledgments): `openai/guided-diffusion` [[4]] (cả
mô hình ADM 256 và mã DDIM), `hojonathanho/diffusion` [[3]],
`openai/improved-diffusion` [[8]], `luping-liu/PNDM` [[7]], `CompVis/ldm` +
`runwayml/stable-diffusion` + `stabilityai/stable-diffusion-2` qua `diffusers` [[9]],
VQ-Diffusion của Microsoft [[10]], và khung thực nghiệm của `peterwang512/CNNDetection`
[[11]] (cấu trúc thư mục `0_real`/`1_fake`, `ImageFolder`, early stopping, validate).

Điểm cần nhấn: **tác giả không train bất kỳ mô hình sinh nào**. Toàn bộ mô hình sinh
là pretrained của người khác. Đóng góp của bài báo là phép biến đổi và thực nghiệm.

## 9.3 Những gì bài báo không làm (để tránh hiểu nhầm)

- Không train lại ADM, không fine-tune ADM cho bài toán phát hiện.
- Không thử detector mạnh hơn ResNet-50 (ViT, ConvNeXt, ensemble).
- Không thử ảnh ngoài 256² (kể cả SD 512² gốc, chỉ dùng bản đã resize).
- Không thử ảnh qua mạng xã hội thật (chỉ mô phỏng nén/làm mờ).
- Không phân tích phổ định lượng cho DIRE map.
- Không báo khoảng tin cậy, không báo nhiều seed cho kết quả chính.

## 9.4 Điểm mạnh về phương pháp

- **Tiền đề có số.** Việc train lại baseline trên cùng dữ liệu (86–91%) là thao tác
  đúng và hiếm; nó loại bỏ cách giải thích "tại baseline train trên dữ liệu cũ".
- **Ablation đúng chỗ.** Bốn ablation (`S`, abs, loại đầu vào, REC vs RGB) trả lời
  đúng bốn câu hỏi thiết kế. Đặc biệt ablation "RGB & DIRE tệ hơn DIRE" là thí
  nghiệm **phản trực giác** và có giá trị cảnh báo.
- **Kiểm chéo nhiều bộ sinh.** 7 bộ sinh chỉ test là thiết kế đúng để đo tổng quát
  hoá; nhiều bài chỉ test trên bộ sinh mình train.
- **Tất định và tái lập được.** `η = 0` làm DIRE tái lập chính xác, điều mà phương
  pháp dựa trên Monte Carlo không có.
- **Phát hành artefact.** Phát hành cả ảnh gốc, ảnh tái tạo và DIRE map giúp cộng
  đồng kiểm chứng và thử biểu diễn mới ở chi phí gần 0.

## 9.5 Điểm yếu về phương pháp

- **Cơ chế chưa được chứng minh.** Phụ lục B là lập luận định tính về sai số Euler;
  không có phép đo độ cong quỹ đạo, không có thí nghiệm tách hai cơ chế (độ cong
  vs phép cắt `[−1,1]`). Và lập luận này **không giải thích** được kết quả
  cross-generator, vốn là kết quả mạnh nhất.
- **Không có khoảng tin cậy.** `n = 1000` mỗi ô; chênh lệch ≤1 điểm không có ý
  nghĩa. Bảng 2 và Bảng 4 vì thế bị đọc quá mức.
- **Không có thí nghiệm phá (stress test) đúng mức.** Robustness chỉ có làm mờ và
  JPEG; thiếu resize lần hai, crop, xoay nhẹ, lưu lại qua mạng xã hội, ảnh chụp
  lại màn hình, ảnh ghép (inpainting), ảnh qua pipeline video (nén H.264).
- **Không xét kẻ tấn công chủ động.** Nếu người dùng biết DIRE, họ có thể tối ưu
  ảnh để DIRE nhỏ (hoặc lớn) — một bài toán đối kháng. Bài báo không thử.
- **Phụ thuộc một mô hình tham chiếu.** Toàn bộ kết quả dùng ADM 256² và `S = 20`.
  Không có thí nghiệm đổi mô hình tham chiếu (ví dụ dùng iDDPM làm `I`/`R`), nên
  không biết DIRE là tính chất của **phép biến đổi** hay của **ADM cụ thể**.
- **Lựa chọn ResNet-50 và crop 224 không được biện luận.** Kế thừa từ CNNDetection.
- **Chi phí không được báo cáo.** Không có bảng GPU-giờ. Với 336k ảnh × 40 NFE,
  đây là chi phí đáng kể mà người tái lập cần biết trước.

## 9.6 Khoảng trống nghiên cứu: đã giải quyết và chưa

**Đã giải quyết (ở mức thực nghiệm):**

- Dấu vết của bộ sinh không chuyển được giữa các họ ⇒ cần tín hiệu độc lập kiến
  trúc. DIRE là một tín hiệu như vậy, và nó chuyển được (99,7% trên SD-v1).
- Abs là cần thiết, không phải tùy chọn.
- Ghép thêm đặc trưng RGB là có hại.
- `S = 20` là điểm cân bằng hợp lý.

**Chưa giải quyết:**

- **Cơ chế.** Vì sao ảnh sinh dễ tái tạo? Chưa có mô hình lý thuyết kiểm được.
- **Miền lạ.** 76,9% khi đổi miền là dấu hiệu yếu; chưa có phương pháp nào xử lý.
- **Bộ sinh mới.** Bài báo dừng ở 2023 (SD 1/2, LDM, VQ-Diffusion). Chưa có kết quả
  cho SDXL, SD3, Flux, DALL·E 3 — những mô hình dùng kiến trúc transformer và
  flow matching, khác hẳn U-Net + DDPM.
- **Đối kháng.** Chưa có nghiên cứu về tấn công/thủ DIRE.
- **Ảnh bị xử lý.** Chưa có kết quả cho JPEG lặp, resize, in–chụp lại, video.
- **Chi phí.** Chưa có phương pháp giảm `2S` NFE mà giữ hiệu năng (distillation,
  caching, hoặc dùng mô hình nhỏ hơn làm `I`/`R`).
- **Định lượng ngưỡng.** Chưa có nghiên cứu về chọn ngưỡng theo `FPR` cho ứng dụng
  thật, nơi chi phí buộc tội oan cao.

## 9.7 Ba câu hỏi nghiên cứu tiếp theo

1. **Cơ chế:** Có thể tách hai nguồn sai số — rời rạc hoá Euler và phép cắt
   `[−1,1]` — bằng thực nghiệm không? Cụ thể: chạy `I`/`R` với `clip_denoised=False`
   và với `S` tăng dần; nếu khoảng cách DIRE(ảnh thật) − DIRE(ảnh sinh) **không**
   giảm khi `S` tăng, thì cơ chế không phải sai số Euler, và Phụ lục B sai.
2. **Bất biến theo mô hình tham chiếu:** DIRE có còn hoạt động nếu thay ADM bằng
   một mô hình khuếch tán khác (iDDPM, hoặc một mô hình nhỏ distill) làm `I`/`R`?
   Nếu có, DIRE là tính chất của **quá trình**; nếu không, nó là tính chất của
   **ADM**. Câu trả lời quyết định hướng triển khai thực tế (dùng mô hình nhỏ ⇒ rẻ
   25×).
## 9.8 Họ hàng gần nhất của DIRE: AEROBLADE

AEROBLADE [[20]] (CVPR 2024) dùng **cùng một tiền đề** với DIRE nhưng khác linh kiện:

| | DIRE [[1]] | AEROBLADE [[20]] |
|---|---|---|
| Phép biến đổi | diffusion model (ADM), đảo ngược + tái tạo, `S` bước | **autoencoder** của LDM, một lượt encode–decode |
| Không gian | ảnh (pixel) | ẩn (latent) |
| Độ đo sai số | `\|x − x̂\|` theo từng điểm ảnh | **LPIPS** (đặc trưng VGG16 tầng 2) |
| Cần train? | có, ResNet-50 trên ảnh DIRE | **không**, chỉ so ngưỡng |
| Chi phí | `2S` NFE | 1 lượt AE |
| Áp dụng cho | mọi mô hình sinh ảnh | chỉ ảnh do **LDM** sinh |

Bài học lý thuyết quan trọng: **"ảnh sinh ra từ mô hình `M` được `M` (hoặc một phần
của `M`) tái tạo tốt hơn ảnh thật" là một quy luật tổng quát**, không phụ thuộc việc
dùng diffusion hay autoencoder, và không phụ thuộc độ đo là pixel hay LPIPS. DIRE
là một cách khai thác quy luật đó với độ phân giải cao nhất (giữ cấu trúc không
gian), AEROBLADE là cách rẻ nhất (một lượt, training-free) nhưng chỉ đúng với ảnh
LDM.

Điều này cũng giải thích vì sao Chương 6 khó: nếu quy luật là tổng quát, thì **cơ
chế** của nó cũng phải tổng quát — và lập luận "sai số Euler" của Phụ lục B chỉ nói
về diffusion, nên nó **không thể** là lời giải thích đầy đủ. Một lời giải thích đầy
đủ phải giải thích được cả trường hợp của AEROBLADE, nơi không có ODE nào cả.

3. **Miền lạ:** Có thể cải thiện trường hợp 76,9% bằng cách chuẩn hoá DIRE theo
   thống kê của miền (ví dụ trừ đi DIRE trung bình của một tập ảnh thật cùng miền)
   không? Đây là dạng "chuẩn hoá theo miền" quen thuộc trong xử lý ảnh, và chưa
   được thử trong bài báo.
