# DIRE từ số 0 — giáo trình cho người mới

> **Đối tượng:** có nền xử lý ảnh truyền thống và học máy cơ bản (biết tích chập,
> biết train một classifier đơn giản), **chưa** biết diffusion model.
>
> **Mục tiêu:** đọc xong hiểu được (a) vì sao bài toán phát hiện ảnh do AI sinh
> lại khó, (b) diffusion model hoạt động thế nào ở mức đủ để tự suy ra công thức,
> (c) DIRE là gì và vì sao nó đúng, (d) đọc được code của tác giả và của repo này
> mà không bị lạc.
>
> **Nguồn:** arXiv:2303.09295v1 (toàn văn + Phụ lục A–C) và repo
> `ZhendongWang6/DIRE`. Mọi chỗ tôi suy luận từ code chứ paper không nói, ghi
> **[suy ra từ code]**.
>
> **Quy ước thuật ngữ:** lần đầu xuất hiện, giữ nguyên tiếng Anh trong ngoặc —
> ví dụ *tích chập* (convolution). Về sau dùng thuật ngữ Việt, trừ khi tiếng Anh
> đã phổ biến hơn (logit, batch, embedding).

---

# Phần 0 — Bản đồ toàn cảnh trước khi đi vào chi tiết

Bạn đã biết pipeline kinh điển của thị giác máy tính:

```
ảnh → tiền xử lý → trích đặc trưng → bộ phân loại → nhãn + điểm tin cậy
```

DIRE **không** thay đổi pipeline này. Nó thay đổi **đầu vào**: thay vì đưa *ảnh*
vào bộ phân loại, nó đưa **ảnh sai số tái tạo** (reconstruction error) — tức là
một ảnh mới, được tạo ra bằng cách cho ảnh gốc đi qua một mô hình diffusion
pretrained rồi trừ ngược lại.

```
ảnh x ──► diffusion model (đảo ngược rồi tái tạo) ──► x̂
                                                      │
                                          DIRE = |x − x̂|
                                                      │
                                              ResNet-50 ──► P(fake)
```

Vậy có **hai** mô hình trong hệ thống, đừng nhầm:

| Thành phần | Là gì | Vai trò | Ai train |
|---|---|---|---|
| **ADM** (diffusion) | mô hình sinh ảnh 256×256 | *tạo ra* biểu diễn DIRE | tác giả ADM (OpenAI), **không** train lại |
| **ResNet-50** | CNN phân loại | quyết định thật/giả từ DIRE | tác giả DIRE train (400 epoch) |

Điểm cần khắc vào đầu: **ADM không được train cho bài toán phát hiện**. Nó chỉ là
một mô hình sinh ảnh bình thường, dùng như một *phép biến đổi đặc trưng* (feature
transform). Đây là lý do DIRE rẻ về mặt ý tưởng: không cần dữ liệu mới, không cần
train lại mô hình sinh.

---

# Phần 1 — Nền tảng xử lý ảnh số

## 1.1 Ảnh số là gì

Một ảnh xám là hàm `I: {0..H−1} × {0..W−1} → {0..255}`. Ảnh màu RGB là ba hàm
như vậy xếp chồng, tức tensor `H × W × 3`.

- **Mẫu hoá** (sampling): rời rạc hoá toạ độ không gian.
- **Lượng tử hoá** (quantization): rời rạc hoá cường độ, thường 8 bit = 256 mức.

Hai phép này **mất thông tin** và mất theo cách khác nhau ở từng thiết bị. Chính
những khác biệt nhỏ đó — dấu vết cảm biến, dấu vết nén — là thứ mà các detector
ảnh giả truyền thống khai thác.

Một điểm rất quan trọng cho toàn bộ tài liệu này: **ảnh do AI sinh không đi qua
cảm biến**. Nó không có nhiễu đọc (read noise), không có nhiễu photon, không có
demosaic, không có méo ống kính. Nó chỉ được *lượng tử hoá* một lần từ số thực.
Đó là khác biệt vật lý mà mọi phương pháp phát hiện đều cố bắt lấy.

## 1.2 Tích chập và lọc tuyến tính

Tích chập rời rạc 2D:

```
(I * K)[i, j] = Σ_m Σ_n I[i+m, j+n] · K[m, n]
```

`K` là **hạt nhân** (kernel / filter). Trực giác: trượt một ô cửa sổ nhỏ khắp ảnh,
mỗi vị trí tính tổng có trọng số của vùng lân cận.

Ba hạt nhân cần nhớ:

| Hạt nhân | Tác dụng | Liên hệ với DIRE |
|---|---|---|
| Trung bình (box) | làm mờ | phá vỡ dấu vết tần số cao ⇒ thử thách robustness |
| Gauss | làm mờ mềm | paper dùng `σ = 1, 2, 3` để kiểm robustness |
| Laplace / Sobel | làm nổi biên = thông cao (high-pass) | DIRE về bản chất cũng là "thông cao không gian" |

**Cửa sổ trượt** (sliding window) là khái niệm bạn sẽ gặp lại hai lần: một lần ở
tích chập, một lần ở cơ chế attention của diffusion (mỗi điểm ảnh "nhìn" các điểm
khác). Cùng một ý: *thông tin cục bộ được tổ hợp có trọng số*.

## 1.3 Miền tần số và vì sao nó quan trọng

Biến đổi Fourier rời rạc (DFT/FFT) đổi ảnh từ miền không gian sang miền tần số:

```
F(u, v) = Σ_x Σ_y I(x, y) · exp(−2πi(ux/H + vy/W))
```

- Tần số thấp = thay đổi chậm (mảng màu lớn, bố cục).
- Tần số cao = thay đổi nhanh (biên, nhiễu, chi tiết nhỏ).

**Vì sao dân phát hiện ảnh giả rất thích miền tần số:** các phép sinh ảnh đều có
bước "phóng to" (upsampling) hoặc "ghép khối", và các phép đó để lại **cấu trúc
tuần hoàn** trong phổ — những đỉnh năng lượng ở tần số cụ thể. Mắt người không
thấy, FFT thấy rõ.

Đó là cơ sở của CNNDet, F3Net, và các phương pháp dựa trên phổ. Nhược điểm: cấu
trúc tuần hoàn **phụ thuộc kiến trúc generator**. Đổi kiến trúc là mất dấu vết.
Đây chính là lý do DIRE phải tìm một tín hiệu khác.

## 1.4 Nén JPEG — ví dụ đầy đủ nhất về "dấu vết"

JPEG chia ảnh thành khối 8×8, biến đổi DCT từng khối, **lượng tử hoá** hệ số theo
bảng lượng tử, rồi mã hoá entropy.

- Lượng tử hoá làm mất thông tin **không đều**: tần số cao bị cắt mạnh hơn.
- Vì chia khối, ảnh JPEG có **tính chu kỳ 8 pixel** — một dấu vết rất mạnh trong
  phổ.
- Nén lại nhiều lần (re-compression) tích luỹ dấu vết; đây là lý do ảnh qua
  Facebook/Zalo nhiều lần rất khác ảnh gốc.

Liên hệ với paper: thí nghiệm robustness nén `quality = 65` và `30`. Kết quả
(Hình 4): DIRE **không giảm** ACC, trong khi `CNNDet*` và `Patchfor*` — vốn dựa
vào dấu vết RGB — **sụp mạnh** trước JPEG. Đây là bằng chứng định tính mạnh nhất
cho luận điểm "học trên RGB là học dấu vết, mà dấu vết thì mong manh".

## 1.5 Tái lấy mẫu (resampling)

Các phép nội suy khi đổi kích thước:

| Phương pháp | Công thức trực giác | Đặc điểm |
|---|---|---|
| Nearest | lấy điểm gần nhất | tạo răng cưa, giữ nguyên giá trị gốc |
| Bilinear | trung bình có trọng số 2×2 | mềm, hơi mờ |
| Bicubic | đa thức bậc 3 trên 4×4 | sắc hơn bilinear, mặc định của nhiều tool |
| Lanczos | cửa sổ sinc | sắc nhất, có thể ringing |

Mọi phép nội suy đều là **lọc thông thấp** (low-pass) ở một mức nào đó ⇒ **xoá
dấu vết tần số cao**. Vì thế:

- Nếu bạn resize ảnh trước khi kiểm tra, bạn có thể tự tay xoá đúng tín hiệu mà
  detector cần.
- Paper chuẩn hoá mọi thứ về 256×256 (SD 512² bị bicubic xuống 256) — nghĩa là
  **một phần tín hiệu đã mất trước khi vào detector**. Đây là điểm yếu tiềm ẩn
  của benchmark, không phải của DIRE.

## 1.6 Thang đo và độ đo chất lượng ảnh

Ba lưu ý thực hành:

- **PSNR** = `10·log10(MAX² / MSE)`, đơn vị dB. Nhược điểm: không khớp cảm nhận
  thị giác, rất "khoan dung" với ảnh mờ.
- **SSIM** so sánh độ tương phản/cấu trúc cục bộ, khớp cảm nhận hơn PSNR.
- **LPIPS** dùng đặc trưng mạng nơ-ron để đo khoảng cách tri giác.

Paper DIRE **không** dùng ba độ đo này. Nó dùng `|x − x̂|` **theo từng điểm ảnh**
— một đại lượng thô, không phải độ đo chất lượng. Điểm cần hiểu: DIRE không hỏi
"ảnh tái tạo đẹp không", mà hỏi "**sai số phân bố thế nào**". Một ảnh tái tạo
PSNR thấp vẫn có thể có cấu trúc sai số rất khác nhau.

---

# Phần 2 — Nền tảng học máy

## 2.1 Bài toán, hàm mất mát, tối ưu

Cho tập `D = {(x_i, y_i)}`, `y_i ∈ {0, 1}`. Ta tìm `f_θ` cực tiểu **rủi ro thực
nghiệm** (empirical risk):

```
min_θ  (1/N) Σ_i  ℓ( f_θ(x_i), y_i )
```

với `ℓ` là hàm mất mát. Ba bước lặp quen thuộc: lan truyền xuôi → tính loss →
lan truyền ngược (backpropagation) → cập nhật θ.

**Nhị phân, một logit.** Nếu `f_θ` cho ra một số thực bất kỳ `z` (gọi là **logit**),
thì:

```
P(y = 1 | x) = σ(z) = 1 / (1 + exp(−z))
ℓ = −[ y·log σ(z) + (1−y)·log(1−σ(z)) ]        (BCE)
```

Đây đúng là Eq. 10 của paper. PyTorch gộp hai bước (sigmoid + BCE) thành
`BCEWithLogitsLoss` để ổn định số học — nếu tách ra, `log(σ(z))` với `z` rất âm
sẽ tràn.

**Hai lớp, hai logit.** Nếu mô hình ra vector `z = (z_0, z_1)`:

```
P(y = k | x) = softmax(z)_k = exp(z_k) / Σ_j exp(z_j)
ℓ = −log softmax(z)_y                          (cross-entropy)
```

Hai quy ước **không hoán đổi được**. Checkpoint DIRE là loại một logit
(`fc.weight` shape `[1, 2048]`); nạp nó vào mô hình hai logit sẽ lỗi
`size mismatch`. Repo này đọc shape từ chính checkpoint để tránh lỗi đó
(`src/dire_core/detector.py::fc_out_features`).

Tại sao sigmoid tràn? `exp(−800) = 0` (dưới hạn số thực) hoặc `exp(800) = inf`.
Cài đặt đúng phải chia hai nhánh theo dấu `z`
(`src/dire_core/score.py::sigmoid`).

## 2.2 Tổng quát hoá, quá khớp, và dịch chuyển phân phối

- **Quá khớp (overfitting):** mô hình học cả nhiễu trong tập train, sai trên tập
  test. Dấu hiệu: loss train giảm, loss val tăng.
- **Tổng quát hoá (generalization):** hiệu năng trên dữ liệu *chưa thấy* nhưng
  **cùng phân phối**.
- **Dịch chuyển phân phối (distribution shift / domain shift):** dữ liệu test
  khác phân phối train.

Bài toán của DIRE là **dịch chuyển phân phối có chủ đích**: `p_g` khi train (ảnh
ADM) khác `p_g` khi test (ảnh SD, LDM, VQ-Diffusion…). Vì vậy mọi thứ trong paper
phải đọc qua lăng kính "cái gì *bất biến* giữa các generator".

Ba chiến lược chống dịch chuyển phân phối, để bạn thấy DIRE đứng ở đâu:

| Chiến lược | Cách làm | Vấn đề |
|---|---|---|
| Tăng dữ liệu train | train trên nhiều generator | mỗi generator mới lại cần dữ liệu mới |
| Học đặc trưng bất biến | học thứ không phụ thuộc kiến trúc | khó, phải tìm đúng tín hiệu |
| Chuẩn hoá theo phép biến đổi | biến ảnh thành biểu diễn "chuẩn" trước khi phân loại | **đây là DIRE** |

## 2.3 Chia tập và rò rỉ dữ liệu

Chuẩn: train / validation / test.

- **Train:** cập nhật tham số.
- **Validation:** chọn siêu tham số, dừng sớm, chọn ngưỡng.
- **Test:** chỉ chạm **một lần** để báo cáo.

Rò rỉ (leakage) xảy ra khi thông tin test lọt vào train hoặc vào quyết định. Ví dụ
nguy hiểm trong dự án này: ảnh `real` của nhánh `lsun_bedroom/lsun_bedroom/` và
nhánh `lsun_bedroom/imagenet/` là **cùng 1000 ảnh** (đã kiểm bằng so từng byte).
Cộng hai nhánh vào một bảng trung bình là **đếm trùng** — không có exception nào
báo, chỉ có bảng số trông hơi lạ.

## 2.4 Độ đo: ACC, AP, ROC, và vì sao cần cả ba

Cho điểm `s = P(fake)` và nhãn `y ∈ {0,1}`:

- **ACC** (accuracy) tại ngưỡng `t`: `(1/N) Σ 1[ (s_i > t) = (y_i = 1) ]`.
  Paper dùng `t = 0.5`. **Chú ý:** so sánh là `>`, không phải `≥`; và ngưỡng 0.5
  chỉ hợp lý khi hai lớp cân bằng (ở đây 40k/40k, và test 1k/1k).
- **TPR** (true positive rate / recall) = `TP / (TP + FN)`: tỉ lệ ảnh giả bắt được.
- **FPR** (false positive rate) = `FP / (FP + TN)`: tỉ lệ ảnh thật bị buộc tội.
- **ROC** là đường `TPR` theo `FPR` khi quét mọi ngưỡng. **AUC** là diện tích dưới
  đường đó.
- **AP** (average precision) là diện tích dưới đường precision–recall:

```
AP = Σ_k (R_k − R_{k−1}) · P_k
```

với `P_k`, `R_k` là precision/recall tại ngưỡng thứ `k`. Điểm dễ sai: sklearn gộp
các điểm có **cùng score** vào một ngưỡng. Nếu tự cài đặt mà không gộp, AP lệch
vài phần nghìn — `src/dire_modal/eval.py::_average_precision` cài đúng biến thể
này và có test đối chiếu.

**Vì sao paper báo cả ACC và AP:** ACC phụ thuộc ngưỡng (một lựa chọn), AP thì
không. Ở mức 99–100%, ACC gần như bão hoà, còn AP vẫn phân biệt được mô hình nào
xếp hạng tốt hơn. Bài toán thực tế lại càng cần AP, vì ngưỡng phải chọn theo
`FPR` chấp nhận được (ví dụ `TPR@FPR=1%`), không phải theo 0.5.

## 2.5 Mất cân bằng lớp

Nếu `n_fake = 0` (đường dẫn sai, nhãn sai), AP **không xác định** nhưng nhiều cài
đặt vẫn trả một con số trông hợp lý. Vì vậy mọi dòng kết quả trong repo này đều
ghi `n_real`/`n_fake`, và cặp lỗi thì để **trống** metric chứ không ghi 0
(`src/dire_modal/eval.py::row_for_result`). Đây là nguyên tắc: *thiếu số thì ghi
thiếu, không điền 0*.

---

# Phần 3 — Mạng tích chập và ResNet-50

## 3.1 Lớp tích chập

Một lớp conv có `C_out` hạt nhân kích thước `k×k×C_in`. Mỗi hạt nhân trượt khắp
ảnh tạo ra một **bản đồ đặc trưng** (feature map). Ba đặc tính quyết định:

1. **Chia sẻ trọng số** (weight sharing): cùng một hạt nhân dùng ở mọi vị trí ⇒
   bất biến tịnh tiến, và số tham số không phụ thuộc kích thước ảnh.
2. **Kết nối cục bộ** (local connectivity): mỗi nơ-ron chỉ nhìn `k×k` vùng.
3. **Trường tiếp nhận** (receptive field): vùng ảnh gốc mà một nơ-ron ở tầng `l`
   "thấy được". Xếp chồng conv 3×3: RF tăng tuyến tính theo số tầng.

Với 224×224 đầu vào, ResNet-50 có RF cuối cùng **lớn hơn ảnh** — tức tầng sâu
nhất nhìn được toàn ảnh. Đây là điều kiện để nó phát hiện cấu trúc toàn cục, không
chỉ dấu vết cục bộ.

## 3.2 Batch normalization và residual

**BatchNorm** chuẩn hoá đầu ra của lớp theo mean/var của cả batch:

```
x̂ = (x − μ_B) / √(σ²_B + ε),   y = γ·x̂ + β
```

Tác dụng: ổn định phân phối đầu vào các tầng sau, cho phép learning rate lớn hơn.
Hai lưu ý thực hành:

- Khi `model.eval()` (đúng phải gọi trước khi đánh giá), BatchNorm dùng running
  statistics, không dùng batch hiện tại. **Quên `.eval()` là lỗi im lặng kinh
  điển**: điểm sẽ thay đổi theo batch size.
- `γ` khởi tạo bằng 0 ở **khối residual cuối** (zero-init residual) làm khối bắt
  đầu như hàm đồng nhất — giúp train sâu dễ hơn. Trong code tác giả, phần này
  nằm sau cờ `zero_init_residual=False` nên **không chạy** `[suy ra từ code]`.

**Kết nối tắt (skip / residual connection):**

```
y = x + F(x)
```

Vì đạo hàm của `x` theo `x` là 1, gradient chảy thẳng qua mọi tầng ⇒ train được
50–152 tầng mà không suy giảm gradient.

## 3.3 Kiến trúc ResNet-50, đọc từ code

`src/dire_modal/vendor/networks/resnet.py`:

```python
class Bottleneck(nn.Module):        # khối "nút cổ chai"
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        self.conv1 = conv1x1(inplanes, planes)          # 1×1 giảm chiều
        self.conv2 = conv3x3(planes, planes, stride)    # 3×3, có thể /2
        self.conv3 = conv1x1(planes, planes * 4)        # 1×1 tăng lại ×4
```

Ba lớp conv thay vì hai: `1×1 → 3×3 → 1×1` giảm số tham số so với hai lớp 3×3
cùng số kênh (nên gọi là "nút cổ chai"), nhưng vẫn giữ trường tiếp nhận 3×3.

```python
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64,  layers[0])          # 3 khối
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # 4 khối
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # 6 khối
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) # 3 khối
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)         # 2048 → C
```

Số khối `[3, 4, 6, 3]` là lý do tên "50": `3+4+6+3 = 16` khối, mỗi khối 3 lớp conv
⇒ `16×3 + 1 (conv1) + 1 (fc) = 50` tầng có tham số.

`forward` đọc thẳng:

```python
def forward(self, x, *args):
    x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
    x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)      # (N, 2048, 1, 1) → (N, 2048)
    return self.fc(x)              # (N, num_classes)
```

`*args` là để nhận thêm `meta` (nhãn phụ) mà không dùng — dấu vết của việc code
được viết cho khung tổng quát hơn.

**Kích thước qua từng tầng với đầu vào 224×224:**

| Tầng | Đầu ra | Ghi chú |
|---|---|---|
| conv1 7×7, stride 2 | 112×112×64 | giảm một nửa |
| maxpool 3×3, stride 2 | 56×56×64 | giảm một nửa |
| layer1 (64 kênh) | 56×56×256 | |
| layer2 (128) | 28×28×512 | |
| layer3 (256) | 14×14×1024 | |
| layer4 (512) | 7×7×2048 | |
| avgpool → fc | 2048 → 1 | một logit |

## 3.4 Transfer learning

Với 80k ảnh (40k thật + 40k giả), học 25,6 triệu tham số từ đầu là quá ít. Cách
làm của tác giả (`vendor/utils/utils.py::get_network`):

```python
model = resnet50(pretrained=True)          # nạp trọng số ImageNet (1000 lớp)
model.fc = nn.Linear(2048, 1)              # thay đầu ra
nn.init.normal_(model.fc.weight, 0.0, 0.02) # khởi tạo nhỏ cho lớp mới
```

Trực giác: các tầng đầu học đặc trưng thị giác phổ quát (biên, texture, hình
khối). Đặc trưng đó dùng lại được cho DIRE map — vốn cũng là "ảnh" theo nghĩa
ma trận 3 kênh. Chỉ lớp cuối cần học khái niệm thật/giả.

Một chi tiết đáng chú ý: **nhánh eval** gọi `resnet(num_classes=1)` **không**
pretrained, vì ngay sau đó nạp checkpoint đè lên. Nếu nạp pretrained rồi mới nạp
checkpoint thì tốn thời gian vô ích (và có thể sai nếu checkpoint có `fc` khác
shape — nhưng ở đây cùng 1 nên vô hại).

---

# Phần 4 — Sinh ảnh: GAN, VAE, và diffusion

Mục này chỉ đủ để bạn hiểu **vì sao diffusion khác về bản chất**, không đi sâu
kiến trúc.

## 4.1 Ba họ mô hình sinh

| Họ | Ý tưởng | Dấu vết để lại |
|---|---|---|
| **GAN** | generator `G` và discriminator `D` đối kháng; `G` biến nhiễu `z` thành ảnh | phép **upsampling** (transposed conv / pixel shuffle) tạo cấu trúc tuần hoàn trong phổ |
| **VAE** | encoder nén vào không gian ẩn (latent), decoder dựng lại; tối ưu ELBO | ảnh thường **mờ**, mất chi tiết tần số cao |
| **Diffusion** | học cách **khử nhiễu** lặp lại nhiều bước | không có phép upsampling một phát; dấu vết nằm ở **quỹ đạo khử nhiễu** |

Điểm mấu chốt: GAN và VAE tạo ảnh trong **một** phép biến đổi (một forward pass
của generator). Diffusion tạo ảnh bằng **hàng trăm phép biến đổi nhỏ**. Vì thế:

- Detector dựa trên artifact của GAN **không chuyển được** sang diffusion: không
  có phép upsampling tương ứng để bắt.
- Nhưng chính tính lặp lại đó lại mở ra một tín hiệu mới: **ảnh do diffusion sinh
  nằm "gần" quỹ đạo mà diffusion model đã học**, nên khi cho nó đi qua một
  diffusion model khác (cùng họ), nó được tái tạo tốt hơn ảnh thật.

Đây là bước ngoặt ý tưởng của DIRE.

## 4.2 Vì sao ảnh sinh "dễ tái tạo" — trực giác trước, công thức sau

Tưởng tượng bạn có một chiếc máy **nén và giải nén ảnh theo một ngôn ngữ cụ thể**
(ngôn ngữ của diffusion model). Ảnh mà máy đó *từng dùng để học* thì nén–giải nén
gần như hoàn hảo. Ảnh lạ thì mất mát nhiều.

Nhưng DIRE không đơn giản là "gần phân phối train". Có hai tầng giải thích trong
paper:

1. **Tầng phân phối:** ảnh sinh ra từ quá trình diffusion nằm trên đa tạp
   (manifold) mà quá trình đó định nghĩa ⇒ tái tạo tốt.
2. **Tầng rời rạc hoá (Phụ lục B):** vì `T = 1000` là hữu hạn, công thức DDIM chỉ
   là **xấp xỉ Euler** cho một ODE. Sai số xấp xỉ **lớn hơn ở ảnh thật** vì quỹ
   đạo của ảnh thật "cong" hơn.

Tầng 2 giải thích được điều tầng 1 không giải thích: vì sao **ảnh thật LSUN** —
vốn cùng domain với ADM — vẫn khó tái tạo. Nó cũng là chỗ paper còn yếu: đây là
lập luận định tính, không có chứng minh hay thí nghiệm đo trực tiếp.

---

# Phần 5 — Diffusion model, từng bước

Đây là phần khó nhất. Tôi đi chậm, mỗi công thức kèm một câu "nghĩa là gì".

## 5.1 Quá trình thuận: thêm nhiễu dần

Định nghĩa một dãy `β_1 … β_T` (ví dụ tuyến tính từ `1e-4` đến `0.02`), đặt
`α_t = 1 − β_t`, `ᾱ_t = α_1·α_2···α_t`.

Một bước thêm nhiễu:

```
q(x_t | x_{t−1}) = N( x_t ; √α_t · x_{t−1} , (1 − α_t)·I )      (Eq. 1)
```

Nghĩa là: giữ ảnh cũ, co lại một chút (`√α_t`), rồi cộng nhiễu Gauss có phương
sai `1−α_t`.

Vì là Markov và Gauss, có thể **nhảy cóc** tới bước bất kỳ:

```
q(x_t | x_0) = N( x_t ; √ᾱ_t · x_0 , (1 − ᾱ_t)·I )              (Eq. 2)
```

Nghĩa là: `x_t = √ᾱ_t·x_0 + √(1−ᾱ_t)·ε` với `ε ~ N(0, I)`.

Ba hệ quả cần nhớ:

1. `ᾱ_t` **giảm** theo `t` ⇒ `x_0` mất dần, nhiễu chiếm dần. Khi `t = T`,
   `ᾱ_T ≈ 0` ⇒ `x_T ≈ N(0, I)` — nhiễu thuần.
2. Công thức nghịch đảo: `x_0 = (x_t − √(1−ᾱ_t)·ε) / √ᾱ_t`. Đây là nền của mọi
   thứ về sau.
3. Vì `x_t` được xác định bởi `(x_0, ε)`, nếu ta biết `ε` thì biết `x_0`. Bài
   toán học trở thành **dự đoán nhiễu**.

## 5.2 Huấn luyện: dự đoán nhiễu

Mạng `ε_θ(x_t, t)` nhận ảnh nhiễu và bước thời gian, trả về nhiễu đã được thêm.
Mất mát (Eq. 4):

```
L = E_{t, x_0, ε} ‖ ε − ε_θ( √ᾱ_t·x_0 + √(1−ᾱ_t)·ε , t ) ‖²
```

Nghĩa là: lấy ảnh thật, thêm nhiễu theo công thức, bắt mạng đoán lại đúng lượng
nhiễu đó.

Vì sao đây là mất mát hợp lý: nó là bản rút gọn (bỏ trọng số) của **ELBO** — cận
dưới biến phân của log-likelihood. Việc bỏ trọng số làm mất tính chất "đúng về
xác suất" nhưng cho kết quả ảnh tốt hơn trong thực tế. Paper ADM và DDPM đều
dùng dạng rút gọn này.

**Kiến trúc ADM** (mà DIRE dùng) là U-Net có attention:

- Encoder giảm dần độ phân giải, decoder tăng lại, với **kết nối tắt** giữa hai
  phía (skip giữa encoder–decoder, khác skip residual trong khối).
- **Attention** ở các độ phân giải 32, 16, 8: mỗi điểm ảnh tổng hợp thông tin từ
  mọi điểm khác — mô hình hoá quan hệ toàn cục.
- `learn_sigma=True`: mạng ra 6 kênh (3 cho mean, 3 cho variance) thay vì 3.
- Classifier-free guidance: không dùng trong cấu hình DIRE (`class_cond=False`).

## 5.3 Sinh ảnh: DDPM có nhiễu, DDIM không nhiễu

**DDPM** lấy mẫu bằng cách lặp:

```
x_{t−1} ~ N( μ_θ(x_t, t), Σ_θ(x_t, t) )
```

Có nhiễu ngẫu nhiên ⇒ mỗi lần chạy ra một ảnh khác.

**DDIM** đặt `σ_t = 0`:

```
x_{t−1} = √ᾱ_{t−1} · x̂_0 + √(1 − ᾱ_{t−1}) · ε_θ(x_t, t)        (Eq. 5)
```

với `x̂_0` là dự đoán ảnh sạch từ `x_t` và `ε_θ`. Không còn nhiễu ⇒ **tất định**.

Tất định là tính chất then chốt của DIRE. Nếu quá trình sinh còn ngẫu nhiên, thì
`R(I(x))` sẽ khác nhau mỗi lần chạy, và DIRE trở thành nhiễu.

## 5.4 Từ rời rạc đến ODE, và vì sao "inversion" khả thi

Khi số bước `T` rất lớn, một bước DDIM xấp xỉ nghiệm của một ODE:

```
d x̄(t) = ε_θ( x̄(t) / √(σ²+1), t ) dσ(t)                        (Eq. 7)
```

Với `x̄ = x/√ᾱ`, `σ = √(1−ᾱ)/√ᾱ`.

Hai ý cần nắm:

1. **ODE có tính thuận nghịch**: nếu biết nghiệm tại một thời điểm và trường vector,
   có thể đi theo cả hai chiều. Vì thế có thể **đảo ngược** quá trình sinh để tìm
   `x_T` từ `x_0` (Eq. 8):

```
x_{t+1}/√ᾱ_{t+1} = x_t/√ᾱ_t + ( √((1−ᾱ_{t+1})/ᾱ_{t+1}) − √((1−ᾱ_t)/ᾱ_t) ) · ε_θ(x_t, t)
```

2. **Rời rạc hoá tạo sai số**: ta chỉ chạy 20 mốc thay vì 1000. Sai số phụ thuộc
   độ cong của quỹ đạo. Đây chính là cơ chế mà Phụ lục B viện dẫn để giải thích
   vì sao DIRE của ảnh sinh không bằng 0.

**Một chi tiết kỹ thuật quan trọng, dễ bỏ qua:**
trong code tác giả, `clip_denoised=True` nghĩa là `x̂_0` bị **cắt vào [−1, 1]** ở
mỗi bước. Phép cắt là **phi tuyến**, nên nó phá tính thuận nghịch chính xác của
ODE. Vì vậy `R(I(x))` không bao giờ bằng `x` tuyệt đối, kể cả với ảnh sinh. Điều
này không sai — nó là một phần của tín hiệu — nhưng cần biết để không kỳ vọng
DIRE = 0.

## 5.5 Rút gọn số bước: respacing

Chạy 1000 bước cho mỗi ảnh là quá đắt. DDIM cho phép chọn một **tập con** các mốc
`τ_1 < … < τ_S` trong `{0..T−1}`. Trong guided-diffusion, đây là `SpacedDiffusion`
với `space_timesteps(1000, "20")` ⇒ stride 50 ⇒ các mốc
`{0, 50, 100, …, 950}`, tức **đúng 20 bước**.

Với S = 20: mỗi ảnh cần **20 lượt forward cho inversion + 20 cho reconstruction =
40 lượt** (NFE — number of function evaluations). Đây là chi phí thật của DIRE,
và nó giải thích vì sao paper phải cân nhắc S (Table 4) chứ không chọn 1000.

## 5.6 Bảng công thức ↔ code (guided-diffusion)

| Toán | Hàm |
|---|---|
| Eq. 2 `q(x_t|x_0)` | `GaussianDiffusion.q_sample`: `√ᾱ_t·x + √(1−ᾱ_t)·noise` |
| Eq. 5 DDIM sampling | `ddim_sample`: `mean = √ᾱ_prev·x̂_0 + √(1−ᾱ_prev−σ²)·ε` |
| Eq. 8 DDIM inversion | `ddim_reverse_sample`: `mean = √ᾱ_next·x̂_0 + √(1−ᾱ_next)·ε` |
| Tiền xử lý `[-1,1]` | `image_datasets.ImageDataset_for_reverse.__getitem__` |
| Eq. 9 DIRE | `compute_dire.main`: `th.abs(imgs − recons)` |

Trong code, `ε` được suy ra từ `x̂_0` bằng:

```python
eps = (sqrt_recip_alphas_cumprod[t] * x - pred_xstart) / sqrt_recipm1_alphas_cumprod[t]
```

tức `ε = (x/√ᾱ_t − x̂_0) / √(1/ᾱ_t − 1)`, hoàn toàn tương đương
`ε = (x − √ᾱ_t·x̂_0)/√(1−ᾱ_t)`. Chỉ là dạng viết khác của Eq. 2 đảo lại.

---

# Phần 6 — DIRE: định nghĩa, trực giác, và các lựa chọn thiết kế

## 6.1 Công thức

```
DIRE(x_0) = | x_0 − R( I(x_0) ) |                                   (Eq. 9)
```

- `I`: DDIM inversion, S bước, `t` tăng.
- `R`: DDIM sampling, S bước, `t` giảm.
- `|·|`: trị tuyệt đối từng điểm ảnh, **giữ 3 kênh** (không chuyển xám).

Kết quả là một ảnh 256×256×3 cùng kích thước đầu vào.

**Vì sao giữ 3 kênh?** Vì sai số tái tạo không giống nhau ở ba kênh: diffusion
xử lý ba kênh như ba kênh độc lập, và mức độ khó tái tạo khác nhau theo kênh.
Chuyển xám sẽ trộn thông tin đó. Paper không thảo luận điểm này, nhưng code giữ
3 kênh và kết quả tốt, nên đây là lựa chọn ngầm định `[suy ra từ code]`.

## 6.2 Vì sao phải lấy trị tuyệt đối

Xét residual `r = x_0 − x̂_0`. Trước khi lấy abs, `r` có **dấu**: chỗ nào tái tạo
sáng hơn ảnh gốc thì dương, tối hơn thì âm.

Vấn đề: dấu phụ thuộc chiều lệch, mà chiều lệch **không ổn định** giữa các ảnh.
Hai ảnh cùng "khó tái tạo" có thể có residual ngược dấu và triệt tiêu nhau nếu
bạn lấy trung bình, hoặc tạo ra hai cụm khác nhau trong không gian đặc trưng.

`|r|` chỉ giữ **độ lớn** — đại lượng có tính hệ thống hơn: ảnh thật ⇒ khó tái tạo
⇒ `|r|` lớn; ảnh sinh ⇒ dễ ⇒ `|r|` nhỏ.

Đo được (Table 6):

| | ADM | DDPM | iDDPM | PNDM | SD-v1 |
|---|---|---|---|---|---|
| không abs | 100/100 | 99,4/100 | 100/100 | 98,2/100 | **87,0/93,0** |
| có abs | 100/100 | 100/100 | 100/100 | 99,7/100 | **99,7/100** |

Chênh lệch 12,7 điểm ACC ở SD-v1. Đây là lý do `|·|` được coi là "critical" trong
paper, không phải chi tiết phụ.

## 6.3 Vì sao 256×256 và vì sao crop 224

- ADM dùng trong paper là **256×256_diffusion_uncond**. Đưa ảnh khác kích thước
  vào sẽ sai kiến trúc.
- Ảnh SD 512² bị **bicubic xuống 256**. Đây là mất mát thông tin có thật; nó
  cũng là lý do mọi thứ trong paper ở 256.
- Crop 224 cho ResNet-50 vì đó là kích thước chuẩn ImageNet và tiết kiệm tính
  toán. Train dùng crop ngẫu nhiên (augmentation), test dùng crop giữa.

## 6.4 Vì sao ResNet-50 chứ không phải mạng lớn hơn

Paper không giải thích. Đọc code thì thấy `arch = "resnet50"` là mặc định, và
CNNDetection (mà DIRE thừa hưởng khung) cũng dùng ResNet-50. Đây là lựa chọn
"đủ tốt, đủ rẻ" chứ không phải kết quả so sánh kiến trúc. Nếu bạn muốn nghiên cứu
tiếp, đây là biến chưa được khảo sát: **kiến trúc classifier có quan trọng không
khi đầu vào là DIRE?**

## 6.5 Số bước S

Table 4 (train trên tập ADM, test 5 generator):

| S | ADM | DDPM | iDDPM | PNDM | SD-v1 |
|---|---|---|---|---|---|
| 5 | 100/100 | 100/100 | 100/100 | 97,5/100 | 87,5/99,8 |
| 10 | 100/100 | 100/100 | 100/100 | 99,4/100 | 98,2/100 |
| **20** | 100/100 | 100/100 | 100/100 | 99,7/100 | **99,7/100** |
| 50 | 100/100 | 100/100 | 100/100 | 100/100 | 99,9/100 |

Đọc bảng này đúng cách:

- S càng lớn, xấp xỉ Euler càng chính xác, DIRE càng "sạch" ⇒ hiệu năng tăng.
- Nhưng chi phí tăng **tuyến tính** theo S (và gấp đôi vì có cả inversion).
- Điểm "khuỷu" là 20: từ 20 lên 50 chỉ thêm 0,2 điểm ở SD-v1 mà tốn 2,5× chi phí.

Đây là ví dụ mẫu của đánh đổi **độ chính xác ↔ chi phí tính toán**, và cách paper
chọn điểm khuỷu thay vì chọn cực trị.

## 6.6 Vì sao DIRE tốt hơn RGB, REC, và RGB&DIRE

Table 5 (train trên ADM subset):

| Đầu vào | DDPM | iDDPM | PNDM | SD-v1 |
|---|---|---|---|---|
| REC (ảnh tái tạo) | 57,1/57,7 | 49,7/92,6 | 87,1/98,7 | 46,9/57,0 |
| RGB (ảnh gốc) | 87,3/99,6 | 100/100 | 77,8/99,1 | 77,4/85,8 |
| RGB & DIRE | 99,8/100 | 99,9/100 | 99,2/100 | **62,4/92,4** |
| **DIRE** | **100/100** | **100/100** | **99,7/100** | **99,7/100** |

Ba kết luận:

1. **REC tệ nhất.** Ảnh tái tạo đã bị "kéo" về phía phân phối model, nên mất
   chính thông tin phân biệt. Nghĩa là thông tin nằm ở **hiệu**, không ở **ảnh
   tái tạo**.
2. **RGB khá nhưng không bền.** Nó học dấu vết; dấu vết đó chuyển được giữa các
   generator *cùng họ* nhưng vỡ khi sang SD-v1.
3. **RGB & DIRE tệ hơn DIRE.** Ghép thêm RGB làm mô hình có thể "lười": nó dùng
   nhánh RGB (dễ học hơn trong tập train) và bỏ qua nhánh DIRE, nên mất tính bất
   biến. Đây là hiện tượng **shortcut learning** — mô hình chọn tín hiệu dễ học
   thay vì tín hiệu đúng.

## 6.7 Chi phí

Một ảnh = 2S lượt forward ADM (S=20 ⇒ 40) + 1 lượt ResNet-50.

Với 42k ảnh × 8 model ≈ 336k ảnh, mỗi ảnh 40 NFE ⇒ **13,4 triệu NFE**. Đây là
hoạt động tốn kém nhất của paper, và là lý do dataset phát hành sẵn DIRE: người
tái lập không phải trả chi phí đó.

Trong repo này, `src/dire_modal/compute_dire.py::estimate_cost` tính đúng đại
lượng đó:

```python
gpu_hours = n_images * sec_per_image / 3600
usd       = gpu_hours * usd_per_gpu_hour
worst_usd = containers * timeout_s / 3600 * usd_per_gpu_hour   # mọi container treo
```

Điểm thiết kế đáng học: guard chặn **cả** chi phí kỳ vọng **và** chi phí xấu
nhất. Cái sau mới là hoá đơn khi chạy hỏng.

---

# Phần 7 — Dataset và thực nghiệm của paper

## 7.1 Thành phần (Table 1)

| Nguồn | Generator | Điều kiện | Số ảnh (real/gen) | Chia |
|---|---|---|---|---|
| LSUN-Bedroom | ADM | unconditional | 42k/42k | 40k/1k/1k |
| | DDPM | unconditional | — /1k | **chỉ test** |
| | iDDPM | unconditional | 42k/42k | 40k/1k/1k |
| | PNDM | unconditional | 42k/42k | 40k/1k/1k |
| | LDM | text2image | 1k/1k | chỉ test |
| | SD-v1 | text2image | 1k/1k | chỉ test |
| | SD-v2 | text2image | 1k/1k | chỉ test |
| | VQ-Diffusion | text2image | 1k/1k | chỉ test |
| ImageNet | ADM | conditional | 50k/50k | 40k/5k/5k |
| | SD-v1 | text2image | 10k/10k | chỉ test |

Hai điều đáng chú ý:

- **Chỉ 4 generator có train split** (ADM, iDDPM, PNDM ở LSUN; ADM ở ImageNet).
  7 generator còn lại **chỉ test** — đó là thiết kế để đo tổng quát hoá.
- Ảnh `real` được chia theo **đúng split của ảnh giả tương ứng**. Nếu không, sẽ
  có rò rỉ: ảnh real trong train trùng ảnh real trong test.

Mọi ảnh phát hành dưới dạng **bộ ba (triplet)**: ảnh nguồn, ảnh tái tạo, ảnh DIRE.
Đây là quyết định rất tốt cho cộng đồng: người sau có thể thử biểu diễn mới mà
không phải chạy ADM 40 NFE/ảnh.

## 7.2 Prompt và ý nghĩa của nó

- LSUN text2image: `"A photo of bedroom"` — cố định, để so sánh công bằng.
- ImageNet SD-v1: `"A photo of {class}"` với 1000 lớp.

Vì prompt cố định, "tổng quát hoá" trong Table 2 **không** bao gồm biến thiên
prompt. Đây là hạn chế cần biết: ảnh SD thực tế có prompt đa dạng, và phân phối
đó rộng hơn nhiều.

## 7.3 Kết quả chính (Table 2, khối DIRE)

| Train (gen) | Recon | ADM | DDPM | iDDPM | PNDM | SD-v1 | SD-v2 | LDM | VQ | Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| ADM | ADM | 100 | 100 | 100 | 99,7 | 99,7 | 100 | 100 | 100 | **99,9** |
| PNDM | ADM | 100 | 100 | 100 | 100 | 89,4 | 100 | 100 | 100 | 98,7 |
| iDDPM | ADM | 99,6 | 100 | 100 | 89,7 | 99,7 | 100 | 99,9 | 99,9 | 98,6 |
| StyleGAN | ADM | 98,8 | 99,8 | 99,9 | 89,6 | 95,2 | 100 | 100 | 100 | 97,9 |

Đọc bảng này đúng cách:

- Đường chéo (train generator = test generator) gần 100 — điều kiện tiên quyết.
- Các ô lệch (89,4 / 89,7 / 89,6) đều nằm ở cột PNDM và SD-v1 — nghĩa là hai
  generator này "khó" nhất.
- Average 97,9–99,9 là rất cao, nhưng test chỉ 1k ảnh/model: **khoảng tin cậy
  xấp xỉ ±0,3–1 điểm** ở mức 99% (với `n = 1000`, sai số chuẩn của một tỉ lệ
  `p ≈ 0,99` là `√(0,99·0,01/1000) ≈ 0,3%`). Vì vậy chênh lệch vài phần mười giữa
  các phương pháp **không có ý nghĩa thống kê**. Paper không báo CI.

## 7.4 Tổng quát hoá chéo (Table 3)

| Train trên | ADM(ImageNet) | SD-v1(ImageNet) | StyleGAN(LSUN) |
|---|---|---|---|
| LSUN ADM | 90,2/97,9 | 97,2/99,8 | 99,9/100 |
| LSUN iDDPM | 90,2/97,9 | 93,7/99,3 | 99,9/100 |
| LSUN StyleGAN | **76,9**/94,4 | 89,7/99,0 | 100/100 |

Bài học: đổi **domain** (LSUN → ImageNet) khó hơn nhiều đổi **generator** trong
cùng domain. 76,9 ACC ở cấu hình StyleGAN→ADM(IN) là mức tụt lớn nhất trong toàn
bộ paper, và paper không phân tích nó.

## 7.5 Robustness (Hình 4)

- Gaussian blur `σ = 1, 2, 3`; JPEG `quality = 65, 30`.
- DIRE "không giảm" — nhưng đây là đọc từ **biểu đồ**, paper không in bảng số.
- Các baseline dựa trên RGB (`CNNDet*`, `Patchfor*`) sụp mạnh trước JPEG.

Về mặt lý thuyết điều này hợp lý: DIRE là hiệu của hai ảnh đã qua cùng một phép
làm mờ/nén, nên thành phần bị phá huỷ **triệt tiêu một phần**. Nhưng cần kiểm
bằng số, không chỉ bằng hình.

---

# Phần 8 — Đọc code: từng hàm, từng dòng

## 8.1 `utils/datasets.py::binary_dataset` — trật tự transform là hợp đồng

```python
def binary_dataset(root, cfg):
    identity_transform = transforms.Lambda(lambda img: img)

    # (1) resize: train hoặc khi aug_resize=True
    if cfg.isTrain or cfg.aug_resize:
        rz_func = transforms.Lambda(lambda img: custom_resize(img, cfg))
    else:
        rz_func = identity_transform

    # (2) crop: train thì ngẫu nhiên, test thì giữa
    if cfg.isTrain:
        crop_func = transforms.RandomCrop(cfg.cropSize)
    else:
        crop_func = transforms.CenterCrop(cfg.cropSize) if cfg.aug_crop else identity_transform

    # (3) lật ngang: chỉ train
    if cfg.isTrain and cfg.aug_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = identity_transform

    return datasets.ImageFolder(root, transforms.Compose([
        rz_func,
        transforms.Lambda(lambda img: blur_jpg_augment(img, cfg)),   # (2')
        crop_func,
        flip_func,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            if cfg.aug_norm else identity_transform,
    ]))
```

Giải thích từng mắt xích:

- **`ImageFolder`** suy nhãn từ **tên thư mục con** theo alphabet: `0_real → 0`,
  `1_fake → 1`. Không có file metadata. Đổi tên thư mục = đổi nhãn, và không có
  lỗi nào báo.
- **`custom_resize`** dùng `TF.resize(img, 256)`: với PIL, tham số nguyên nghĩa
  là **cạnh ngắn = 256**, giữ tỉ lệ. Ảnh 4:3 thành 256×341 chẳng hạn.
- **`blur_jpg_augment` nằm trước crop**, tức nhiễu được thêm ở ảnh lớn. Đây là
  chi tiết quan trọng nếu bật `jpg_prob > 0`: vị trí đặt khác nhau cho phân phối
  nhiễu khác nhau. Mặc định cả hai xác suất = 0 nên không ảnh hưởng paper.
- **`RandomCrop(224)` từ ảnh 256**: tương đương crop 224 rồi có 32 pixel dư để
  dịch. Đây là augmentation mạnh hơn "resize thẳng 224" vì nó tạo dịch chuyển.
- **`Normalize` là bước cuối và bắt buộc.** Thiếu nó, đầu vào ResNet lệch phân
  phối so với lúc train ImageNet ⇒ điểm tụt mà không có lỗi nào.
- **`ImageFile.LOAD_TRUNCATED_IMAGES = True`** ở đầu module: ảnh JPEG tải dở vẫn
  đọc được thay vì ném `OSError`. Cần cho dữ liệu tải từ Baidu — nhưng cũng có
  nghĩa là ảnh hỏng **âm thầm** lọt vào train. Đánh đổi cần biết.

## 8.2 `train.py` — vòng lặp đầy đủ

```python
val_cfg = get_val_cfg(cfg, split="val", copy=True)   # (1) chụp cấu hình val TRƯỚC
cfg.dataset_root = os.path.join(cfg.dataset_root, "train")
data_loader = create_dataloader(cfg)
```

- **(1)** `deepcopy` rồi mới đổi `dataset_root` cho train. Nếu đổi trước, `val_cfg`
  cũng trỏ vào `train/` ⇒ **đánh giá trên tập train**, điểm cao giả tạo. Đây là
  lỗi kinh điển và code này tránh được đúng.

```python
trainer = Trainer(cfg)
early_stopping = EarlyStopping(patience=cfg.earlystop_epoch, delta=-0.001, verbose=True)

for epoch in range(cfg.nepoch):
    for data in tqdm(data_loader):
        trainer.total_steps += 1
        trainer.set_input(data)           # đưa lên GPU, label.float()
        trainer.optimize_parameters()      # forward → loss → zero_grad → backward → step
        train_writer.add_scalar("loss", trainer.loss, trainer.total_steps)

        if trainer.total_steps % cfg.save_latest_freq == 0:
            trainer.save_networks("latest")
```

`set_input`:

```python
def set_input(self, input):
    img, label, meta = input if len(input) == 3 else (input[0], input[1], {})
    self.input = img.to(self.device)
    self.label = label.to(self.device).float()   # BCE cần float
```

`optimize_parameters`:

```python
def optimize_parameters(self):
    self.forward()                                        # output = model(input, meta)
    self.loss = self.loss_fn(self.output.squeeze(1), self.label)
    self.optimizer.zero_grad()                            # xoá gradient cũ
    self.loss.backward()                                  # lan truyền ngược
    self.optimizer.step()                                 # cập nhật θ
```

`squeeze(1)` biến `(N, 1)` thành `(N,)` để khớp shape của `label` — nếu quên,
BCE sẽ broadcast và tính sai **mà không báo lỗi**.

Validation và early stopping:

```python
    trainer.eval()                                    # tắt dropout/BN dùng batch stats
    val_results = validate(trainer.model, val_cfg)    # ACC, AP trên val
    if cfg.earlystop:
        early_stopping(val_results["ACC"], trainer)
        if early_stopping.early_stop:
            if trainer.adjust_learning_rate():        # chia LR cho 10
                early_stopping = EarlyStopping(patience=..., delta=-0.002)
            else:
                break                                 # LR < 1e-6 ⇒ dừng hẳn
    trainer.train()                                   # quay lại train mode
```

`early_stopping` nhận **ACC**, không phải loss. Đây là lựa chọn có ý thức: ACC là
metric cuối cùng, nhưng nó **rời rạc** (nhảy từng bước 1/1000) nên tín hiệu dừng
nhiễu hơn loss.

## 8.3 `utils/earlystop.py` — dấu của `delta`

```python
def __call__(self, score, trainer):
    if self.best_score is None:
        self.best_score = score; self.save_checkpoint(score, trainer)
    elif score < self.best_score - self.delta:     # delta = -0.001
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
    else:
        self.best_score = score; self.save_checkpoint(score, trainer)
        self.counter = 0
```

Vì `delta` **âm**, điều kiện là `score < best + 0.001`. Nghĩa là **một lần tăng
0,0005 cũng bị coi là không cải thiện**. Thiết kế này chống dừng sớm do dao động
nhỏ; nhưng nếu bạn tái lập với `delta = 0`, số epoch dừng sẽ khác và checkpoint
khác ⇒ đừng ngạc nhiên khi điểm lệch.

Ngoài ra `save_checkpoint` lưu `model_epoch_best.pth` **mỗi lần** tốt hơn ⇒ I/O
nhiều. Với ResNet-50 (269 MB/checkpoint) điều này không nhỏ.

## 8.4 `utils/eval.py::validate` — metric

```python
predict = model(in_tens, meta).sigmoid()          # 1 logit → P(fake)
y_pred.extend(predict.flatten().tolist())

r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
acc   = accuracy_score(y_true, y_pred > 0.5)
ap    = average_precision_score(y_true, y_pred)
```

- `sigmoid()` không tham số: nhận tensor, trả tensor cùng shape.
- `> 0.5` **strict**: `P = 0.5` tính là real.
- `r_acc` / `f_acc` tách riêng để phát hiện mô hình "đoán bừa một lớp" — nếu
  `r_acc = 1.0` và `f_acc = 0.0`, ACC tổng có thể vẫn 0.5 nhưng mô hình vô dụng.

## 8.5 `test.py` — một vòng lặp cho mọi dataset con

```python
for i, dataset in enumerate(cfg.datasets_test):
    cfg.dataset_root = os.path.join(dataset_root, dataset)
    cfg.datasets = [""]
    model = get_network(cfg.arch)                    # dựng lại mỗi lần
    model.load_state_dict(torch.load(cfg.ckpt_path)["model"])
    test_results = validate(model, cfg)
    rows.append([dataset] + list(test_results.values()))
```

- Dựng lại model mỗi dataset là **lãng phí** nhưng vô hại (chỉ tốn thời gian nạp).
- `["model"]` là bóc lớp bọc — khớp với `save_networks` lưu
  `{"model":…, "optimizer":…, "total_steps":…}`.
- Ghi CSV một dòng/dataset. Không có ngưỡng chọn trên test — đúng nguyên tắc.

## 8.6 `demo.py` — suy luận một ảnh

```python
trans = transforms.Compose((
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
))
for img_path in file_list:
    img = Image.open(img_path).convert("RGB")
    img = trans(img)
    if args.aug_norm:
        img = TF.normalize(img, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    prob = model(in_tens).sigmoid().item()
    print(f"Prob of being synthetic: {prob:.4f}")
```

**Chú ý:** `aug_norm` mặc định `True`, nên demo **có** normalize — chỉ là viết
thành bước riêng sau `ToTensor` thay vì đặt trong `Compose`. Về mặt toán học
tương đương. Khác biệt thật giữa demo và eval chỉ là **không có augmentation**
(không random crop/flip) — hoàn toàn đúng cho suy luận.

> Trong repo này, `src/dire_core/preprocess.py` có docstring nói ngược lại
> ("demo.py của tác giả thiếu Normalize"). Đó là **lỗi trích dẫn** và cần sửa
> (xem Phần 10).

## 8.7 `guided-diffusion/compute_dire.py` — sinh DIRE

Đây là phần quan trọng nhất, đi từng khối:

```python
model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
model.to(dist_util.dev())
model.convert_to_fp16()
model.eval()
```

- `create_model_and_diffusion` dựng U-Net + đối tượng `SpacedDiffusion` theo
  `timestep_respacing` (`ddim20` từ shell ⇒ 20 mốc).
- `convert_to_fp16()` chuyển trọng số sang half precision: nhanh gần 2×, đổi lại
  sai số nhỏ hơn 1/255 ở hầu hết pixel. Với DIRE, sai số này nằm **trong** tín
  hiệu nên cần biết khi so số.

```python
imgs, out_dicts, paths = next(data)      # generator vô hạn
imgs = reshape_image(imgs, args.image_size)
```

`reshape_image`: nếu ảnh không vuông ⇒ `CenterCrop(256)`, nếu vẫn sai ⇒
`F.interpolate(mode="bicubic")`. Đây là chỗ ảnh SD 512² bị thu nhỏ.

```python
latent = reverse_fn(model, (B, 3, 256, 256), noise=imgs,
                    clip_denoised=args.clip_denoised, model_kwargs=model_kwargs,
                    real_step=args.real_step)
```

- `reverse_fn = diffusion.ddim_reverse_sample_loop`.
- **Tham số thứ hai là `shape`, tham số thứ ba là `noise`.** Truyền ảnh vào vị trí
  `shape` là lỗi; đây chính là lỗi trong `src/dire_modal/compute_dire.py:241`
  của repo này.
- `real_step=0` ⇒ đảo qua **toàn bộ** `range(T)`.

```python
recons = sample_fn(model, (B, 3, 256, 256), noise=latent,
                   clip_denoised=..., model_kwargs=..., real_step=args.real_step)
```

- `sample_fn = diffusion.ddim_sample_loop` (vì `use_ddim=True`).
- `noise=latent` ⇒ bắt đầu từ `x_T` đã tìm được, chạy ngược về `x_0`.

```python
dire = th.abs(imgs - recons)                       # trong [-1,1], ∈ [0,2]
dire = (dire * 255.0 / 2.0).clamp(0, 255).to(th.uint8)
recons = ((recons + 1) * 127.5).clamp(0, 255).to(th.uint8)
imgs  = ((imgs + 1) * 127.5).clamp(0, 255).to(th.uint8)
```

Phân tích thang đo — **chỗ dễ sai nhất**:

- `imgs, recons ∈ [−1, 1]` ⇒ `(x+1)·127.5 ∈ [0, 255]` là đúng.
- `|imgs − recons| ∈ [0, 2]` ⇒ để thành `[0, 255]` phải nhân `127.5 = 255/2`.
  Vậy `dire_u8 ≈ |Δpixel|` **thang 1×** — không khuếch đại.

```python
cv2.imwrite(f"{dire_save_dir}/{fn}", cv2.cvtColor(dire[i].numpy(), cv2.COLOR_RGB2BGR))
```

OpenCV dùng BGR, Python/PIL dùng RGB. Quên đổi ⇒ ảnh lưu ra đảo kênh đỏ–xanh;
với DIRE (vốn là ảnh sai số) thì **không ai nhận ra bằng mắt**, nhưng classifier
học trên kênh đảo ⇒ lệch phân phối so với lúc test nếu test cũng đảo — hoặc tệ
hơn, train/test đảo khác nhau.

## 8.8 `src/dire_modal/compute_dire.py` — bản của repo này

Đối chiếu ba điểm quan trọng:

**a) Chỗ gọi inversion (lỗi):**

```python
noise = diffusion.ddim_reverse_sample_loop(model, x0, clip_denoised=True, ...)
```

Chữ ký: `(self, model, shape, noise=None, …)`. `x0` rơi vào `shape` ⇒
`th.randn(*x0)` với `x0` là tensor ⇒ `TypeError`.
Đúng: `(model, x0.shape, noise=x0, …)`.

**b) Đối tượng dùng cho inversion (lỗi chi phí):**

```python
ddim = gd.SpacedDiffusion(use_timesteps=gd.space_timesteps(diffusion.num_timesteps, str(steps)), ...)
noise = diffusion.ddim_reverse_sample_loop(...)   # ← `diffusion` (1000 bước)
recon = ddim.ddim_sample_loop(...)                # ← `ddim` (20 bước)
```

Không đối xứng: `x_T` sinh bằng thang 1000 đem tái tạo bằng thang 20 **không phải**
cặp nghịch đảo. Sửa: dùng `ddim` cho cả hai ⇒ 40 NFE/ảnh thay vì 1020.

**c) Thang lưu (lỗi biểu diễn):**

```python
diff_u8 = np.clip(np.abs(orig - recon) * 4, 0, 255).astype(np.uint8)
```

Hệ số 4 là để **hiển thị gallery**, nhưng hàm này cũng là hàm được
`process_one` dùng để lưu `.npy`. `clip` làm bão hoà ở 64 ⇒ mất thông tin. Nếu
dùng `.npy` để train classifier, biểu diễn khác hẳn DIRE phát hành.

**Sửa đề nghị:** tách `dire_raw()` (thang 1×, để lưu) và `dire_display()` (×4,
chỉ để vẽ).

## 8.9 `src/dire_core/` — phần đã đúng và có test

| Hàm | Làm gì | Vì sao đáng tin |
|---|---|---|
| `build_transform` | `Resize(256) → CenterCrop(224) → ToTensor → Normalize` | test AST khoá đủ 4 bước ở **một** chỗ |
| `fc_out_features` | đọc `fc.weight.shape[0]` | không đoán 1 hay 2 logit |
| `prob_fake` | sigmoid / softmax ổn định | test giá trị đã biết, không cần torch |
| `batch_logits` | chia chunk, `inference_mode` | CLI/API/eval dùng chung một vòng lặp |
| `metrics` | ACC/AP/TPR@FPR | có test đối chiếu sklearn |

`preprocess.py` docstring có một câu **sai** về `demo.py` (xem 8.6) — cần sửa.
Ngoài câu đó, logic đúng.

---

# Phần 9 — Pipeline của repo này, chạy thật

## 9.1 Ba đường vào một lõi

```
                 ┌──────────────────────────────┐
                 │  dire_core (thuần Python)    │
                 │  preprocess / detector /     │
                 │  score / batch / provenance  │
                 └───────┬──────────────┬───────┘
                         │              │
              ┌──────────▼───┐   ┌──────▼──────────┐
              │  dire_local  │   │  dire_modal     │
              │  CLI + API   │   │  eval / api     │
              └──────────────┘   └─────────────────┘
```

Quy tắc: **không** bản sao thứ hai của tiền xử lý, cách đọc checkpoint, hay cách
đổi logit → `P(fake)`. Test `tests/test_core_parity.py` khoá điều này bằng AST.

## 9.2 Các lệnh

```bash
# 1) Dữ liệu (0 byte nếu chỉ xem kế hoạch)
python -m dire_modal.local_fetch --tiers e2min --mode plan
python -m dire_modal.local_fetch --tiers e2min --mode fetch

# 2) Xem trước sẽ chấm bao nhiêu cặp (không cần dữ liệu)
modal run -m dire_modal.eval::local_pairs --tiers e2

# 3) Chấm điểm
bash scripts/01_repro.sh --quick        # e2min: 3 file ⇒ 1 ô + cổng S2
bash scripts/01_repro.sh                # e2,ckpts: 22 file ⇒ 32 ô

# 4) Trên máy nhà, không cần Modal
pip install -e .
dire info                               # không cần torch
dire detect anh.jpg --weights checkpoints/lsun_adm.pth
dire batch ./thu_muc --out scores.csv
```

## 9.3 Ba cổng phải qua trước khi tin một con số

| Cổng | Câu hỏi | Kiểm bằng |
|---|---|---|
| **S2** | `ACC(adm) ≥ 0.99`? | sai ⇒ sai checkpoint/tiền xử lý, **đừng đọc AP** |
| **S6** | chạy lại có ra cùng CSV? | diff = 0 |
| **S9** | local và cloud có cùng lõi? | `pytest tests/test_core_parity.py` |

Nếu S2 không đạt, thứ tự kiểm: `n_real`/`n_fake` trong CSV → `preproc` trong
`run.json` → sha256 checkpoint → thứ tự nhãn. Đừng bắt đầu bằng việc nghi ngờ DIRE.

## 9.4 Mức tái lập (ADR-0006)

Đường eval mặc định dùng **DIRE map của tác giả** và **trọng số của tác giả**, nên
nó là **R0 — chấm lại** (re-scoring): kiểm đường đo của mình khớp đường đo của họ.
Nó **không** chứng minh:

- DIRE map đúng (nếu tác giả tính sai, mình dùng đúng map sai đó);
- detector train lại được (đó là R2 / FR-8);
- ảnh sinh lại được (R3).

Ba trục **độc lập**, không phải cấp bậc. Chi tiết: `docs/adr/adr-0006-reproduction-levels.md`.

---

# Phần 10 — Sai lệch đã tìm thấy, và cách tự kiểm

| # | Vị trí | Vấn đề | Cách sửa |
|---|---|---|---|
| F1 | `src/dire_modal/compute_dire.py:241` | `ddim_reverse_sample_loop(model, x0, …)` — `x0` vào vị trí `shape` | `(model, x0.shape, noise=x0, …)` |
| F2 | cùng file | inversion dùng `diffusion` (1000 bước), reconstruction dùng `ddim` (20) ⇒ 1020 NFE, mất đối xứng | dùng `ddim` cho cả hai ⇒ 40 NFE |
| F3 | cùng file:257 | `|Δ|·4` rồi `clip` ⇒ bão hoà 64, và **lưu** thang 4× vào `.npy` | tách `dire_raw` (1×) / `dire_display` (4×) |
| F4 | `src/dire_core/preprocess.py:7-8` | nói `demo.py` thiếu `Normalize` — sai, `aug_norm` mặc định True | sửa docstring |
| F5 | `compute_dire.py` `_load_adm` | `dropout=0.0` còn shell gốc `0.1`; không gọi `convert_to_fp16()` | ghi rõ trong `run.json`, hoặc khớp cấu hình |

Ba lệnh tự kiểm, không cần GPU:

```bash
# F1 — chữ ký hàm
python - <<'PY'
import inspect
from guided_diffusion import gaussian_diffusion as gd
print(inspect.signature(gd.GaussianDiffusion.ddim_reverse_sample_loop))
PY
# kỳ vọng: (self, model, shape, noise=None, ...)

# F2 — đếm NFE
grep -n "timestep_respacing\|ddim_reverse_sample_loop" src/dire_modal/compute_dire.py

# F3 — thang đo
grep -n "\* 4" src/dire_modal/compute_dire.py
```

---

# Phần 11 — Thuật ngữ (Anh–Việt) và những hiểu nhầm thường gặp

## 11.1 Bảng thuật ngữ

| English | Tiếng Việt | Ghi chú một dòng |
|---|---|---|
| convolution | tích chập | tổng có trọng số trên cửa sổ trượt |
| kernel / filter | hạt nhân / bộ lọc | ma trận trọng số nhỏ |
| receptive field | trường tiếp nhận | vùng ảnh gốc một nơ-ron "thấy" |
| stride / padding | bước nhảy / đệm | điều khiển kích thước đầu ra |
| pooling | gộp | giảm độ phân giải (max/avg) |
| batch normalization | chuẩn hoá theo batch | chuẩn hoá theo mean/var của batch |
| residual connection | kết nối tắt (tàn dư) | `y = x + F(x)` |
| logit | logit | đầu ra thô trước sigmoid/softmax |
| sigmoid / softmax | — | hai cách đổi logit thành xác suất |
| binary cross-entropy | mất mát entropy chéo nhị phân | loss cho 1 logit |
| overfitting | quá khớp | học cả nhiễu của tập train |
| distribution shift | dịch chuyển phân phối | test khác train |
| data leakage | rò rỉ dữ liệu | thông tin test lọt vào train |
| accuracy / precision / recall | độ chính xác / độ chuẩn xác / độ phủ | recall = TPR |
| average precision (AP) | độ chính xác trung bình | diện tích dưới đường precision–recall |
| ROC / AUC | đường cong ROC / diện tích | TPR theo FPR |
| TPR@FPR | — | tỉ lệ bắt được khi khống chế báo nhầm |
| Markov chain | xích Markov | trạng thái sau chỉ phụ thuộc trạng thái trước |
| isotropic Gaussian | Gauss đẳng hướng | phương sai như nhau mọi chiều |
| ELBO | cận dưới biến phân | mục tiêu tối ưu của mô hình sinh xác suất |
| score function | hàm điểm | gradient của log-mật độ |
| ODE / Euler | phương trình vi phân thường / phương pháp Euler | rời rạc hoá để giải |
| discretization error | sai số rời rạc hoá | do chia bước thô |
| inversion | phép đảo ngược | tìm nhiễu tương ứng một ảnh |
| guidance | dẫn hướng (classifier / classifier-free) | tăng độ "bám" điều kiện |
| ablation | thí nghiệm loại bỏ | bỏ một thành phần để đo vai trò |
| concat / fusion | ghép / hợp nhất | nối nhiều nguồn đặc trưng |
| shortcut learning | học đường tắt | dùng tín hiệu dễ thay vì tín hiệu đúng |
| artefact / fingerprint | dấu vết / vân tay | đặc trưng do quá trình sinh để lại |

## 11.2 Năm hiểu nhầm hay gặp

1. **"DIRE là một detector."** Không hẳn. DIRE là **biểu diễn**; detector là
   ResNet-50 đứng sau. Có thể đổi classifier mà giữ DIRE.
2. **"ADM dùng để phát hiện ảnh giả."** Không. ADM là mô hình sinh bình thường,
   dùng như phép biến đổi. Nó không biết gì về bài toán thật/giả.
3. **"DIRE cao = ảnh giả."** Ngược lại: DIRE **thấp** ⇒ dễ tái tạo ⇒ **ảnh giả**.
   Nhầm chiều là lỗi từng có trong repo này (`compute_dire.py` từng ghi
   `|x − x̂|` nhưng diễn giải ngược).
4. **"Dùng lại dataset của tác giả là tái lập."** Không — đó là **chấm lại**
   (R0). Tái lập cần tự tính map hoặc tự train lại.
5. **"ACC 100% nghĩa là hoàn hảo."** Với `n = 1000`, ACC 100% có khoảng tin cậy
   ±0,3% (quy tắc ba-sigma cho tỉ lệ). 99,9 và 100,0 không khác nhau về ý nghĩa.

## 11.3 Lộ trình đọc tiếp

| Bước | Đọc gì | Mục đích |
|---|---|---|
| 1 | Phần 1–2 tài liệu này | chắc nền xử lý ảnh + metric |
| 2 | `ddpm` (Ho et al. 2020), §1–2 | hiểu forward/ELBO |
| 3 | `ddim` (Song et al. 2020) | hiểu vì sao tất định và đảo ngược được |
| 4 | Phần 5–6 tài liệu này | nối vào DIRE |
| 5 | `docs/paper-code-walkthrough.md` | đọc code gốc từng dòng |
| 6 | Tự chạy `--tiers e2min` | thấy một con số thật |

---

# Phụ lục — Bảng "một ảnh đi qua những gì"

| Bước | Dạng dữ liệu | Kích thước | Code |
|---|---|---|---|
| Đọc ảnh | PIL RGB | H×W×3, uint8 | `Image.open(...).convert("RGB")` |
| Resize/crop | PIL | 256×256 | `reshape_image` |
| Chuẩn hoá vào diffusion | tensor CHW | 3×256×256, float `[-1,1]` | `arr/127.5 − 1` |
| Inversion (S=20) | tensor | 3×256×256, `[-1,1]` | `ddim_reverse_sample_loop(..., noise=x)` |
| Reconstruction (S=20) | tensor | 3×256×256, `[-1,1]` | `ddim_sample_loop(..., noise=x_T)` |
| DIRE | mảng | 3×256×256, uint8 | `abs(x−x̂)·255/2` |
| Tiền xử lý classifier | tensor | 3×224×224, chuẩn hoá ImageNet | `Resize(256)→CenterCrop(224)→ToTensor→Normalize` |
| ResNet-50 | tensor | 2048 → 1 | `model(x)` |
| Xác suất | số thực | `[0,1]` | `sigmoid(logit)` |

Tổng chi phí một ảnh: **40 forward ADM + 1 forward ResNet-50**.
