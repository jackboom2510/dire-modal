# Phần 0 — Công cụ toán tối thiểu

Phần này gom **đúng** những công cụ toán mà các phần sau dùng. Không chứng minh
định lý; mỗi mục gồm: phát biểu, một câu "nghĩa là gì", và một chỗ dùng thật trong
tài liệu. Ai đã quen có thể bỏ qua, nhưng nên đọc mục 0.5–0.7 vì đó là chỗ nối
toán với hàm mất mát và với độ đo.

## 0.1 Véc-tơ, ma trận, tensor — và chuyện shape

- **Véc-tơ** `x ∈ R^n`: danh sách `n` số. Một ảnh xám 224×224 là véc-tơ 50176 chiều.
- **Ma trận** `A ∈ R^(m×n)`: `m·n` số, xếp thành lưới.
- **Tensor**: mảng nhiều chiều. PyTorch dùng quy ước `(N, C, H, W)` = (số ảnh trong
  lô, kênh, cao, rộng). Một lô 64 ảnh RGB 224² là tensor `(64, 3, 224, 224)`.

Phép toán hay gặp:

$$ a \cdot b = \sum_i a_i b_i, \qquad (Ax)_i = \sum_j A_{ij} x_j $$

**Phát sóng** (broadcasting): khi trừ một tensor `(N, 1)` cho `(N,)`, PyTorch **mở
rộng** cái sau thành `(N, N)` chứ không báo lỗi. Đây không phải chi tiết lý thuyết
suông: nó chính là lý do phải có `output.squeeze(1)` trước `BCEWithLogitsLoss` ở
Phần 10.4. Nếu quên, mất mát được tính trên **ma trận** `N×N` và mô hình vẫn train
"bình thường" nhưng sai hoàn toàn.

## 0.2 Chuẩn, khoảng cách, và vì sao DIRE dùng L1 chứ không L2

$$ \|x\|_2 = \sqrt{\sum_i x_i^2}, \qquad \|x\|_1 = \sum_i |x_i|, \qquad \|x\|_\infty = \max_i |x_i| $$

- `‖·‖₂` (Euclid) phạt mạnh thành phần lớn; `‖·‖₁` phạt đều mọi thành phần.
- **MSE** là `‖·‖₂²/N`; **MAE** là `‖·‖₁/N`.

DIRE dùng `|x − x̂|` **theo từng điểm ảnh** rồi để **mạng tích chập** tự tổng hợp —
tức không nén ngay thành một số như MSE. Lý do: thông tin phân biệt nằm ở **vị trí**
của sai số (biên, texture), và phép lấy trung bình sẽ xoá mất cấu trúc đó. Đây là
cùng lý do SSIM/LPIPS tồn tại: một con số vô hướng không mô tả được cấu trúc sai số.

## 0.3 Đạo hàm, lan truyền ngược, và ba dòng giải thích Adam

Một lớp mạng là hàm hợp. Đạo hàm của hàm hợp là **tích các đạo hàm thành phần**
(chain rule):

$$ \frac{\partial \mathcal{L}}{\partial w_1} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial h} \cdot \frac{\partial h}{\partial w_1} $$

Lan truyền ngược (backpropagation) chỉ là chain rule áp **từ cuối về đầu**, dùng
lại kết quả trung gian nên chi phí cùng bậc với lượt đi.

**Gradient descent** cập nhật `θ ← θ − η·∇θ`. Ba biến thể cần phân biệt:

| Thuật toán | Cập nhật (ý tưởng) | Vì sao dùng |
|---|---|---|
| SGD | `θ ← θ − η·g` | đơn giản, nhiều nhiễu |
| SGD + momentum | cộng dồn vận tốc `v ← βv + g` | vượt qua vùng phẳng, giảm dao động |
| **Adam** | chia `g` cho căn trung bình bình phương của chính nó, theo từng tham số | tự chỉnh learning rate cho từng tham số |

Adam giữ hai trung bình trượt: `m` (moment bậc 1, trung bình gradient) và `v`
(moment bậc 2, trung bình bình phương gradient). Bước cập nhật có dạng

$$ \theta \leftarrow \theta - \eta\,\frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon} $$

với `m̂`, `v̂` là hai moment đã hiệu chỉnh độ lệch. Trực giác: tham số nào có
gradient **dao động mạnh** thì `v` lớn ⇒ bước nhỏ lại; tham số nào gradient **đều**
thì bước gần bằng `η`. Bài báo dùng Adam `lr = 1e-4` ([[19]]).

Một chi tiết bắt buộc: **phải zero_grad** mỗi bước. PyTorch cộng dồn gradient vào
`.grad`; quên xoá thì gradient là tổng của mọi batch trước đó.

## 0.4 Xác suất: biến ngẫu nhiên, kỳ vọng, phương sai

- **Kỳ vọng** `E[X]` là trung bình có trọng số theo xác suất; với mẫu hữu hạn, xấp
  xỉ bằng trung bình cộng (luật số lớn).
- **Phương sai** `Var[X] = E[(X − E[X])²]`; **độ lệch chuẩn** là căn của nó.
- Hai biến **độc lập** thì `Var[X+Y] = Var[X] + Var[Y]` — tính chất này là toàn bộ
  cơ sở của công thức `q(x_t | x_0)` ở Phần 5.1.

**Phân phối Gauss một chiều:**

$$ \mathcal{N}(x;\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) $$

**Gauss đẳng hướng nhiều chiều** (mọi chiều cùng phương sai, không tương quan):

$$ \mathcal{N}(x;\mu,\sigma^2\mathbf{I}) \;\propto\; \exp\!\left(-\frac{\|x-\mu\|^2}{2\sigma^2}\right) $$

Nghĩa là: "đám mây" hình cầu quanh `μ`. Chính vì thế `ε ~ N(0, I)` trong diffusion
được cộng vào **mọi** điểm ảnh với cùng mức nhiễu.

**Ba lý do Gauss có mặt ở khắp nơi** (cần biết để không thấy nó "từ trên trời rơi
xuống"):

1. **Định lý giới hạn trung tâm**: tổng nhiều nguồn nhiễu nhỏ, độc lập ⇒ xấp xỉ
   Gauss. Nhiễu cảm biến đúng là như vậy.
2. **Tính đóng**: tổng hai Gauss độc lập vẫn là Gauss, với phương sai **cộng**.
   Đây là lý do `x_t = √ᾱ_t x₀ + √(1−ᾱ_t)ε` có dạng đóng chính xác, không cần xấp
   xỉ gì.
3. **Entropy cực đại** với phương sai cho trước: Gauss là phân phối "ít giả định
   nhất". Nên khi chỉ biết phương sai nhiễu, chọn Gauss là lựa chọn trung lập.

## 0.5 Từ hợp lý cực đại (MLE) tới BCE — vì sao hàm mất mát có dạng đó

Giả sử mô hình nói "xác suất ảnh này là giả" là `p = σ(z)`. Với nhãn `y ∈ {0,1}`,
xác suất của quan sát là phân phối **Bernoulli**:

$$ P(y \mid z) = \sigma(z)^{y}\,\big(1-\sigma(z)\big)^{1-y} $$

**Hợp lý** (likelihood) của cả tập là tích các xác suất này. Tích dễ tràn số, nên
lấy log (đơn điệu tăng, không đổi chỗ cực trị) rồi đổi dấu để thành bài toán cực
tiểu:

$$ \mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_i \Big[\, y_i \log \sigma(z_i) + (1-y_i)\log\big(1-\sigma(z_i)\big) \Big] $$

Đây **đúng là** âm log-hợp lý (negative log-likelihood) của Bernoulli — nên nó
không phải một công thức được "chọn cho tiện", mà là hệ quả của giả định xác suất.
Đạo hàm của nó rất gọn:

$$ \frac{\partial \mathcal{L}}{\partial z} = \sigma(z) - y $$

Nghĩa là: sai bao nhiêu (theo xác suất) thì sửa bấy nhiêu — và đây là lý do sigmoid
đi cặp với BCE. (Nếu dùng MSE với sigmoid, đạo hàm có thêm `σ′(z) = σ(1−σ)`, gần 0
khi mô hình rất tự tin nhưng **sai** ⇒ học cực chậm. Đó là "vanishing gradient" của
lớp ra.)

**Ổn định số học.** `log σ(z) = −log(1+e^(−z))`. Với `z = −800`, `e^800` tràn.
PyTorch gộp phép tính trong `BCEWithLogitsLoss` để dùng dạng ổn định
`max(z,0) − z*y + log(1+e^(−|z|))`. Khi tự cài, phải chia hai nhánh theo dấu — như
`src/dire_core/score.py` làm.

## 0.6 Entropy, entropy chéo, KL — vì sao "khớp phân phối" là mục tiêu đúng

$$ H(p) = -\sum_k p_k \log p_k, \qquad H(p,q) = -\sum_k p_k \log q_k, \qquad D_{\mathrm{KL}}(p\,\|\,q) = \sum_k p_k \log\frac{p_k}{q_k} $$

Ba đại lượng này liên hệ đúng một đẳng thức:

$$ H(p, q) = H(p) + D_{\mathrm{KL}}(p\,\|\,q) $$

Vì `D_KL ≥ 0` (bất đẳng thức Jensen) và bằng 0 **chỉ khi** `p = q`, cực tiểu hoá
entropy chéo theo `q` chính là kéo `q` về `p`. Nói cách khác: **cross-entropy là
một độ đo khoảng cách giữa hai phân phối, sai khác một hằng số**. Đó là lý do mọi
bài phân loại dùng nó, và cũng là lý do ELBO trong diffusion có dạng "kỳ vọng log
tỉ lệ".

**ELBO, một dòng.** Với biến ẩn `z` và dữ liệu `x`:

$$ \log p(x) \;\ge\; \mathbb{E}_{q(z\mid x)}\big[\log p(x,z) - \log q(z\mid x)\big] \;=\; \text{ELBO} $$

Chênh lệch giữa hai vế **đúng bằng** `D_KL(q(z|x) ‖ p(z|x))`. Nên tối đa ELBO =
tối đa hợp lý + tối thiểu khoảng cách giữa xấp xỉ và hậu nghiệm thật. Hàm mất mát
[11] của diffusion là ELBO bị **bỏ trọng số theo `t`** (xem Phần 5.2).

**Entropy như "độ bất định".** Phân phối đều trên `K` lớp có `H = log K` (bất định
nhất); phân phối "chắc chắn 100%" có `H = 0`. Vì thế `H` xuất hiện trong mọi tiêu
chí chọn ngưỡng, và trong phân tích "mô hình có đang đoán bừa không".

## 0.7 Lấy mẫu, Nyquist, và định lý tích chập

**Định lý lấy mẫu (Nyquist–Shannon).** Một tín hiệu có tần số cao nhất `f_max` chỉ
khôi phục được nếu lấy mẫu với tần số `≥ 2·f_max`. Lấy mẫu thưa hơn ⇒ **chồng phổ**
(aliasing): tần số cao "gập" xuống thành tần số thấp giả.

Hệ quả thực hành, dùng ở Phần 1.5:

- Thu nhỏ ảnh **phải** lọc thông thấp trước khi bỏ điểm ảnh (đó là `antialias=True`
  của torchvision). Bỏ qua ⇒ ảnh nhỏ bị moiré, và bạn vừa tạo ra một dấu vết **giả**
  mà detector có thể học nhầm.
- Vì sao dấu vết JPEG nằm ở tần số cụ thể: chia khối 8×8 là một phép nhân với lưới
  chu kỳ 8 trong miền không gian ⇔ tạo các đỉnh cách đều trong miền tần số.

**Định lý tích chập.** Tích chập trong miền không gian = phép nhân trong miền tần số:

$$ \mathcal{F}\{f * g\} = \mathcal{F}\{f\}\cdot\mathcal{F}\{g\} $$

Nghĩa là "lọc" và "tích chập" là một. Nên mọi phép làm mờ, làm nét, nội suy, và cả
tích chập của mạng CNN đều là **phép nhân phổ** — đó là lý do nói "CNN tầng đầu học
bộ lọc tần số" không phải ẩn dụ.

## 0.8 ROC, AUC, AP — và một cách hiểu AUC rất gọn

Cho điểm `s` và hai phân phối điểm: của ảnh thật `p_r(s)`, của ảnh giả `p_g(s)`. Với
ngưỡng `t`:

$$ \mathrm{TPR}(t) = \int_t^\infty p_g(s)\,ds, \qquad \mathrm{FPR}(t) = \int_t^\infty p_r(s)\,ds $$

Đường ROC là `(FPR(t), TPR(t))` khi `t` quét toàn miền. Hai kết quả cần nhớ:

1. **AUC = xác suất một ảnh giả ngẫu nhiên được cho điểm cao hơn một ảnh thật ngẫu
   nhiên** (đẳng thức Mann–Whitney). Nên AUC 0,5 = xếp hạng ngẫu nhiên; AUC 1 = tách
   hoàn hảo. Hiểu theo nghĩa này, AUC không phụ thuộc ngưỡng và không phụ thuộc tỉ lệ
   hai lớp.
2. **AP** là diện tích dưới đường precision–recall, và **phụ thuộc tỉ lệ lớp** — khác
   AUC ở điểm này. Vì thế một mô hình có thể có AP cao mà TPR@FPR thấp nếu tập test
   lệch lớp. Đây là lý do bài báo báo **cả** ACC và AP, và lý do tài liệu này luôn
   ghi `n_real`/`n_fake`.

**Chọn ngưỡng trong ứng dụng thật.** Không dùng 0,5 (đó là quy ước cho tập cân
bằng). Người ta chọn `t` sao cho `FPR ≤ 1%`, rồi báo `TPR` tại đó. Ngưỡng **phải**
chọn trên validation, không trên test.

## 0.9 Chi phí tính toán: vì sao NFE là đơn vị đúng

Một lượt chạy mạng trên một ảnh tốn xấp xỉ `O(số tham số × số điểm ảnh)`. Khi so
sánh phương pháp, đại lượng trung tính nhất là **số lượt chạy mạng** (NFE), vì nó
không phụ thuộc GPU cụ thể.

- ResNet-50 chấm một ảnh: **1 NFE**.
- DIRE tính một map với `S` bước: **2S NFE** (đảo ngược + tái tạo).
- Với `S = 20`: DIRE đắt gấp **40 lần** một lượt phân loại thường.

Đây là con số phải nhớ khi đọc mọi ước lượng chi phí ở Phần 7.7. Và là lý do một lỗi
cấu hình (chạy 1000 bước thay vì 20) làm hoá đơn tăng 25 lần mà **không** làm kết
quả "trông sai" — nó vẫn ra một ảnh DIRE, chỉ là một biểu diễn khác.
