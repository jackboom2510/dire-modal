# Phần 1 — Ảnh số và những phép biến đổi làm mất thông tin

## 1.1 Ảnh số: hai phép rời rạc hoá

Một ảnh xám là hàm hai biến `I(x, y)` cho cường độ sáng tại toạ độ `(x, y)`. Để
lưu vào máy tính phải làm hai việc, và cả hai đều **mất thông tin**:

- **Mẫu hoá** (sampling): rời rạc hoá toạ độ không gian thành lưới `H × W`. Mất
  chi tiết nhỏ hơn một điểm ảnh (pixel).
- **Lượng tử hoá** (quantization): rời rạc hoá cường độ thành `2^b` mức, thường
  `b = 8` nên có 256 mức từ 0 đến 255.

Ảnh màu RGB là ba ảnh xám xếp chồng, tức tensor `H × W × 3`. Trong học sâu người
ta đổi sang `C × H × W` (kênh trước) — đây là lý do mọi tensor trong PyTorch đều
có dạng `(N, C, H, W)`.

**Vì sao điều này quan trọng với bài toán phát hiện ảnh giả.** Một ảnh chụp thật đi
qua chuỗi thiết bị vật lý:

```
cảm biến (Bayer) → khử nhiễu → demosaic → cân bằng trắng → gamma → nén JPEG
```

Mỗi bước để lại dấu vết **có tính vật lý và có tính thiết bị**: nhiễu đọc của cảm
biến, nhiễu photon tỉ lệ với căn bậc hai cường độ, méo quang học, và cuối cùng là
lượng tử hoá 8 bit. Một ảnh do mô hình sinh tạo ra **không đi qua bất kỳ bước
nào** trong chuỗi đó: nó ra đời dưới dạng số thực trong `[−1, 1]`, được cắt và
lượng tử hoá **một lần duy nhất**.

Hệ quả: mọi detector ảnh giả đều đang cố bắt một khác biệt **vật lý**, và mọi
phương pháp đều yếu đi khi ảnh bị xử lý lại (nén, làm mờ, phóng to) vì các phép
đó xoá chính dấu vết vật lý. Ta sẽ thấy DIRE né được điểm yếu này ở mức nào
(Phần 8.5), nhưng không né được hoàn toàn.

## 1.2 Tích chập: phép toán nền của mọi mạng tích chập

Tích chập rời rạc 2D của ảnh `I` với hạt nhân `K` kích thước `k × k`:

$$[2] (I * K)[i, j] \;=\; \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} I[i+m,\; j+n]\; K[m, n] $$

Trực giác: trượt một ô cửa sổ `k × k` khắp ảnh; tại mỗi vị trí, đầu ra là tổng có
trọng số của vùng lân cận. Ba hạt nhân nên nhớ:

| Hạt nhân | Nội dung | Tác dụng | Liên hệ với DIRE |
|---|---|---|---|
| Trung bình (box) | mọi trọng số bằng `1/k²` | làm mờ | xoá chi tiết tần số cao |
| Gauss | trọng số giảm theo khoảng cách | làm mờ mềm | bài báo dùng `σ = 1, 2, 3` để kiểm robustness |
| Laplace | `[0,−1,0; −1,4,−1; 0,−1,0]` | làm nổi biên | DIRE cũng là một phép "thông cao không gian" |

Tích chập có ba tính chất giải thích vì sao nó phù hợp với ảnh:

- **Bất biến tịnh tiến**: cùng hạt nhân dùng ở mọi vị trí, nên một mẫu hình được
  nhận ra dù nằm ở đâu.
- **Kết nối cục bộ**: mỗi nơ-ron chỉ nhìn `k × k` điểm ảnh, phù hợp với việc
  tương quan không gian của ảnh là tương quan gần.
- **Chia sẻ trọng số**: số tham số không phụ thuộc kích thước ảnh, nên mạng dùng
  được cho ảnh lớn hơn lúc train.

Một khái niệm đi kèm mà tài liệu sẽ dùng lại: **trường tiếp nhận** (receptive
field) — vùng ảnh gốc mà một nơ-ron ở tầng `l` "thấy" được. Công thức truy hồi:

```
RF_l = RF_{l−1} + (k_l − 1) · Π(stride từ tầng 1 đến l)
```

Với ResNet-50 và ảnh 224×224, trường tiếp nhận ở tầng cuối **lớn hơn** ảnh. Nghĩa
là tầng sâu nhất có thể phát hiện cấu trúc toàn cục, không chỉ dấu vết cục bộ.

## 1.3 Miền tần số: vì sao dân pháp y ảnh rất thích nó

Biến đổi Fourier rời rạc (DFT) đổi biểu diễn từ miền không gian sang miền tần số:

$$[3] F(u, v) \;=\; \sum_{x=0}^{H-1}\sum_{y=0}^{W-1} I(x, y)\; e^{-2\pi i\,(ux/H \,+\, vy/W)} $$

- **Tần số thấp** = thay đổi chậm: mảng màu lớn, bố cục, ánh sáng.
- **Tần số cao** = thay đổi nhanh: biên, chi tiết nhỏ, nhiễu.

Phổ biên độ `|F(u,v)|` cho biết năng lượng ở từng tần số. Các bộ sinh ảnh đời đầu
dùng phép **phóng to** (upsampling) để tăng độ phân giải: transposed convolution
hoặc pixel-shuffle. Mọi phép phóng to đều là một phép chèn mẫu (zero-insertion)
theo sau là lọc, và chèn mẫu tạo ra **cấu trúc tuần hoàn** trong phổ — những đỉnh
năng lượng ở vị trí cố định. Mắt người không thấy; FFT thấy rõ.

Đây là cơ sở của cả một thế hệ detector: CNNDetection [[11]] dùng đặc trưng phổ,
F3-Net [[13]] phân tích phổ tần số hai nhánh, Patchfor [[12]] dùng classifier trên
từng mảng để chỉ ra vùng nào dễ phát hiện.

**Điểm yếu của cả họ này:** vị trí đỉnh phổ **phụ thuộc kiến trúc bộ sinh**. Đổi
từ GAN sang diffusion là đổi hoàn toàn cấu trúc phổ. DIRE ra đời đúng để tránh
phụ thuộc đó: nó không đọc phổ của bộ sinh, mà đọc **sai số của một mô hình
khuếch tán độc lập**.

## 1.4 JPEG: ví dụ đầy đủ nhất về "dấu vết"

JPEG chia ảnh thành khối 8×8, biến đổi DCT từng khối, **lượng tử hoá** hệ số theo
bảng lượng tử, rồi mã hoá entropy (Huffman hoặc số học). Ba hệ quả:

1. **Mất thông tin không đều**: bảng lượng tử cắt tần số cao mạnh hơn nhiều so
   với tần số thấp. Đó là lý do ảnh JPEG trông "mượt" chứ không "nhiễu".
2. **Tính chu kỳ 8 pixel**: vì chia khối, ảnh JPEG có cấu trúc tuần hoàn chu kỳ 8
   trong phổ. Đây là dấu vết mạnh và rất ổn định.
3. **Nén nhiều lần tích luỹ**: mỗi lần nén lại, lượng tử hoá cắt thêm; ảnh qua
   Facebook/Zalo nhiều lần khác xa ảnh gốc.

Với bài toán phát hiện ảnh giả, JPEG là **con dao hai lưỡi**. Nó tạo dấu vết mới
(giúp phát hiện ảnh đã qua xử lý), nhưng đồng thời **xoá** dấu vết của bộ sinh
(làm hại detector). Bài báo kiểm đúng điểm này: nén `quality = 65` và `30`
(Hình 4 của [[1]]); các baseline dựa trên RGB sụp mạnh, DIRE gần như không đổi.

Vì sao DIRE bền hơn với JPEG? Vì DIRE là hiệu của hai ảnh đã đi qua **cùng một
phép biến đổi** (inversion rồi reconstruction). Thành phần bị JPEG phá huỷ giống
nhau ở hai vế nên **triệt tiêu một phần** khi trừ. Đây là lập luận định tính hợp
lý, nhưng bài báo không đo trực tiếp phần triệt tiêu đó.

## 1.5 Tái lấy mẫu: phép biến đổi âm thầm nhất

Khi đổi kích thước ảnh, ta phải nội suy:

| Phương pháp | Công thức trực giác | Đặc điểm |
|---|---|---|
| Nearest | lấy điểm gần nhất | giữ nguyên giá trị gốc, tạo răng cưa |
| Bilinear | trung bình có trọng số 2×2 | mềm, hơi mờ |
| Bicubic | đa thức bậc 3 trên 4×4 | sắc hơn bilinear, mặc định của nhiều công cụ |
| Lanczos | cửa sổ sinc | sắc nhất, có thể tạo ringing |

Điểm cần khắc vào đầu: **mọi phép nội suy đều là lọc thông thấp ở một mức nào
đó**, nên đều **xoá dấu vết tần số cao**. Nếu bạn resize ảnh trước khi kiểm tra,
bạn có thể tự tay xoá đúng tín hiệu mà detector cần.

Ba chỗ trong pipeline của bài báo liên quan trực tiếp:

- Ảnh Stable Diffusion là 512×512, bị **bicubic xuống 256×256** cho vừa ADM
  (`reshape_image` trong `compute_dire.py`). Một phần tín hiệu mất trước khi vào
  detector.
- Hai mô hình sinh ảnh của LSUN và ImageNet đều được đưa về 256, nên so sánh giữa
  các generator là so sánh **sau cùng một phép làm mờ**.
- Khi đánh giá, ảnh được `Resize(256)` rồi `CenterCrop(224)`. Với ảnh 256×256 thì
  resize là không đổi; với ảnh DIRE cũng 256×256 nên không đổi. Nhưng nếu bạn tự
  tạo DIRE map ở độ phân giải khác rồi đưa qua pipeline này, bạn đang đo một thứ
  khác.

Một chi tiết cài đặt dễ sai: trong torchvision, `Resize(256)` với **một số nguyên**
có nghĩa là "cạnh **ngắn** bằng 256, giữ tỉ lệ". Với ảnh 4:3, kết quả là 256×341
chứ không phải 256×256. Chỉ sau `CenterCrop(224)` mới thành 224×224.

## 1.6 Ba độ đo chất lượng ảnh — và vì sao DIRE không dùng chúng

- **MSE / PSNR.** `PSNR = 10·log₁₀(MAX²/MSE)` với `MAX = 255`, đơn vị dB. Nhược
  điểm: không khớp cảm nhận thị giác, rất "khoan dung" với ảnh mờ (làm mờ giảm
  MSE mạnh).
- **SSIM.** So sánh trung bình, phương sai và hiệp phương sai cục bộ; khớp cảm
  nhận hơn PSNR, nhưng vẫn là độ đo toàn cục.
- **LPIPS.** Dùng đặc trưng của một mạng pretrained để đo khoảng cách tri giác;
  gần với đánh giá của người nhất, nhưng phụ thuộc mạng dùng làm "trọng tài".

DIRE **không** dùng ba độ đo này, và đó là lựa chọn có chủ đích: DIRE không hỏi
"ảnh tái tạo có đẹp không" mà hỏi "**sai số phân bố thế nào**". Hai ảnh tái tạo có
cùng PSNR vẫn có thể có cấu trúc sai số rất khác nhau — một cái sai đều khắp ảnh,
một cái sai tập trung ở vùng biên. Chính cấu trúc đó mới là thứ ResNet-50 học.

## 1.7 Không gian màu, gamma, và vì sao JPEG không nén RGB trực tiếp

### Ba kênh không độc lập

Mắt người nhạy với **độ sáng** hơn nhiều so với **màu**. Vì vậy mọi chuẩn nén ảnh
đều đổi RGB sang một không gian tách hai thứ đó ra:

$$ Y = 0{,}299R + 0{,}587G + 0{,}114B, \qquad C_b = 0{,}564\,(B-Y), \qquad C_r = 0{,}713\,(R-Y) $$

`Y` là độ sáng (luma), `C_b`/`C_r` là hai kênh màu (chroma). JPEG sau đó **lấy mẫu
thưa** hai kênh màu (4:2:0 — giữ 1/4 số điểm ảnh) vì mắt không phân biệt được. Đây
là lý do ảnh JPEG 8 bit thực chất chỉ chứa đầy đủ thông tin độ sáng.

**Hệ quả cho pháp y ảnh:** mọi phép xử lý ảnh (nén, chỉnh màu, phóng to) đều tác
động **không đều** lên `Y` và `C_b`/`C_r`. Một số detector chỉ dùng kênh `Y`. Với
DIRE thì khác: nó giữ **3 kênh RGB**, nên sai số trên kênh màu cũng vào biểu diễn —
và đây là một trong những lý do tín hiệu của nó phong phú hơn các phương pháp chỉ
nhìn độ sáng.

### Gamma: vì sao "giá trị 128" không phải "một nửa độ sáng"

Cảm biến ghi ánh sáng **tuyến tính** theo số photon, nhưng mắt cảm nhận độ sáng
theo **luỹ thừa** gần bậc 1/2,2. Nếu lưu tuyến tính, dải tối sẽ bị dồn vào vài giá
trị và dải sáng chiếm quá nhiều. Vì vậy ảnh được **mã hoá gamma**:

$$ I_{\mathrm{enc}} = 255\cdot\left(\frac{I_{\mathrm{lin}}}{255}\right)^{1/\gamma}, \qquad \gamma \approx 2{,}2 $$

trong đó `I_lin` là giá trị tuyến tính (tỉ lệ số photon) và `I_enc` là giá trị sau
mã hoá (giá trị mà ta lưu). Hệ quả định lượng cần nhớ: giá trị 8 bit `128` tương
ứng **khoảng 22%** độ sáng tuyến tính, không phải 50%.

Ba chỗ gamma ảnh hưởng tới tài liệu này:

1. **`|x − x̂|` không phải sai số vật lý.** DIRE tính trên giá trị đã mã hoá gamma,
   nên một sai lệch `Δ` ở vùng tối "nặng" hơn cùng `Δ` ở vùng sáng (theo nghĩa vật
   lý). Bài báo không bàn điểm này; nó là một nguồn thiên lệch có hệ thống giữa ảnh
   tối và ảnh sáng.
2. **`pixel/127,5 − 1`** là phép đổi **thẳng** trên giá trị đã mã hoá, không giải
   gamma. ADM cũng được train trên dữ liệu như vậy, nên nhất quán — nhưng nếu bạn
   tự giải gamma trước khi đưa vào, bạn đang lệch khỏi miền mà mô hình biết.
3. **Làm mờ và nén** đều tác động khác nhau lên vùng tối/sáng vì gamma; đó là một
   phần lý do "ảnh thật khó tái tạo" không đồng đều trên toàn ảnh.

### Vì sao 8 bit là đủ — và khi nào không

Lượng tử hoá 8 bit cho 256 mức. Nếu nhiễu cảm biến lớn hơn 1 mức (trường hợp phổ
biến), lượng tử hoá gần như không mất gì. Nhưng nếu ảnh được tạo **tổng hợp** (như
ảnh diffusion) rồi lượng tử hoá **một lần** từ số thực, thì bước lượng tử trở thành
một dấu vết: phân phối phần dư làm tròn của ảnh tổng hợp **khác** ảnh chụp (ảnh chụp
đã bị nhiễu che lấp trước khi lượng tử hoá). Đây là cơ sở của cả một nhóm phương
pháp pháp y ảnh dựa trên "dấu vết lượng tử hoá", và là lý do các phép phân tích
**không được** làm mất bit thấp (đừng chuyển sang JPEG chất lượng cao trước khi
phân tích).

# Phần 2 — Học máy: những gì thực sự cần cho bài toán này

## 2.1 Bài toán tối ưu

Cho tập $D = \{(x_i, y_i)\}$ với $y_i \in \{0,1\}$ ($0$ = thật, $1$ = giả), ta
tìm tham số $\theta$ cực tiểu **rủi ro thực nghiệm**:

$$ \min_\theta \; \frac{1}{N}\sum_{i=1}^{N} \ell\big(f_\theta(x_i),\, y_i\big) $$

Quy trình lặp quen thuộc: lan truyền xuôi → tính mất mát → lan truyền ngược →
cập nhật. Ba siêu tham số quyết định tốc độ:

- **Learning rate** $\eta$: bước cập nhật $\theta \leftarrow \theta - \eta\nabla_\theta$.
- **Batch size**: số mẫu dùng cho một lần tính gradient. Bài báo dùng 64.
- **Optimizer**: bài báo dùng Adam [[19]], `lr = 1e-4`. Bộ điều hoà learning
  rate mặc định là `CosineAnnealingLR` [[15]].

Một con số cần tính đúng khi tái lập: **số bước mỗi epoch**. Với 40k ảnh thật +
40k ảnh giả = 80k ảnh, batch 64 ⇒ `80 000 / 64 = 1 250` bước/epoch. Bài báo
train 400 epoch ⇒ 500 000 bước. Con số này là cơ sở của mọi ước lượng chi phí ở
Phần 7.7, và nó thường bị tính sai thành `40 000/64 = 625`.

## 2.2 Từ logit tới xác suất: hai quy ước không hoán đổi được

Mạng cho ra số thực `z` gọi là **logit**. Có hai cách biến nó thành xác suất, ứng
với hai cách train khác nhau:

**Một logit, mất mát BCE** (bài báo dùng cách này):

$$[4] \sigma(z) = \frac{1}{1+e^{-z}}, \qquad \mathcal{L} = -\Big[\, y\log\sigma(z) + (1-y)\log\big(1-\sigma(z)\big) \Big] $$

**Hai logit, mất mát cross-entropy:**

$$[5] P(y=k \mid x) = \frac{e^{z_k}}{\sum_j e^{z_j}}, \qquad \mathcal{L} = -\log P(y \mid x) $$

Hai quy ước này **không hoán đổi**. Checkpoint DIRE có `fc.weight` shape
`[1, 2048]` — tức một logit, phải dùng sigmoid. Nếu bạn dựng mô hình với
`num_classes=2`, `load_state_dict` sẽ nổ `size mismatch for fc.weight` **sau khi**
đã tải 269 MB. Kho này đọc shape từ chính checkpoint để tránh lỗi đó
(`src/dire_core/detector.py::fc_out_features`).

**Vì sao dùng `BCEWithLogitsLoss` thay vì sigmoid rồi BCE.** Hợp của hai phép có
dạng ổn định hơn nhiều: với `z` rất âm, `log(σ(z))` tính trực tiếp sẽ ra `log(0)`
tức `−∞`; công thức gộp dùng `logaddexp` nên không tràn. Khi tự cài đặt phép đổi
logit → xác suất (như `src/dire_core/score.py`), phải chia hai nhánh theo dấu `z`:

```python
def sigmoid(z: float) -> float:
    """Sigmoid chịu được |z| lớn (sigmoid(-800) = 0, không raise)."""
    z = float(z)
    if z >= 0:
        return 1.0 / (1.0 + _math.exp(-z))
    e = _math.exp(z)          # nhánh này tránh exp(800) = inf
    return e / (1.0 + e)
```

## 2.3 Quá khớp, tổng quát hoá, và dịch chuyển phân phối

- **Quá khớp** (overfitting): mô hình học cả nhiễu của tập train; loss train giảm
  nhưng loss validation tăng.
- **Tổng quát hoá**: tốt trên dữ liệu chưa thấy nhưng **cùng phân phối**.
- **Dịch chuyển phân phối** (distribution shift): dữ liệu test khác phân phối
  train. Đây là trục chính của bài toán phát hiện ảnh giả.

Bài toán DIRE là dịch chuyển phân phối **có chủ đích**: `p_g` lúc train (ảnh ADM)
khác `p_g` lúc test (SD, LDM, VQ-Diffusion…). Vì vậy khi đọc kết quả, câu hỏi
không phải "độ chính xác bao nhiêu" mà "**cái gì bất biến giữa các bộ sinh**".

Ba chiến lược chống dịch chuyển, để thấy DIRE đứng ở đâu:

| Chiến lược | Cách làm | Vấn đề |
|---|---|---|
| Tăng dữ liệu train | train trên nhiều bộ sinh | mỗi bộ sinh mới lại cần dữ liệu mới |
| Học đặc trưng bất biến | tìm tín hiệu không phụ thuộc kiến trúc | khó, phải đúng tín hiệu |
| Chuẩn hoá bằng phép biến đổi | biến ảnh thành biểu diễn "chuẩn" rồi mới phân loại | **đây là DIRE** |

Một khái niệm liên quan mà Phần 7.6 sẽ dùng: **học đường tắt** (shortcut
learning) — khi có nhiều nguồn tín hiệu, mô hình chọn nguồn **dễ học** chứ không
phải nguồn **đúng**. Đó là lý do ghép thêm RGB vào DIRE lại làm kết quả **tệ đi**.

## 2.4 Chia tập và rò rỉ dữ liệu

Chuẩn: train (cập nhật tham số) / validation (chọn siêu tham số, dừng sớm) / test
(báo cáo một lần). Rò rỉ xảy ra khi thông tin test lọt vào quá trình quyết định.

Ba dạng rò rỉ có thật trong bối cảnh này:

1. **Rò rỉ qua augmentation**: nếu `val_cfg` bị tạo **sau** khi đổi
   `cfg.dataset_root` sang `train/`, thì "validation" thực chất là đánh giá trên
   tập train. `train.py` của tác giả làm đúng: `get_val_cfg(..., copy=True)` được
   gọi **trước**. Đây là lỗi kinh điển và code này tránh được.
2. **Rò rỉ qua ảnh thật dùng chung**: trong bộ dữ liệu phát hành, nhánh
   `lsun_bedroom/lsun_bedroom/real` và `lsun_bedroom/imagenet/real` là **cùng
   1000 ảnh** (đã kiểm bằng so từng byte). Cộng hai nhánh vào một bảng trung bình
   là **đếm trùng** — không có exception nào báo, chỉ có bảng số trông hơi lạ.
3. **Rò rỉ qua chọn ngưỡng**: chọn ngưỡng trên test rồi báo cáo trên test là tự
   lừa. Ngưỡng phải chọn trên validation.

## 2.5 Sáu độ đo, và khi nào dùng cái nào

Với `s = P(fake)` và nhãn `y ∈ {0,1}`:

- **ACC** tại ngưỡng `t`: `(1/N) Σ 1[(s_i > t) = (y_i = 1)]`. Bài báo dùng
  `t = 0.5`, so sánh **nghiêm ngặt** (`>`), nên `s = 0.5` được tính là **thật**.
- **TPR** (recall) `= TP/(TP+FN)`: tỉ lệ ảnh giả bắt được.
- **FPR** `= FP/(FP+TN)`: tỉ lệ ảnh thật bị buộc tội oan.
- **ROC** là đường TPR theo FPR khi quét mọi ngưỡng; **AUC** là diện tích dưới nó.
- **AP** (average precision) là diện tích dưới đường precision–recall:

$$[6] \mathrm{AP} \;=\; \sum_{k} \big(R_k - R_{k-1}\big)\, P_k $$

với `P_k`, `R_k` là precision và recall tại ngưỡng thứ `k`. Chi tiết dễ sai:
sklearn gộp các điểm có **cùng score** thành một ngưỡng. Tự cài đặt mà không gộp
sẽ lệch vài phần nghìn — kho này cài đúng biến thể đó và có test đối chiếu
(`src/dire_modal/eval.py::_average_precision`).

**Vì sao bài báo báo cả ACC và AP.** ACC phụ thuộc một lựa chọn (ngưỡng 0,5), AP
thì không. Ở mức 99–100% ACC đã bão hoà, AP còn phân biệt được mô hình nào **xếp
hạng** tốt hơn. Trong ứng dụng thật, người ta còn dùng `TPR@FPR=1%` vì chi phí
buộc tội oan cao hơn chi phí bỏ sót.

**Đọc con số cho đúng.** Test chỉ có 1 000 ảnh mỗi bộ sinh. Với $p \approx 0{,}99$, sai số
chuẩn của tỉ lệ là $\sqrt{0{,}99\cdot 0{,}01/1000} \approx 0{,}3\%$, nên khoảng tin cậy 95% rộng
khoảng ±0,6 điểm. Chênh lệch 99,7 so với 100,0 **không có ý nghĩa thống kê**. Bài
báo không báo khoảng tin cậy, và đó là một hạn chế (Phần 9.5).

## 2.6 Mất cân bằng lớp và nguyên tắc "thiếu thì ghi thiếu"

Nếu lớp giả vắng mặt (đường dẫn sai, nhãn sai), AP không xác định nhưng nhiều cài
đặt vẫn trả một con số trông hợp lý. Vì vậy mọi dòng kết quả trong kho này ghi
kèm `n_real` và `n_fake`, và cặp lỗi thì để **trống** chỉ số chứ không ghi 0.

::: key
Nguyên tắc: **thiếu số thì ghi thiếu, không điền 0**. Một số 0 trông giống "kém"
sẽ bị đọc là kết quả; một ô trống buộc người đọc phải hỏi.
:::

# Phần 3 — Mạng tích chập và ResNet-50

## 3.1 Lớp tích chập trong thực tế

Một lớp conv có `C_out` hạt nhân, mỗi hạt nhân kích thước `k × k × C_in`. Số tham
số: `k² · C_in · C_out + C_out` (nếu có bias). Với `k = 3`, `C_in = C_out = 256`:
`3²·256·256 ≈ 590 000` tham số cho **một** lớp — lý do mạng sâu cần "nút cổ chai".

Kích thước đầu ra:

$$ H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2p - k}{s} \right\rfloor + 1 $$

Ba siêu tham số này (`kernel`, `padding`, `stride`) đủ để kiểm mọi lỗi shape.
Ví dụ lớp đầu của ResNet-50: `k=7, stride=2, padding=3` trên 224 ⇒ `(224+6−7)/2+1
= 112`.

## 3.2 Batch normalization và kết nối tắt

**BatchNorm** chuẩn hoá đầu ra của lớp theo thống kê của **cả batch**:

$$[7] \hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad y = \gamma\,\hat{x} + \beta $$

`γ`, `β` là tham số học được, cho phép mạng "hoàn tác" chuẩn hoá nếu cần. Tác
dụng: ổn định phân phối đầu vào các tầng sau, cho phép learning rate lớn hơn.

Hai lưu ý thực hành:

- Khi `model.eval()`, BatchNorm dùng **running statistics** (trung bình trượt
  trong lúc train), không dùng batch hiện tại. **Quên `.eval()` là lỗi im
  lặng**: điểm thay đổi theo batch size, và không có cảnh báo nào.
- BatchNorm làm mỗi mẫu phụ thuộc các mẫu khác trong batch — điều này ảnh hưởng
  đến diễn giải, nhưng không ảnh hưởng đến kết quả nếu batch đủ lớn.

**Kết nối tắt** (residual): $y = x + F(x)$. Vì đạo hàm của `x` theo `x` bằng 1,
gradient chảy thẳng qua mọi tầng, nên train được 50–152 tầng mà gradient không
triệt tiêu. Đây là ý tưởng cốt lõi của ResNet [[14]].

## 3.3 ResNet-50, đọc từ mã

Khối cơ bản của ResNet-50 là **Bottleneck**: ba lớp conv `1×1 → 3×3 → 1×1` thay vì
hai lớp `3×3`. Số kênh giữa bị nén, nên số tham số giảm mạnh mà vẫn giữ trường
tiếp nhận `3×3`:

```python
class Bottleneck(nn.Module):
    expansion = 4                      # kênh ra = 4 × kênh giữa

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)        # nén chiều
        self.conv2 = conv3x3(planes, planes, stride)  # tích chập 3×3, có thể /2
        self.conv3 = conv1x1(planes, planes * 4)      # bung lại ×4
        self.bn1, self.bn2, self.bn3 = (BatchNorm2d(planes),
                                        BatchNorm2d(planes),
                                        BatchNorm2d(planes * 4))
        self.downsample = downsample     # nhánh tắt, dùng khi đổi số kênh
```

Khởi tạo và `forward`:

```python
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64,  layers[0])            # 3 khối
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 4 khối
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 6 khối
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 3 khối
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)          # 2048 → C

    def forward(self, x, *args):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)        # (N, 2048, 1, 1) → (N, 2048)
        return self.fc(x)                # (N, num_classes)
```

Tên "50" đến từ số tầng có tham số: `3+4+6+3 = 16` khối, mỗi khối 3 conv
⇒ `16×3 + 1 (conv1) + 1 (fc) = 50`.

Kích thước tensor qua từng tầng với đầu vào `224×224`:

| Tầng | Đầu ra | Ghi chú |
|---|---|---|
| `conv1` 7×7, stride 2 | 112×112×64 | giảm một nửa |
| `maxpool` 3×3, stride 2 | 56×56×64 | giảm một nửa |
| `layer1` | 56×56×256 | 3 khối, không đổi kích thước |
| `layer2` | 28×28×512 | stride 2 ở khối đầu |
| `layer3` | 14×14×1024 | |
| `layer4` | 7×7×2048 | |
| `avgpool` → `fc` | 2048 → 1 | một logit cho bài toán nhị phân |

Tổng tham số: khoảng **25,6 triệu** (97,8 MB ở fp32). Checkpoint của tác giả là
**269,4 MB** vì lưu thêm optimizer state và metadata. Con số này giải thích tại
sao mỗi lần lưu checkpoint "best" lại tốn I/O đáng kể (Phần 10.5).

Một chi tiết trong mã tác giả: tham số `zero_init_residual=False` nên khối
residual **không** được khởi tạo bằng 0 ở `bn3.weight`. Khởi tạo 0 làm khối bắt
đầu như hàm đồng nhất, thường giúp train sâu dễ hơn; ở đây không dùng. Không sai,
chỉ là khác mặc định của bài báo ResNet gốc.

## 3.4 Transfer learning và quy ước một logit

Với 80k ảnh, học 25,6 triệu tham số từ đầu là quá ít dữ liệu. Cách làm của tác
giả (`vendor/utils/utils.py::get_network`):

```python
def get_network(arch, num_classes=1, pretrained=True):
    model = resnet50(pretrained=pretrained)      # trọng số ImageNet [[16]]
    if num_classes == 1:                         # nhánh train
        model.fc = nn.Linear(2048, 1)
        nn.init.normal_(model.fc.weight, 0.0, 0.02)
    return model
```

Trực giác: các tầng đầu học đặc trưng thị giác phổ quát — biên, texture, hình
khối — dùng lại được cho một "ảnh" mới là DIRE map. Chỉ lớp cuối cần học khái
niệm thật/giả.

Khởi tạo `normal_(0, 0.02)` cho lớp mới là quan trọng: nếu lớp `fc` khởi tạo
ngẫu nhiên lớn, gradient đầu tiên sẽ lớn và có thể phá các đặc trưng đã học ở
những bước đầu.

**Một điểm dễ nhầm trong nhánh đánh giá.** Khi load checkpoint để chấm điểm, mã
tác giả gọi `resnet(num_classes=1)` **không** pretrained, rồi nạp checkpoint đè
lên. Nạp pretrained trước là vô ích (tốn thời gian, và nguy hiểm nếu shape khác),
nhưng không sai vì checkpoint phủ toàn bộ tham số.

## 3.5 Vì sao là ResNet-50 chứ không phải mạng lớn hơn

Bài báo không giải thích. Đọc mã thì thấy `arch = "resnet50"` là mặc định, và
CNNDetection [[11]] — nguồn gốc của khung thực nghiệm này — cũng dùng ResNet-50.
Đây là lựa chọn "đủ tốt, đủ rẻ" chứ không phải kết quả so sánh kiến trúc.

Với người muốn nghiên cứu tiếp, đây là một biến **chưa được khảo sát** và khá dễ
làm: khi đầu vào là DIRE map, kiến trúc classifier có quan trọng không? Nếu
ResNet-18 cho cùng kết quả, chi phí train giảm 3–4 lần. Nếu ViT tốt hơn, có nghĩa
thông tin nằm ở quan hệ toàn cục chứ không ở texture cục bộ — một phát hiện có ý
nghĩa cơ chế, không chỉ là con số.
