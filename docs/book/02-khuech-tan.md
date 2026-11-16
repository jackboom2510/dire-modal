# Phần 4 — Ba họ mô hình sinh, và vì sao khuếch tán khác về bản chất

## 4.1 GAN, VAE, khuếch tán: ba cách "nghĩ" khác nhau

| Họ | Cơ chế | Số lần biến đổi để ra ảnh | Dấu vết để lại |
|---|---|---|---|
| **GAN** | bộ sinh `G` và bộ phân biệt `D` đối kháng; `G` biến nhiễu `z` thành ảnh trong **một** lượt | 1 | phép phóng to (transposed conv) tạo cấu trúc tuần hoàn trong phổ |
| **VAE** | bộ mã hoá nén ảnh vào không gian ẩn, bộ giải mã dựng lại; tối ưu ELBO | 1 | ảnh thường mờ, thiếu chi tiết tần số cao |
| **Khuếch tán** | học cách khử nhiễu lặp lại `T` lần, từ nhiễu trắng tới ảnh | 20–1000 | dấu vết nằm ở **quỹ đạo khử nhiễu**, không ở một phép biến đổi |

Điểm khác biệt cốt lõi: GAN tạo ảnh bằng **một** phép ánh xạ; khuếch tán tạo ảnh
bằng **một quỹ đạo** gồm nhiều bước nhỏ. Hệ quả kép:

- Detector dựa trên dấu vết của GAN **không chuyển được** sang khuếch tán, vì
  không có phép phóng to tương ứng để bắt. Đây chính là lý do các baseline trong
  bài báo tụt từ 86–91% xuống 51–59% khi chuyển sang ảnh khuếch tán.
- Chính tính lặp lại đó mở ra tín hiệu mới: **ảnh do khuếch tán sinh ra nằm "gần"
  quỹ đạo mà mô hình khuếch tán đã học**, nên khi cho nó đi qua một mô hình khuếch
  tán khác cùng họ, nó được tái tạo tốt hơn ảnh thật.

## 4.2 Trực giác trước khi vào công thức

Tưởng tượng một chiếc máy nén–giải nén ảnh theo "ngôn ngữ" của một mô hình khuếch
tán cụ thể. Ảnh mà chiếc máy đó **đã học** thì nén–giải nén gần như hoàn hảo.
Ảnh lạ thì mất mát nhiều.

Nhưng "gần phân phối train" chưa đủ để giải thích kết quả, vì ảnh thật LSUN-Bedroom
cũng nằm trong phân phối mà ADM học (ADM được train trên chính LSUN-Bedroom).
Vậy tại sao ảnh **thật** LSUN vẫn khó tái tạo hơn ảnh **ADM sinh**?

Bài báo đưa ra hai tầng giải thích (Phần 6.2 phân tích kỹ):

1. **Tầng phân phối:** ảnh sinh nằm đúng trên đa tạp mà quá trình sinh định nghĩa.
2. **Tầng rời rạc hoá (Phụ lục B):** vì `T` hữu hạn, công thức DDIM chỉ là xấp xỉ
   Euler cho một ODE, và sai số xấp xỉ **lớn hơn ở ảnh thật** vì quỹ đạo của ảnh
   thật "cong" hơn.

Tầng thứ hai là phần thú vị nhất về mặt lý thuyết, và cũng là chỗ yếu nhất của
bài báo, vì nó là lập luận định tính không có thí nghiệm trực tiếp.

# Phần 5 — Khuếch tán: quá trình thuận, hàm mất mát, kiến trúc

## 5.1 Quá trình thuận: thêm nhiễu có kiểm soát

Chọn một dãy `β_1, …, β_T` (với ADM: tuyến tính từ `1e-4` đến `0.02`, `T = 1000`),
đặt `α_t = 1 − β_t` và `ᾱ_t = α₁·α₂···α_t` (tích luỹ). Một bước thêm nhiễu:

$$[8] q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t;\ \sqrt{\alpha_t}\,x_{t-1},\ (1-\alpha_t)\,\mathbf{I}\right) $$

Nghĩa là: co ảnh cũ lại một chút (`√α_t`) rồi cộng nhiễu Gauss phương sai
`1−α_t`. Vì đây là Markov và Gauss, có thể **nhảy cóc** tới bước bất kỳ:

$$[9] q(x_t \mid x_0) = \mathcal{N}\left(x_t;\ \sqrt{\bar{\alpha}_t}\,x_0,\ (1-\bar{\alpha}_t)\,\mathbf{I}\right) $$

hay viết dưới dạng mẫu (chỉ cần một lần lấy nhiễu `ε`):

$$[10] x_t = \sqrt{\bar{\alpha}_t}\; x_0 + \sqrt{1-\bar{\alpha}_t}\; \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \mathbf{I}) $$

Công thức [10] là công thức quan trọng nhất của cả tài liệu. Ba hệ quả:

1. `ᾱ_t` **giảm** theo `t`: ảnh mất dần, nhiễu chiếm dần. Bảng dưới là số tính
   thật với lịch tuyến tính của ADM.
2. Đảo công thức [10] được: $x_0 = \big(x_t - \sqrt{1-\bar{\alpha}_t}\,\varepsilon\big)/\sqrt{\bar{\alpha}_t}$.
   Nếu biết $\varepsilon$ thì biết ảnh sạch.
3. Vì `x_t` được xác định hoàn toàn bởi cặp `(x₀, ε)`, bài toán "sinh ảnh" trở
   thành bài toán "**đoán nhiễu**".

Bảng `ᾱ_t` với lịch tuyến tính `β: 1e-4 → 0.02`, `T = 1000` (số tính từ công thức,
không phải từ bài báo):

| `t` | 0 | 50 | 100 | 250 | 500 | 750 | 950 | 999 |
|---|---|---|---|---|---|---|---|---|
| `ᾱ_t` | 0,9999 | 0,9700 | 0,8951 | 0,5214 | 0,0778 | 0,00330 | **1,06·10⁻⁴** | 4,04·10⁻⁵ |
| `√ᾱ_t` | 0,9999 | 0,985 | 0,946 | 0,722 | 0,279 | 0,0574 | 0,0103 | 0,0064 |

Hai điều rút ra từ bảng, sẽ dùng lại ở Phần 6 và Phần 8:

- Ở `t = 500`, ảnh vẫn còn giữ **7,8%** năng lượng tín hiệu. "Nửa đường" của quá
  trình nhiễu không phải là "nửa thông tin".
- Ở `t = 950` (mốc cuối khi dùng `ddim20`), ảnh còn **1,03%** tín hiệu, **không
  phải 0%**. Nghĩa là phép đảo ngược không đưa ảnh về "nhiễu trắng hoàn toàn", mà
  về một điểm rất gần nhiễu trắng nhưng vẫn mang dấu vết của ảnh ban đầu. Chi
  tiết này chính là cơ chế khiến phép đảo ngược **khôi phục được** ảnh gốc.

## 5.2 Huấn luyện: đoán nhiễu thay vì đoán ảnh

Mạng `ε_θ(x_t, t)` nhận ảnh nhiễu và bước thời gian, trả về nhiễu đã được thêm.
Hàm mất mát:

$$[11] \mathcal{L}(\theta) = \mathbb{E}_{t,\,x_0,\,\varepsilon}\left[\, \left\| \varepsilon - \varepsilon_\theta\left(\sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\varepsilon,\ t\right) \right\|^2 \,\right] $$

Cách đọc: lấy ảnh thật, thêm nhiễu theo [10], bắt mạng đoán lại đúng lượng nhiễu
đó. Không cần nhãn, không cần dữ liệu cặp — đây là **học tự giám sát**
(self-supervised): nhãn chính là nhiễu ta vừa thêm.

Vì sao đây là mất mát hợp lý: nó là bản rút gọn (bỏ trọng số theo `t`) của **ELBO**
— cận dưới biến phân của log-likelihood. Dạng đầy đủ có trọng số cho mỗi `t`;
DDPM [[3]] nhận thấy bỏ trọng số cho ảnh sinh chất lượng hơn, dù mất tính "đúng về
xác suất". ADM cũng dùng dạng rút gọn.

Một điểm lý thuyết mà người mới hay bỏ qua: mạng không cần đoán **ảnh sạch**, chỉ
cần đoán **nhiễu**. Hai việc tương đương về toán (vì biết `ε` là suy ra `x₀` theo
[10]), nhưng khác nhau về **điều kiện số**: đoán nhiễu ở `t` lớn dễ hơn đoán ảnh,
vì nhiễu có phân phối chuẩn hoá, còn ảnh sạch ở `t` lớn gần như không còn thông
tin để đoán. Đây là lý do thực nghiệm DDPM chọn dự đoán `ε`.

Mã của ADM còn cho phép ba kiểu dự đoán khác (`ModelMeanType`):

```python
class ModelMeanType(enum.Enum):
    PREVIOUS_X = enum.auto()   # mô hình đoán x_{t-1}
    START_X    = enum.auto()   # mô hình đoán x_0
    EPSILON    = enum.auto()   # mô hình đoán epsilon  ← ADM dùng cái này
```

Và hàm đổi từ `x̂₀` sang `ε` (dùng cả trong lúc lấy mẫu, xem 5.6):

$$[12] \hat{\varepsilon} = \frac{x_t - \sqrt{\bar{\alpha}_t}\,\hat{x}_0}{\sqrt{1-\bar{\alpha}_t}} $$

Trong mã, công thức này được viết ở dạng tương đương nhưng tránh chia cho số nhỏ:

```python
eps = (sqrt_recip_alphas_cumprod[t] * x - pred_xstart) / sqrt_recipm1_alphas_cumprod[t]
#     (1/√ᾱ_t)·x − x̂₀                     ÷ √(1/ᾱ_t − 1)
```

Hai dạng bằng nhau vì `√(1−ᾱ)/√ᾱ = √(1/ᾱ − 1)`. Chỉ là biến đổi đại số, nhưng
dạng trong mã ổn định hơn khi `ᾱ` rất nhỏ (khi đó `√(1/ᾱ)` không tràn).

## 5.3 Kiến trúc ADM: U-Net có attention

Mô hình `ε_θ` mà DIRE dùng là ADM [[4]] — một U-Net:

- **Bộ mã hoá** giảm dần độ phân giải (256 → 128 → 64 → 32 → 16 → 8), tăng số
  kênh. **Bộ giải mã** làm ngược lại. Có **kết nối tắt giữa hai phía** ở mỗi độ
  phân giải, nên thông tin chi tiết không bị mất qua cổ chai.
- **Attention** ở các độ phân giải 32, 16, 8 (`attention_resolutions="32,16,8"`):
  mỗi điểm ảnh tổng hợp thông tin từ mọi điểm khác trong bản đồ đặc trưng. Đây là
  cơ chế mô hình hoá quan hệ **toàn cục** — cần thiết để giữ bố cục phòng ngủ hợp
  lý, không chỉ texture.
- **Nhúng thời gian** (timestep embedding): `t` được mã hoá thành vector và đưa vào
  mọi khối, để cùng một mạng xử lý được 1000 mức nhiễu khác nhau.
- `learn_sigma=True`: mạng ra **6 kênh** (3 cho trung bình, 3 cho log-phương sai)
  thay vì 3.
- `class_cond=False` trong cấu hình DIRE: mô hình không điều kiện hoá theo lớp,
  nên ảnh sinh là ảnh "phòng ngủ nói chung", không có nhãn.

Ba hyperparameter này (`num_channels=256`, `num_res_blocks=2`,
`num_head_channels=64`) quyết định kích thước mô hình — khoảng **552 triệu** tham
số, tức khoảng 1,1 GB ở fp32. Đó là lý do checkpoint `256x256_diffusion_uncond.pt`
nặng khoảng **2,1 GB** (kèm optimizer state và metadata).

## 5.4 Lấy mẫu: DDPM có nhiễu, DDIM không nhiễu

**DDPM** lấy mẫu bằng cách lặp từ `t = T` về `t = 1`:

```
x_{t-1} ~ N( μ_θ(x_t, t), Σ_θ(x_t, t) )
```

Có thành phần ngẫu nhiên, nên mỗi lần chạy cho một ảnh khác. Cần `T` bước, với
`T = 1000` là 1000 lượt chạy mạng.

**DDIM** [[5]] đặt phương sai bằng 0 ở mỗi bước:

$$[13] x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\;\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}}\;\hat{\varepsilon} $$

Không còn nhiễu ngẫu nhiên ⇒ ánh xạ **tất định**: cùng một `x_T` luôn cho cùng một
`x_0`. Đây là tính chất mà DIRE sống nhờ. Nếu quá trình còn ngẫu nhiên,
`R(I(x))` sẽ khác nhau mỗi lần chạy và DIRE trở thành nhiễu không tái lập.

DDIM cũng cho phép dạng có tham số `η`:

$$[14] \sigma_t = \eta\,\sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}}\;\sqrt{1-\frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}} $$

`η = 0` ⇒ DDIM tất định; `η = 1` ⇒ tương đương DDPM. Mã của tác giả gọi
`ddim_sample(..., eta=0.0)`, và hàm `ddim_reverse_sample` có `assert eta == 0.0`
kèm thông báo "Reverse ODE only for deterministic path" — tức phép đảo ngược
**chỉ** hợp lệ ở nhánh tất định.

## 5.5 Từ rời rạc tới ODE, và vì sao phép đảo ngược tồn tại

Khi số bước `T → ∞`, quá trình DDIM tiến tới nghiệm của một phương trình vi phân
thường (ODE) [[6]][[17]]:

$$[15] \mathrm{d}\bar{x}(t) = \varepsilon_\theta\left(\frac{\bar{x}(t)}{\sqrt{\sigma^2+1}},\ t\right)\mathrm{d}\sigma(t), \qquad \bar{x} = \frac{x}{\sqrt{\bar{\alpha}}},\ \ \sigma = \sqrt{\frac{1-\bar{\alpha}}{\bar{\alpha}}} $$

Hai tính chất của ODE là nền tảng của DIRE:

1. **Tất định**: một điều kiện đầu cho đúng một quỹ đạo.
2. **Thuận nghịch**: nếu biết trường vector, có thể đi theo cả hai chiều. Vì thế
   có thể tìm `x_T` từ `x_0` — gọi là **phép đảo ngược** (inversion):

$$[16] x_{t+1} = \sqrt{\bar{\alpha}_{t+1}}\;\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t+1}}\;\hat{\varepsilon} $$

So sánh [13] và [16]: cùng một công thức, chỉ khác **chiều** — [13] đi từ `t` về
`t−1` (dùng `ᾱ_{t−1}`, tức mốc trước), [16] đi từ `t` lên `t+1` (dùng `ᾱ_{t+1}`,
tức mốc sau). Đây là điểm dễ nhầm nhất khi đọc mã, vì `ddim_sample` và
`ddim_reverse_sample` chỉ khác nhau đúng một biến:

```python
# gaussian_diffusion.py — ddim_sample (đi xuống)
alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
mean_pred = out["pred_xstart"] * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev - sigma**2) * eps

# gaussian_diffusion.py — ddim_reverse_sample (đi lên)
alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)
mean_pred = out["pred_xstart"] * th.sqrt(alpha_bar_next) + th.sqrt(1 - alpha_bar_next) * eps
```

`alphas_cumprod_prev` chứa `ᾱ_{t−1}`, `alphas_cumprod_next` chứa `ᾱ_{t+1}` — hai
mảng được dựng sẵn trong `__init__` của `GaussianDiffusion`:

```python
self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
```

Lưu ý `next` được nối thêm `0.0` ở cuối: ở `t = T−1`, `ᾱ_T` không tồn tại nên đặt
bằng 0 (tương ứng "nhiễu hoàn toàn"). Phép đảo ngược ở bước cuối cùng vì thế dồn
ảnh về gần nhiễu trắng nhất có thể.

### Vì sao đảo ngược không hoàn hảo — ba nguồn sai số

Đây là phần cần hiểu để không kỳ vọng `R(I(x)) = x`:

1. **Sai số rời rạc hoá.** Ta chạy `S = 20` mốc thay vì 1000. Công thức [16] là
   một bước Euler; sai số cục bộ tỉ lệ với bình phương khoảng cách giữa các mốc,
   và sai số tích luỹ phụ thuộc độ cong của quỹ đạo.
2. **Phép cắt `clip_denoised`.** Mã của tác giả đặt `clip_denoised=True`, nghĩa là
   `x̂₀` bị cắt vào `[−1, 1]` ở **mỗi** bước. Phép cắt là **phi tuyến**, nên nó
   phá tính thuận nghịch chính xác của ODE: thông tin bị cắt đi không thể tái tạo.
3. **Số học fp16.** Khi bật `use_fp16`, trọng số và phép tính ở nửa độ chính xác;
   sai số làm tròn cỡ `10⁻³` tương đối, tức vài phần nghìn trên thang `[−1,1]`.

Ba nguồn này cộng lại tạo ra một "sàn" sai số mà **cả ảnh thật và ảnh sinh đều
phải chịu**. DIRE hoạt động được vì phần **vượt trên sàn** khác nhau giữa hai loại
ảnh — không phải vì một loại có sai số bằng 0.

## 5.6 Rút gọn số bước: respacing và chi phí thật

Chạy 1000 bước cho mỗi ảnh là quá đắt. DDIM cho phép chọn một **tập con** các mốc
`τ₁ < … < τ_S` trong `{0, …, T−1}`. Trong guided-diffusion, đây là `SpacedDiffusion`
với `space_timesteps(1000, "20")`:

```python
stride = num_timesteps // section_counts      # 1000 // 20 = 50
used = list(range(0, num_timesteps, stride))  # 0, 50, 100, …, 950
```

Tập mốc là `{0, 50, 100, …, 950}` — **20 phần tử**, mốc lớn nhất là **950**, không
phải 999. Hai hệ quả định lượng:

- **Chi phí:** mỗi ảnh cần $S$ lượt cho đảo ngược + $S$ lượt cho tái tạo = $2S$
  lượt chạy mạng. Với `S = 20`: **40 NFE/ảnh** (NFE = number of function
  evaluations). Đây là hoá đơn thật của DIRE, lớn hơn nhiều so với 1 lượt ResNet-50.
- **Điểm bắt đầu tái tạo:** vì mốc cuối là 950, `x_950` vẫn còn `√ᾱ_950 ≈ 1,03%`
  tín hiệu. Nếu bạn tự cài đặt và dùng mốc 999, `√ᾱ_999 ≈ 0,64%` — khác một chút,
  nhưng **khác mốc là khác biểu diễn**, nên DIRE map sẽ không trùng với map phát
  hành dù trông giống nhau.

::: warn
`S` không phải một tham số truyền vào hàm sinh. Nó là **kích thước của tập mốc**
được dựng khi tạo `SpacedDiffusion`. Truyền `real_step=20` mà vẫn dùng đối tượng
1000 bước sẽ chạy **20 bước đầu của thang 1000** (mốc 0…19), tức một quỹ đạo hoàn
toàn khác — không phải "20 mốc rải đều". Đây là lỗi cài đặt đã có thật trong kho
này (Phần 11.3, lỗi F2).
:::

## 5.7 Lịch nhiễu: tuyến tính, cosine, và hệ quả lên DIRE

Cách chọn dãy `β_t` **không** phải chi tiết phụ. Nó quyết định:

- bao nhiêu "thời gian" dành cho việc phá cấu trúc ảnh so với việc phá chi tiết;
- giá trị `ᾱ_t` ở mốc cuối của tập respacing (tức điểm bắt đầu tái tạo);
- do đó, **DIRE map của hai mô hình dùng hai lịch khác nhau không so sánh trực tiếp
  được**.

### Hai lịch phổ biến

**Tuyến tính** (DDPM, ADM):

$$ \beta_t = \beta_{\min} + \frac{t-1}{T-1}\left(\beta_{\max}-\beta_{\min}\right), \qquad \beta_{\min}=10^{-4},\ \beta_{\max}=0{,}02 $$

**Cosine** (iDDPM [[8]]): định nghĩa trực tiếp qua `ᾱ` thay vì `β`:

$$ \bar{\alpha}_t = \frac{f(t)}{f(0)}, \qquad f(t) = \cos^{2}\!\left(\frac{t/T + s}{1+s}\cdot\frac{\pi}{2}\right), \quad s = 0{,}008 $$

và `β_t = 1 − ᾱ_t/ᾱ_{t−1}`.

### So sánh định lượng (số tính từ công thức, `T = 1000`)

| `t` | `ᾱ_t` tuyến tính | `ᾱ_t` cosine |
|---|---|---|
| 250 | 0,521 | 0,847 |
| 500 | 0,078 | 0,494 |
| 750 | 0,0033 | 0,144 |

Đọc bảng: lịch tuyến tính **phá ảnh quá nhanh ở nửa đầu** — đến `t = 500` đã mất
92% tín hiệu — rồi dành nửa sau cho việc "dọn" gần như không còn gì. Cosine giữ
tín hiệu lâu hơn, phân bố công việc đều hơn. iDDPM chứng minh cosine cho ảnh sinh
tốt hơn; đây là một trong những khác biệt lớn giữa DDPM và iDDPM.

### Ba hệ quả trực tiếp lên DIRE

1. **`√ᾱ` ở mốc cuối khác nhau.** Với `ddim20`, mốc cuối là `t = 950`:
   `√ᾱ_950 ≈ 0,0103` (tuyến tính). Nếu mô hình dùng cosine, cùng mốc đó có
   `√ᾱ_950` lớn hơn **nhiều lần** ⇒ `x_T` còn giữ nhiều tín hiệu hơn ⇒ phép đảo ngược
   "dễ" hơn ⇒ DIRE có xu hướng **nhỏ hơn** ở mọi ảnh. Nói cách khác: **thang đo DIRE
   phụ thuộc lịch nhiễu**.
2. **DIRE map của ADM và của iDDPM không cùng thang.** Đây là lý do khi làm thí
   nghiệm E2 (thay ADM bằng iDDPM làm `I`/`R`, Phần 12.2) phải **huấn luyện lại**
   detector, và phải chuẩn hoá lại ngưỡng; không thể dùng checkpoint cũ.
3. **So sánh giữa các bài báo phải kiểm lịch.** Nếu một phương pháp báo "DIRE với
   `S = 20`" mà không nói lịch nhiễu, con số chưa đủ để tái lập. Trong bài báo DIRE,
   lịch là tuyến tính (mặc định của `guided-diffusion`), và điều này chỉ đọc được từ
   **mã**, không có trong bài.

::: warn
Một cái bẫy khi tự cài: `β` được khai theo `T = 1000`, nhưng khi dùng respacing 20
mốc, **không** được tính lại `β`. `SpacedDiffusion` chỉ **chọn** các mốc từ lịch đã
tính. Tính lại `β` cho 20 bước (tức giãn lịch) là một thay đổi hoàn toàn khác, và sẽ
cho DIRE map khác hẳn — dù "chỉ là 20 bước" trong cả hai trường hợp.
:::

# Phần 6 — Vì sao ảnh sinh "dễ tái tạo": hai tầng giải thích

## 6.1 Tầng phân phối

Ảnh do quá trình sinh tạo ra nằm trên quỹ đạo do chính phép sinh định nghĩa. Nếu
ta đảo ngược bằng **cùng một phép** (cùng mô hình, cùng lịch, cùng số mốc), ta
đang đi ngược lại đúng quỹ đạo đó, nên khôi phục gần hoàn hảo.

Ảnh thật không nằm trên quỹ đạo đó. Nó chỉ "gần" phân phối mà mô hình học. Khi đảo
ngược, mô hình phải **xấp xỉ** một điểm ngoài quỹ đạo quen thuộc, và sai số lớn
hơn.

Lập luận này có một lỗ hổng rõ ràng: ADM được train trên LSUN-Bedroom, nên ảnh
thật LSUN **thuộc** phân phối mà ADM học. Nếu lập luận trên là đủ, DIRE không phân
biệt được ảnh thật LSUN với ảnh ADM. Thực tế nó phân biệt được (100% ACC với
`n = 1000` mỗi lớp). Vậy cần tầng giải thích thứ hai.

## 6.2 Tầng rời rạc hoá (Phụ lục B của bài báo)

Bài báo lập luận: với `T` hữu hạn, DDIM là xấp xỉ Euler của ODE [15], và sai số
xấp xỉ phụ thuộc **độ cong của quỹ đạo**. Quỹ đạo của ảnh sinh "thẳng" hơn (vì nó
được tạo bởi chính lưới rời rạc đó), quỹ đạo của ảnh thật "cong" hơn, nên sai số
lớn hơn. Do đó `|x − R(I(x))|` lớn hơn với ảnh thật.

Đây là lập luận **hợp lý nhưng chưa được chứng minh**, và có ba vấn đề:

1. Không có thí nghiệm trực tiếp đo độ cong hay sai số Euler. Bài báo chỉ đo kết
   quả cuối (ACC/AP) và một ablation về `S` (Table 4), trong đó `S` càng lớn thì
   kết quả càng tốt — **phù hợp** với giả thuyết nhưng không **chứng minh** nó, vì
   tăng `S` cũng làm mọi thứ khác tốt lên.
2. Có một cơ chế khác không được xét: **phép cắt `[−1,1]`** (Phần 5.5). Ảnh thật
   đã nằm sát biên `[−1,1]` (vì pixel 0 và 255 là phổ biến ở ảnh chụp), nên phép
   cắt tác động mạnh hơn lên ảnh thật. Cơ chế này giải thích cùng hiện tượng mà
   không cần đến độ cong quỹ đạo.
3. Bản thân "ảnh sinh có quỹ đạo thẳng" cũng chỉ đúng khi ảnh sinh ra từ **cùng
   mô hình, cùng lịch, cùng số mốc**. Với ảnh SD-v1 (mô hình 512×512, train trên
   LAION), quỹ đạo khác hẳn, vậy tại sao DIRE vẫn đạt 99,7%? Lập luận Phụ lục B
   **không** giải thích được kết quả cross-generator, mà đó lại là kết quả mạnh
   nhất của bài báo.

Kết luận trung thực: hiện tượng được xác nhận bằng thực nghiệm rất vững chắc
(100% ACC, 5 seed không báo), nhưng **cơ chế chưa được chứng minh**. Đây là một
khoảng trống nghiên cứu có thật, và Phần 9.7 sẽ nêu nó như câu hỏi mở.

## 6.3 Một cách nhìn khác: DIRE như "khoảng cách tới đa tạp"

Có thể phát biểu lại DIRE mà không cần ODE: mô hình khuếch tán định nghĩa một
**đa tạp** (manifold) các ảnh mà nó có thể tạo ra. Phép `R ∘ I` là phép **chiếu**
ảnh lên đa tạp đó (gần đúng, qua 20 bước). Khi đó `|x − R(I(x))|` là **khoảng cách
từ ảnh tới đa tạp**.

Cách nhìn này giải thích được nhiều quan sát hơn Phụ lục B:

- Ảnh thật nằm **ngoài** đa tạp của ADM (vì ADM chỉ học LSUN, không học ảnh chụp
  nói chung) ⇒ khoảng cách lớn.
- Ảnh sinh bởi bất kỳ mô hình khuếch tán nào cũng nằm **gần** đa tạp của ADM, vì
  mọi quá trình khuếch tán đều tạo ảnh bằng cách khử nhiễu từ nhiễu trắng, nên
  ảnh kết quả có cấu trúc phổ và thống kê cục bộ giống nhau ⇒ khoảng cách nhỏ.
- Điều này giải thích được tổng quát hoá sang SD-v1 mà Phụ lục B không giải thích.

Tuy vậy, đây là **diễn giải của tài liệu này**, không phải phát biểu của bài báo.
Bài báo không dùng ngôn ngữ đa tạp và không có thí nghiệm nào kiểm trực tiếp. Cần
ghi rõ như vậy để không biến một diễn giải thành một kết quả.

::: note
Cách kiểm được giả thuyết "khoảng cách tới đa tạp": lấy ảnh thật, chiếu lên đa tạp
bằng `R(I(·))` với `S` lớn (ví dụ 200), rồi đo khoảng cách còn lại. Nếu khoảng cách
giảm mạnh khi `S` tăng, đó là bằng chứng cho thành phần "sai số rời rạc hoá"; nếu
không giảm, đó là bằng chứng cho "ảnh thật nằm ngoài đa tạp". Bài báo không làm thí
nghiệm này.
:::
