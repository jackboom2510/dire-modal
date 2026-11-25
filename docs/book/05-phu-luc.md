# Phụ lục A — Thuật ngữ Anh–Việt

## A.1 Xử lý ảnh và mạng tích chập

| English | Tiếng Việt | Ghi chú một dòng |
|---|---|---|
| sampling (spatial) | mẫu hoá | rời rạc hoá toạ độ không gian |
| quantization | lượng tử hoá | rời rạc hoá cường độ, thường 8 bit |
| convolution | tích chập | tổng có trọng số trên cửa sổ trượt |
| kernel / filter | hạt nhân / bộ lọc | ma trận trọng số nhỏ |
| stride / padding | bước nhảy / đệm | điều khiển kích thước đầu ra |
| pooling | gộp | giảm độ phân giải (max/avg) |
| receptive field | trường tiếp nhận | vùng ảnh gốc một nơ-ron "thấy" |
| high-pass / low-pass | thông cao / thông thấp | giữ tần số cao / thấp |
| upsampling / downsampling | tăng / giảm mẫu | phóng to / thu nhỏ |
| interpolation | nội suy | bilinear, bicubic, Lanczos… |
| aliasing | răng cưa (hiện tượng chồng phổ) | do lấy mẫu dưới ngưỡng Nyquist |
| demosaic | khử mosaic | dựng ảnh màu từ cảm biến Bayer |
| quantization table | bảng lượng tử | bảng hệ số của JPEG |
| DCT | biến đổi cosin rời rạc | phép biến đổi trong JPEG |
| re-compression | nén lại | tích luỹ dấu vết JPEG |
| bathnorm | chuẩn hoá theo batch | chuẩn hoá theo mean/var của batch |
| residual connection | kết nối tắt / tàn dư | `y = x + F(x)` |
| bottleneck | nút cổ chai | khối `1×1 → 3×3 → 1×1` |
| transfer learning | học chuyển giao | dùng trọng số pretrained |

## A.2 Học máy và đánh giá

| English | Tiếng Việt | Ghi chú một dòng |
|---|---|---|
| logit | logit | đầu ra thô trước sigmoid/softmax |
| sigmoid / softmax | — | hai cách đổi logit thành xác suất |
| binary cross-entropy (BCE) | mất mát entropy chéo nhị phân | loss cho một logit |
| overfitting | quá khớp | học cả nhiễu của tập train |
| generalization | tổng quát hoá | tốt trên dữ liệu chưa thấy |
| distribution shift | dịch chuyển phân phối | test khác train |
| data leakage | rò rỉ dữ liệu | thông tin test lọt vào train |
| shortcut learning | học đường tắt | dùng tín hiệu dễ thay vì tín hiệu đúng |
| accuracy | độ chính xác | ACC |
| precision / recall | độ chuẩn xác / độ phủ | recall = TPR |
| false positive / negative | dương tính giả / âm tính giả | báo nhầm / bỏ sót |
| ROC / AUC | đường cong ROC / diện tích | TPR theo FPR |
| average precision (AP) | độ chính xác trung bình | diện tích dưới precision–recall |
| TPR@FPR | — | tỉ lệ bắt được khi khống chế báo nhầm |
| ablation | thí nghiệm loại bỏ | bỏ một thành phần để đo vai trò |
| cross-generator | tổng quát hoá chéo bộ sinh | train một bộ, test bộ khác |
| confidence interval | khoảng tin cậy | độ bất định của một con số |

## A.3 Khuếch tán

| English | Tiếng Việt | Ghi chú một dòng |
|---|---|---|
| diffusion model | mô hình khuếch tán | sinh ảnh bằng khử nhiễu lặp lại |
| forward process | quá trình thuận | thêm nhiễu dần |
| reverse process | quá trình nghịch | khử nhiễu dần |
| Markov chain | xích Markov | trạng thái sau chỉ phụ thuộc trạng thái trước |
| isotropic Gaussian | Gauss đẳng hướng | phương sai như nhau mọi chiều |
| noise schedule | lịch nhiễu | dãy `β_t` |
| respacing | rút gọn số bước | chọn tập con các mốc |
| ε-prediction | dự đoán nhiễu | mạng ra `ε` thay vì ảnh |
| classifier-free guidance | dẫn hướng không cần classifier | tăng độ bám điều kiện |
| U-Net | — | kiến trúc mã hoá–giải mã có kết nối tắt |
| attention | cơ chế chú ý | tổng hợp thông tin toàn cục |
| timestep embedding | nhúng bước thời gian | đưa `t` vào mọi khối |
| ELBO | cận dưới biến phân | mục tiêu của mô hình sinh xác suất |
| score function | hàm điểm | gradient của log-mật độ |
| ODE / SDE | phương trình vi phân thường / ngẫu nhiên | hai cách nhìn liên tục |
| Euler discretization | rời rạc hoá Euler | bước đi bậc một |
| discretization error | sai số rời rạc hoá | do chia bước thô |
| inversion | phép đảo ngược | tìm nhiễu tương ứng một ảnh |
| deterministic | tất định | cùng đầu vào ⇒ cùng đầu ra |
| NFE | số lượt chạy mạng | thước đo chi phí |

## A.4 Năm hiểu nhầm hay gặp

1. **"DIRE là một detector."** Không hẳn. DIRE là **biểu diễn**; detector là
   ResNet-50 đứng sau. Có thể đổi classifier mà giữ nguyên DIRE.
2. **"ADM dùng để phát hiện ảnh giả."** Không. ADM là mô hình sinh bình thường, dùng
   như một phép biến đổi đặc trưng. Nó không biết gì về bài toán thật/giả.
3. **"DIRE cao = ảnh giả."** Ngược lại: DIRE **thấp** ⇒ dễ tái tạo ⇒ **ảnh giả**.
   Nhầm chiều là lỗi từng có trong chính kho này.
4. **"Dùng lại dữ liệu của tác giả là tái lập."** Không — đó là **chấm lại** (`R0`).
5. **"ACC 100% nghĩa là hoàn hảo."** Với `n = 1000`, sai số chuẩn ở mức 99% là
   khoảng 0,3%; 99,9 và 100,0 không khác nhau về ý nghĩa thống kê.

# Phụ lục B — Bảng tra: công thức ↔ khái niệm ↔ mã

| # | Nội dung | Nơi cài đặt |
|---|---|---|
| (1) | `DIRE = \|x₀ − R(I(x₀))\|` | `guided-diffusion/compute_dire.py`: `dire = th.abs(imgs - recons)` |
| (2) | tích chập 2D | `networks/resnet.py`: `conv3x3`, `conv1x1` |
| (3) | DFT | không dùng trong mã (chỉ để hiểu miền tần số) |
| (4) | sigmoid + BCE | `utils/trainer.py`: `nn.BCEWithLogitsLoss()`; `dire_core/score.py: sigmoid` |
| (5) | softmax + CE | nhánh `num_classes=2` (không dùng cho checkpoint DIRE) |
| (6) | AP | `dire_modal/eval.py: _average_precision` |
| (7) | batch normalization | `networks/resnet.py`: `nn.BatchNorm2d` |
| (8) | `q(x_t \| x_{t−1})` | `gaussian_diffusion.py`: `p_mean_variance`, `q_posterior_mean_variance` |
| (9) | `q(x_t \| x_0)` | `gaussian_diffusion.py: q_mean_variance` |
| (10) | `x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε` | `gaussian_diffusion.py: q_sample` |
| (11) | hàm mất mát huấn luyện | `gaussian_diffusion.py: training_losses` |
| (12) | `ε` từ `x̂₀` | `gaussian_diffusion.py: _predict_eps_from_xstart` |
| (13) | DDIM sampling | `gaussian_diffusion.py: ddim_sample` |
| (14) | DDIM có `η` | cùng hàm, biến `sigma` |
| (15) | ODE giới hạn | không cài trực tiếp (chỉ dùng để giải thích) |
| (16) | DDIM inversion | `gaussian_diffusion.py: ddim_reverse_sample` |
| (17) | DIRE (định nghĩa đầy đủ) | `compute_dire.py` + `dire_modal/compute_dire.py::_dire_for_image` |

Bảng tra ngược — từ một dòng mã tới công thức:

| Dòng mã | Công thức | Ý nghĩa |
|---|---|---|
| `sqrt_alphas_cumprod[t] * x_start` | (10) | co ảnh sạch |
| `+ sqrt_one_minus_alphas_cumprod[t] * noise` | (10) | cộng nhiễu |
| `pred_xstart.clamp(-1, 1)` | — | phép cắt phi tuyến, phá thuận nghịch |
| `sqrt_recip_alphas_cumprod[t] * x - pred_xstart) / sqrt_recipm1_alphas_cumprod[t]` | (12) | suy `ε` |
| `pred_xstart * sqrt(alpha_bar_prev) + sqrt(1 - alpha_bar_prev - sigma²) * eps` | (13)(14) | bước DDIM xuống |
| `pred_xstart * sqrt(alpha_bar_next) + sqrt(1 - alpha_bar_next) * eps` | (16) | bước DDIM lên |
| `mean_flat((noise - model_output) ** 2).mean()` | (11) | MSE trên nhiễu |
| `th.abs(imgs - recons)` | (1)(17) | DIRE |
| `(dire * 255.0 / 2.0)` | — | đổi `[0,2] → [0,255]`, thang 1× |

# Phụ lục C — Danh mục kiểm trước khi báo cáo một con số

Dùng danh mục này cho mọi bảng kết quả, kể cả bảng nhỏ.

**Trước khi chạy**

- [ ] Nhãn `R0/R1/R2/R3` đã xác định và ghi vào `run.json`.
- [ ] `preproc` khớp `resize256_centercrop224_normalize` (hoặc ghi rõ nếu khác).
- [ ] sha256 checkpoint khớp manifest.
- [ ] Tập test **không** được dùng để chọn ngưỡng hay chọn epoch.
- [ ] Nếu dùng lại DIRE map của tác giả: đã ghi "chấm lại", chưa ghi "tái lập".

**Sau khi chạy**

- [ ] `n_real` và `n_fake` có mặt và hợp lý (không phải 0, không trùng nhau).
- [ ] Cổng S2 đạt (`ACC` ô ADM→ADM ≥ 0,99). Nếu không, dừng và gỡ lỗi.
- [ ] Chạy lại cho cùng CSV (cổng S6).
- [ ] Ô không đo được để **trống**, không ghi 0.
- [ ] Mọi con số ước lượng (chi phí, thời gian) ghi rõ là ước lượng và kèm giả định.

**Khi viết**

- [ ] Không trích số của bài báo mà không ghi số hiệu bảng.
- [ ] Không so sánh hai con số có khoảng tin cậy chồng lấn như thể chúng khác nhau.
- [ ] Không dùng cụm "tái lập" khi chưa đạt `R1+R2+R3` tương ứng.
- [ ] Mọi khẳng định về cơ chế (vì sao) có ghi rõ là giả thuyết nếu chưa có thí
      nghiệm kiểm.

# Phụ lục D — Tài liệu tham khảo

[1] Z. Wang, J. Bao, W. Zhou, W. Wang, H. Hu, X. Chen, C. Li. *DIRE for
Diffusion-Generated Image Detection.* ICCV 2023. arXiv:2303.09295.
<https://arxiv.org/abs/2303.09295>

[2] Z. Wang. *DIRE — mã nguồn chính thức.* GitHub, `ZhendongWang6/DIRE` (lưu trữ
2025-07-08). <https://github.com/ZhendongWang6/DIRE>

[3] J. Ho, A. Jain, P. Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS
2020. arXiv:2006.11239.

[4] P. Dhariwal, A. Nichol. *Diffusion Models Beat GANs on Image Synthesis.* NeurIPS
2021. arXiv:2105.05233. Mã: `openai/guided-diffusion`.

[5] J. Song, C. Meng, S. Ermon. *Denoising Diffusion Implicit Models.* ICLR 2021.
arXiv:2010.02502.

[6] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, B. Ommer. *High-Resolution Image
Synthesis with Latent Diffusion Models.* CVPR 2022. arXiv:2112.10752.

[7] L. Liu, Y. Ren, Z. Lin, Z. Zhao. *Pseudo Numerical Methods for Diffusion Models
on Manifolds.* ICLR 2022. arXiv:2202.09778.

[8] A. Nichol, P. Dhariwal. *Improved Denoising Diffusion Probabilistic Models.*
ICML 2021. arXiv:2102.09672.

[9] Stable Diffusion: `runwayml/stable-diffusion-v1-5` và
`stabilityai/stable-diffusion-2` (model card trên Hugging Face); bài báo nền là [6].

[10] S. Gu, D. Chen, J. Bao, F. Wen, B. Zhang, D. Chen, L. Yuan, B. Guo. *Vector
Quantized Diffusion Model for Text-to-Image Synthesis.* CVPR 2022.
arXiv:2111.14822.

[11] S.-Y. Wang, O. Wang, R. Zhang, A. Owens, A. A. Efros. *CNN-Generated Images Are
Surprisingly Easy to Spot… for Now.* CVPR 2020. arXiv:1912.11035.

[12] L. Chai, D. Bau, S.-N. Lim, P. Isola. *What Makes Fake Images Detectable?
Understanding Properties That Generalize.* ECCV 2020. arXiv:2008.10588.

[13] Y. Qian, G. Yin, L. Sheng, Z. Chen, J. Shao. *Thinking in Frequency: Face
Forgery Detection by Mining Frequency-Aware Clues (F3-Net).* ECCV 2020.

[14] K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.*
CVPR 2016. arXiv:1512.03385.

[15] I. Loshchilov, F. Hutter. *SGDR: Stochastic Gradient Descent with Warm
Restarts.* ICLR 2017. arXiv:1608.03983. (Cơ sở của `CosineAnnealingLR`.)

[16] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, L. Fei-Fei. *ImageNet: A
Large-Scale Hierarchical Image Database.* CVPR 2009.

[17] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, B. Poole.
*Score-Based Generative Modeling through Stochastic Differential Equations.*
ICLR 2021. arXiv:2011.13456.

[18] F. Yu, A. Seff, Y. Zhang, S. Song, T. Funkhouser, J. Xiao. *LSUN: Construction
of a Large-scale Image Dataset using Deep Learning with Humans in the Loop.* 2015.
arXiv:1506.03365.

[19] D. P. Kingma, J. Ba. *Adam: A Method for Stochastic Optimization.* ICLR 2015.
arXiv:1412.6980.

[20] J. Ricker, D. Lukovnikov, A. Fischer. *AEROBLADE: Training-Free Detection of
Latent Diffusion Images Using Autoencoder Reconstruction Error.* CVPR 2024.
arXiv:2401.17879. (Cùng ý tưởng "sai số tái tạo là tín hiệu", nhưng dùng autoencoder
của LDM thay cho diffusion model, và LPIPS thay cho `|Δ|` theo điểm ảnh.)

::: note
Các tài liệu nội bộ của kho này nên đọc kèm: `docs/adr/adr-0006-reproduction-levels.md`
(định nghĩa `R0`–`R3`), `docs/paper-code-walkthrough.md` (bản đối chiếu code ngắn
hơn), `spec.md` và `plan.md` (yêu cầu và kế hoạch của dự án).
:::
