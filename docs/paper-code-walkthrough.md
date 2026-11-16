# Từ lý thuyết đến từng dòng code — DIRE (Wang et al., ICCV 2023)

> Tài liệu này trả lời một câu hỏi hẹp hơn "bài báo nói gì": **mỗi bước toán học
> được hiện thực ở đâu, bằng dòng code nào, và có khớp không**.
>
> **Nguồn được đọc:**
> - `arXiv:2303.09295v1` (toàn văn + Phụ lục A–C).
> - Repo tác giả `ZhendongWang6/DIRE` (đã archive): `train.py`, `test.py`,
>   `demo.py`, `train.sh`, `test.sh`, `utils/{config,datasets,trainer,eval,earlystop,utils}.py`,
>   `guided-diffusion/compute_dire.{py,sh}`, `guided-diffusion/guided_diffusion/{gaussian_diffusion,image_datasets}.py`.
> - Repo này: `src/dire_core/*`, `src/dire_modal/compute_dire.py`,
>   `src/dire_modal/vendor/*` (bản vendor của tác giả).
>
> **Quy ước:** `file:dòng` là dòng trong repo này; `gốc:file` là repo tác giả.
> Chỗ nào tôi suy luận từ code mà paper không nói, ghi rõ **[suy ra từ code]**.

---

## Phần I — Lý thuyết cơ sở

### 1. Bài toán, phát biểu đúng

Cho ảnh `x`, cần quyết định `x ~ p_r` (thật) hay `x ~ p_g` (do máy sinh).

- Đây là **phân loại nhị phân** với hai rủi ro khác nhau về chi phí: bỏ sót ảnh
  giả (false negative) và buộc tội ảnh thật (false positive). Vì vậy paper báo
  **ACC** *và* **AP**; AP không phụ thuộc ngưỡng nên đo được chất lượng xếp hạng
  tách khỏi lựa chọn ngưỡng.
- Khó khăn không nằm ở việc tách `p_r` và `p_g` khi đã biết `p_g`, mà ở việc tách
  khi **`p_g` lúc test khác `p_g` lúc train**. Đây là bài toán **domain shift**,
  và nó là lý do toàn bộ kiến trúc của paper xoay quanh "đừng học dấu vết của
  generator cụ thể".
- Bằng chứng định lượng cho nhận định đó nằm ngay trong Table 2: các detector
  hiện đại (`CNNDet*`, `Patchfor*`, `F3Net*`) được **train lại trên đúng ảnh ADM**
  vẫn chỉ đạt lần lượt 86,5 / 90,9 / 91,6 ACC trung bình trên 8 generator, và sụp
  ở generator chưa thấy.

### 2. Ảnh số và ba phép đổi miền giá trị

Toàn bộ pipeline có **ba** lần đổi thang đo, và lẫn chúng là nguồn lỗi phổ biến
nhất khi tái lập:

| Miền | Công thức | Xuất hiện ở |
|---|---|---|
| uint8 `[0,255]` | ảnh gốc trên đĩa | PNG/JPG |
| float `[-1,1]` | `x = pixel/127.5 − 1` | đầu vào diffusion (ADM) |
| tensor đã chuẩn hoá ImageNet | `(x − μ)/σ`, μ=(0.485,0.456,0.406), σ=(0.229,0.224,0.225) | đầu vào ResNet-50 |

Điểm cần nhớ: **diffusion làm việc ở `[-1,1]`, ResNet làm việc ở miền chuẩn
hoá ImageNet, còn DIRE được lưu ở uint8**. Ba thang này không thay thế nhau được.

### 3. Vì sao ResNet-50

- Tích chập giữ **bất biến tịnh tiến cục bộ**; xếp chồng nhiều tầng cho trường
  tiếp nhận lớn dần, nên tầng sâu nhìn được cấu trúc toàn ảnh.
- Khối residual `y = x + F(x)` cho gradient đi thẳng qua hàng trăm tầng — điều
  kiện để train 50 tầng mà không suy giảm.
- `AdaptiveAvgPool2d(1)` gộp không gian → vector 2048 chiều; `fc` biến nó thành
  **một** logit. Một logit + `BCEWithLogits` là quy ước của DIRE, và nó khác với
  "2 lớp + cross-entropy" ở chỗ: `sigmoid(z)` là xác suất lớp `1_fake`, còn
  `softmax(z)[1]` cần hai đầu ra. Hai quy ước này **không hoán đổi được** — đây
  chính là lỗi `size mismatch for fc.weight: [1,2048] vs [2,2048]` mà
  `src/dire_core/detector.py:1-18` ghi lại.
- Transfer learning: khởi tạo từ trọng số ImageNet rồi thay `fc`, vì 80k ảnh là
  quá ít để học 25 triệu tham số từ đầu.

### 4. Diffusion: từ Markov đến ODE

**Quá trình thuận** (Eq. 1–2): thêm nhiễu dần,

```
q(x_t | x_0) = N( x_t ; √ᾱ_t · x_0 , (1 − ᾱ_t) I )
```

với `ᾱ_t = ∏ α_i`. Hệ quả: `x_0` **được xác định** bởi `(x_t, ε)` qua
`x_0 = (x_t − √(1−ᾱ_t)·ε)/√ᾱ_t` — mọi thứ về sau chỉ là biến đổi của đẳng thức này.

**Huấn luyện** (Eq. 4): mạng `ε_θ(x_t, t)` học dự đoán nhiễu đã thêm, mất mát MSE.

**Sinh ảnh** (DDPM, Eq. 3): lặp `x_{t−1} ~ N(μ_θ, Σ_θ)`. Có nhiễu ngẫu nhiên ⇒
mỗi lần chạy ra một ảnh khác.

**DDIM** (Eq. 5): đặt `σ_t = 0` ⇒

```
x_{t−1} = √ᾱ_{t−1} · x̂_0 + √(1 − ᾱ_{t−1}) · ε_θ(x_t, t)
```

Không còn nhiễu ngẫu nhiên ⇒ **tất định** ⇒ ánh xạ `x_0 ↦ x_T` **đảo ngược được**.
Đảo ngược chính là Eq. 8 (chạy cùng công thức theo chiều `t` tăng).

Khi `T` lớn, Eq. 5 là **Euler** cho một ODE (Eq. 6–7). Hai hệ quả:

1. Rời rạc hoá 1000 bước thành 20 bước (S=20) là **xấp xỉ Euler bậc thấp** ⇒ có
   sai số, và sai số phụ thuộc độ "cong" của quỹ đạo tại điểm xuất phát.
2. Vì tất định, sai số ấy **lặp lại được** — cùng ảnh, cùng S ⇒ cùng `x̂`. Đó là
   điều kiện để DIRE dùng được làm đặc trưng (không phải nhiễu ngẫu nhiên).

### 5. Vì sao "tái tạo được" lại là tín hiệu

Paper lập luận: ảnh do diffusion sinh nằm trên quỹ đạo mà model đã học, nên
`R(I(x)) ≈ x`; ảnh thật thì không. Phụ lục B bổ sung: vì `T` hữu hạn, Euler có
sai số, và sai số **lớn hơn ở ảnh thật** do phân phối thật phức tạp hơn.

Cần nói thẳng: đây là **lập luận định tính**, không phải định lý. Nó cũng có một
lỗ hổng nội tại — ADM được train trên LSUN-Bedroom, và ảnh real trong benchmark
**cũng** là LSUN-Bedroom; theo chính giả thuyết "nằm trong phân phối đã học thì
tái tạo tốt", ảnh real phải tái tạo **tốt**, không phải kém. Paper không giải
quyết mâu thuẫn này (xem `docs/` phần phản biện nếu đã lưu).

### 6. Định nghĩa DIRE

```
DIRE(x_0) = | x_0 − R(I(x_0)) |          (Eq. 9)
```

- `I(·)`: S bước DDIM inversion, `t: 0 → S`.
- `R(·)`: S bước DDIM sampling, `t: S → 0`.
- `|·|`: trị tuyệt đối **theo từng điểm ảnh, từng kênh**, giữ ảnh 3 kênh.
- Kết quả: một ảnh cùng kích thước đầu vào (256×256×3), trong đó **giá trị lớn =
  khó tái tạo = giống ảnh thật**.

Vì sao `|·|` quan trọng (Table 6): nếu bỏ abs, dấu của residual phụ thuộc chiều
lệch — một đại lượng không ổn định giữa các ảnh. Abs chỉ giữ **độ lớn**, thứ có
tính hệ thống hơn. Đo được: SD-v1 từ 87,0/93,0 lên 99,7/100 ACC/AP.

### 7. Chuỗi đầy đủ, một dòng

```
ảnh uint8
  → (center crop + resize 256) → float [-1,1]
  → I: 20 bước DDIM inversion  → x_T
  → R: 20 bước DDIM sampling   → x̂  (cùng [-1,1])
  → |x − x̂| * 127.5            → DIRE uint8 256×256
  → Resize(256)→CenterCrop(224)→Normalize
  → ResNet-50 (fc 2048→1) → logit z → sigmoid(z) = P(fake)
```

---

## Phần II — Ánh xạ công thức ↔ code

| Công thức | Ý nghĩa | Hàm trong code |
|---|---|---|
| Eq. 2 | `q(x_t|x_0)` | `gaussian_diffusion.q_sample` (chỉ dùng khi train ADM; DIRE **không** gọi) |
| Eq. 5 | DDIM sampling `x_{t−1}` | `gaussian_diffusion.ddim_sample` |
| Eq. 8 | DDIM inversion `x_{t+1}` | `gaussian_diffusion.ddim_reverse_sample` |
| Eq. 9 | `DIRE = |x − R(I(x))|` | `compute_dire.main` (gốc), `_dire_for_image` (repo này) |
| Eq. 10 | BCE | `nn.BCEWithLogitsLoss` trong `vendor/utils/trainer.py` |
| §4.1 tiền xử lý | resize/crop/normalize | `vendor/utils/datasets.py::binary_dataset`, `dire_core/preprocess.py::build_transform` |
| §4.1 metric | ACC/AP | `vendor/utils/eval.py::validate`; bản thuần Python ở `dire_modal/eval.py::metrics` |

---

## Phần III — Pipeline thực tế A: **sinh DIRE** (`guided-diffusion/compute_dire.py`)

Đây là phần đắt nhất và là phần **quyết định toàn bộ dữ liệu train**. Chạy một
lần, dùng mãi.

### A.1 `compute_dire.sh` — bốn dòng quyết định

1. `SAMPLE_FLAGS="--batch_size 16 --num_samples 1000 --timestep_respacing ddim20 --use_ddim True"`
   - `ddim20`: **respace** 1000 bước xuống đúng 20 mốc `{0, 50, 100, …, 950}`.
     Đây là chỗ S=20 của paper được hiện thực — không phải tham số của hàm, mà là
     **tập timestep** của một `SpacedDiffusion`.
   - `use_ddim True`: chọn `ddim_sample_loop` thay vì `p_sample_loop`.
2. `MODEL_FLAGS` khớp kiến trúc ADM 256² unconditional: `num_channels 256`,
   `num_res_blocks 2`, `num_head_channels 64`, `attention_resolutions 32,16,8`,
   `resblock_updown True`, `use_scale_shift_norm True`, `learn_sigma True`,
   `class_cond False`, `diffusion_steps 1000`, `noise_schedule linear`.
3. `MPI` + `mpiexec -n 4`: chia ảnh theo rank; `dist_util.setup_dist` chọn device.
4. `--has_subfolder True`: giữ cấu trúc thư mục con khi ghi (`lsun_bedroom/`, `sdv1/`…).

### A.2 `reshape_image(imgs, image_size)` — chuẩn bị tensor

- Nếu tensor 3 chiều → thêm batch.
- Nếu **không vuông** → `CenterCrop(image_size)`, rồi nếu vẫn khác kích thước →
  `F.interpolate(mode="bicubic")` về 256².
- Hệ quả: ảnh SD 512² bị **crop giữa rồi resize bicubic** — mất viền và mất dải
  tần số cao. Đây là lý do mọi thứ trong paper đều 256².

### A.3 `load_data_for_reverse` + `ImageDataset_for_reverse` — đọc ảnh

- Liệt kê file đệ quy (`_list_image_files_recursively`), chỉ nhận
  `jpg/jpeg/png/gif`.
- `shard=k, num_shards=n` ⇒ `local_images = all_files[shard::n]` — chia việc theo
  MPI, không trùng ảnh.
- `center_crop_arr(pil, 256)`: crop giữa (không pad) rồi mới đổi sang mảng.
- `arr.astype(np.float32)/127.5 − 1` → **đúng** miền `[-1,1]` mà ADM cần.
- Trả `(tensor CHW, {"y":...}, path)` — `path` để ghi lại đúng tên file.

### A.4 Vòng lặp chính — 8 bước

```
model, diffusion = create_model_and_diffusion(...)      # SpacedDiffusion 20 bước
model.load_state_dict(load_state_dict(model_path))      # 256x256_diffusion_uncond.pt
model.to(dev); model.convert_to_fp16(); model.eval()    # fp16 để tăng throughput
```

1. `imgs, out_dicts, paths = next(data)` — generator vô hạn, luôn trả batch.
2. `model_kwargs["y"]` chỉ khi `class_cond` (với LSUN unconditional thì rỗng).
3. `imgs = reshape_image(imgs, 256)`.
4. **Inversion**:
   `latent = diffusion.ddim_reverse_sample_loop(model, shape, noise=imgs, real_step=...)`
   - `indices = range(num_timesteps)` (tăng dần) khi `real_step == 0`.
   - Mỗi bước gọi `ddim_reverse_sample`, tính:
     `eps = (√(1/ᾱ_t)·x − x̂_0) / √(1/ᾱ_t − 1)` rồi
     `x_{t+1} = √ᾱ_{t+1}·x̂_0 + √(1−ᾱ_{t+1})·eps` — **đúng Eq. 8**.
5. **Reconstruction**:
   `recons = diffusion.ddim_sample_loop(model, shape, noise=latent, ...)`
   - `indices` giảm dần; công thức Eq. 5 với `σ = 0` (vì `eta=0`).
   - Cùng `diffusion` (cùng 20 mốc) ⇒ inversion và reconstruction **đối xứng**.
6. **`dire = th.abs(imgs − recons)`** — tính ở miền `[-1,1]`, **trước** khi lượng tử hoá.
7. Lượng tử hoá:
   - `recons = ((recons+1)·127.5).clamp(0,255).uint8`
   - `dire = (dire·255/2).clamp(0,255).uint8` ⇒ vì `dire ∈ [0,2]`, hệ số `127.5`
     biến nó thành **đúng |Δpixel|** trong thang `[0,255]`. Không khuếch đại.
8. Ghi file: `cv2.cvtColor(..., RGB2BGR)` rồi `cv2.imwrite` (vì `imwrite` nhận BGR).

**Chi phí:** 20 bước inversion + 20 bước reconstruction = **40 lượt forward
ADM/ảnh** (NFE). Với 42k ảnh × 8 model, đây là hoá đơn thật của paper — không
phải phần train ResNet.

**`real_step`:** nếu `> 0`, inversion chỉ chạy `range(T)[:real_step]` — tức đảo
ngược **một phần** (dừng ở bước thứ `real_step`) rồi reconstruct từ đó. Đây là
biến thể thử nghiệm, mặc định 0 = đảo toàn phần.

---

## Phần IV — Pipeline thực tế B: **dataset & augmentation** (`utils/datasets.py`)

### B.1 `binary_dataset(root, cfg)` — thứ tự transform là hợp đồng

Chuỗi transform, đúng thứ tự thực thi:

| # | Train | Val/Test | Ghi chú |
|---|---|---|---|
| 1 | `custom_resize` → cạnh ngắn 256, `bilinear` | như train (vì `aug_resize=True`) | resize **giữ tỉ lệ**, không ép vuông |
| 2 | `blur_jpg_augment` | no-op (vì `isTrain=False`) | `blur_prob=0`, `jpg_prob=0` mặc định |
| 3 | `RandomCrop(224)` | `CenterCrop(224)` | crop **sau** resize ⇒ 224 từ 256 |
| 4 | `RandomHorizontalFlip()` | identity | p=0.5 |
| 5 | `ToTensor()` | như train | `[0,255] → [0,1]` |
| 6 | `Normalize(ImageNet)` | như train (`aug_norm=True`) | bước hay bị bỏ sót nhất |

- `datasets.ImageFolder` suy nhãn từ **tên thư mục con**, sắp theo alphabet:
  `0_real → 0`, `1_fake → 1`. Đây là toàn bộ cơ chế gán nhãn — không có file
  metadata nào. Xáo trộn tên thư mục là đổi nhãn.
- `ImageFile.LOAD_TRUNCATED_IMAGES = True` (đầu file): ảnh JPEG hỏng một phần
  vẫn đọc được thay vì ném lỗi — cần thiết với dataset tải từ Baidu.
- `blur_jpg_augment` dùng `scipy.ndimage.gaussian_filter` cho từng kênh và
  `cv2.imencode/imdecode` cho JPEG; mặc định **tắt** nên không ảnh hưởng số paper,
  nhưng nếu bật thì nó là augmentation **trong miền uint8 trước khi crop**.

### B.2 `create_dataloader`

- `shuffle = isTrain and not class_bal` (val: `serial_batches=True` ⇒ tất định).
- `batch_size=64`, `num_workers=20`, `sampler=None` (không cân bằng lớp — hai lớp
  vốn đã 40k/40k).

---

## Phần V — Pipeline thực tế C: **huấn luyện** (`train.py` + `trainer.py` + `earlystop.py`)

### C.1 `config.py::DefaultConfigs` — những con số paper **không in**

| Dòng | Giá trị | Ý nghĩa |
|---|---|---|
| `seed` | 3407 | seed |
| `arch` | `resnet50` | |
| `datasets` | `["zhaolian_train"]` → bị `train.sh` ghi đè `lsun_adm` | |
| `batch_size` | 64 | |
| `loadSize` / `cropSize` | 256 / 224 | |
| `nepoch` | 400 | |
| `lr` | 1e-4 | Adam, `beta1=0.9` |
| `earlystop_epoch` | 5 | patience |
| `warmup` | False | tắt ⇒ **không** dùng cosine/warmup |
| `aug_norm`, `aug_crop`, `aug_flip`, `aug_resize` | True | |
| `blur_prob`, `jpg_prob` | 0.0 | augmentation nhiễu **tắt** |

### C.2 `get_network` — điểm khởi tạo

```
resnet(pretrained=True)        # 1000 lớp ImageNet, tải từ model_zoo
model.fc = nn.Linear(2048, 1)  # thay đầu ra
nn.init.normal_(model.fc.weight, 0.0, 0.02)
```

- Nhánh eval (`isTrain=False`): `resnet(num_classes=1)` **không** pretrained —
  vì sắp nạp checkpoint.
- `init_weights()` định nghĩa trong `trainer.py` **không được gọi** ở đâu
  [suy ra từ code: không có call site] — code chết, đừng tưởng nó chạy.

### C.3 `train.py` — vòng lặp, từng dòng

1. `val_cfg = get_val_cfg(cfg, "val")` **trước** khi đổi `dataset_root` ⇒ val
   trỏ `.../val`, train trỏ `.../train`.
2. `create_dataloader(cfg)`; `dataset_size = len(data_loader)` — đây là số **batch**
   (log ghi `batch × batch_size` = số ảnh).
3. `Trainer(cfg)`: dựng model, `BCEWithLogitsLoss`, `Adam(lr=1e-4, betas=(0.9,0.999))`.
4. `EarlyStopping(patience=5, delta=-0.001)`.
5. Vòng epoch × batch:
   - `total_steps += 1`
   - `set_input`: đưa ảnh/nhãn lên device, `label.float()` (BCE cần float).
   - `optimize_parameters`: `forward → loss → zero_grad → backward → step`.
   - `train_writer.add_scalar("loss", ...)` — log TensorBoard.
   - mỗi 2000 step: `save_networks("latest")`.
6. Mỗi 20 epoch: lưu `latest` + `model_epoch_{n}.pth`.
7. **Validation mỗi epoch** bằng `validate(trainer.model, val_cfg)`.
8. Early stopping:
   - `early_stopping(val_acc, trainer)`; nếu kích hoạt thì `adjust_learning_rate()`
     (chia 10) và **tạo mới** `EarlyStopping` với `delta=-0.002` (nghiêm hơn).
   - `adjust_learning_rate` trả `False` khi `lr < 1e-6` ⇒ với `lr=1e-4` chỉ giảm
     được **3 lần** (1e-5, 1e-6, rồi dừng).
9. `trainer.train()` cuối mỗi epoch (đưa model về train mode — cần vì bước 7 đã
   gọi `eval()`).

**Chi tiết dễ sai:** `delta` **âm**. Điều kiện dừng là
`score < best_score − delta` = `score < best + 0.001`, nên **một lần tăng ít hơn
0.001 cũng bị tính là không cải thiện**. Nếu tái lập mà không biết điều này, số
epoch dừng sẽ khác, và checkpoint khác.

### C.4 `EarlyStopping` — hành vi thật

- Lần đầu: lưu `best` (`model_epoch_best.pth`) và ghi `best_score`.
- Mỗi lần tốt hơn: lưu đè `best`, reset `counter`.
- Không tốt hơn: `counter += 1`; đủ `patience` ⇒ `early_stop = True`.

### C.5 Checkpoint lưu gì

`save_networks` lưu `{"model": state_dict, "optimizer": ..., "total_steps": ...}`
⇒ file ~269 MB cho ResNet-50 (vì gấp 3 lần trọng số). Đây là lý do
`dire_core/detector.py::unwrap_state` phải bóc khoá `"model"` trước khi
`load_state_dict` — nạp thẳng sẽ lỗi "unexpected key".

---

## Phần VI — Pipeline thực tế D: **đánh giá** (`test.py` + `utils/eval.py`)

### D.1 `get_val_cfg(cfg, split)`

- `deepcopy` để không làm bẩn config train.
- `dataset_root = dataset_root + "/" + split`.
- `datasets = datasets_test` (tên thư mục con dưới `data/test/`).
- `isTrain=False`, `aug_flip=False`, `serial_batches=True` ⇒ **không xáo, không flip**.
- `jpg_method=["pil"]` (để nhiễu JPEG tất định nếu có bật).

### D.2 `validate(model, cfg)`

```
predict = model(in_tens, meta).sigmoid()     # 1 logit → P(fake)
acc      = accuracy(y, pred > 0.5)           # ngưỡng 0.5, so sánh STRICT >
r_acc    = accuracy trên tập nhãn 0 (ảnh thật)
f_acc    = accuracy trên tập nhãn 1 (ảnh sinh)
ap       = average_precision_score(y, pred)   # sklearn
roc_auc  = roc_auc_score(...)
```

- `> 0.5` (không phải `>=`) ⇒ `pred == 0.5` được tính là **real**. Repo này dùng
  cùng quy ước: `(s > threshold) == (y == FAKE)` ở `src/dire_modal/eval.py:184`.
- AP của sklearn dùng **trung bình có trọng số theo recall**, không phải
  trapezoid — nếu tự viết lại phải khớp chỗ này, nếu không sẽ lệch vài phần nghìn.
  `src/dire_modal/eval.py:107-140` cài đặt đúng biến thể đó và có test đối chiếu.

### D.3 `test.py`

- Mỗi dataset con: dựng lại model, `load_state_dict(state_dict["model"])`, `cuda`, `eval`.
- Ghi CSV `TestSet, <metric>…` một dòng/dataset.
- Không có bước chọn ngưỡng trên test — ngưỡng cố định 0.5, đúng như paper.

---

## Phần VII — Pipeline trong **repo này**: đối chiếu từng hàm

### E.1 `dire_core/preprocess.py` — một pipeline duy nhất

| Dòng | Nội dung | Vì sao |
|---|---|---|
| 21 | `PREPROC = "resize256_centercrop224_normalize"` | định danh ghi vào CSV/`run.json`; hai lần chạy khác pipeline không thể lẫn |
| 23–24 | `RESIZE=256`, `CROP=224` | khớp `loadSize/cropSize` |
| 26–27 | `MEAN/STD` ImageNet | khớp `datasets.py` |
| 37–44 | `Compose([Resize, CenterCrop, ToTensor, Normalize])` | **đúng** thứ tự của `binary_dataset` |
| 47–58 | `preprocess(img, torch, transforms)` | nhận `torch` để không import ở tầng module |

Khác biệt có ý thức: `transforms.Resize(256)` (cạnh ngắn → 256) thay cho
`custom_resize` + `rz_interp` ngẫu nhiên. Vì `rz_interp=["bilinear"]` chỉ có một
lựa chọn nên hai cách **tương đương** khi eval.

⚠ **Sai sót cần sửa:** docstring dòng 7–8 nói "`demo.py` của tác giả **thiếu**
`Normalize`". Đọc `demo.py` bản `main` hôm nay: nó dùng
`transforms.Resize(256), CenterCrop(224), ToTensor()` rồi `TF.normalize(...)` với
`--aug_norm` **mặc định True**. Nghĩa là demo **có** normalize; khác biệt thật
chỉ là nó chuẩn hoá **sau** `ToTensor` (tương đương) và không có augmentation
train. Câu trong docstring là **sai**, và nó đang được dùng để giải thích "vì sao
demo lệch paper" — cần thay bằng lý do đúng hoặc bỏ.

### E.2 `dire_core/detector.py`

- `unwrap_state` (23–44): bóc `model`/`state_dict`/`net`/`model_state_dict`, bỏ
  tiền tố `module.` do `DataParallel`. **Cần** vì checkpoint gốc lưu `{"model": …}`.
- `fc_out_features` (47–71): đọc `fc.weight.shape[0]` từ chính checkpoint.
  `[1,2048] → 1` (BCE + sigmoid), `[2,2048] → 2` (CE + softmax). Không đoán.
- `load_detector` (74–105): `resnet50(num_classes=n_out)` → `load_state_dict` →
  `.to(device).eval()`. Lỗi được dịch thành `SystemExit` có nội dung, vì lỗi
  `size mismatch` thô rất khó đọc.

### E.3 `dire_core/score.py`

- `sigmoid` (22–28): hai nhánh theo dấu để tránh `exp` tràn (`exp(800)` → `inf`).
- `prob_fake` (31–48): `n_out == 1` → sigmoid; `n_out ≥ 2` → softmax **trừ max**
  trước khi `exp` (ổn định số học). Không đủ cột ⇒ `ValueError`, không bịa điểm.

### E.4 `dire_core/batch.py`

- `batch_logits` (17–32): chia chunk, `torch.inference_mode()` (nhẹ hơn
  `no_grad`), trả **logit thô** — tách khỏi bước đổi xác suất để test được riêng.
- `Detector.load` (55–71): chọn `cuda` nếu có; import torch trong hàm.
- `logits_of` (74–78): `preprocess` từng PIL → stack → `batch_logits`.
- `score_pils` (80–82): `prob_fake(logits, n_out)`.
- `info` (85–97): trả luôn `loss` và `prob` để response API tự mô tả quy ước.

### E.5 `dire_modal/eval.py` — đường chấm điểm

- `score_pair`: mở hai tar, yield `(name, bytes)` → `PIL.open` → `preprocess` →
  `batch_logits` → `prob_fake` → gom `y_true/y_score` → `metrics`.
- `label_of`: nhãn suy từ **tên archive** (`real.tar.gz` → 0, còn lại → 1), không
  từ tên member — tránh trường hợp tar trộn.
- `metrics`: ACC (strict `>`), AP (biến thể sklearn), `tpr_at_5fpr`, `r_acc`, `f_acc`.
- `row_for_result`: dòng lỗi để **trống** metric, không ghi 0.

### E.6 `dire_modal/compute_dire.py` — bản tự viết lại A

Đối chiếu với `compute_dire.py` gốc:

| Bước | Gốc | Repo này | Khớp? |
|---|---|---|---|
| Kiến trúc ADM | `MODEL_FLAGS` trong .sh | `_load_adm(...)` tham số hoá | ✅ (trừ `dropout`: gốc 0.1, đây 0.0 — vô hại khi eval) |
| Respace S=20 | `--timestep_respacing ddim20` | `_load_adm(timestep_respacing="")` + `SpacedDiffusion(space_timesteps(..., "20"))` | ✅ cùng tập 20 mốc `{0,50,…,950}` |
| fp16 | `model.convert_to_fp16()` | không gọi ⇒ fp32 | ⚠ chậm ~2×, chính xác hơn |
| Inversion | `diffusion.ddim_reverse_sample_loop(model, shape, noise=imgs)` | `diffusion.ddim_reverse_sample_loop(model, x0, ...)` | ❌ **sai** (xem F1) |
| Reconstruction | cùng `diffusion` (20 mốc) | `ddim.ddim_sample_loop` (20 mốc) | ⚠ đối xứng chỉ khi F2 được sửa |
| DIRE | `abs(x−x̂)·255/2` = `|Δpixel|` | `abs(orig_u8−recon_u8) · 4` | ❌ khác thang (xem F3) |
| Ghi | PNG uint8 | `.npy` uint8 | — |

---

## Phần VIII — Sai lệch cụ thể tìm được khi đọc code

### F1. `compute_dire.py:245` — gọi sai vị trí tham số ⇒ **không chạy được**

```python
noise = diffusion.ddim_reverse_sample_loop(model, x0, clip_denoised=True, ...)
```

Chữ ký thật: `ddim_reverse_sample_loop(self, model, shape, noise=None, ...)`.
`x0` rơi vào **`shape`**, còn `noise` = `None` ⇒ hàm đi vào nhánh
`th.randn(*shape)` với `shape` là **tensor 4 chiều** ⇒ `TypeError`.

Đúng phải là:

```python
noise = ddim.ddim_reverse_sample_loop(model, x0.shape, noise=x0, ...)
```

Đây khớp với cảnh báo trong docstring đầu file ("CHƯA được chạy thử … kiểm
`ddim_reverse_sample_loop` trả (x, _) hay chỉ x") — nhưng vấn đề không phải giá
trị trả về, mà là **tham số truyền vào**.

### F2. Inversion dùng `diffusion` (1000 bước) thay vì `ddim` (20 bước)

`_load_adm` tạo `diffusion` với `timestep_respacing=""` ⇒ `num_timesteps = 1000`.
`ddim` (SpacedDiffusion 20) chỉ được dùng cho bước reconstruction. Nếu chỉ sửa
F1, mỗi ảnh sẽ chạy **1000 bước đảo + 20 bước tái tạo = 1020 NFE**, gấp **25×**
chi phí paper, và tệ hơn: `x_T` từ thang 1000 bước đem tái tạo bằng thang 20 bước
**không phải** cặp nghịch đảo ⇒ DIRE mất ý nghĩa.

Sửa: dùng `ddim` cho **cả hai** bước.

Hệ quả lên ngân sách: con số "8 M NFE cho 200k ảnh" (= 200k × 40) chỉ đúng sau
khi F2 được sửa. Trước đó nó là 204 M NFE.

### F3. Hệ số `* 4` khi lưu `.npy`

`_dire_for_image` trả `diff_u8 = |orig − recon| · 4`, và `process_one` lưu nguyên
`diff_u8` vào `.npy`. Với ảnh thật, `|Δpixel|` trung bình cỡ 10–30 ⇒ nhân 4 rồi
`clip(0,255)` làm **bão hoà ở 63,75** (mọi giá trị ≥ 64 thành 255).

- Hệ số 4 là để **hiển thị** gallery — hợp lý cho demo.
- Nhưng nếu dùng `.npy` làm **đầu vào train**, biểu diễn sẽ khác hẳn DIRE phát
  hành (thang 1×), và classifier học một phân phối khác ⇒ không so được với paper.

Sửa: tách hai hàm — `dire_raw()` (thang 1×, để lưu) và `dire_display()`
(khuếch đại, chỉ để vẽ).

### F4. Nhỏ nhưng ảnh hưởng tái lập

- `dropout 0.0` thay vì `0.1`: không ảnh hưởng khi `eval()`, **có** ảnh hưởng nếu
  ai dùng file này để train ADM (không ai làm, nhưng nên khớp).
- Không gọi `convert_to_fp16()`: fp32 ⇒ chậm hơn, nhưng tránh lệch số do fp16.
  Nếu chọn fp32, phải ghi vào `run.json` để người khác biết.
- `space_timesteps(num_timesteps, "20")` và `"ddim20"` cho **cùng** tập mốc
  (stride 50) — nhưng cách thứ hai là cách tác giả dùng; nên ghi comment để
  người đọc không tưởng là hai cấu hình khác nhau.

### F5. Docstring sai về `demo.py` (mục E.1)

Không phải lỗi code, nhưng là **lỗi trích dẫn**: nó khẳng định một điều không
đúng về repo tác giả, và dùng điều đó để giải thích lệch điểm. Trong một dự án
mà tiêu chí là "không bịa số, không suy diễn quá mức", loại lỗi này cần sửa
ngang với lỗi code.

---

## Phần IX — Những gì **chỉ code** mới nói (paper không in)

| Thông tin | Giá trị | Nguồn |
|---|---|---|
| Số epoch | 400 | `config.py::nepoch` |
| Optimizer / LR | Adam / 1e-4 | `config.py`, `trainer.py` |
| Batch size | 64 | `config.py` |
| Early stop | theo **val ACC**, patience 5, `delta=-0.001` | `train.py`, `earlystop.py` |
| LR schedule | chỉ giảm khi early-stop kích hoạt, ÷10, tối đa 3 lần | `train.py` + `adjust_learning_rate` |
| Warmup / cosine | **tắt** | `warmup=False` |
| Seed | 3407 | `config.py` |
| Augmentation nhiễu | blur=0, jpeg=0 (tắt) | `config.py` |
| Số bước | 20 (respace `ddim20`), 40 NFE/ảnh | `compute_dire.sh` |
| Định dạng DIRE | uint8, `= |Δpixel|`, đã `clip` | `compute_dire.py` |
| Checkpoint | chứa cả optimizer + `total_steps` ⇒ 269 MB | `trainer.save_networks` |
| Nhãn | từ **tên thư mục** `0_real`/`1_fake` | `datasets.ImageFolder` |
| Ngưỡng ACC | 0.5, so sánh **strict** `>` | `utils/eval.py` |

---

## Phần X — Cách tự kiểm chứng ba phát hiện ở Phần VIII (không cần GPU)

```bash
# F1: chữ ký hàm — tham số thứ hai là `shape`, không phải ảnh
python - <<'PY'
import inspect, sys
sys.path.insert(0, "src/dire_modal/vendor")   # hoặc clone guided-diffusion
from guided_diffusion import gaussian_diffusion as gd
print(inspect.signature(gd.GaussianDiffusion.ddim_reverse_sample_loop))
PY

# F2: đếm NFE — 1000 hay 20?
grep -n "timestep_respacing" src/dire_modal/compute_dire.py      # "" ⇒ 1000
grep -n "ddim_reverse_sample_loop" src/dire_modal/compute_dire.py # dùng `diffusion`

# F3: thang đo — 4× và bão hoà
grep -n "\* 4" src/dire_modal/compute_dire.py
```

Kỳ vọng: F1 in ra `(self, model, shape, noise=None, ...)`; F2 cho thấy chỗ gọi
dùng `diffusion` chứ không phải `ddim`; F3 cho thấy `* 4` nằm trên **đường lưu
dữ liệu**, không chỉ trên đường hiển thị.

---

## Phần XI — Bảng tổng kết: lý thuyết ↔ code ↔ trạng thái

| Lý thuyết | Code (gốc) | Code (repo này) | Trạng thái |
|---|---|---|---|
| DDIM inversion Eq. 8 | `ddim_reverse_sample` | `_dire_for_image` (F1) | ❌ sai chỗ gọi |
| DDIM reconstruction Eq. 5 | `ddim_sample` | `_dire_for_image` | ⚠ đúng công thức, sai cấu hình |
| DIRE Eq. 9 | `abs(x−x̂)·255/2` | `abs(u8−u8)·4` | ❌ khác thang khi lưu |
| ResNet-50 + BCE | `get_network` + `BCEWithLogits` | `detector.load_detector` | ✅ |
| Tiền xử lý §4.1 | `binary_dataset` | `preprocess.build_transform` | ✅ |
| Nhãn `0_real/1_fake` | `ImageFolder` | `labels.label_of` | ✅ (khác cơ chế, cùng kết quả) |
| ACC/AP | `validate` | `eval.metrics` | ✅ (có test đối chiếu sklearn) |
| S=20 | `ddim20` | `space_timesteps(..., "20")` | ✅ cùng tập mốc |

**Kết luận ngắn:** phần *đọc và chấm điểm* (Phần IV–VI, E.1–E.5) khớp tác giả và
đã có test khoá. Phần *sinh DIRE* (E.6) **chưa chạy được**, và nếu sửa máy móc
theo kiểu "chạy được là xong" thì sẽ tốn 25× chi phí mà vẫn sai về mặt toán.
Ba việc phải làm trước khi tin bất kỳ DIRE map tự tính nào: sửa F1, sửa F2, tách
F3 thành hai hàm.
