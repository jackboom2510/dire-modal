# Phần 10 — Đọc mã của tác giả: từng hàm, từng dòng

## 10.1 Bản đồ kho mã

Repo của tác giả [[2]] có hai phần **rời nhau**, đừng nhầm:

```
DIRE/
├── train.py                 # pipeline 1: train detector ResNet-50 trên ảnh DIRE
├── test.py                  #            đánh giá detector
├── demo.py                  #            suy luận một ảnh
├── train.sh                 #            lệnh shell gọi train.py
├── utils/                   #            config, datasets, trainer, eval, earlystop
└── guided-diffusion/        # pipeline 2: SINH ra DIRE map
    ├── compute_dire.py
    ├── compute_dire.sh
    └── guided_diffusion/    # bản fork của openai/guided-diffusion
        ├── gaussian_diffusion.py   # DDPM + DDIM + phép đảo ngược
        ├── respace.py              # SpacedDiffusion, space_timesteps
        ├── script_util.py          # create_model_and_diffusion
        └── image_datasets.py       # đọc ảnh và đưa về [-1,1]
```

Hai pipeline **không** chạy chung một tiến trình và **không** chia sẻ cấu hình. Đó
là lý do có những khác biệt nhỏ giữa chúng (ví dụ `demo.py` không có augmentation
train), và là lý do phải cẩn thận khi so sánh con số.

**Đường đi của dữ liệu qua hai pipeline:**

```
ảnh gốc ──[compute_dire.py]──► dire/*.png ──[train.py]──► ResNet-50 ──[test.py]──► ACC/AP
    │
    └── đồng thời lưu recons/*.png (để kiểm và để phát hành)
```

## 10.2 `utils/datasets.py` — trật tự transform là hợp đồng

```python
def binary_dataset(root, cfg):
    identity_transform = transforms.Lambda(lambda img: img)

    # (1) resize: chỉ khi train, hoặc khi aug_resize=True
    if cfg.isTrain or cfg.aug_resize:
        rz_func = transforms.Lambda(lambda img: custom_resize(img, cfg))
    else:
        rz_func = identity_transform

    # (2) crop: train thì ngẫu nhiên, test thì crop giữa (nếu aug_crop)
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]) if cfg.aug_norm else identity_transform,
    ]))
```

Giải thích từng mắt xích, kèm cái bẫy của nó:

- **`ImageFolder`** suy nhãn từ **tên thư mục con** theo thứ tự alphabet:
  `0_real → 0`, `1_fake → 1`. Không có file metadata nào. Đổi tên thư mục là đổi
  nhãn, và **không có lỗi nào báo**. Đây là lý do mọi báo cáo trong kho này ghi
  `n_real`/`n_fake`.
- **`custom_resize`** gọi `TF.resize(img, cfg.resizeSize)` với `resizeSize = 256`.
  Với một số nguyên, torchvision hiểu là **cạnh ngắn = 256, giữ tỉ lệ**. Ảnh 4:3
  thành 256×341.
- **`blur_jpg_augment`** nằm **trước** crop, tức nhiễu được thêm trên ảnh lớn.
  Mặc định `blur_prob = 0` và `jpg_prob = 0` nên nó là hàm đồng nhất trong thí
  nghiệm của bài báo. Nếu bạn bật, vị trí này khác với "thêm nhiễu sau crop" và
  phân phối nhiễu sẽ khác.
- **`RandomCrop(224)` từ 256**: tương đương crop 224 với 32 pixel dư để dịch. Đây
  là augmentation **mạnh hơn** "resize thẳng về 224", vì nó tạo dịch chuyển.
- **`RandomHorizontalFlip()`** với `p = 0.5` mặc định: tăng gấp đôi dữ liệu hiệu
  dụng. Hợp lệ vì tính thật/giả không phụ thuộc chiều trái–phải.
- **`ToTensor()`** đổi `HWC uint8 [0,255]` → `CHW float [0,1]`, chia 255.
- **`Normalize`** là bước **cuối và bắt buộc**: `(x − μ)/σ` với `μ`, `σ` của
  ImageNet. Thiếu bước này, đầu vào ResNet lệch phân phối so với lúc train
  ImageNet ⇒ điểm tụt, không có lỗi nào.

Đầu module còn hai dòng quan trọng:

```python
ImageFile.LOAD_TRUNCATED_IMAGES = True     # ảnh JPEG tải dở vẫn đọc được
```

Đánh đổi: dữ liệu tải từ Baidu có ảnh hỏng, và dòng này cho chúng **âm thầm** lọt
vào train. Nếu tái lập, nên kiểm lại số ảnh đọc được so với manifest.

## 10.3 `train.py` — vòng lặp đầy đủ và một chi tiết tinh tế

```python
val_cfg = get_val_cfg(cfg, split="val", copy=True)   # (1) chụp cấu hình val TRƯỚC
cfg.dataset_root = os.path.join(cfg.dataset_root, "train")
data_loader = create_dataloader(cfg)
```

- **(1)** `deepcopy` rồi **mới** đổi `dataset_root` cho train. Nếu làm ngược lại,
  `val_cfg` cũng trỏ vào `train/` ⇒ đánh giá trên tập train, điểm cao giả tạo. Đây
  là lỗi kinh điển, và mã này tránh được đúng.
- `dataset_size = len(data_loader)` là số **batch**, không phải số ảnh; dòng log
  `dataset_size * cfg.batch_size` mới ra số ảnh (≈ 80 000).

Vòng lặp:

```python
trainer = Trainer(cfg)
early_stopping = EarlyStopping(patience=cfg.earlystop_epoch, delta=-0.001, verbose=True)
for epoch in range(cfg.nepoch):
    for data in tqdm(data_loader, dynamic_ncols=True):
        trainer.total_steps += 1
        trainer.set_input(data)            # lên GPU, label.float()
        trainer.optimize_parameters()      # forward → loss → zero_grad → backward → step
        train_writer.add_scalar("loss", trainer.loss, trainer.total_steps)
        if trainer.total_steps % cfg.save_latest_freq == 0:
            trainer.save_networks("latest")
```

Sau mỗi epoch:

```python
    trainer.eval()                                    # tắt dropout/BN dùng batch stats
    val_results = validate(trainer.model, val_cfg)    # ACC, AP trên val
    if cfg.earlystop:
        early_stopping(val_results["ACC"], trainer)
        if early_stopping.early_stop:
            if trainer.adjust_learning_rate():        # chia LR cho 10
                early_stopping = EarlyStopping(patience=cfg.earlystop_epoch,
                                               delta=-0.002, verbose=True)
            else:
                break                                 # LR < 1e-6 ⇒ dừng hẳn
    if cfg.warmup:
        trainer.scheduler.step()
    trainer.train()
```

Ba điểm cần hiểu đúng:

1. **Theo dõi ACC, không theo dõi loss.** ACC là metric cuối cùng, nhưng nó **rời
   rạc** (nhảy từng bước `1/1000`), nên tín hiệu dừng nhiễu hơn loss.
2. **Early stop không dừng ngay, mà giảm LR rồi train tiếp.** Đây là "giảm bậc"
   (step decay) thủ công: mỗi lần early stop, `lr ← lr/10`, tạo `EarlyStopping`
   mới với `delta = −0.002` (khắt khe hơn). Với `lr = 1e-4`, chỉ giảm được **3
   lần** (1e-4 → 1e-5 → 1e-6 → dừng), nên tổng số epoch tối đa bị chặn trên.
3. **`warmup=False`** trong cấu hình, nên `scheduler.step()` **không chạy**. Đừng
   mô tả pipeline của tác giả là "cosine warmup" — scheduler tồn tại nhưng không
   được dùng.

## 10.4 `utils/trainer.py` — nơi dễ sai shape nhất

```python
def set_input(self, input):
    img, label, meta = input if len(input) == 3 else (input[0], input[1], {})
    self.input = img.to(self.device)
    self.label = label.to(self.device).float()      # BCE cần float, không phải long

def optimize_parameters(self):
    self.forward()                                   # output = model(input, meta)
    self.loss = self.loss_fn(self.output.squeeze(1), self.label)
    self.optimizer.zero_grad()                       # phải xoá gradient cũ
    self.loss.backward()
    self.optimizer.step()
```

Ba cái bẫy:

- **`.float()`**: nhãn từ `ImageFolder` là `int64`. `BCEWithLogitsLoss` cần `float`.
  Không đổi thì PyTorch báo lỗi dtype (dễ phát hiện), nhưng nếu bạn tự viết công
  thức BCE bằng tay thì có thể ra kết quả sai **không báo**.
- **`.squeeze(1)`**: đầu ra `(N, 1)` phải thành `(N,)` để khớp `label` `(N,)`. Nếu
  quên, PyTorch **broadcast** `(N,1)` với `(N,)` thành `(N,N)` và tính loss sai
  hoàn toàn — đây là lỗi im lặng nguy hiểm nhất trong cả file.
- **`zero_grad()` trước `backward()`**: PyTorch **cộng dồn** gradient. Quên
  `zero_grad` thì gradient tích luỹ qua các batch và mô hình phân kỳ.

Phần còn lại: `init_weights` định nghĩa nhưng **không có chỗ gọi** (mã chết),
`save_networks` lưu `{"model", "optimizer", "total_steps"}`, `adjust_learning_rate`
chia LR cho 10 và trả `False` khi `lr < 1e-6`.

## 10.5 `utils/earlystop.py` — dấu của `delta` là cái bẫy

```python
def __call__(self, score: float, trainer: Trainer):
    if self.best_score is None:
        self.best_score = score
        self.save_checkpoint(score, trainer)
    elif score < self.best_score - self.delta:     # delta = -0.001  ⇒  score < best + 0.001
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
    else:
        self.best_score = score
        self.save_checkpoint(score, trainer)
        self.counter = 0
```

Vì `delta` **âm**, điều kiện là `score < best + 0.001`. Nghĩa là **một lần tăng
0,0005 cũng bị coi là không cải thiện**. Cách viết này ngược với trực giác "delta =
ngưỡng cải thiện tối thiểu" trong tài liệu về early stopping. Hệ quả khi tái lập:
nếu bạn dùng `delta = 0`, số epoch dừng khác, checkpoint khác, điểm khác — và
**không có gì báo** là bạn đã đổi một siêu tham số.

Ngoài ra `save_checkpoint` gọi `trainer.save_networks("best")` **mỗi lần** điểm tốt
hơn: với ResNet-50 (97,8 MB trọng số, 269,4 MB cho cả file checkpoint) đây là I/O
đáng kể trên ổ chậm (như ổ NTFS gắn ngoài hoặc volume mạng).

## 10.6 `utils/eval.py` — sáu dòng quyết định mọi con số

```python
predict = model(in_tens, meta).sigmoid()          # 1 logit → P(fake)
y_pred.extend(predict.flatten().tolist())
y_true.extend(label.tolist())

r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
acc   = accuracy_score(y_true, y_pred > 0.5)
ap    = average_precision_score(y_true, y_pred)
```

- `> 0.5` là so sánh **nghiêm ngặt**: `P = 0.5` được tính là **thật**.
- `r_acc`/`f_acc` tách riêng để phát hiện mô hình "đoán bừa một lớp": nếu
  `r_acc = 1,0` và `f_acc = 0,0`, ACC tổng có thể vẫn 0,5 nhưng mô hình vô dụng.
- `average_precision_score` của sklearn gộp các điểm có cùng score; tự cài đặt mà
  không gộp sẽ lệch.

## 10.7 `test.py` — một vòng lặp cho mọi bộ dữ liệu con

```python
for i, dataset in enumerate(cfg.datasets_test):
    cfg.dataset_root = os.path.join(dataset_root, dataset)
    cfg.datasets = [""]
    model = get_network(cfg.arch)                        # dựng lại mỗi bộ
    model.load_state_dict(torch.load(cfg.ckpt_path)["model"])
    test_results = validate(model, cfg)
    rows.append([dataset] + list(test_results.values()))
```

- `["model"]` bóc lớp bọc, khớp với `save_networks`.
- Dựng lại model mỗi bộ dữ liệu là lãng phí (chỉ tốn thời gian nạp), không sai.
- Ghi CSV một dòng/bộ. **Không** có ngưỡng nào được chọn trên test — đúng nguyên
  tắc.

## 10.8 `demo.py` — và một trích dẫn sai cần sửa

```python
trans = transforms.Compose((
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
))
for img_path in file_list:
    img = Image.open(img_path).convert("RGB")
    img = trans(img)
    if args.aug_norm:                                   # mặc định True
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    prob = model(in_tens).sigmoid().item()
    print(f"Prob of being synthetic: {prob:.4f}")
```

`--aug_norm` mặc định **True**, nên demo **có** chuẩn hoá ImageNet — chỉ viết thành
bước riêng sau `ToTensor` thay vì lồng trong `Compose`. Về toán là tương đương.
Khác biệt thật giữa demo và eval chỉ là **không có augmentation** (không random
crop, không lật), hoàn toàn đúng cho suy luận.

::: warn
Trong kho này, docstring của `src/dire_core/preprocess.py` viết rằng "`demo.py` của
tác giả **thiếu** `Normalize`; đó là lý do điểm trong demo không khớp điểm trong
paper". Câu đó **sai** về mặt sự kiện, và cần được sửa (lỗi F4, Phần 11.6). Đây là
ví dụ của một loại lỗi khó thấy: một giả thuyết hợp lý được viết vào tài liệu rồi
được trích lại như sự thật.
:::

## 10.9 `guided-diffusion/compute_dire.py` — sinh DIRE, từng khối

**a) Nạp mô hình**

```python
model, diffusion = create_model_and_diffusion(
    **args_to_dict(args, model_and_diffusion_defaults().keys()))
model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
model.to(dist_util.dev())
if args.use_fp16:
    model.convert_to_fp16()
model.eval()
```

- `create_model_and_diffusion` dựng U-Net **và** đối tượng diffusion theo
  `--timestep_respacing` (`ddim20` từ shell).
- `convert_to_fp16()` chuyển trọng số sang nửa độ chính xác: nhanh gần 2×, đổi lại
  sai số làm tròn cỡ vài phần nghìn trên thang `[−1,1]`.
- `model.eval()` tắt dropout. Thiếu bước này, kết quả sẽ thay đổi mỗi lần chạy
  (dù dropout chỉ 0,1).

**b) Chuẩn hoá kích thước**

```python
def reshape_image(imgs, image_size):
    if imgs.shape[2] != imgs.shape[3]:                 # không vuông
        imgs = transforms.CenterCrop(image_size)(imgs)
    if imgs.shape[2] != image_size:                    # vẫn sai
        imgs = F.interpolate(imgs, size=(image_size, image_size), mode="bicubic")
    return imgs
```

Đây là chỗ ảnh SD 512² bị thu về 256². Lưu ý thứ tự: **crop trước, resize sau** —
khác với pipeline của ResNet (resize rồi crop). Hai thứ tự này cho kết quả hơi khác
nhau, và bài báo dùng cả hai ở hai chặng khác nhau của pipeline.

**c) Đảo ngược rồi tái tạo**

```python
reverse_fn = diffusion.ddim_reverse_sample_loop
imgs = reshape_image(imgs, args.image_size)

latent = reverse_fn(
    model,
    (batch_size, 3, args.image_size, args.image_size),   # ← shape
    noise=imgs,                                          # ← ảnh gốc vào đây
    clip_denoised=args.clip_denoised,
    model_kwargs=model_kwargs,
    real_step=args.real_step,
)
sample_fn = diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
recons = sample_fn(
    model,
    (batch_size, 3, args.image_size, args.image_size),
    noise=latent,                                        # ← nhiễu vừa tìm
    clip_denoised=args.clip_denoised,
    model_kwargs=model_kwargs,
    real_step=args.real_step,
)
```

Hai điểm phải khớp tuyệt đối, và đây là nguồn của lỗi F1/F2 ở kho này:

- Tham số thứ hai là **`shape`** (tuple), tham số thứ ba là **`noise`** (tensor ảnh).
  Truyền ảnh vào vị trí `shape` là lỗi kiểu — sẽ nổ ngay, may mắn là lỗi ồn ào.
- `real_step=0` (mặc định) nghĩa là chạy **toàn bộ** `range(T)`. Nếu bạn đặt
  `real_step = 20` thay vì dùng `--timestep_respacing ddim20`, bạn chạy **20 bước
  đầu của thang 1000** (mốc 0…19), không phải 20 mốc rải đều. Hai thứ hoàn toàn
  khác nhau.

**d) Tính và lưu**

```python
dire = th.abs(imgs - recons)                          # trong [-1,1] ⇒ [0,2]
dire = (dire * 255.0 / 2.0).clamp(0, 255).to(th.uint8)
recons = ((recons + 1) * 127.5).clamp(0, 255).to(th.uint8)
imgs = ((imgs + 1) * 127.5).clamp(0, 255).to(th.uint8)
```

Phân tích thang đo — **chỗ dễ sai nhất của cả tài liệu**:

| Đại lượng | Miền giá trị | Phép đổi | Kết quả |
|---|---|---|---|
| `imgs`, `recons` | `[−1, 1]` | `(x+1)·127.5` | `[0, 255]` |
| `dire = \|imgs − recons\|` | `[0, 2]` | `·255/2 = ·127.5` | `[0, 255]` |

Vậy `dire_u8 ≈ |Δpixel|` — **thang 1×, không khuếch đại**. Đây là điểm mấu chốt để
đánh giá lỗi F3: bản trong kho này nhân **4** rồi `clip(0,255)`, tức bão hoà mọi
giá trị ≥ 64.

**e) Lưu ảnh: bẫy BGR**

```python
cv2.imwrite(f"{dire_save_dir}/{fn_save}",
            cv2.cvtColor(dire[i].numpy(), cv2.COLOR_RGB2BGR))
```

OpenCV đọc/ghi theo thứ tự **BGR**; PIL/torch theo **RGB**. Quên đổi thì ảnh lưu ra
bị đảo kênh đỏ–xanh. Với ảnh thường thì thấy ngay; với DIRE (ảnh sai số) thì **không
ai nhận ra bằng mắt**, nhưng classifier học trên kênh đảo — và nếu lúc test không
đảo thì lệch phân phối.

**f) Chia việc bằng MPI**

```python
dist_util.setup_dist(os.environ["CUDA_VISIBLE_DEVICES"])
# ... và chạy: mpiexec -n 4 python compute_dire.py ...
```

`image_datasets.py::ImageDataset_for_reverse` chia ảnh theo `shard[shard::num_shards]`,
nên 4 tiến trình chia nhau 4 phần rời nhau. Đây là cách tăng thông lượng đơn giản
nhất, và là lý do script chạy trên 4 GPU.

## 10.10 `gaussian_diffusion.py` — ba hàm quyết định DIRE

**DDIM sampling (đi xuống) — công thức [13] của tài liệu này:**

```python
def ddim_sample(self, model, x, t, clip_denoised=True, ..., eta=0.0):
    out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, ...)
    eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
    alpha_bar      = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
    alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
    sigma = (eta * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                 * th.sqrt(1 - alpha_bar / alpha_bar_prev))
    mean_pred = (out["pred_xstart"] * th.sqrt(alpha_bar_prev)
                 + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps)
    sample = mean_pred + nonzero_mask * sigma * noise      # eta=0 ⇒ sample = mean_pred
    return {"sample": sample, "pred_xstart": out["pred_xstart"]}
```

**DDIM inversion (đi lên) — công thức [16]:**

```python
def ddim_reverse_sample(self, model, x, t, clip_denoised=True, ...):
    assert eta == 0.0, "Reverse ODE only for deterministic path"
    out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, ...)
    eps = (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
           - out["pred_xstart"]) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
    alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)
    mean_pred = (out["pred_xstart"] * th.sqrt(alpha_bar_next)
                 + th.sqrt(1 - alpha_bar_next) * eps)      # không có sigma: tất định
    return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}
```

Đối chiếu từng dòng với toán:

| Dòng mã | Toán |
|---|---|
| `sqrt_recip_alphas_cumprod[t] * x − pred_xstart) / sqrt_recipm1_alphas_cumprod[t]` | `ε = (x/√ᾱ_t − x̂₀)/√(1/ᾱ_t − 1)`, tương đương [12] |
| `pred_xstart * sqrt(alpha_bar_prev) + sqrt(1 − alpha_bar_prev − sigma²) * eps` | [13] + thành phần $\sigma$ của [14] |
| `pred_xstart * sqrt(alpha_bar_next) + sqrt(1 − alpha_bar_next) * eps` | [16] |

**Vòng lặp và `real_step`:**

```python
def ddim_sample_loop_progressive(self, model, shape, noise=None, ..., real_step=0):
    if noise is not None:
        img = noise
    else:
        img = th.randn(*shape, device=device)      # ← nếu noise=None thì tạo nhiễu
    indices = (list(range(self.num_timesteps))[::-1] if not real_step
               else list(range(self.num_timesteps))[:real_step][::-1])
    for i in indices:
        t = th.tensor([i] * shape[0], device=device)
        out = self.ddim_sample(model, img, t, ...)
        yield out
        img = out["sample"]
```

Ba chi tiết đọc kỹ sẽ thấy:

1. `th.randn(*shape, ...)` — vì `shape` phải là **tuple/list** (`assert isinstance`).
   Truyền tensor vào đây là `TypeError` ở `th.randn(*tensor)`. Đây chính là lỗi F1.
2. `indices` đi **ngược** (`[::-1]`) vì sampling đi từ `T` về 0.
3. Với `real_step = k > 0`, `indices = list(range(T))[:k][::-1]` = `{k−1, …, 0}` —
   tức **k bước cuối trong thang T** (mốc gần 0), **không** phải `k` mốc rải đều.
   Đây là chi tiết mà lỗi F2 đã hiểu sai.

**`p_mean_variance` — nơi phép cắt xảy ra:**

```python
if clip_denoised:
    pred_xstart = pred_xstart.clamp(-1, 1)
```

Đây là phép cắt phi tuyến đã nói ở Phần 5.5: nó làm ODE mất tính thuận nghịch
chính xác, và nó tác động **mạnh hơn lên ảnh thật** (vì ảnh thật có nhiều pixel ở
biên `[−1,1]` do bão hoà 0/255).

## 10.11 Những điều chỉ đọc mã mới biết

| Điều | Vì sao quan trọng |
|---|---|
| `S = 20` là **tập mốc** `{0, 50, …, 950}`, không phải tham số hàm | quyết định NFE = 40 và điểm bắt đầu tái tạo |
| `real_step=k` lấy **k bước cuối** của thang `T` | "20" theo nghĩa này khác "ddim20" |
| `assert eta == 0.0` trong hàm đảo ngược | phép đảo ngược chỉ hợp lệ ở nhánh tất định |
| `alphas_cumprod_next` nối thêm `0.0` | ở `t = T−1`, `ᾱ_T` được định nghĩa là 0 |
| `clip_denoised=True` là mặc định | phá thuận nghịch chính xác, tạo "sàn" sai số |
| `convert_to_fp16()` chỉ chạy khi `use_fp16` | đổi độ chính xác ⇒ đổi DIRE ở mức vài phần nghìn |
| `reshape_image`: crop **trước**, resize **sau** | khác thứ tự của pipeline ResNet |
| `cv2.cvtColor(..., RGB2BGR)` | ảnh DIRE lưu ra là BGR trên đĩa |
| Tiền xử lý DIRE giữ **3 kênh** | không chuyển xám; bài báo không thảo luận |
| `EarlyStopping(delta=−0.001)` | tăng 0,0005 bị coi là không cải thiện |
| `warmup=False` | `scheduler.step()` không chạy; không có warmup |
| `init_weights` không có chỗ gọi | mã chết, đừng mô tả là "được dùng" |

# Phần 11 — Đọc mã của kho này: đối chiếu và các sai lệch

## 11.0 Mức tái lập: đọc nhãn trước khi đọc số

Đường đánh giá mặc định của kho này dùng **DIRE map phát hành** và **trọng số của
tác giả**, nên nó là **`R0` — chấm lại** (re-scoring): kiểm đường đo của mình có
khớp đường đo của họ. Nó **không** chứng minh:

- DIRE map đúng (nếu tác giả tính sai, mình dùng đúng map sai đó);
- detector train lại được (đó là `R2`);
- ảnh sinh lại được (`R3`).

Ba trục **độc lập**, không phải cấp bậc. Nhãn `R0` **cấm** dùng cụm "tái lập DIRE".
Chi tiết: `docs/adr/adr-0006-reproduction-levels.md`.

## 11.1 Kiến trúc: ba đường vào, một lõi

```
                 ┌─────────────────────────────┐
                 │  dire_core (Python thuần)   │
                 │  preprocess · detector      │
                 │  score · batch · provenance │
                 └──────┬───────────────┬──────┘
                        │               │
             ┌──────────▼───┐    ┌──────▼─────────┐
             │  dire_local  │    │  dire_modal    │
             │  CLI + API   │    │  eval · api    │
             └──────────────┘    └────────────────┘
```

Quy tắc: **không** có bản sao thứ hai của tiền xử lý, cách đọc checkpoint, hay cách
đổi logit → `P(fake)`. `tests/test_core_parity.py` khoá điều này bằng phân tích AST.

Lý do tồn tại của quy tắc này rất cụ thể: trong mã của tác giả, `demo.py` và
`datasets.py` là hai bản gần giống nhau của cùng một pipeline — và chúng đã lệch
nhau (không phải ở `Normalize`, mà ở augmentation).

## 11.2 `src/dire_core/` — phần đã đúng và có test

| Hàm | Làm gì | Vì sao đáng tin |
|---|---|---|
| `preprocess.build_transform` | `Resize(256) → CenterCrop(224) → ToTensor → Normalize` | test AST khoá đủ 4 bước ở **một** chỗ |
| `preprocess.PREPROC` | chuỗi `"resize256_centercrop224_normalize"` | ghi vào CSV/`run.json` để lần chạy không lẫn |
| `detector.unwrap_state` | bóc `model`/`state_dict`/`net`, bỏ `module.` | file gốc 269,4 MB có bọc optimizer |
| `detector.fc_out_features` | đọc `fc.weight.shape[0]` | không đoán 1 hay 2 logit |
| `score.sigmoid` | sigmoid ổn định số học | test giá trị đã biết, không cần torch |
| `score.prob_fake` | sigmoid hay softmax theo `n_out` | quy ước lấy từ checkpoint |
| `batch.batch_logits` | chia lô 64, `inference_mode` | CLI/API/eval dùng chung một vòng lặp |
| `provenance` | nhãn `R0/R1/R2/R3` + metadata | ADR-0006 khoá ngữ nghĩa |

Ba đoạn đáng đọc kỹ:

```python
# detector.py — đọc quy ước lớp cuối từ chính checkpoint, không đoán
def fc_out_features(state: dict) -> int:
    state = {(k[len("module."):] if k.startswith("module.") else k): v
             for k, v in state.items()}
    for key in ("fc.weight", "fc.0.weight", "classifier.weight", "last_linear.weight"):
        if key in state:
            return int(state[key].shape[0])
    return 0        # không nhận ra ⇒ chỗ gọi quyết định báo lỗi
```

```python
# preprocess.py — một pipeline duy nhất cho mọi đường
def build_transform(transforms):
    return transforms.Compose([
        transforms.Resize(RESIZE),          # 256
        transforms.CenterCrop(CROP),        # 224
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),    # ImageNet
    ])
```

```python
# score.py — sigmoid ổn định, không bao giờ tràn
def sigmoid(z: float) -> float:
    z = float(z)
    if z >= 0:
        return 1.0 / (1.0 + _math.exp(-z))
    e = _math.exp(z)
    return e / (1.0 + e)
```

`preprocess.py` có **một câu sai** trong docstring (về `demo.py`, lỗi F4). Ngoài
câu đó, logic đúng.

## 11.3 `src/dire_modal/compute_dire.py` — ba lỗi nằm cạnh nhau

Đây là hàm lõi, khoảng 20 dòng:

```python
ddim = gd.SpacedDiffusion(
    use_timesteps=gd.space_timesteps(diffusion.num_timesteps, str(steps)),   # (a)
    model_mean_type=diffusion.model_mean_type,
    model_var_type=diffusion.model_var_type,
    loss_type=diffusion.loss_type,
    rescale_timesteps=diffusion.rescale_timesteps,
)
noise = diffusion.ddim_reverse_sample_loop(          # (b)  LỖI F2: dùng `diffusion`
    model, x0, clip_denoised=True, model_kwargs={}, eta=0.0, progress=False
)                                                     # (c)  LỖI F1: `x0` là `shape`
if isinstance(noise, tuple):
    noise = noise[0]
recon = ddim.ddim_sample_loop(                        # (d)  dùng `ddim` — không đối xứng
    model, noise.shape, noise=noise, clip_denoised=True, model_kwargs={}, progress=False
)
...
diff_u8 = np.clip(np.abs(orig_u8.astype(np.int16) - recon_u8.astype(np.int16)) * 4,
                  0, 255).astype(np.uint8)            # (e)  LỖI F3: ×4 rồi clip
```

Bốn nhận xét, theo thứ tự nghiêm trọng:

1. **(c) F1 — không chạy được.** Chữ ký thật là
   `ddim_reverse_sample_loop(self, model, shape, noise=None, …)`. Truyền `x0` vào vị
   trí `shape` ⇒ `noise=None` ⇒ rơi vào nhánh `th.randn(*x0)` với `x0` là tensor
   ⇒ `TypeError`. Đúng phải là `(model, x0.shape, noise=x0, …)`.
2. **(b) F2 — sai chi phí và sai biểu diễn.** `ddim` được tạo đúng (20 mốc) nhưng
   **không được dùng** cho phép đảo ngược; đảo ngược chạy trên `diffusion` (1000
   mốc). Nếu chỉ sửa F1, mỗi ảnh tốn `1000 + 20 = 1020` NFE (25× bài báo) và `x_T`
   của thang 1000 đem tái tạo bằng thang 20 **không phải cặp nghịch đảo** ⇒ DIRE
   mất nghĩa. Sửa: dùng `ddim` cho **cả hai** chiều.
3. **(e) F3 — sai thang dữ liệu.** Hệ số 4 là để nhìn rõ trên gallery, nhưng hàm
   này cũng là hàm `process_one` dùng để lưu `.npy`. `clip(0,255)` làm bão hoà mọi
   giá trị ≥ 64 ⇒ mất thông tin. Sửa: tách `dire_raw()` (thang 1×, để lưu) khỏi
   `dire_display()` (×4, chỉ để vẽ).
4. **(a) Đúng.** `SpacedDiffusion` với `space_timesteps(1000, "20")` cho đúng tập
   mốc `{0, 50, …, 950}` như `--timestep_respacing ddim20`.

Ngoài ba lỗi này, hai điểm cần kiểm tra thêm (chưa có kết quả kiểm):

- `_load_adm` đặt `dropout=0.0` trong khi shell gốc dùng `0.1`. Với `model.eval()`
  dropout bị tắt nên **không ảnh hưởng** kết quả — nhưng nó làm cấu hình khác
  `run.json` khai báo, và cần ghi rõ.
- `_load_adm` không gọi `dist_util.setup_dist(...)` và không gọi
  `convert_to_fp16()` dù `use_fp16=True` khai báo. Cái sau làm chậm ~2× (không sai
  kết quả); cái trước có thể làm sai **thiết bị** (`dist_util.dev()` trả CPU nếu
  chưa setup).

## 11.4 `src/dire_modal/eval.py` — nơi con số được sinh ra

Cấu trúc: `pairs_from_raw` dò cặp `(ảnh gốc, DIRE map)` từ cây dữ liệu →
`score_pair` chấm từng cặp → `metrics` tính 6 độ đo → `row_for_result` ghi một dòng
CSV kèm metadata → `run_json` ghi bối cảnh của cả lần chạy.

Những điểm thiết kế đáng chú ý:

- **`label_of` suy nhãn từ đường dẫn** (`0_real`/`1_fake`), không từ file metadata —
  giống `ImageFolder`, nhưng có kiểm.
- **`ACC_THRESHOLD = 0.5`** và `_accuracy` dùng `>` nghiêm ngặt, khớp `validate`.
- **`_average_precision`** cài biến thể sklearn (gộp score trùng), có test đối chiếu.
- **`row_for_result`** ghi `n_real`, `n_fake`, `preproc`, `threshold`, `git_sha`,
  `created_at`. Cặp lỗi ⇒ ô metric để trống, không ghi 0.
- **`run_json`** ghi `preproc`, `acc_threshold`, tiers, số ảnh — để một lần chạy
  không thể lẫn với lần khác.

## 11.5 Chạy thật: bốn lệnh và ba cổng

```bash
# 1) Dữ liệu — xem kế hoạch trước, tải 0 byte
python -m dire_modal.local_fetch --tiers e2min --mode plan
python -m dire_modal.local_fetch --tiers e2min --mode fetch

# 2) Xem trước sẽ chấm bao nhiêu cặp (không cần dữ liệu)
modal run -m dire_modal.eval::local_pairs --tiers e2

# 3) Chấm điểm trên Modal
bash scripts/01_repro.sh --quick      # e2min: 3 file ⇒ 1 ô + cổng S2
bash scripts/01_repro.sh              # e2,ckpts: 22 file ⇒ 32 ô

# 4) Trên máy nhà, không cần Modal
pip install -e .
dire info                                        # không cần torch
dire detect anh.jpg --weights checkpoints/lsun_adm.pth
dire batch ./thu_muc --out scores.csv
```

Ba cổng phải qua **trước khi** đọc bất kỳ con số nào:

| Cổng | Câu hỏi | Kiểm bằng | Nếu trượt |
|---|---|---|---|
| **S2** | `ACC` của ô ADM→ADM ≥ 0,99? | dòng `lsun_adm/adm` trong CSV | sai checkpoint hoặc sai tiền xử lý; **đừng đọc AP** |
| **S6** | chạy lại có ra cùng CSV? | `diff` hai file = rỗng | có yếu tố ngẫu nhiên chưa tắt |
| **S9** | local và cloud dùng cùng lõi? | `pytest tests/test_core_parity.py` | có bản sao thứ hai của pipeline |

Thứ tự kiểm khi S2 trượt — theo thứ tự chi phí tăng dần:

1. `n_real`/`n_fake` trong CSV (nhãn có bị đảo không?).
2. `preproc` trong `run.json` (có phải `resize256_centercrop224_normalize`?).
3. sha256 của checkpoint so với manifest.
4. Đường dẫn DIRE map có đúng nhánh (không lấy `recons` làm `dire`?).
5. Cuối cùng mới nghi ngờ DIRE.

## 11.6 Bảng sai lệch F1–F5

| # | Vị trí | Vấn đề | Hệ quả | Cách sửa |
|---|---|---|---|---|
| **F1** | `compute_dire.py:241` | `ddim_reverse_sample_loop(model, x0, …)` — `x0` vào vị trí `shape` | `TypeError`, hàm không chạy | `(model, x0.shape, noise=x0, …)` |
| **F2** | cùng file | đảo ngược trên `diffusion` (1000 mốc), tái tạo trên `ddim` (20 mốc) | 1020 NFE (25×); không phải cặp nghịch đảo ⇒ DIRE vô nghĩa | dùng `ddim` cho cả hai ⇒ 40 NFE |
| **F3** | `compute_dire.py:257` | `|Δ|·4` rồi `clip(0,255)`, và **lưu** vào `.npy` | bão hoà ở 64; biểu diễn khác DIRE phát hành | tách `dire_raw` (1×) / `dire_display` (4×) |
| **F4** | `preprocess.py:7-8` | nói `demo.py` thiếu `Normalize` — sai | lý do "vì sao điểm demo lệch" là bịa | sửa hoặc bỏ câu đó |
| **F5** | `_load_adm` | `dropout=0.0` (gốc 0.1); không `convert_to_fp16()`; không `setup_dist` | chậm ~2×; rủi ro sai thiết bị; cấu hình lệch `run.json` | khớp cấu hình hoặc ghi rõ trong `run.json` |

Ba lệnh tự kiểm, không cần GPU:

```bash
# F1 — chữ ký thật của hàm
python - <<'PY'
import inspect
from guided_diffusion import gaussian_diffusion as gd
print(inspect.signature(gd.GaussianDiffusion.ddim_reverse_sample_loop))
PY
# kỳ vọng: (self, model, shape, noise=None, ...)

# F2 — hai đối tượng khác nhau cho hai chiều
grep -n "ddim_reverse_sample_loop\|ddim_sample_loop" src/dire_modal/compute_dire.py

# F3 — hệ số khuếch đại trên đường lưu
grep -n "\* 4" src/dire_modal/compute_dire.py
```

## 11.7 Bảng "chỉ code mới nói" (không có trong bài báo)

| Sự thật | Nguồn | Vì sao phải biết |
|---|---|---|
| `S = 20` = tập mốc `{0,50,…,950}` | `respace.py` | quyết định NFE và điểm bắt đầu |
| `real_step=k` = k bước **cuối** thang `T` | `gaussian_diffusion.py` | khác "k mốc rải đều" |
| `clip_denoised=True` mặc định | `compute_dire.py` | phá thuận nghịch chính xác |
| lưu BGR qua OpenCV | `compute_dire.py` | đảo kênh nếu quên |
| `EarlyStopping(delta=−0.001)` | `earlystop.py` | số epoch dừng phụ thuộc dấu |
| `warmup=False` | `config.py` | không có warmup dù có scheduler |
| `init_weights` không được gọi | `trainer.py` | mã chết |
| `Blur+JPEG` đặt **trước** crop | `datasets.py` | vị trí đặt đổi phân phối nhiễu |
| 1 250 bước/epoch (80k ảnh, batch 64) | suy ra từ `train.py` | cơ sở của mọi ước lượng chi phí |
| checkpoint 269,4 MB vs trọng số 97,8 MB | `detector.py` docstring | chi phí I/O khi lưu "best" |

# Phần 12 — Từ đọc hiểu tới nghiên cứu: lộ trình thực hành

## 12.1 Bốn bài tập kiểm hiểu (không cần GPU)

1. **Đo NFE.** Viết một "mô hình giả" có `forward` đếm số lần gọi, chạy
   `ddim_sample_loop` với `SpacedDiffusion(space_timesteps(1000, "20"))` và đếm.
   Kỳ vọng: 20. Rồi chạy với `SpacedDiffusion` 1000 mốc và `real_step=20`; đếm lại
   và giải thích vì sao vẫn là 20 nhưng **khác tập mốc**.
2. **Cài AP không dùng sklearn.** Trên 1 000 điểm score giả, cài AP hai cách (gộp
   score trùng / không gộp) và đo chênh lệch. Kỳ vọng: chênh lệch nhỏ nhưng khác 0.
3. **Kiểm thang đo.** Cho `x = 0,5` và `x̂ = 0,4` trên thang `[−1,1]`. Tính DIRE
   theo mã tác giả và theo mã nhân 4. Kỳ vọng: 12,75 và 51,0 — cái sau bắt đầu
   chạm trần `clip` khi `|Δ| ≥ 0,5` (tức 64 pixel).
4. **Kiểm rò rỉ.** Đọc `pairs_from_raw` và chỉ ra chỗ nào đảm bảo ảnh thật và ảnh
   giả cùng split. Rồi tìm một cách sửa **sai** (ví dụ lấy mọi file `real`) và mô
   tả hệ quả lên con số.

## 12.2 Bốn thí nghiệm đề xuất, xếp theo chi phí

| # | Thí nghiệm | Trả lời câu hỏi | Chi phí ước lượng |
|---|---|---|---|
| E1 | `clip_denoised=False` vs `True`, `S ∈ {10, 20, 50, 100}` | cơ chế: sai số Euler hay phép cắt? | 1 000 ảnh × 8 cấu hình × 2S NFE |
| E2 | thay ADM bằng iDDPM làm `I`/`R` | DIRE là tính chất của quá trình hay của ADM? | như trên, thêm 1 checkpoint |
| E3 | 1 kênh (xám) vs 3 kênh | kênh màu có mang thông tin? | train 2 detector |
| E4 | chuẩn hoá DIRE theo miền (trừ trung bình của ảnh thật cùng miền) | có cứu được ô 76,9%? | 1 detector × 2 miền |

E1 là thí nghiệm **rẻ nhất và có giá trị lý thuyết cao nhất**: nó trực tiếp kiểm
Phụ lục B. Nếu khoảng cách DIRE(thật) − DIRE(sinh) **không** giảm khi `S` tăng, thì
lập luận sai số Euler sai, và cần tìm cơ chế khác.

## 12.3 Cách viết kết quả để không tự lừa

Bốn quy tắc, rút từ chính các lỗi đã gặp trong kho này:

1. **Ghi mức tái lập trong mọi bảng** (`R0`/`R1`/`R2`/`R3`). Không có nhãn thì
   không có cách nào phân biệt "tôi đo được" với "tôi chép lại".
2. **Ghi `n_real`/`n_fake` cạnh mọi metric.** Một ACC 99% trên 10 ảnh và trên
   1 000 ảnh là hai thứ khác nhau.
3. **Không chọn ngưỡng trên test.** Chọn trên val, đóng băng, rồi báo test.
4. **`null` thay vì 0.** Nếu một ô không đo được, để trống và ghi lý do.

## 12.4 Đọc tiếp

| Bước | Đọc gì | Mục đích |
|---|---|---|
| 1 | Phần 1–3 tài liệu này | chắc nền xử lý ảnh + metric |
| 2 | DDPM [[3]], Mục 1–2 | hiểu quá trình thuận và ELBO |
| 3 | DDIM [[5]] | hiểu vì sao tất định và đảo ngược được |
| 4 | Bài báo DIRE [[1]], Mục 3 | đọc phương pháp bằng ngôn ngữ gốc |
| 5 | Phần 10–11 tài liệu này | đọc mã, đối chiếu |
| 6 | Tự chạy `--tiers e2min` | thấy một con số thật của chính mình |
