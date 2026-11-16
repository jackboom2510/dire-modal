# Tái lập **từ đầu** (R1+R2+R3) — không dùng một byte nào của tác giả

> Đường này trả lời câu: *"nếu không có map và checkpoint của tác giả, mình có
> dựng lại được toàn bộ phép đo không?"*
>
> Khác với `scripts/01_repro.sh`: bản đó nạp DIRE map **đã phát hành** rồi chấm
> bằng detector **đã phát hành** ⇒ **chấm lại (R0)**, chỉ chứng minh harness của
> mình khớp harness của họ.

## 1. Ba thành phần, và cái gì *không* phải scratch

`src/dire_core/provenance.py` tách ba thành phần độc lập:

| Thành phần | Nhãn | Đường này tự làm bằng |
|---|---|---|
| DIRE map | `R1` | ADM-LSUN + DDIM `S = 20`, trên ảnh **mình** sinh |
| Detector | `R2` | ResNet-50, 1 logit + BCE, siêu tham số của tác giả |
| Ảnh sinh | `R3` | sinh từ checkpoint công khai, seed **mình** chọn |

Nhãn được phép nói là **`R1+R2+R3`**. Ba thứ **không** nằm trong "scratch",
phải ghi rõ trong báo cáo:

- **Trọng số diffusion** (ADM) là của OpenAI. Train ADM-LSUN từ đầu là ~250k
  step × batch 128 trên 8×A100 — ngoài ngân sách 0đ, và cũng không phải câu hỏi
  của dự án. Đây là *công cụ*, không phải *kết quả*.
- **LSUN-Bedroom** là dữ liệu thật duy nhất; giấy phép **chỉ cho nghiên cứu**.
- **Kiến trúc** ResNet-50 / U-Net lấy từ torchvision / guided-diffusion.

## 2. Chín cổng vàng (G0–G8)

Cổng nào cũng được **thi hành bằng code**, không bằng lời hứa:

| # | Điều kiện | Thi hành ở đâu |
|---|---|---|
| **G0** | Không đọc `dire/**` | `dire_core/scratch.py::assert_no_published` — chặn theo **từng thành phần đường dẫn**, bắt cả `raw/dire/x`, `D:/data/dire/...`, `./dire/x` |
| **G1** | Ảnh thật + checkpoint có sha256 khớp manifest | `tools/scratch_repro.sh --stage fetch` ghim vào `manifests/scratch_sha256.tsv`, lần sau lệch là **dừng** |
| **G2** | Seed train/test rời nhau | `split_seeds()` + `assert_disjoint()`; `gen.json` ghi cả hai danh sách |
| **G3** | `gen.json` đủ trường | `check_gen_meta()` — thiếu `seeds`/`sampler`/`steps` là `AssertionError`, không phải cảnh báo |
| **G4** | DIRE đúng giao thức | `ddim.Schedule` lấy mốc từ `ddim_timesteps(1000, 20)` ⇒ `{0, 50, …, 950}`; map lưu ở thang gốc `[0, 2]` |
| **G5** | NFE/ảnh `= 2S` | `check_nfe()` đếm **thật** trong `_predict`; sai là `AssertionError` |
| **G6** | Ngưỡng chọn trên val | `scratch_train.train` chỉ nhìn val; test chấm **một lần**, ghi `train.json` |
| **G7** | Guard chi phí trước container | `compute_dire.preflight_cost` (có sẵn) |
| **G8** | `sec/step` là số **đo** | `--stage bench` chạy 3 ảnh rồi ghi `sec_per_step.json`; chưa có file này thì `--stage gen/dire/train` **từ chối chạy** |

G5 và G8 là hai cổng về tiền. Lỗi cũ chạy 1020 NFE thay vì 40 — sai **25 lần**
mà không có exception nào; mọi ước lượng chi phí trong repo lại đang dựa vào một
`sec_per_step = 0,35` **chưa ai đo**.

## 3. Chạy

```bash
export DIRE_LOCAL_ROOT=D:/dire-data          # Windows: đường dẫn có ổ đĩa

bash tools/scratch_repro.sh                  # in kế hoạch, 0 đồng
bash tools/scratch_repro.sh --stage gates    # kiểm 9 cổng
bash tools/scratch_repro.sh --stage fetch    # tải LSUN + ADM (~vài GB), ghim sha256
bash tools/scratch_repro.sh --stage bench    # ĐO sec/step (bắt buộc trước GPU)
bash tools/scratch_repro.sh --stage gen      # R3: sinh 500 ảnh
bash tools/scratch_repro.sh --stage dire     # R1: tính DIRE map
bash tools/scratch_repro.sh --stage train    # R2: train detector
```

Đẩy dữ liệu lên Volume (sau khi `fetch`, và sau khi bạn đồng ý chờ):

```bash
modal volume put dire-raw "$DIRE_LOCAL_ROOT/scratch/ckpt/adm_lsun.pt" ckpt/adm_lsun.pt
modal volume put dire-raw "$DIRE_LOCAL_ROOT/scratch/lsun/png" scratch/real
```

## 4. Chi phí — tính bằng GPU-giờ, không bằng lời

Công thức duy nhất:

```
GPU-giờ = N ảnh × NFE/ảnh × sec_per_step / 3600
NFE/ảnh = 2S cho DIRE map, S cho sinh ảnh
```

Với `N = 1000 + 1000` ảnh, `S = 20`:

| Bước | NFE | Ở 0,004 s/step | Ở 0,027 s/step |
|---|---|---|---|
| sinh 1 000 ảnh | 20 000 | 0,022 GPU-h | 0,15 GPU-h |
| DIRE 2 000 ảnh | 80 000 | 0,089 GPU-h | 0,60 GPU-h |
| train detector (~800 map × 40 epoch) | — | ~0,1 GPU-h | ~0,1 GPU-h |

Tức **dưới 1 GPU-giờ** cho cả đường, ở mọi mức `sec/step` đã từng ghi trong
repo. Con số này rẻ hơn nhiều so với ước lượng cũ ($22–150) vì ở đây `N` là
1 000 ảnh, không phải 336 000. Quy mô đầy đủ (LSUN val 30k + 30k) thì nhân 30.

> Đừng dùng bảng này như báo giá. Nó chỉ đúng khi `sec_per_step` là số **đo**
> ở máy bạn — đó chính là việc của `--stage bench`.

## 5. Kết quả kỳ vọng, và câu **không được** nói

Kỳ vọng hợp lý: ACC 95–99 trên cặp `ADM-sinh vs LSUN-thật` (bài toán dễ nhất —
cùng một họ mô hình). Bài báo đạt 99,9 vì ảnh sinh của họ **chính là** ảnh họ
tạo ra bằng đúng checkpoint đó, seed của họ.

Không được nói:

- "DIRE đạt 99,9" — đó là số của bài báo, không phải số của đường này;
- "tái lập được DIRE" khi mới có `R1+R2` mà ảnh vẫn của tác giả;
- "tốt hơn / kém hơn DIRE" — khác `N`, khác seed, khác generator thì không so được;
- bất kỳ số nào **chưa có `train.json` + `dire.json` + `gen.json`** đi kèm.

## 6. Hạn chế (ghi rõ, không giấu)

- **Chỉ có ADM.** `--generator` khác `adm` sẽ `NotImplementedError` — thêm
  iDDPM/PNDM cần checkpoint + loader riêng, và phải ghi metadata từng generator.
  Hệ quả: đường này mới chứng minh *in-distribution*, chưa chứng minh
  cross-generator (Table 3 của bài).
- **256×256 cố định.** Ảnh khác cỡ bị bicubic về 256 trước khi vào mô hình.
- **`R3` không tái lập được ảnh của tác giả.** Seed khác, sampler khác tham số
  thư viện ⇒ ảnh khác. Đó là chủ ý: mình cần *một* tập ảnh sinh hợp lệ, không
  phải *đúng* tập của họ.
- **Chưa chạy trên GPU thật.** Toàn bộ logic thuần (lịch, NFE, seed, cổng) có
  29 test chạy mỗi commit; phần torch (`ddim.py`, `scratch_gen.py`,
  `scratch_train.py`) **chưa được thực thi** trong sandbox soạn code. Lần chạy
  đầu phải kiểm: tên tham số U-Net khớp checkpoint, `ddim_reverse` trả đúng
  shape, VRAM ở batch 1.
- **LSUN giấy phép nghiên cứu** — không dùng cho sản phẩm.

## 7. Danh mục kiểm trước khi báo cáo một con số

- [ ] `gen.json` có `ckpt_sha256` và `seeds`; test seeds **không** giao train.
- [ ] `dire.json` có `nfe_total = 2·S·N` và `s_per_image` là số đo.
- [ ] `train.json` có `best_val_acc` và `test_acc_at_0.5`; test chỉ chấm một lần.
- [ ] `run.json` ghi `label: "R1+R2+R3"` và `sec_per_step_is_measured: true`.
- [ ] Báo cáo ghi rõ: trọng số diffusion của OpenAI, LSUN cho nghiên cứu,
      kiến trúc từ torchvision.
