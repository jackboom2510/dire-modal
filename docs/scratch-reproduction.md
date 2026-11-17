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

## 2b. Bản clean-room (không dùng code tác giả)

Đường trong tài liệu này vẫn mượn `guided_diffusion` (clone repo tác giả) để
dựng U-Net và ResNet-50. Nếu câu hỏi là "**code** của tác giả có đúng không",
dùng `src/dire_zero/` — bản viết lại độc lập, kiểm bằng 566 cặp tên→shape, và
tìm ra ba lỗi trong chính `ddim.py` ở đây (map `NaN`, NFE thiếu 2, nhiễu dư
0,5 %). Xem [`docs/from-zero-clean-room.md`](from-zero-clean-room.md).

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

### 4.1 Vì sao `::dire` mất ~49 phút cho 800 ảnh — kiểm bằng FLOP, không bằng cảm giác

Một lượt forward của ADM 256² (552,8 M tham số) là **415 GMAC = 829 GFLOP**: đo
bằng cách dựng đúng kiến trúc trong `dire_zero.unet` trên thiết bị `meta` rồi cộng
FLOP của từng `Conv2d`/`Conv1d`/`Linear` (4 phép chú ý ở 16² và 8² chỉ chiếm 6
GMAC — conv mới là phần đắt). Từ đó:

| Đại lượng | Giá trị |
|---|---|
| forward / ảnh (S = 20) | 40 (`2S`: một lượt đảo ngược, một lượt tái tạo) |
| FLOP / ảnh | 40 × 829 GFLOP = **33,2 TFLOP** |
| 800 ảnh | **26,5 PFLOP** |
| `sec/step` đo được (`::bench`, A10G, fp16, batch 1) | 0,0917 s ⇒ **9,0 TFLOPS** |
| ⇒ thời gian | 26,5 PFLOP / 9,0 TFLOPS = **2 934 s ≈ 49 phút** |
| **sàn vật lý** (100 % đỉnh fp16 A10G = 31,2 TFLOPS) | **851 s ≈ 14 phút** |

9,0 TFLOPS là **29 %** đỉnh fp16 của A10G — đúng mức một U-Net batch 1 thường đạt
(GroupNorm chạy fp32, softmax fp32, batch 1 không lấp đầy tensor core). Nên
0,0917 s/step **không phải số bất thường**, và 49 phút không phải ước lượng phóng
đại: nó là 29 % hiệu suất của một phép tính 26,5 PFLOP.

Bảng độ nhạy, để thấy con số nào là khả dĩ:

| `sec/step` | TFLOPS | % đỉnh A10G | 800 ảnh | đọc thế nào |
|---|---|---|---|---|
| 0,014 | 59 | 190 % | 7,5 phút | **bất khả** — vượt đỉnh |
| 0,030 | 27,6 | 89 % | 16 phút | chỉ khi đã tối ưu mạnh |
| 0,050 | 16,6 | 53 % | 27 phút | cần batch ≥ 8 |
| **0,0917** | **9,0** | **29 %** | **49 phút** | **số đang dùng** |
| 0,150 | 5,5 | 18 % | 80 phút | máy chia sẻ / tải nhiệt |

Điểm quan trọng: **không có kịch bản nào dưới 14 phút** trên A10G ở batch 1, vì
đó là sàn của 26,5 PFLOP. Còn nếu ai nghi 49 phút là sai, cách kết luận nhanh
nhất không phải tranh luận mà là đo hai lần:

```bash
modal run -m dire_zero.modal_app::bench                  # ~2 phút, ~0,03 USD
modal run -m dire_zero.modal_app::dire --splits test --generator adm --n 5 --max-usd 0.2
# 20 ảnh: đo trực tiếp giây/ảnh, rồi nhân 40 để ra 800 ảnh
```

### 4.2 Đổi GPU rẻ hơn cả đổi thuật toán

Cùng phép tính 26,5 PFLOP, giá niêm yết trong `dire_zero.budget.PRICES`:

| GPU | đỉnh fp16 | ở 29 % ⇒ TFLOPS | 800 ảnh | giá/giờ | tiền |
|---|---|---|---|---|---|
| A10G | 31,2 | 9,0 | 49 phút | 1,10 USD | **0,90 USD** |
| A100-40GB | 312 | 90 | **5 phút** | 2,50 USD | **0,20 USD** |

A100 nhanh gấp ~10 lần nhưng chỉ đắt gấp 2,3 lần ⇒ **rẻ hơn ~4,5 lần** cho cùng
phép đo. Đây là suy luận từ FLOP và giá niêm yết, **chưa đo** trên A100: muốn
dùng thì chạy `::bench` với `gpu="A100-40GB"` trước, đừng tin bảng này như báo giá.

Các đòn khác, xếp theo mức đáng làm:

| Đòn | Tác dụng | Giá phải trả |
|---|---|---|
| `S = 10` thay `S = 20` | nhanh đúng 2× | đổi phép đo: Table 4 của bài có 98,2 (S=10) so với 99,7 (S=20) trên SD-v1 |
| gộp 8 ảnh / batch | **1,5–2×** (không phải 3–4×: trần là 3,5× và batch nhỏ không đạt) | thêm bộ nhớ, thêm một đường code phải test |
| ít ảnh hơn (`--n 50` ⇒ 200 ảnh) | 4× | khoảng tin cậy rộng hơn |
| A100-40GB | **~10×** | đổi GPU, phải đo lại `sec/step` |

`sec/step` **luôn** là số đo tại chỗ: `budget.load_sec_per_step` từ chối chạy nếu
thiếu `bench.json`, và mọi ước lượng tiền trong repo đi qua đúng một công thức.

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
