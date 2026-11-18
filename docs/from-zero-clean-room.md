# From zero — bản viết lại **thực sự** từ con số 0

> Đối tượng: người đã đọc `docs/dire-from-scratch.md` và muốn **chạy** DIRE mà
> không dựa vào một dòng code nào của tác giả.
>
> Code: `src/dire_zero/`. Test: `tests/test_zero_*.py` (100 test) +
> `tests/test_ddim_protocol.py` (11 test). Suite đầy đủ: **572 passed, 3 skipped**.
> Lệnh chạy: `python -m dire_zero …` hoặc `modal run -m dire_zero.modal_app::…`.

---

## 1. "From scratch" nghĩa là gì — và **không** nghĩa là gì

Repo này đã có một đường from-scratch (`docs/scratch-reproduction.md`, nhãn
`R1+R2+R3`). Nhưng đường đó vẫn **mượn code**:

| Thứ | Đường cũ | Đường này |
|---|---|---|
| U-Net 256 | `guided_diffusion` clone từ repo tác giả vào `/opt/dire` | tự viết, 566 tensor khớp từng shape |
| Attention | `guided_diffusion.unet.AttentionBlock` | tự viết (`blocks.Attention`) |
| DDIM + DIRE | `ddim.py` gọi `guided_diffusion` | tự viết (`diffusion.py`) |
| ResNet-50 | `vendor/networks/resnet.py` (chép từ repo tác giả) | tự viết (`detector.py`) |
| ACC/AP | `sklearn` | tự viết (`metrics.py`), có CI bootstrap |
| **Trọng số ADM** | checkpoint của OpenAI | **vẫn là checkpoint của OpenAI** |
| **LSUN-Bedroom** | dữ liệu thật | **vẫn là dữ liệu thật** |

Hai dòng cuối là **có chủ ý**, và phải nói rõ thay vì gọi tất cả là "từ số 0":

* **Trọng số diffusion không phải kết quả của dự án.** Train ADM-LSUN từ đầu là
  ~250 000 step × batch 128 trên 8×A100 — ngoài ngân sách 0 đồng, và cũng không
  phải câu hỏi "DIRE có tái lập được không". Đây là *công cụ*, như cái cân: mua
  cân không phải là gian lận khi cân.
* **LSUN-Bedroom là dữ liệu thật duy nhất** cho phép so sánh với bài báo. Tự
  chụp ảnh phòng ngủ rồi bảo đó là LSUN thì con số không còn nghĩa.

Nên câu đúng là: **bản này không dùng một dòng code nào của tác giả**, chứ không
phải "không dùng gì của ai". Nhãn vẫn là `R1+R2+R3`; cái mới là *độc lập triển
khai*: nếu code DIRE của tác giả có lỗi, bản này vẫn cho số khác — và chính vì
thế nó kiểm chứng được họ.

## 2. Bốn lớp chứng minh (không phải lời hứa)

Tự viết lại mà không kiểm thì chỉ là *một* cách hiểu khác, chưa chắc đúng. Bốn
lớp dưới đây chạy bằng `pytest` trên máy không có GPU:

**(1) Hình dạng tham số.** `tests/fixtures/adm_lsun_state_keys.json` chứa 566
cặp *tên → shape* của checkpoint ADM, sinh từ kiến trúc tham chiếu. Đây là **dữ
liệu**, không phải code, và không chứa một byte trọng số nào. Test đòi bảng ánh
xạ `enc.3.1.res.conv_in.weight ↔ input_blocks.7.0.in_layers.2.weight` là **song
ánh** và khớp shape từng cái.

```text
$ python -m dire_zero keys
OK: 566 tensor, 552,814,086 tham số — khớp adm_lsun_state_keys.json
```

Con số 552 814 086 không phải gõ tay: nó là tổng tham số của kiến trúc tham
chiếu. Sai một khối, sai một phép giảm mẫu, hay dùng conv stride 2 thay
average-pool, đều lệch hàng chục triệu — thấy ngay, không cần tải 2,5 GB.

**(2) Song ánh bắt lỗi im lặng.** `check_bijection` kiểm ba điều: không key nào
của checkpoint bị bỏ, không key nào của mình trỏ hụt, và shape khớp. Test có
sẵn hai ca "phá" (`test_bijection_catches_a_missing_block`,
`test_bijection_catches_wrong_shape`) để chắc rằng nó **đỏ** khi đáng đỏ.

**(3) Số học có đáp án đóng.** `tests/test_zero_schedule.py` khoá `ᾱ[0] = 0,9999`,
`ᾱ[250] = 0,5214`, mốc cuối `950` (không phải 999), `NFE/ảnh = 2S`.

**(4) Parity số với bản tham chiếu** (`test_parity_with_reference_if_available`).
Nạp **cùng bộ trọng số** vào hai mô hình rồi so đầu ra từng phần tử:

```text
$ DIRE_REFERENCE_UNET=… pytest tests/test_zero_unet.py -k parity
max abs diff = 0.0
```

Test này tự skip nếu máy không có bản tham chiếu — nhưng nó **đã được chạy** khi
phát triển, và nó là test duy nhất bắt được lỗi *ngữ nghĩa*: ví dụ chạy attention
**sau** phép tăng mẫu vẫn ra ảnh đúng hình dạng, vẫn hội tụ, nhưng tốn gấp 4 lần
compute và lệch hẳn kết quả.

## 3. Tám lỗi tìm được khi viết lại

Viết lại từ công thức không chỉ để "sạch". Bảy lỗi dưới đây lộ ra vì phải suy
nghĩ lại từng chỉ số — và phần lớn nằm trong **đường đang chạy** của repo, tức
là đường mà `docs/runbook-local-manual.md` bảo người dùng chạy.

**Trạng thái:** 3.1–3.3 đã sửa trong `src/dire_modal/ddim.py`, có test riêng
(`tests/test_ddim_protocol.py`, 11 test, chạy bằng mô hình "tiên tri" nên không
cần GPU lẫn checkpoint). 3.4–3.7 lộ ra khi chạy thật: 3.4 qua `::selfcheck` trên
A10G, 3.5 và 3.6 qua lượt `::dire` đầu tiên trên A10G (cùng một traceback), 3.7
qua lượt `::gen` đầu tiên trên A10G.

### 3.1 `ᾱ[0] = 1,0` ⇒ DIRE map toàn `NaN` (đường cũ)

`dire_core.ddim_schedule.alphas_cumprod` trả list có `ᾱ[0] = 1,0` (đúng nghĩa
"chưa thêm nhiễu"), nhưng `dire_modal.ddim._predict` lại dùng `ᾱ[t]` với `t = 0`
là mốc đầu của phép **đảo ngược**:

```text
ε̂ = (x₀ − √ᾱ₀ · x̂₀) / √(1 − ᾱ₀) = (x₀ − x₀) / 0 = NaN
```

Một `NaN` duy nhất ở bước đầu lan qua toàn bộ đường, nên **mọi** DIRE map của
đường scratch là `NaN` — không exception, không cảnh báo, chỉ có kết quả vô
nghĩa. Không test nào bắt được vì không test nào chạy `invert` với một mô hình.

Cách sửa: `Schedule._ab(t)` trả `abar[t + 1]` (mốc 0 → 0,99988 thay vì 1,0).
`test_no_nan_at_the_first_milestone` khoá lại: map **phải hữu hạn** ở mọi `S`.

### 3.2 NFE đếm sai ở hai phía

Đường cũ: `invert` chạy `S−1` lượt (zip của các cặp mốc), `reconstruct` cũng
`S−1` ⇒ thực tế 38 NFE cho `S = 20`, trong khi `check_nfe` đòi 40 — nên hoặc
assert nổ, hoặc (nếu ai đó hạ ngưỡng xuống 38) chi phí bị báo thấp hơn thực tế
5 %. Cả hai phía nay chạy **đúng S lượt**; `dire_zero` còn đếm ngay trong hàm
gọi mô hình (`ModelCounter`) thay vì đếm bằng công thức.

### 3.3 Bước cuối của tái tạo dùng `ᾱ[0]` thay vì `ᾱ = 1`

Bản tham chiếu, ở bước cuối, dùng `ᾱ_prev = 1` (ảnh sạch) nên `x̂₀` trả về **đúng**
`x̂₀`. Dùng `ᾱ[0] = 0,9999` thì đầu ra còn ~0,5 % nhiễu, và phần nhiễu đó vào
thẳng DIRE map — sai lệch nhỏ nhưng **có hệ thống trên mọi ảnh**, tức là nó dịch
chuyển cả phân bố chứ không thêm nhiễu ngẫu nhiên.

> Ba lỗi trên đã sửa. Nhưng điều đáng nhớ hơn: **không lỗi nào trong ba lỗi này
> làm chương trình dừng.** Chúng cho ra ảnh, ra file `.npy`, ra báo cáo — chỉ là
> báo cáo toàn `NaN` hoặc lệch vài phần trăm. Đó là lý do tồn tại của
> `selfcheck` và của các test dùng mô hình tiên tri.

### 3.4 fp16: hai lỗi dtype chỉ tồn tại trên GPU

`selfcheck` nạp ADM thật (2,2 GB, `strict=True` qua 566 tensor) rồi chết ở lượt
mạng **đầu tiên**, hai lần liên tiếp, với hai thông báo khác nhau:

```text
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and Half
RuntimeError: mixed dtype (CPU): expect parameter to have scalar type of Float
```

Cả hai đều là hệ quả của một quyết định **đúng**: `time_embedding` và `GroupNorm32`
cố ý tính ở float32, vì sin/cos của `f_i·t` (t tới 1000) và chuẩn hoá theo nhóm
đều mất chính xác ở fp16 — đúng chỗ tín hiệu DIRE sống. Nhưng:

* đầu ra float32 của `time_embedding` đi thẳng vào `time_mlp` fp16;
* `model.half()` biến tham số của chính `GroupNorm32` thành fp16, trong khi lớp
  đó đẩy `x` lên float32 để tính.

Sửa: nhúng thời gian đổi dtype **trước** `time_mlp`; `ZeroUNet.half()` giữ các
lớp norm ở float32 (đúng như ADM, nơi chỉ conv được chuyển fp16). `GroupNorm32`
vẫn tự đổi tham số nếu ai đó gọi `.to(torch.float16)` thẳng.

> Vì sao hai lỗi này lọt: **toàn bộ** test chạy float32 trên CPU, còn lỗi chỉ
> xuất hiện ở fp16 trên GPU. Nay có ba test chạm đúng đường đó —
> `test_group_norm_accepts_half_parameters` (không cần conv fp16 nên không bị
> skip), `test_half_keeps_norms_in_float32`, và `test_full_fp16_pipeline_is_finite`
> chạy trọn `dire_map` 4 bước trên mô hình fp16. Bài học giống mục 3.1: **một
> cấu hình chưa từng được chạy thì coi như chưa có test.**

### 3.5 Thiết bị: `load_image` trả CPU, mô hình trên GPU

Lượt `::dire` thật đầu tiên trên A10G chết ở lượt mạng **đầu tiên**:

```text
File "/root/dire_zero/unet.py", line 224, in forward
    emb = self.time_mlp(time_embedding(timesteps, self.cfg.model_channels).to(dtype))
RuntimeError: Expected all tensors to be on the same device, but found at least
two devices, cuda:0 and cpu! (when checking argument for argument mat1 ...)
```

Nguyên nhân không nằm ở `unet.py`. `ModelCounter.__call__` tạo `timesteps` trên
`x.device`; `compute_maps` lấy `img = load_image(path)` — mà `load_image` trả
tensor **CPU** — rồi dùng thẳng. Vậy `timesteps` ở CPU ⇒ `time_embedding` ở CPU ⇒
`time_mlp` (đã ở CUDA) nổ ở `addmm`. Dòng `.to(dtype)` trong `unet.py` không đổi
thiết bị, nên nó không cứu được.

Vì sao lọt: đường duy nhất từng chạy trên GPU là `selfcheck`, và nó gọi
`load_image(...).cuda()` **trước** khi vào `dire_map`; mọi test khác chạy CPU nên
hai bên cùng thiết bị.

Sửa hai lớp:

* `compute_maps` đưa ảnh sang `next(f.model.parameters()).device` (sau khi kiểm
  kích thước, để ảnh sai cỡ không bị chuyển lên GPU vô ích), và `.cpu()` trước
  `np.save` — map nằm trên GPU, `np.save` cần bộ nhớ CPU;
* `dire_map` tự chuẩn hoá: ảnh khác `sched.device` thì `.to(sched.device)`, để
  mọi đường gọi đều đúng thay vì nổ ở giữa `time_mlp`.

Test canh: `test_compute_maps_moves_images_to_the_model_device` đặt mô hình trên
`meta` (một thiết bị khác hẳn CPU) rồi kiểm ảnh được chuyển sang đúng đó — trên
CPU thuần, test kiểu "hai bên cùng CPU" không bắt được gì.

### 3.6 Ước lượng tiền đếm cả cây, bỏ qua `--n`

Cùng lượt chạy đó in:

```text
[tiền] 82000 ảnh × 40 NFE = 3280000 NFE · 83.55 GPU-h · 91.90 USD
```

trong khi lệnh là `--n 200` trên `train,test` với **một** generator: thực tế 800
ảnh (0,32 GPU-h, ~0,90 USD). Sai 102 lần, và sai theo hướng **chặn oan**: với
`--max-usd` mặc định 5, lượt chạy rẻ này bị `require_estimate` từ chối.

Nguyên nhân: ước lượng đếm toàn bộ cây (`n_pairs + n_fake`) còn vòng lặp dựng
`items` cắt `[:n]` cho từng ô. Sửa: `_plan_images()` đếm **đúng thứ vòng lặp sẽ
lấy** (`min(len(ô), n)` theo từng cặp split × lớp), và `::dire` in thêm dòng
`[kế hoạch] real … + fake …` để chênh lệch lộ ra ngay thay vì nằm trong một con số
trông hợp lý.

Test canh: `test_plan_images_caps_every_cell_by_n_and_not_the_whole_tree`.

### 3.7 `save_png` gọi `.numpy()` trên tensor CUDA — chết ở ảnh đầu tiên

Lượt `::gen` đầu tiên:

```text
[tiền] 500 ảnh × 50 NFE = 25000 NFE · 0.64 GPU-h · 0.70 USD (khoảng 0.56–1.12) @ A10G 1.10/h
[nạp] /zero/ckpt/adm_lsun.pt · float16 · 10.7s · sha256 a37c32fffd31…
Traceback (most recent call last):
  File "/root/dire_zero/pipeline.py", line 111, in generate
    save_png(img, out / name)
  File "/root/dire_zero/pipeline.py", line 66, in save_png
    Image.fromarray(to_uint8(arr).squeeze(0).permute(1, 2, 0).numpy()).save(path, format="PNG")
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
```

Đã trả tiền nạp checkpoint và chạy xong bước sinh **đầu tiên** rồi mới chết, vì
`to_uint8` cố ý **giữ nguyên thiết bị** của đầu vào (nó là hàm số học thuần, không
phải hàm I/O). Đường CPU không bao giờ thấy lỗi này — đúng như 3.4, 3.5: ba lỗi
liên tiếp, cùng một dạng "chỉ tồn tại trên GPU".

Sửa: mọi đường ghi ảnh đi qua **một** hàm `pipeline.to_hwc_uint8()`, làm đúng ba
bước theo thứ tự `.detach() → .cpu() → .contiguous()`, và nhận cả `(1, C, H, W)`
lẫn `(C, H, W)` (map 1 kênh trả ảnh `L`). Cùng lỗi trong
`src/dire_modal/scratch_gen.py` cũng được sửa — nó sẽ chết y như vậy ở lần chạy
sau, chỉ muộn hơn.

**Test canh, và vì sao phải có hai lớp:**

* `test_save_png_moves_the_tensor_to_cpu_before_numpy` — ghi thật một PNG rồi đọc
  lại (đúng số học), **và** kiểm thứ tự `.cpu()` trước `.numpy()` trong chuỗi gọi;
* `test_no_module_converts_a_device_tensor_without_cpu` — quét **từng hàm** trong
  `src/`: hàm nào có `.numpy()` sau `permute`/`to_uint8` mà `.cpu()` không đứng
  trước thì fail. Bài này bắt **cả lớp lỗi**, nên lần sau ai viết lại ở module
  khác cũng bị chặn — đó là điều test chức năng không làm được khi máy không có GPU.

> Bài học chung của 3.4–3.7: **bốn lỗi, một nguyên nhân.** CPU và GPU khác nhau ở
> dtype (fp16), ở thiết bị (CUDA vs host), và ở chỗ nào *ghi ra đĩa*. Mọi hàm
> `to_*`/`save_*` phải tự nói rõ nó trả tensor ở đâu, và mọi `.numpy()` phải có
> `.cpu()` ngay trước — không có ngoại lệ "chắc là CPU rồi".

### 3.8 Bốn lỗi nữa của `::dire`, lộ ra khi thêm ảnh sinh sau lần chạy đầu

Câu hỏi đặt ra: *đã chạy `::dire` trước `::gen`; giờ có phải chạy lại để có map
cho ảnh vừa sinh không?* Có — nhưng làm đúng thì không phải trả tiền hai lần.

**a) Ảnh sinh ra sau thì không thể có map từ trước.** `::dire` chỉ thấy thư mục
đang tồn tại lúc nó chạy; `::gen` đặt ảnh vào `data/<split>/ours_adm/` **sau** đó
⇒ lần chạy đầu không có `ours_adm` trong danh sách. Không có cách nào "tính bù"
từ map cũ: map là hàm của **ảnh**, không phải của tên thư mục.

**b) Generator bị gộp thành `fake`.** `::dire` dựng item 3 phần tử
`(path, label, split)`; `_as_item` thấy thiếu generator nên gán `"fake"`. Hệ quả:
`maps/<split>/ours_adm/` không bao giờ được tạo, map của `adm` và `ours_adm` ghi
đè nhau nếu tên file trùng (ảnh sinh tên là `seed.png`), và
`::train --generator ours_adm` báo "chưa có map". Sửa: item mang generator
(`_items_to_map`), một hàm duy nhất `map_rel_path()` quyết định đường dẫn.
Test canh: `test_items_carry_the_generator_not_a_single_fake_bucket`.

Map của lần chạy cũ thì **không phải tính lại**: `dire.json` của lần đó ghi
`source` cho từng map (đường dẫn ảnh gốc, tức là có tên generator), nên sửa nhãn
là đổi tên file:

```bash
python tools/relabel_maps.py --maps <volume>/maps            # xem trước, không đụng đĩa
python tools/relabel_maps.py --maps <volume>/maps --apply    # đổi tên thật
# hoặc trên Modal (volume đã mount ở /zero), không cần GPU:
modal run -m dire_zero.modal_app::relabel --apply
```

Đích đã tồn tại ⇒ **không ghi đè**, báo xung đột; item không suy được generator
⇒ để nguyên và báo. 0 USD thay vì ~1 USD cho 1 000 ảnh. Test:
`tests/test_zero_maps_index.py`.

**c) `dire.json` bị ghi đè ⇒ mất dấu lần chạy trước.** `compute_maps` ghi
`maps/dire.json` mỗi lần, nên bản ghi của lần đầu (n, NFE, giây) biến mất trong
khi các `.npy` của lần đó vẫn còn — một cây map không còn tài liệu giải thích.
Sửa: `dire.json` thành **sổ** (`runs`, cộng dồn NFE, dedupe theo `path`) và mỗi
lần chạy còn một bản đầy đủ ở `maps/dire-runs/<created>-NNN.json`. Hai lần chạy
khác `steps` ⇒ `steps` cấp ngoài là `null` + `steps_mixed`, không lấy bừa một giá trị.

**d) Không có `--skip-existing` ⇒ trả tiền lại cho ảnh `real`.** Ảnh `real` nằm
trong danh sách **mọi** lần chạy, nên thêm một generator là một lần tính lại toàn
bộ `real` rồi ghi đè đúng file vừa tạo. Với cây 500/500 mỗi split, đó là ~1 000
ảnh × 3,67 s ≈ 1 giờ GPU và ~1 USD cho **zero** map mới. Nay: `--skip-existing`
bỏ qua map đã có (file khác rỗng), `n_new`/`n_reused` nói rõ cái nào mới, và ước
lượng tiền đếm theo số ảnh **sẽ tính thật**. Nếu không còn gì mới thì `::dire`
in "không có gì mới — không gọi GPU" và trả về, không bật máy.

`::report` thêm `maps.by_split_generator`: tồn kho `.npy` **trên đĩa**, để một
thư mục `maps/<split>/fake/` còn sót từ bản cũ lộ ra ngay thay vì lẫn vào tập train.

### 3.9 Hai cổng của `::dire`: tiền **và** thời gian

Lượt thật, 15 generator × 2 split, `--n 500 --steps 20`:

```text
[kế hoạch] 10100 ảnh (n ≤ 500 mỗi ô, 2 split, 15 generator: adm, dalle2, …)
[kế hoạch] sẽ tính 9700, dùng lại 400 map đã có
[tiền] 9700 ảnh × 40 NFE = 388000 NFE · 9.88 GPU-h · 10.87 USD @ A10G 1.10/h
GateError: ước tính 10.87 USD > trần 5.00 USD.
```

Cổng tiền hoạt động đúng: nó dừng **trước** khi nạp checkpoint và trước NFE đầu
tiên, nên lượt bị chặn tốn ~0 (chỉ vài giây CPU của container). Nhưng nâng
`--max-usd 12` sẽ **không** làm lượt chạy thành công — và nay cổng **thời gian**
chạy trước cổng tiền, vì khi cả hai cùng vượt thì lời khuyên "nâng `--max-usd`"
là lời khuyên sai (nó chỉ làm hoá đơn của phần chạy dở đắt hơn):

| | Số | Kết luận |
|---|---|---|
| Ước tính | 9,88 GPU-h | |
| `timeout` của function | 6 h | bị kill ở giờ thứ 6 |
| Ngân sách an toàn (75 %) | 4,5 h | phải chia nhỏ |
| Số giây/ảnh ở `S=20` | 1,834 s | `2S × 0,0917` |
| Tiền/ảnh | **0,00112 USD** | đọc dòng `sẽ tính N` rồi nhân |

Ba con số cuối là công cụ duy nhất cần: `N × 0,00112 USD` và `N × 1,834 s`.
`::dire` nay tự kiểm cả hai và, khi vượt thời gian, in **lệnh copy–paste được**
cho từng nhóm, kèm ước tính của chính nhóm đó.

Bản đầu của tính năng này in ra nhóm **sai**, và người dùng bị chặn lần thứ hai
với đúng bản gợi ý vừa đọc — một vòng lặp tốn thời gian mà lỗi nằm ở chỗ khó thấy:

```text
[kế hoạch] 5500 ảnh … 7 generator …          # 5,20 GPU-h, trần 4,50 h
  --generator adm,iddpm,dalle2,ddpm,diff-projectedgan,diff-stylegan   ← 4 000 ảnh = 4,08 h
  --generator if
```

Nhóm đầu "vừa dưới trần" 4,08 h — nhưng **mỗi lệnh còn mang theo `real`**, và 600
ảnh `real` chưa có map ⇒ lệnh thật là 4 000 + 600 = 4,69 h > 4,50 h ⇒ `GateError`
y như cũ. Bài học: một bản gợi ý mà **không tự kiểm bằng chính cổng của mình** thì
không phải là gợi ý.

Hai sửa:

1. `real` được tính vào **nhóm chạy đầu tiên** (chỉ nhóm đó trả tiền cho nó; lệnh
   sau dùng lại map nhờ `--skip-existing`);
2. chia **cân bằng** (LPT: việc lớn nhất vào nhóm đang nhẹ nhất) thay vì "nhét
   đầy" nhóm đầu — nhóm sát trần chỉ cần chậm 10 % là bị kill. Số nhóm bắt đầu từ
   `ceil(tổng / trần)` và **tăng dần cho tới khi mọi nhóm nằm dưới trần** (LPT chỉ
   bảo đảm ≤ 4/3 OPT, không bảo đảm ≤ trần: 10 000 ảnh với trần 2 940 ảnh chia 4
   nhóm vẫn ra 3 000 ảnh/nhóm).

Kết quả cho đúng ca trên:

```text
  modal run -m …::dire --n 500 --steps 20 --splits train,test --skip-existing \
      --generator diff-projectedgan,iddpm,if
      # 2600 ảnh (gồm 600 real chưa có map) = 2.65 GPU-h ≈ 2.92 USD
  modal run -m …::dire --n 500 --steps 20 --splits train,test --skip-existing \
      --generator adm,dalle2,ddpm,diff-stylegan
      # 2500 ảnh = 2.55 GPU-h ≈ 2.81 USD
```

2,65 h và 2,55 h — cách trần 4,5 h gần 2 h, và mỗi lệnh đều dưới `--max-usd` mặc
định 5 USD (2,92 USD).

**Chia nhỏ không làm tăng tiền** — tổng NFE như cũ — nhưng làm ba điều:

1. mỗi container nằm dưới timeout, nên `vol.commit()` cuối **chạy được**;
2. hỏng ở nhóm thứ 5 thì 4 nhóm trước đã nằm trên Volume, lần sau dùng lại;
3. mỗi lệnh tự in bảng tiền của riêng nó, không phải tin một con số 10,87 USD.

`--generator` nhận danh sách (`adm,pndm`), và tên gõ sai là **lỗi**: `sdv1` trong
khi cây chỉ có `sdv2` mà vẫn chạy thì sẽ trả tiền cho `real` rồi báo "xong" với 0
map mới.

> Cùng một bài học với 3.6: một con số "trông hợp lý" (10,87 USD) không nói gì về
> việc lượt chạy có **kết thúc** được hay không. Trần tiền và trần thời gian là
> hai câu hỏi khác nhau — và bản gợi ý chia nhỏ phải được kiểm bằng **chính cổng
> vừa chặn**, nếu không thì nó chỉ là một vòng lặp thứ hai.

> Bài học chung của 3.5–3.8: **ước lượng, thiết bị và tính "đã có hay chưa" đều
> là thứ "trông đúng" khi chỉ đọc code.** Cái thứ nhất chỉ sai trên GPU, cái thứ
> hai chỉ sai khi `--n` nhỏ hơn kích thước cây, cái thứ ba chỉ sai ở **lần chạy
> thứ hai** — và cả ba đều phải chạy thật mới lộ.

## 4. Chạy

### 4.1 Trên máy, không cần GPU

```bash
pip install -e ".[zero]"          # torch + numpy + pillow
python -m dire_zero info          # lịch, mốc, NFE, bảng tiền
python -m dire_zero keys          # đối chiếu 566 key
python -m dire_zero selftest      # chạy trọn 4 bước trên U-Net 32×32, ~20 giây
```

`selftest` là bước nên chạy **trước khi** thuê GPU: nó gọi đúng bốn hàm mà
`modal_app` gọi, chỉ khác kích thước. Xanh ở đây thì lỗi còn lại chỉ có thể là
môi trường.

### 4.2 Dùng dataset riêng (ổ `F:`) thay vì tải LSUN

Nếu bạn đã có cây `DiffusionForensics` trên ổ, không cần tải lại gì:

```bash
python -m dire_zero peek    --tar F:/du-lieu/.../dire/val/lsun_bedroom/adm.tar.gz  # bên trong có gì
python -m dire_zero dataset --root F:/du-lieu --license research-only   # kiểm, 0 đồng
python -m dire_zero ingest  --root F:/du-lieu --license research-only \
                            --splits train val test   # → ./data/dire-zero-norm (F: khi thiếu chỗ)
```

`dataset` in bảng `split × generator` (đã gộp `stylegan_official` → `stylegan`),
cảnh báo bất đối xứng + cảnh báo thư mục ảnh giải nén dở nằm cạnh tar, rồi **đếm
thật** số ảnh của phần được chọn. Nó **dừng** nếu thiếu giấy phép, nếu dataset
không cùng phân bố với ADM, hoặc nếu bộ lọc rỗng. `ingest` đổ tar.gz ra PNG 256²,
tên file theo sha256 pixel, kèm manifest + `data_card.json`.

Cây **có `images/val/`** (5 archive) ⇒ dùng val của người đóng gói, không cắt từ
train. Cây cũng **có `images/test/lsun_bedroom/adm.tar.gz`** ⇒ dòng headline ADM
làm được bằng dữ liệu phát hành, không cần `::gen`.

Nhãn phụ thuộc **ảnh giả lấy từ đâu**: `--fake-source author` (dùng ảnh của tác
giả) ⇒ tối đa `R1+R2`; `--fake-source gen` (tự sinh) ⇒ `R1+R2+R3`;
`--fake-source published` (map có sẵn) ⇒ `R0` — chỉ là *chấm lại*, không phải
tái lập. Chi tiết, gồm đường R0 trên CPU ~$0 và bảng lệch giữa bản kê với
inventory: [`docs/dataset-own.md`](dataset-own.md).

### 4.3 Volume: **một** cái duy nhất (`dire-zero`)

Đường cũ dùng hai volume (`dire-raw` ⇒ `/raw`, `dire-ckpt` ⇒ `/ckpt`) và dữ liệu liền
lạc ở `/raw/zero/**`. Hậu quả không phải thẩm mỹ: cùng một checkpoint 2,5 GB đã
nằm ở **sáu** đường khác nhau tuỳ lần chạy —

```text
/raw/checkpoints/lsun_adm.pth     /ckpt/pretrained/lsun_adm.pth
/raw/ckpt/adm_lsun.pt             /raw/zero/ckpt/adm_lsun.pt
zero/ckpt/adm_lsun.pt             ckpt/adm_lsun.pt
```

⇒ không ai biết bản nào là bản đang dùng. Từ nay bố cục do **một** module khai
báo (`src/dire_zero/volume.py`) và mọi app đọc từ đó:

```text
dire-zero → /zero
  ckpt/adm_lsun.pt   ckpt/detector.pth   ckpt/author/*.pth
  data/<split>/<gen>/*.png   data/manifest.tsv   data/data_card.json
  maps/<split>/<gen>/*.npy   gen/<split>/<gen>/*.png   runs/*.json   runs/author/*.json
  _migration/receipt.json
```

**Bê dữ liệu cũ sang, không tải lại 2,5 GB.** `dire_zero.migrate` là app riêng
(image `debian_slim`, **không** torch ⇒ không chạm GPU, 0 đồng):

```bash
modal run -m dire_zero.migrate::survey          # liệt kê, 0 ghi
modal run -m dire_zero.migrate::plan            # n_copy / n_conflict / GB
modal run -m dire_zero.migrate::apply --confirm        # chỉ checkpoint
modal run -m dire_zero.migrate::apply --confirm --data  # thêm cây dữ liệu
modal run -m dire_zero.migrate::verify                  # sha hai đầu
```

Ba bất biến, mỗi cái đều có test:

| Bất biến | Cách thực hiện |
|---|---|
| **không xoá** volume cũ | chỉ mở `/raw`, `/ckpt` để **đọc**; lệnh xoá duy nhất trong toàn file là `unlink` file tạm của chính nó (test bằng AST) |
| **không ghi đè** đích đã có | đích tồn tại ⇒ `conflict`, `apply` dừng và báo, không tự ý thay |
| **không để file cụt** | ghi `.migrating` rồi `os.replace` (nguyên tử trên cùng FS), sha kiểm lại sau khi đổi tên |

**Tên file không đủ để biết nó là gì.** Lần `survey` đầu tiên trên volume thật
tìm ra ba "ứng viên checkpoint", và cả ba đều là `checkpoints/lsun_adm.pth` —
**282.534.013 B**, tức 283 MB. Trọng số ADM 256² có 552,8 M tham số ⇒ ~2,2 GB.
283 MB là ResNet-50 của tác giả (23,5 M tham số). Nếu migration chỉ khớp tên,
`apply` sẽ bê nhầm file đó vào `ckpt/adm_lsun.pt`, ghi biên lai "sha khớp", rồi
`selfcheck` nổ `Missing key(s) in state_dict` — hỏng **muộn** nhất có thể, sau
khi mọi thứ trông như đã xong.

Vì vậy có **sàn kích thước**: ứng viên ADM phải ≥ 1,5 GB. File cùng tên mà nhỏ
hơn thì rơi vào `adm_rejected_by_size` của `survey`/`plan` với lý do rõ ràng, và
**không** được coi là ADM. Nếu `survey` báo `KHÔNG có ứng viên ADM nào đủ lớn`
thì trọng số ADM **chưa có trên Modal** — migration không thể tạo ra thứ không
tồn tại; phải đẩy `256x256_diffusion_uncond.pt` (~2,2 GB) lên bằng
`modal volume put dire-zero … ckpt/adm_lsun.pt`.

**Ba trạng thái của đích ADM, ba thông báo khác nhau** (bản đầu gộp cả ba thành
"1 xung đột", và lần chạy thật đã ra đúng thông báo sai đó):

| Trạng thái | `apply` nói gì |
|---|---|
| `ckpt/adm_lsun.pt` đã có, ≥ 1,5 GB | `[adm] đã có (2,xx GB) — không có gì để chuyển` |
| file ở đó nhưng < 1,5 GB | **dừng**, nói rõ "không phải trọng số ADM", không xoá, không ghi đè |
| không có ở đích **và** không có ở nguồn | **dừng** kèm lệnh `modal volume put …` |

Thiếu nguồn **không phải** xung đột: xung đột là có dữ liệu khác ở đích và cần
người quyết định; thiếu nguồn thì chỉ cần biết đích đã có hay chưa.

Detector tác giả đi đường riêng: `checkpoints/*.pth` trong khoảng 100–400 MB ⇒
`ckpt/author/<tên gốc>.pth` (giữ tên = tên generator: `lsun_adm`, `lsun_pndm`,
`imagenet_adm`, …), và `results/repro_*` ⇒ `runs/author/`. Đây là đường **R0**
— chấm lại trên map có sẵn, 0 đồng GPU — và nó **không bao giờ** ghi vào
`ckpt/adm_lsun.pt` hay `ckpt/detector.pth`. Nạp bằng
`dire_zero.detector.load_dire_detector`, hàm này đòi đúng 23.510.081 tham số và
từ chối `fc` 1000 lớp.

`discover_ckpts` khớp theo **tên file** (`adm_lsun.pt`, `lsun_adm.pth`,
`256x256_diffusion_uncond.pt`), không theo đường dẫn — vì đường dẫn chính là thứ
đang sai. Nhiều bản cùng tên ⇒ lấy bản **lớn nhất**, các bản còn lại ghi vào
`skipped` của biên lai để sau này còn truy được. `_migration/receipt.json` ghi
nguồn, đích, size, sha hai đầu, `layout_version: 1` và `deleted_anything: false`.

Dọn volume cũ **không** nằm trong app: đó là việc tay, chỉ sau khi `::verify` xanh
và đã có bản trên `dire-zero`. Không có cờ nào để migration xoá hộ.

#### Ba cái bẫy của `modal volume put`

Đã làm hỏng một lần chạy thật, nên ghi lại (kiểm từ source `modal/cli/volume.py`):

| Bẫy | Sự thật |
|---|---|
| `-r` | **không tồn tại** ở Modal 1.x. `put` đã đệ quy sẵn khi nguồn là thư mục; cờ lạ làm lệnh dừng **trước khi upload byte nào** |
| chạy lại | thiếu `--force` ⇒ `FileExistsError` → lệnh từ chối, kể cả khi chỉ một file đã có |
| đích là thư mục | `put … <dir> data` đặt **nội dung** vào `data/`, không sinh `data/<tên-thư-mục-local>/` |

#### Bẫy lớn nhất: `put` cả cây 200 nghìn file

`modal volume put <vol> <thư mục> <đích>` **không** ghi dần. Trong
`modal/volume.py` (1.6.x), mọi file được đăng ký trong bộ nhớ rồi ở cuối context
manager mới gửi **một** request `VolumePutFiles`:

```python
async def __aexit__(self, exc_type, exc_val, exc_tb):
    if not exc_val:                      # có exception ⇒ KHÔNG gửi gì
        ... upload từng file ...
        await self._client._stub.VolumePutFiles(request)
```

Đã xảy ra thật: 217 579/217 585 file rồi treo ⇒ **không file nào** vào FS, dù
thanh tiến trình đã gần đầy (nó đếm byte đã gửi). Chạy lại dừng đúng cùng con số.
Một file hỏng làm mất cả 22 GB.

Dùng `tools/put_resume.py` thay thế:

```bash
python tools/put_resume.py --local F:/dire-zero-norm --remote data --dry-run
python tools/put_resume.py --local F:/dire-zero-norm --remote data
python tools/put_resume.py --local F:/dire-zero-norm --remote data --only train/pndm --chunk 1
```

Nó liệt kê Volume trước (chỉ metadata), chỉ đẩy file **thiếu hoặc khác kích
thước**, mỗi lô 500 file là một lần ghi riêng, và đẩy `manifest.tsv` +
`data_card.json` **sau cùng**. `--chunk 1` là chế độ khoanh vùng file cứng đầu:
mỗi file một request, file hỏng chỉ mất chính nó.

Hai file metadata nằm ở **gốc** cây local, nên đẩy thủ công từng thư mục con sẽ
bỏ sót chúng — và `::checkdata` sau đó báo "thiếu manifest.tsv" (nay phân biệt
được với "manifest rỗng", mỗi ca một cách sửa).

Và một bẫy Windows: `MSYS_NO_PATHCONV=1` **tắt** đúng phép đổi `/f/…` → `F:/…`,
nên khi đã đặt biến đó thì phải viết `F:/dire-zero-norm`, không viết `/f/…`.
Đường dẫn `/f/…` chỉ đúng khi **để** MSYS đổi hộ (tức không đặt biến).

Mặc định `apply` **không** bê cây dữ liệu (`data/`, `maps/`, `gen/`, `runs/`) —
`--data` mới bê. Lý do: người gọi migration là để cứu checkpoint 2,5 GB, không
phải để một lệnh tự động sửa cả cây dữ liệu; `plan` vẫn in sẵn `n_copy` của phần
dữ liệu để ai muốn thì quyết định có căn cứ. Dữ liệu sinh từ máy local vẫn đi
đường `modal volume put` quen thuộc (mục 4.4).

#### Cân bằng lớp phải xét **từng bài toán**, không xét tổng

Manifest của cây DiffusionForensics có 1 000 ảnh thật và 12 368 ảnh giả của **14
generator** (mỗi cái ~1 000). Nhìn tổng là 1:12,4 — và bản đầu của `checkdata`
chặn đúng vì con số đó. Nhưng:

* Table 2 của bài báo là **từng dòng = từng generator**: một detector cho
  `real vs adm`, một detector cho `real vs pndm`, …
* `::dire --generator adm` đã dựng đúng bài toán đó, và ở đó tỉ lệ là **1000:1000**.

Nay `checkdata` dựng `tasks` = danh sách `{split, generator, real, fake, ratio, ok}`
và chỉ chặn khi **một bài toán** lệch quá `--max-ratio` (mặc định 5) hoặc khi một
generator không có ảnh thật cùng split. Gộp mọi generator thành một lớp giả vẫn
được phép, nhưng báo cáo thêm cảnh báo rằng con số đó không so được với Table 2.

`::train --generator <tên>` chọn đúng một bài toán; `train.json` ghi
`generators_used` và `note_scope` để không ai đọc nhầm phạm vi. `auto` (mặc định)
gộp mọi generator — đúng khi muốn đo "real vs bất kỳ fake", sai khi muốn so bảng.

#### `::checkdata` trên 200 nghìn file: timeout là bài toán độ trễ, không phải băng thông

Lần chạy thật với `cpu=2`, **một luồng**, timeout 1800 s đã bị cắt giữa đường.
22,5 GB không phải vấn đề (đọc tuần tự chỉ vài phút); vấn đề là **217 585 lần mở
file** trên FUSE của Volume — mỗi lần vài ms ⇒ hơn 30 phút.

Nay `checkdata` băm bằng **64 luồng** trên `cpu=8`, timeout 4 giờ, in tiến trình
`file/s` + thời gian còn lại. Ba mức dùng:

```bash
modal run -m dire_zero.modal_app::checkdata --splits test   # vài giây, đủ để lặp nhanh
modal run -m dire_zero.modal_app::checkdata --sample 2000   # mẫu cách đều, báo cáo ghi "sample"
modal run -m dire_zero.modal_app::checkdata                 # toàn bộ, cần cho con số cuối
```

`--sample` **không** được đọc thành "đã kiểm toàn bộ": báo cáo ghi
`sha_scope: "sample"` và `sha_checked_rows`. Đếm PNG trên đĩa nay mặc định
**tắt** (`--count-png` mới bật) vì một vòng `os.walk` qua 217 585 entry cũng tốn
hàng phút mà không chặn bước nào.

### 4.4 Trên Modal

```bash
# 0. đẩy checkpoint (2,5 GB) — dùng lại bản đã tải tay cho đường cũ
modal volume put dire-zero "$DIRE_LOCAL_ROOT/data/raw/sources/ckpt/adm_lsun.pt" ckpt/adm_lsun.pt

# 1. cổng nạp trọng số: strict=True + 1 ảnh + kiểm NaN/NFE
modal run -m dire_zero.modal_app::selfcheck

# 2. ĐO sec/step (bắt buộc: mọi ước lượng tiền dùng số này)
modal run -m dire_zero.modal_app::bench

# 3. R3 — sinh ảnh (bài báo dùng 50 bước cho sinh)
modal run -m dire_zero.modal_app::gen --n 500 --steps 50 --seed-base 1000

# 4. R1 — DIRE map (bài báo dùng 20 bước cho DIRE)
modal run -m dire_zero.modal_app::dire --n 500 --steps 20

# 4b. thêm map cho ảnh vừa sinh, KHÔNG tính lại ảnh real đã có map
modal run -m dire_zero.modal_app::dire --n 500 --steps 20 --skip-existing

# 5. R2 — train detector: **mỗi generator một dòng của Table 2**
modal run -m dire_zero.modal_app::train --epochs 40 --generator all
#    (một generator: --generator adm; gộp tất cả: --generator auto — bài toán KHÁC)

# 6. gom báo cáo
modal run -m dire_zero.modal_app::report
```

### 4.4.1 Sau `::dire` thì làm gì — đúng thứ tự

`::dire` chỉ tạo `.npy`; **chưa có một con số nào** cho tới bước 2 dưới đây.
Thứ tự này là thứ tự phụ thuộc dữ liệu, không phải thứ tự tuỳ thích:

| # | Việc | Lệnh | Ra cái gì | Chặn bởi |
|---|---|---|---|---|
| 0 | **Kiểm tồn kho map** | `::maps --n 500` | `by_split_generator`, số map thiếu, lệnh chạy tiếp | — |
| 1 | **Train từng generator** | `::train --generator all` | `runs/train/<gen>/detector.pth` + `train.json` mỗi cái | 0 phải đủ generator |
| 2 | **Chấm điểm từng ảnh** | `::evaluate --generator all` | `runs/eval/<gen>/{eval.json,scores.json}` | 1 |
| 3 | **Gom bảng** | `::report` | `train_by_generator` + `eval_by_generator` | 1, 2 |
| 4 | **Gallery + demo** | `tools/make_gallery.py --scores …` rồi Space | trang 2 có mẫu thật **và** có `p_fake` | 2 |
| 5 | **So với bài báo** | chưa có `::author_eval` (R0) | cột "bài báo" | cần checkpoint tác giả |

**Bước 0 — trước khi train, xem mình có gì.** `::dire` chạy theo nhóm nên rất dễ
kết thúc với 12 generator thay vì 15 mà không biết:

```text
"maps": {"total": 10200, "by_split_generator": {"test/adm": 500, "test/dalle2": 500, …}}
```

Thiếu ô nào thì chạy bù đúng ô đó (`--generator <tên> --skip-existing`), đừng train
rồi mới phát hiện.

**Bước 2 — `::evaluate`: bước dễ bị bỏ quên nhất.** `::train` chỉ ghi *số tổng hợp*
(`acc`, `ap`, `auc`) vào `train.json`; nó **không** ghi điểm của từng ảnh. Nhưng
`tools/make_gallery.py --scores` cần đúng một bảng `{tên mẫu: p_fake}`, và
`platforms/hf-spaces/demolib/provenance.py` từ chối hiện điểm nếu thiếu `gate`/`threshold`.
Không chạy bước này thì gallery vẫn dựng được, chỉ là mọi mẫu hiện "chưa có điểm" —
trang 2 mất đúng phần muốn khoe.

```bash
modal run -m dire_zero.modal_app::evaluate --generator all      # hoặc --generator adm
```

Ghi `runs/eval/<tag>/eval.json` (metric, có `acc_ci95`) và `scores.json` (điểm từng ảnh,
khoá `f"{generator}_{stem}"` — khớp đúng khoá `make_gallery` sinh ra). Ngưỡng trong
`train.json` là **logit**; `::evaluate` đổi sang xác suất bằng sigmoid rồi mới ghi, vì
`make_gallery --threshold` chỉ nhận `[0, 1]`. Đổi thang không đổi ACC/AP/AUC (hàm đơn
điệu tăng) nhưng đổi cách đọc — trộn hai thang là hiện điểm sai mà không có lỗi nào báo.

`gate` chỉ `passed` khi ngưỡng thật sự đến từ `val` với ≥ 20 mẫu; không đạt thì
`gate: failed` và `provenance` **từ chối** hiện điểm. Đó là hành vi đúng: ngưỡng chọn
trên test là tự khen mình.

```bash
# nối vào gallery, dùng đúng ngưỡng mà ::evaluate đã ghi
modal volume get dire-zero runs/eval/gen-adm /tmp/eval
python tools/make_gallery.py --maps maps/test --orig data/test \
  --scores /tmp/eval/scores.json --gate passed \
  --threshold "$(python -c 'import json;print(json.load(open("/tmp/eval/eval.json"))["threshold_prob"])')"
```

### 4.4.2 Mất mạng giữa lúc `::dire` đang chạy — chuyện gì xảy ra, kiểm thế nào

```text
Stopping app - local client disconnected. Use `modal run --detach` to keep apps
running even if your local client disconnects.
```

`modal run` (không có `--detach`) **giết app** khi client mất kết nối: container
dừng, function không "chạy tiếp". Không có lệnh nào nối lại một app đã bị dừng —
cách duy nhất là chạy lại, và nhờ `--skip-existing` thì lần chạy lại chỉ tính phần
còn thiếu.

Nhưng có một chỗ chết người nằm sau thông báo đó: **Volume v1 không tự ghi**.
Thay đổi chỉ bền sau `vol.commit()`, và bản trước chỉ commit **một lần ở cuối
hàm**. Nghĩa là một lượt bị dừng giữa đường mất **toàn bộ** map đã tính — đã trả
tiền cho từng giây GPU rồi mất sạch, và lần chạy lại trả tiền lại từ đầu.

Nay `::dire` commit theo hai lớp:

| Lớp | Khi nào | Mất tối đa |
|---|---|---|
| định kỳ | mỗi `COMMIT_EVERY = 200` ảnh | 200 ảnh (~6 phút GPU ≈ 0,11 USD) |
| `except` | exception mềm (lỗi mạng, Ctrl-C) | 0 |

Kill **cứng** (OOM, hết `timeout` của container) không chạy được `except`, nên lớp
định kỳ mới là lớp chính.

**Kiểm đã đủ chưa — không cần GPU, không cần nối lại app:**

```bash
modal run -m dire_zero.modal_app::maps --n 500          # CÙNG --n với lệnh bạn định chạy
modal run -m dire_zero.modal_app::maps --n 500 --verify 20
```

```text
ô                               ảnh    cần    map  thiếu  %
test/adm                          500    500    500      0  100.0
train/adm                         500    500    320    180   64.0
train/real                        500    500    500      0  100.0

tổng: 1820 map · thiếu 180 · ~0.18 GPU-h ≈ 0.20 USD (S=20, dùng lại 1820 map đã có)
  ! train/real: thiếu 60 map — chạy bất kỳ lệnh `::dire` nào với `--skip-existing`
  ! test/fake: còn thư mục `fake/` từ bản cũ — chạy `::relabel --apply` trước khi train

chạy tiếp (1 lệnh, mỗi lệnh đã có --skip-existing):
  modal run -m …::dire --n 500 --steps 20 --splits test,train --skip-existing --generator adm  # …
```

**`--n` PHẢI khớp `::dire`.** Đây là lỗi đã xảy ra thật: chạy `::maps` trần trên
cây 40 000 ảnh/ô ⇒ "thiếu 212 583 map". Đúng về số học, vô dụng về quyết định —
không ai định tính map cho 212 583 ảnh, và con số đó không nói phải chạy lệnh gì.
Mặc định nay là **2 000**, đúng mặc định của `::dire`; `--n 0` mới là "cả cây".

**Không có `--generator real`.** `real` không phải generator hợp lệ (nó có mặt
trong **mọi** lệnh `::dire`), nên bản đầu của `::maps` in ra một lệnh không chạy
được. Nay dòng gợi ý cho `real` chỉ nói "chạy bất kỳ lệnh `::dire` nào với
`--skip-existing`", và có test chặn đúng chuỗi đó.

**In luôn lệnh chạy tiếp**, dùng chung `group_generators` + `plan_commands` với
cổng thời gian của `::dire` ⇒ nếu phần còn thiếu vượt 4,5 h thì nó chia nhóm hệt
như `::dire` sẽ làm, thay vì để người đọc tự chia.

`--verify` đọc thử bằng `np.load`: file ghi **dở** do bị kill vẫn tồn tại và
**khác rỗng**, nên đếm file là chưa đủ — một `.npy` cụt sẽ làm `::train` nổ giữa
epoch. `::maps` cũng cảnh báo **mất cân bằng lớp** (ví dụ `test/midjourney` 100
ảnh vs `test/real` 1 000 ⇒ 10:1): recall của dòng đó phản ánh số mẫu, không phải
chất lượng DIRE.

Ba cách kiểm nhanh khác, không cần chạy hàm nào:

| Cách | Lệnh | Thấy gì |
|---|---|---|
| Đếm file trên Volume | `modal volume ls dire-zero maps/train/adm` | danh sách `.npy` (không đếm hộ) |
| Xem app còn sống | `modal app list` | `ephemeral` + đang chạy hay đã dừng |
| Log của app đã chết | `modal app logs ap-…` | dừng ở ảnh thứ mấy |

**Lần sau, chạy dài thì `--detach`:**

```bash
modal run --detach -m dire_zero.modal_app::dire --n 500 --steps 20 --skip-existing \
    --generator dalle2,ddpm,diff-stylegan
```

`--detach` làm app sống độc lập với máy local: đổi Wi-Fi, gập laptop, mất mạng
đều không giết container. Theo dõi bằng `modal app logs <app-id>`, và nếu muốn
dừng thì `modal app stop <app-id>` — dừng chủ động vẫn commit phần đã xong nhờ lớp
`except` (nếu là SIGTERM) hoặc mất ≤ 200 ảnh (nếu kill cứng).

**Bước 1 — `--generator all`, không phải `auto`.** Hai bài toán khác nhau:

| | `--generator adm` | `--generator all` | `--generator auto` |
|---|---|---|---|
| Bài toán | `real vs adm` | 15 detector, mỗi cái `real vs <một generator>` | `real vs bất kỳ fake` |
| So được Table 2? | **có**, đúng một dòng | **có**, cả bảng | **không** — bảng không có dòng gộp |
| Số detector | 1 | 15 | 1 |

`all` chạy lần lượt trong **một** container, mỗi generator ghi vào thư mục riêng
(`runs/train/gen-adm/`, `runs/train/gen-pndm/`, …). Bản trước ghi tất cả vào
`runs/train.json` + `runs/detector.pth`: chạy 15 lệnh thì dòng thứ 15 là dòng duy
nhất còn lại, và `::report` đọc đúng một con số mà không nói 14 dòng kia đã mất.

`runs/train.json` nay là **bảng**: `by_generator` (mỗi mục là một dòng Table 2) +
`runs` (mọi lượt, kể cả lượt cũ cùng generator — để so lại sau khi đổi seed).

**Thời gian**: mỗi lượt là ResNet-50 trên ~2 000 map, **giây đến phút**, không
phải GPU-giờ như DIRE — nên không có cổng tiền ở đây. `--generator all` in số giây
thật của từng lượt và **dừng trước khi chạm timeout**, liệt kê generator còn lại;
chạy tiếp lệnh sau là được (mỗi generator đã xong nằm trong thư mục riêng).

**Bước 2 — đọc bảng, không đọc một số.** `train_by_generator` có `acc` + `acc_ci95`,
`ap`, `auc`, `n_real`/`n_fake` cho từng generator. Ba điều phải nói khi báo cáo:

* `acc_ci95` rộng cỡ ±2,5 điểm với 1 000 ảnh test (±7 điểm với 200) ⇒ chênh lệch
  2–3 điểm giữa hai generator là **nhiễu**, không phải thứ hạng. `::report` tự
  tính nửa độ rộng từ `n` thật thay vì chép tay một con số;
* mỗi detector được train lại từ đầu (`init: scratch`), không ImageNet như bài báo;
* `midjourney` chỉ có ~100 ảnh/ô ⇒ tỉ lệ 5:1, recall là số yếu và nó phản ánh **số
  mẫu**, không phải chất lượng DIRE. `::train` in cảnh báo khi tỉ lệ > 5:1.

**Bước 3–4** (gallery/demo và cột R0 của bài báo) là việc riêng, chưa xong; đừng
trộn vào bảng kết quả.

Bước 3 và 4 **của khối lệnh trên** (sinh ảnh và DIRE map) **cố ý khác `--steps`**:
Phụ lục A của bài báo sinh bằng DDIM 50
bước rồi tính DIRE bằng 20 bước. Dùng chung một số cho cả hai (như code cũ) là
lệch giao thức — sai lệch này không làm chương trình hỏng, chỉ làm con số không
so được với bài báo.

### 4.5 Chi phí

`dire_zero` không tự đoán giá. Hai cổng cứng, cả hai đều **dừng** chương trình:

* chưa có `bench.json` ⇒ `dire/gen/train` từ chối chạy;
* ước tính > `--max-usd` (mặc định 5 USD) ⇒ từ chối, in rõ vượt bao nhiêu.

Công thức duy nhất:

```text
GPU-giờ = N × NFE/ảnh × sec_per_step / 3600
USD     = GPU-giờ × 1,10   (A10G)
NFE/ảnh = 2S  (DIRE)   ·   S  (sinh ảnh)
```

Với `S = 20`, 1 000 ảnh DIRE = 40 000 NFE. Nếu `sec_per_step` đo được là 0,35 s
⇒ 3,9 GPU-h ⇒ ~4,3 USD (khoảng 3,4–6,9 USD). Nếu là 2,5 s ⇒ ~30 USD. Chênh 7
lần — đó là lý do bước 2 không được bỏ qua.

## 5. Đọc kết quả cho đúng

`train.json` tự khai những gì nó **không** nói:

| Trường | Nghĩa |
|---|---|
| `label` | do **người gọi** truyền vào, lấy từ `data_card.json`: `R1+R2` khi ảnh giả là của tác giả, `R1+R2+R3` khi ảnh giả do `::gen` sinh ra |
| `init` | `scratch` ⇒ **không** nạp ImageNet (bài báo có). Kém hơn là bình thường |
| `test.acc` | ACC tại ngưỡng chọn trên val |
| `test.acc_at_0.5` | ACC tại 0,5 — con số so được với Table 2 |
| `test.acc_ci95` | khoảng tin cậy bootstrap 95 % **của chính `test.acc`** (cùng ngưỡng), kèm `acc_ci95_threshold` và `acc_ci95_covers_point` |
| `threshold_source` | `val` hay `default (val quá nhỏ)` |

Hai con số ACC có thể lệch nhau nhiều khi mô hình còn non: sigmoid chưa hiệu
chuẩn (mọi điểm ~0,02) làm ACC@0,5 = 0,5 trong khi AP/AUC = 1,0. Đó là vấn đề
**hiệu chỉnh**, không phải phân biệt — và báo cáo nào chỉ in một trong hai đều
gây hiểu sai.

Vì thế `acc_ci95` **phải** là CI của đúng con số đứng cạnh nó. Bản trước tính CI
tại 0,5 trong khi `acc` tính tại ngưỡng chọn trên val; với detector đã hội tụ hai
số trùng nhau nên lỗi không lộ, nhưng với detector còn non, log thật cho ra
`acc 0,781` đi kèm `acc_ci95 [0,47, 0,529]` — một khoảng tin cậy **không chứa**
điểm ước lượng của chính nó. Không có cách đọc nào làm con số đó đúng. Nay
`metrics.evaluate` truyền đúng ngưỡng vào bootstrap, ghi `acc_ci95_threshold`, và
tự đặt cờ `acc_ci95_covers_point`; `::report` biến cờ đó thành cảnh báo.

**Hai cặp số trong `::report` dễ đọc sai:**

```text
dire.n = 5 900   n_new = 5 869   n_reused = 2 431
```

`n` là số `.npy` **phân biệt**; `n_new`/`n_reused` **cộng dồn qua mọi lượt chạy**,
nên một map tính ở lượt trước rồi dùng lại ở lượt sau được đếm ở cả hai phía.
`5 869 + 2 431 ≠ 5 900` là bình thường, không phải mất map — nhưng bản trước in ba
số cạnh nhau mà không nói, đọc ra như một phép cộng sai. Nay có
`n_distinct_maps` / `n_new_cumulative` / `n_reused_cumulative` / `n_runs` và một
dòng `n_note`.

Ba câu **không** được nói từ kết quả này:

* "DIRE đạt 99,9 %" — đó là số của bài báo, với ADM được train đủ và detector
  khởi tạo ImageNet;
* "tái lập được bảng 2" — khác seed, khác generator, khác init;
* "DIRE không hoạt động" vì ACC thấp trên 200 ảnh — CI 95 % ở mức 200 mẫu rộng
  ±7 điểm; dưới ngưỡng đó không kết luận được gì.

## 6. Cấu trúc

```text
src/dire_zero/
├── schedule.py     lịch β, ᾱ (0-based), tập mốc, NFE      — không cần torch
├── blocks.py       ResBlock / Attention / GroupNorm32
├── unet.py         ZeroUNet (552 814 086 tham số), EncEntry/DecEntry
├── checkpoint.py   bảng ánh xạ 566 key + load strict=True
├── diffusion.py    invert / reconstruct / sample / dire_map, NFE đếm thật
├── detector.py     ResNet-50 (23,5 M), chuẩn hoá ImageNet
├── data.py         chia tập theo hash (tất định), sha256
├── metrics.py      ACC, AP, AUC, balanced ACC, bootstrap CI
├── budget.py       công thức tiền + hai cổng cứng
├── pipeline.py     gen → dire → train → evaluate
├── modal_app.py    image KHÔNG có repo tác giả; 6 entrypoint
└── cli.py          info / keys / selftest
```

Package tự chứa: chỉ `numpy`, `torch`, `Pillow`. Có test canh việc đó
(`test_zero_isolation.py`): không import `dire_core`, `dire_modal`,
`dire_local`, `guided_diffusion`; không nhắc `/opt/dire`.

## 7. Hạn chế — nói trước

1. **Chưa chạy trên GPU thật.** Toàn bộ phần torch được kiểm trên CPU với mô
   hình 32×32 và mô hình "tiên tri"; `modal_app.selfcheck` là lần chạy thật đầu
   tiên. Rủi ro còn lại: tên tham số U-Net khớp checkpoint thật, `invert` trả
   đúng shape ở 256², VRAM ở batch 1.
2. **Một checkpoint ADM** ⇒ chỉ đo in-distribution. DIRE báo 99,9 % cả
   cross-model; bản này không chạm tới phần đó.
3. **Detector không ImageNet** ⇒ không so được với Table 2, và nói rõ trong
   `train.json`.
4. **`sections` của bài báo**: số ảnh thật không nêu, nên số ảnh dùng (500) là
   lựa chọn của mình, ghi trong `gen.json`.
5. **Không hỗ trợ `resblock_updown=False`** (cấu hình khác của ADM) — bản này
   `NotImplementedError` thay vì đoán.
