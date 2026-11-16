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

## 3. Ba lỗi tìm được khi viết lại

Viết lại từ công thức không chỉ để "sạch". Ba lỗi dưới đây lộ ra vì phải suy
nghĩ lại từng chỉ số — và cả ba đều nằm trong **đường đang chạy** của repo, tức
là đường mà `docs/runbook-local-manual.md` bảo người dùng chạy.

**Trạng thái:** cả ba đã sửa trong `src/dire_modal/ddim.py`, có test riêng
(`tests/test_ddim_protocol.py`, 11 test, chạy bằng mô hình "tiên tri" nên không
cần GPU lẫn checkpoint). Chưa lần nào chạy với trọng số ADM thật.

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
python -m dire_zero dataset --root F:/du-lieu --license research-only   # kiểm, 0 đồng
python -m dire_zero ingest  --root F:/du-lieu --out F:/dire-zero-norm --license research-only
```

`dataset` in bảng `split × generator`, cảnh báo bất đối xứng, và **dừng** nếu
thiếu giấy phép hoặc nếu dataset không cùng phân bố với ADM. `ingest` đổ tar.gz
ra PNG 256², tên file theo sha256 pixel, kèm manifest + `data_card.json`.

Nhãn phụ thuộc **ảnh giả lấy từ đâu**: `--fake-source author` (dùng ảnh của tác
giả) ⇒ tối đa `R1+R2`; `--fake-source gen` (tự sinh) ⇒ `R1+R2+R3`.
Chi tiết + phát hiện "ảnh ADM test chưa từng được phát hành":
[`docs/dataset-own.md`](dataset-own.md).

### 4.3 Trên Modal

```bash
# 0. đẩy checkpoint (2,5 GB) — dùng lại bản đã tải tay cho đường cũ
modal volume put dire-raw "$DIRE_LOCAL_ROOT/data/raw/sources/ckpt/adm_lsun.pt" zero/ckpt/adm_lsun.pt

# 1. cổng nạp trọng số: strict=True + 1 ảnh + kiểm NaN/NFE
modal run -m dire_zero.modal_app::selfcheck

# 2. ĐO sec/step (bắt buộc: mọi ước lượng tiền dùng số này)
modal run -m dire_zero.modal_app::bench

# 3. R3 — sinh ảnh (bài báo dùng 50 bước cho sinh)
modal run -m dire_zero.modal_app::gen --n 500 --steps 50 --seed-base 1000

# 4. R1 — DIRE map (bài báo dùng 20 bước cho DIRE)
modal run -m dire_zero.modal_app::dire --n 500 --steps 20

# 5. R2 — train detector
modal run -m dire_zero.modal_app::train --epochs 40

# 6. gom báo cáo
modal run -m dire_zero.modal_app::report
```

Bước 3 và 4 **cố ý khác `--steps`**: Phụ lục A của bài báo sinh bằng DDIM 50
bước rồi tính DIRE bằng 20 bước. Dùng chung một số cho cả hai (như code cũ) là
lệch giao thức — sai lệch này không làm chương trình hỏng, chỉ làm con số không
so được với bài báo.

### 4.4 Chi phí

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
| `label` | `R1+R2+R3` — ba thành phần đều của mình |
| `init` | `scratch` ⇒ **không** nạp ImageNet (bài báo có). Kém hơn là bình thường |
| `test.acc` | ACC tại ngưỡng chọn trên val |
| `test.acc_at_0.5` | ACC tại 0,5 — con số so được với Table 2 |
| `test.acc_ci95` | khoảng tin cậy bootstrap 95 % |
| `threshold_source` | `val` hay `default (val quá nhỏ)` |

Hai con số ACC có thể lệch nhau nhiều khi mô hình còn non: sigmoid chưa hiệu
chuẩn (mọi điểm ~0,02) làm ACC@0,5 = 0,5 trong khi AP/AUC = 1,0. Đó là vấn đề
**hiệu chỉnh**, không phải phân biệt — và báo cáo nào chỉ in một trong hai đều
gây hiểu sai.

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
