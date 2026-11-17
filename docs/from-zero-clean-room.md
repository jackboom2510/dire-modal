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

## 3. Bốn lỗi tìm được khi viết lại

Viết lại từ công thức không chỉ để "sạch". Bốn lỗi dưới đây lộ ra vì phải suy
nghĩ lại từng chỉ số — và cả ba đều nằm trong **đường đang chạy** của repo, tức
là đường mà `docs/runbook-local-manual.md` bảo người dùng chạy.

**Trạng thái:** 3.1–3.3 đã sửa trong `src/dire_modal/ddim.py`, có test riêng
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

Mặc định `apply` **không** bê cây dữ liệu (`data/`, `maps/`, `gen/`, `runs/`) —
`--data` mới bê. Lý do: người gọi migration là để cứu checkpoint 2,5 GB, không
phải để một lệnh tự động sửa cả cây dữ liệu; `plan` vẫn in sẵn `n_copy` của phần
dữ liệu để ai muốn thì quyết định có căn cứ. Dữ liệu sinh từ máy local vẫn đi
đường `modal volume put` quen thuộc (mục 4.4).

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

# 5. R2 — train detector
modal run -m dire_zero.modal_app::train --epochs 40

# 6. gom báo cáo
modal run -m dire_zero.modal_app::report
```

Bước 3 và 4 **cố ý khác `--steps`**: Phụ lục A của bài báo sinh bằng DDIM 50
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
