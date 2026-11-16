# Dùng dataset riêng trên ổ `F:` (cây DiffusionForensics)

> Dành cho: bạn đã có `F:\…\DiffusionForensics` và **không** muốn tải lại LSUN
> từ Baidu. Mọi lệnh dưới đây chạy trên máy bạn, không tốn GPU.
>
> Code: `src/dire_zero/dataset.py`. Test: `tests/test_zero_dataset.py` (22 test).

---

## 1. Đọc cây dữ liệu — cái gì là gì

```text
DiffusionForensics/
├── images/{train,test}/{lsun_bedroom,imagenet,celebahq}/<generator>.tar.gz
├── dire/{train,val,test}/<dataset>/<generator>.tar.gz      ← DIRE map của TÁC GIẢ
├── recons/test/<dataset>/<dataset>/<generator>.tar.gz      ← reconstruction của TÁC GIẢ
└── checkpoints/*.pth                                        ← detector của TÁC GIẢ
```

Ba quy ước, tất cả đều **do cấu trúc quyết định**, không do mình thích:

| Thư mục | Là gì | Dùng được cho |
|---|---|---|
| `images/…/real.tar.gz` | ảnh **thật** (LSUN-Bedroom / ImageNet) | train + test, nhãn 0 |
| `images/…/<gen>.tar.gz` | ảnh do **<gen> sinh** | test/train, nhãn 1 |
| `dire/…` | **DIRE map** của tác giả | **chỉ** chấm lại R0 — bị từ chối cho from-scratch |
| `recons/…` | reconstruction của tác giả | chỉ để đối chiếu số học DDIM (việc B) |
| `checkpoints/…` | detector của tác giả | chỉ chấm lại R0 |

Nhãn lấy từ **tên file**: `real` ⇒ thật, mọi tên khác ⇒ giả, và tên đó chính là
tên generator. `sdv1_test.tar.gz` được hiểu là generator `sdv1` (hậu tố `_test`
chỉ là chỗ đóng gói khác đi).

## 2. Điều quan trọng nhất rút ra từ chính cây của bạn

```text
dire/test/lsun_bedroom/adm.tar.gz        ← CÓ map của ảnh ADM
images/test/lsun_bedroom/                ← KHÔNG có adm.tar.gz (chỉ ldm, pndm, real)
```

**Ảnh ADM dùng cho test chưa từng được phát hành.** Tác giả phát hành *map* DIRE
của chúng, không phát hành *ảnh*. Hệ quả trực tiếp:

| Câu hỏi | Dữ liệu có sẵn trên ổ F | Nhãn được nói |
|---|---|---|
| ADM in-distribution (dòng headline của paper) | **phải tự sinh ảnh** (`::gen`) | `R1+R2+R3` |
| Cross-model pndm / ldm / sdv1 | `images/test/lsun_bedroom/*` | `R1+R2` |
| Chấm lại bảng 2 (R0) | `dire/` + `checkpoints/` | `R0` |

Đây không phải chi tiết vụn: nó có nghĩa **không thể** làm dòng "ADM" bằng dữ
liệu phát hành. Muốn có dòng đó thì phải tự sinh — đúng việc mà bản clean-room
làm, và cũng là lý do `dire_zero` có `::gen`.

Ngoài ra, cây này **không có `images/val/`**. Bước train sẽ tự cắt 15 % từ train
làm val (ghi rõ trong `train.json`), và test không bao giờ được nhìn.

## 3. Ba lệnh

### 3.1 Kiểm trước — 0 đồng, ~2 giây

```bash
python -m dire_zero dataset --root F:/du-lieu --dataset lsun_bedroom \
                            --license research-only --splits train test
```

In ra bảng `split × generator` kèm **số ảnh thật** trong từng archive, cảnh báo
bất đối xứng, và kế hoạch NFE. Cổng **đỏ ⇒ exit 3**, không đi tiếp:

| Cổng | Vì sao |
|---|---|
| thiếu `--license` | repo có đường mirror công khai; `data_card.json` ghi thẳng giá trị này |
| `--dataset imagenet` mà không `--ack-ood` | ADM 256 học LSUN-Bedroom; ảnh thật ngoài phân bố ⇒ khoảng cách DIRE co lại ⇒ ACC về mức đoán mò |
| chưa có `sec_per_step` | **không báo tiền**. Khoảng thực tế 0,05–2,5 s/step ⇒ chênh 50× |

`--root` nhận cả ba cách gõ: `F:/du-lieu`, `F:\du-lieu`, `/f/du-lieu`; trỏ vào
thư mục cha cũng được (tự lùi vào `DiffusionForensics/` nếu thấy `images/`).

### 3.2 Chuẩn hoá — 0 đồng, vài phút

```bash
python -m dire_zero ingest --root F:/du-lieu --out F:/dire-zero-norm \
                           --dataset lsun_bedroom --license research-only \
                           --splits train test --limit 500
```

Việc nó làm:

- giải nén **trực tiếp** từ `.tar.gz` (không cần giải ra đĩa trước);
- RGB, cạnh ngắn = 256, cắt giữa 256×256 — đúng phép biến đổi của DIRE;
- tên file = **16 ký tự đầu sha256 của pixel**, không phải tên gốc: hai ảnh trùng
  nội dung không thể thành hai mẫu, và chia train/test theo hash vẫn ổn định qua
  các máy;
- ghi `manifest.tsv` (10 cột, có `source_tar` + `member` để truy ngược) và
  `data_card.json`;
- **chặn rò rỉ**: cùng một ảnh ở cả hai lớp ⇒ `IntegrityError`, dừng, không ghi
  gì. (Đây là lỗi làm ACC 100 % và vô nghĩa, và nó không bao giờ tự báo.)
- in sẵn lệnh `modal volume put`.

`--limit 500` cắt **500 ảnh đầu** mỗi archive (thứ tự trong tar là ổn định, nên
cắt tiền tố là tái lập được — khác với cắt ngẫu nhiên).

### 3.3 Đẩy lên và kiểm trên Volume

```bash
modal volume put dire-raw F:/dire-zero-norm zero/data -r
modal run -m dire_zero.modal_app::checkdata
```

`checkdata` kiểm lại **trên Volume** (nơi có thể bị ghi đè từ ngoài): sha256 từng
PNG, đủ hai lớp, tỉ lệ hai lớp ≤ 5:1, và `data_card.label` có khớp nguồn ảnh giả
không. Sai ⇒ dừng trước khi tiêu GPU.

## 4. Nhãn phụ thuộc **ảnh giả lấy từ đâu**

```bash
--fake-source author   # dùng images/test/<gen>.tar.gz   ⇒ R1+R2
--fake-source gen      # để ::gen tự sinh                  ⇒ R1+R2+R3
--fake-source auto     # mặc định: có ảnh giả thì author, không thì gen
```

Ảnh thật luôn là dữ liệu của người khác (LSUN) — không bao giờ tự tính vào
"của mình". Vì thế **R3 chỉ đúng khi ảnh giả do mình sinh**; dùng ảnh giả của tác
giả mà nói `R3` là tự khen một thành phần chưa làm. `data_card.json` ghi cứng
điều này, và `checkdata` từ chối nếu thấy `fake_source="author"` mà nhãn có `R3`.

## 5. Đường chạy đầy đủ

```bash
# 0) máy: kiểm + chuẩn hoá
python -m dire_zero dataset --root F:/du-lieu --license research-only
python -m dire_zero ingest  --root F:/du-lieu --out F:/dire-zero-norm --license research-only

# 1) đẩy checkpoint ADM (2,5 GB) và dữ liệu
modal volume put dire-raw "F:/du-lieu/…/256x256_diffusion_uncond.pt" zero/ckpt/adm_lsun.pt
modal volume put dire-raw F:/dire-zero-norm zero/data -r

# 2) cổng
modal run -m dire_zero.modal_app::checkdata
modal run -m dire_zero.modal_app::selfcheck     # nạp strict=True + kiểm NaN/NFE
modal run -m dire_zero.modal_app::bench         # ĐO sec/step

# 3) ba bước đo
modal run -m dire_zero.modal_app::gen   --n 500 --steps 50     # R3 (chỉ khi cần ảnh ADM)
modal run -m dire_zero.modal_app::dire  --splits train,test --steps 20
modal run -m dire_zero.modal_app::train --epochs 40
modal run -m dire_zero.modal_app::report
```

## 6. Chi phí cho đúng cây này

Với 500 ảnh mỗi phía, `S = 20` cho DIRE, `S = 50` cho sinh, GPU A10G:

| Việc | NFE | GPU-h @0,35 s | USD @$1,10 |
|---|---|---|---|
| sinh 500 ảnh ADM (S=50) | 25 000 | 2,43 | 2,67 |
| DIRE 500 thật + 500 giả | 40 000 | 3,89 | 4,28 |
| train detector (~40 epoch) | — | 0,3–1 | 0,33–1,10 |
| **tổng** | | **~7** | **~7–8** |

Ba lưu ý:

1. **0,35 s/step là giả định, chưa đo.** Bảng trên chỉ để biết bậc độ lớn. Số
   thật lấy từ `::bench`; nếu là 2,5 s/step thì tổng thành ~50 USD.
2. Bỏ bước `gen` (dùng ảnh giả của tác giả) thì rẻ hơn ~2,7 USD, nhưng nhãn tụt
   xuống `R1+R2`.
3. Nếu chỉ muốn kiểm đường ống trước khi tiêu tiền: `--limit 50` cho mỗi archive
   ⇒ toàn bộ dưới 1 USD.

## 7. Hạn chế của đường này

- **Không có `images/val/`** ⇒ val cắt từ train; early-stop vì thế kém tin cậy
  hơn, ghi rõ trong `train.json`.
- **`imagenet/` không cùng phân bố** với ADM-LSUN ⇒ kết quả trên ImageNet chỉ có
  nghĩa "đo lại bằng mô hình sai phân bố", không so được với paper.
- **Ảnh test của tác giả không do mình sinh** ⇒ mọi kết quả trên `images/test/`
  tối đa là `R1+R2`.
- **`--limit` cắt tiền tố** trong tar ⇒ nếu tar được đóng gói theo thứ tự lớp
  con (vd. toàn ảnh sáng trước), mẫu của bạn bị lệch. Muốn chắc thì đừng cắt,
  hoặc kiểm phân bố độ sáng của mẫu (chưa có trong repo).
