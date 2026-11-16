# Tải dữ liệu **bằng tay** (không BaiduPCS-Go) — tải gì, đặt ở đâu

> Quy ước duy nhất: **đường dẫn trong dataset = đường dẫn dưới `data/raw/`**.
> `checkpoints/lsun_adm.pth` → `data/raw/checkpoints/lsun_adm.pth`.
>
> Không đổi tên, không giải nén, không gộp phẳng. Ba lệnh dưới đây kiểm tra
> đúng những lỗi đó.

```bash
python tools/manual_data.py list   --need a,gallery,e2min,e2,ckpts   # in danh sách
python tools/manual_data.py verify --need a,gallery,e2min,e2,ckpts   # kiểm offline
python tools/manual_data.py push   --need e2min --execute            # đẩy lên Volume
```

---

## A. Đường từ đầu (R1+R2+R3) — **không lấy gì từ dataset**

Dataset **không chứa** weight diffusion. Hai file này tải từ nguồn ngoài:

| # | Nguồn | Đặt vào | Size |
|---|---|---|---|
| 1 | `https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt` | `data/raw/sources/ckpt/adm_lsun.pt` | 2,5 GB |
| 2 | `http://dl.yf.io/lsun/objects/bedroom_val_lmdb.zip` | `data/raw/sources/lsun/bedroom_val_lmdb.zip` | chưa đo |

Sau đó: `bash tools/scratch_repro.sh --stage fetch` (giải nén LMDB → PNG, ghim sha256).

---

## B. Kiểm chứng reconstruction — đối chiếu `recons/` của tác giả

Bộ tối thiểu (3 generator × 3 nhánh, ~0,67 GB):

| # | Đường dẫn trong dataset | Đặt vào |
|---|---|---|
| 1 | `images/test/lsun_bedroom/real.tar.gz` | `data/raw/images/test/lsun_bedroom/real.tar.gz` |
| 2 | `recons/test/lsun_bedroom/lsun_bedroom/real.tar.gz` | `data/raw/recons/test/lsun_bedroom/lsun_bedroom/real.tar.gz` |
| 3 | `dire/test/lsun_bedroom/lsun_bedroom/real.tar.gz` | `data/raw/dire/test/lsun_bedroom/lsun_bedroom/real.tar.gz` |
| 4–6 | `…/pndm.tar.gz` (ba nhánh) | `data/raw/{images,recons/…,dire/…}/pndm.tar.gz` |
| 7–9 | `…/ldm.tar.gz` (ba nhánh) | `data/raw/{images,recons/…,dire/…}/ldm.tar.gz` |

Mở rộng lên đủ **7 generator** có `recons/` (+10 file):

| generator | `images/` | `recons/` |
|---|---|---|
| sdv2 | `sdv2.tar.gz` | `sdv2.tar.gz` |
| vqdiffusion | `vqdiffusion.tar.gz` | `vqdiffusion.tar.gz` |
| projectedgan | `projectedgan.tar.gz` | `projectedgan.tar.gz` |
| midjourney | **`midjourney.tar.gz`** | **`midjourney.tar.gz`** |
| stylegan | **`stylegan_official.tar.gz`** | `stylegan.tar.gz` |

> `dire/` của `midjourney` là **`midjouney.tar.gz`** (thiếu chữ `r`) — typo gốc,
> không sửa. Xem §E.

**Không có `recons/` cho:** `adm`, `ddpm`, `iddpm`, `sdv1`, `dalle2`, `if`,
`diff-projectedgan`, `diff-stylegan`. Nghĩa là generator **headline của paper
(ADM)** không đối chiếu được bằng đường này — muốn có phải tự tính, tức mục A.

---

## C. Chấm lại R0 (tuỳ chọn, để có số tham chiếu)

### C1 — con số đầu tiên: `--need e2min` (3 file, ~0,40 GB)

| Đường dẫn trong dataset | Đặt vào |
|---|---|
| `checkpoints/lsun_adm.pth` (269,4 MB) | `data/raw/checkpoints/lsun_adm.pth` |
| `dire/test/lsun_bedroom/lsun_bedroom/real.tar.gz` | `data/raw/dire/test/…/real.tar.gz` |
| `dire/test/lsun_bedroom/lsun_bedroom/adm.tar.gz` | `data/raw/dire/test/…/adm.tar.gz` |

### C2 — hàng ADM đủ 8 cột: `--need e2` (17 file, ~1,94 GB)

`checkpoints/lsun_adm.pth` + 16 tar trong `dire/test/lsun_bedroom/lsun_bedroom/`:

```
real  adm  ddpm  iddpm  pndm  ldm  sdv1  sdv2
vqdiffusion  stylegan  projectedgan  diff-projectedgan  diff-stylegan
dalle2  if  midjouney
```

### C3 — đủ 32 ô Table 2: `--need ckpts` (6 file, ~1,58 GB)

```
checkpoints/lsun_adm.pth      checkpoints/lsun_iddpm.pth
checkpoints/lsun_pndm.pth     checkpoints/lsun_stylegan.pth
checkpoints/imagenet_adm.pth  checkpoints/celebahq_sdv2.pth
```

---

## D. **Đừng tải**

| Thứ | Vì sao |
|---|---|
| `images/train/`, `dire/train/`, `recons/train/` (**47 GB = 77% cây**) | chỉ cần nếu train detector trên dữ liệu *của tác giả*. Đường R2 tự sinh ảnh nên không cần. |
| 8 tar `sdv1_*ddim*` | nằm trong `NEVER`. Hai file chỉ **130 B** (rỗng ngay trong dataset gốc). |
| `sdv1_new2.tar.gz` (361 MB) | biến thể chưa rõ khác gì `sdv1_new`. |
| `celebahq/`, `imagenet/` | chỉ cho cross-domain; ảnh gốc CelebA-HQ **không có** trong `images/`, nên không tự tính DIRE lại được. |

---

## E. Tên file dễ sai — checklist

- [ ] `images/test/lsun_bedroom/midjourney.tar.gz` — **đúng** chính tả
- [ ] `recons/test/lsun_bedroom/lsun_bedroom/midjourney.tar.gz` — **đúng**
- [ ] `dire/test/lsun_bedroom/lsun_bedroom/midjouney.tar.gz` — **thiếu `r`** (typo gốc)
- [ ] `images/test/lsun_bedroom/stylegan_official.tar.gz` — khác tên hai nhánh kia
- [ ] `dire/…/sdv1.tar.gz` nhưng `images/…/sdv1_new.tar.gz`; **không** có `recons/sdv1`
- [ ] Ba tầng thư mục khác nhau: `images/` **một tầng**, `recons/` và `dire/` lồng thêm `lsun_bedroom/`

Baidu hay thêm hậu tố `(1)`, `(2)`. `verify` tìm theo tên khi khớp **duy nhất**
và in ra `[theo tên → …]`, nhưng hãy đổi lại cho đúng cây — đường `push` dùng
đúng đường dẫn để đặt file lên Volume.

---

## F. Sau khi tải: kiểm rồi mới đẩy

```bash
# 1. kiểm cấu trúc: gzip/tar hợp lệ, đếm member, phát hiện HTML/JSON lỗi, file .part
python tools/manual_data.py verify --need a,gallery,e2min --strict

# 2. xem lệnh đẩy (mặc định CHỈ IN, chưa chạy)
python tools/manual_data.py push --need e2min

# 3. chạy thật
python tools/manual_data.py push --need e2min --execute
```

`verify` bắt bốn loại hỏng mà `ls -l` không thấy:

| Dấu hiệu | Nguyên nhân thật |
|---|---|
| `nội dung là TRANG HTML` | phiên Baidu hết hạn, file thực chất là trang đăng nhập |
| `tar.gz hỏng: …` | tải hụt / đứt kết nối |
| `tar.gz rỗng (0 member)` | tar 130 B của dataset (biết trước, không phải lỗi bạn) |
| `không phải zip checkpoint` | `.pth` thực chất là HTML, hoặc file 0 byte |

---

## G. Dung lượng nếu làm cả A + B + C

2,5 GB (ADM) + LSUN val + 0,67 GB (gallery) + 1,94 GB (e2) + 1,58 GB (ckpts)
≈ **8–9 GB** — vừa máy nhẹ. Nếu ổ là FAT32 (giới hạn 4 GB/file) thì ADM 2,5 GB
vẫn lọt, nhưng đừng để `.part` cộng dồn.

Thứ tự đề nghị: **A → B → (C1 nếu cần số tham chiếu)**. C chỉ để biết harness
của mình không hỏng; nó không phải tái lập.
