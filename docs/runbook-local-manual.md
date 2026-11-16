# Runbook — chạy trên **máy cá nhân**, tải dữ liệu bằng tay

> Dành cho: Windows + Git Bash (MSYS2), ổ NTFS, tải dữ liệu bằng GUI/trình duyệt
> rồi đẩy lên Modal. Mỗi bước có **lệnh**, **kết quả mong đợi**, và **cổng** —
> cổng đỏ thì dừng, đừng đi tiếp.
>
> Chi phí GPU chỉ bắt đầu từ bước 5. Bước 0–4 là 0 đồng.

---

## Bước 0 — Kiểm máy (5 giây, 0 đồng)

```bash
cd ~/dire-modal
export DIRE_LOCAL_ROOT=D:/dire-data      # đổi theo ổ của bạn; đừng hard-code trong code
bash tools/preflight_local.sh
```

**Mong đợi:** `MÁY ĐỦ ĐIỀU KIỆN.` Nếu đỏ:

| Thông báo | Sửa |
|---|---|
| `Python 3.9` | cần ≥ 3.10 (pyproject) |
| `DIRE_LOCAL_ROOT chưa đặt` | `export DIRE_LOCAL_ROOT=...`, thêm vào `~/.bashrc` |
| `trống ... MB — cần ~9 GB` | A+B+C ≈ 9 GB; dọn hoặc đổi ổ |
| ổ `fat`/`msdos` | **dùng NTFS** — FAT32 chặn file 4 GB, `.part` cộng dồn rất dễ tràn |

Thêm vào `~/.bashrc` để khỏi gõ lại mỗi phiên:

```bash
export DIRE_LOCAL_ROOT=D:/dire-data
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'
```

---

## Bước 1 — In danh sách cần tải (0 đồng)

```bash
bash tools/scratch_repro.sh --stage list --need a,gallery,e2min
```

**Mong đợi:** bảng `đường dẫn trong dataset → data/raw/...` + cảnh báo bốn tên
dễ sai. Chi tiết đầy đủ: [`docs/download-manual.md`](download-manual.md).

Quy ước duy nhất: **đường dẫn trong dataset = đường dẫn dưới `data/raw/`**.

---

## Bước 2 — Tải tay A (bắt buộc, 1 lần)

| Tải từ | Đặt vào | Size |
|---|---|---|
| `256x256_diffusion_uncond.pt` (openaipublic) | `data/raw/sources/ckpt/adm_lsun.pt` | 2,5 GB |
| `bedroom_val_lmdb.zip` (`dl.yf.io/lsun/objects/`) | `data/raw/sources/lsun/bedroom_val_lmdb.zip` | — |

> Hai file này **không có** trong dataset DiffusionForensics — dataset chỉ có
> checkpoint *detector*, không có weight diffusion.

Kiểm ngay sau khi tải:

```bash
python tools/manual_data.py verify --need a --strict
```

`verify` bắt được thứ `ls -l` không thấy: **trang HTML đăng nhập** lưu thành
`.pt` (magic không phải `PK`), file 0 byte, `.part` còn sót.

---

## Bước 3 — Ingest: LMDB → PNG (local, 0 đồng, ~5–10 phút)

```bash
bash tools/scratch_repro.sh --stage ingest --n-real 500
```

Việc nó làm: kiểm A → hardlink ADM vào `$DIRE_LOCAL_ROOT/scratch/ckpt/` (cùng ổ
thì 0 byte, khác ổ thì copy) → giải nén LMDB → `lsun_to_png.py` xuất PNG 256² +
`manifests/lsun_val.tsv` (sha256 từng ảnh).

**Mong đợi:** `[xong] 500 ảnh → .../lsun/png · bỏ N ảnh (cỡ/kênh khác 256² RGB)`.

Nếu `unzip` báo lỗi: `pip install lmdb pillow` rồi chạy lại; script tự tìm thư
mục có `data.mdb` nên tên thư mục con sau giải nén không quan trọng.

---

## Bước 4 — Đẩy lên Volume (upload, 0 đồng GPU)

```bash
# ADM (2,5 GB) → /raw/ckpt/adm_lsun.pt  (đúng đường `scratch_gen.CKPT` đọc)
python tools/manual_data.py push --need a --dest "$DIRE_LOCAL_ROOT"

# Ảnh thật → /raw/scratch/real
modal volume put dire-raw "$DIRE_LOCAL_ROOT/scratch/lsun/png" scratch/real
```

`push` mặc định chỉ **in lệnh**; thêm `--execute` để chạy. Lần đầu nên xem rồi
mới chạy — 2,5 GB upload sai chỗ là 2,5 GB phải xoá.

Kiểm trên Volume:

```bash
modal volume ls dire-raw ckpt
modal volume ls dire-raw scratch/real | head
```

---

## Bước 5 — **ĐO** `sec/step` (bắt buộc, cổng G8) ⚠

```bash
bash tools/scratch_repro.sh --stage bench --steps 20
```

Đây là cổng duy nhất **không được bỏ qua**. Mọi ước lượng tiền trong repo đang
dựa vào `0,35 s/step` chưa ai đo; khoảng cách giữa hai đầu là **7×** ($21 so với
$141 cho 131k ảnh).

**Mong đợi:** dòng `s_per_image=... (sec/step=...)` và file
`$DIRE_LOCAL_ROOT/scratch/sec_per_step.json`.

Chưa có file này thì các bước sau **từ chối chạy** — đó là chủ ý.

---

## Bước 6 — R3: sinh ảnh (GPU, ~0,1 GPU-giờ)

```bash
bash tools/scratch_repro.sh --stage gen --n-gen 500 --seed-base 1000
```

**Mong đợi:** `[xong] 500 ảnh → /raw/scratch/gen/adm (20 000 NFE)` + `gen.json`
có `ckpt_sha256`, `seeds`, `split.train`, `split.test`.

**Cổng G2/G3 tự kiểm:** seed train/test rời nhau; thiếu trường trong `gen.json`
là `AssertionError`, không phải cảnh báo.

---

## Bước 7 — R1: tính DIRE map (GPU, ~0,1 GPU-giờ)

```bash
bash tools/scratch_repro.sh --stage dire --steps 20
```

**Mong đợi:** `[DIRE] real: 500 map`, `[DIRE] fake: 500 map`,
`s_per_image=... (sec/step=...)`, và `dire.json` với `nfe_total = 2·20·1000`.

NFE được **đếm thật** trong lúc chạy và assert `= 2S`. Nếu thấy 1020 thay vì 40,
dừng ngay — đó là lỗi F2 cũ, sai 25 lần tiền.

---

## Bước 8 — R2: train detector (GPU, ~0,1 GPU-giờ)

```bash
bash tools/scratch_repro.sh --stage train --epochs 40
```

**Mong đợi:** log từng epoch `loss` + `val ACC`; kết thúc bằng
`[xong] /ckpt/scratch/scratch_adm/detector.pth · val 0.9x · test 0.9x`.

Đọc kết quả:

```bash
modal volume get dire-ckpt scratch/scratch_adm/train.json /tmp/train.json
cat /tmp/train.json
```

**Điều được nói:** `test_acc_at_0.5` với nhãn `R1+R2+R3` — cả ba thành phần đều
của bạn. **Điều không được nói:** "DIRE đạt 99,9" (đó là số của paper), hay so
sánh trực tiếp với bảng của họ (khác seed, khác generator).

---

## Bước 9 (tuỳ chọn) — B: đối chiếu reconstruction với tác giả

Chỉ làm khi bạn đã tải `images/` + `recons/` (xem `--need gallery`). Đây là cách
duy nhất kiểm **số học DDIM** của mình, độc lập với detector.

```bash
python tools/manual_data.py push --need gallery --dest "$DIRE_LOCAL_ROOT" --execute
bash tools/scratch_repro.sh --stage verify-recons
```

**Mong đợi:** `median`/`p95` cho từng generator. **Lần chạy đầu chỉ ĐO** — lấy
chính số đó làm ngưỡng cho các lần sau (`--fail-above`). Đặt ngưỡng trước rồi
chỉnh theo kết quả là tự lừa mình.

`adm` không có `recons/` nên không kiểm được — ghi rõ trong báo cáo.

---

## Bước 10 (tuỳ chọn) — C: chấm lại R0

```bash
bash tools/scratch_repro.sh --stage r0 --need e2min
```

Cho ra một con số **để biết harness không hỏng**, nhãn `R0`. Không phải tái lập.

---

## Tóm tắt thời gian và tiền

| Bước | Ở đâu | Thời gian | Tiền |
|---|---|---|---|
| 0–1 | máy bạn | 1 phút | 0 |
| 2 | máy bạn (tải tay) | tuỳ mạng | 0 |
| 3 | máy bạn | 5–10 phút | 0 |
| 4 | upload | tuỳ mạng | 0 |
| 5 bench | Modal A10G | ~3 phút | < $0,01 |
| 6 gen 500 ảnh | Modal A10G | 5–20 phút | ~$0,02–0,15 |
| 7 DIRE 1000 ảnh | Modal A10G | 10–40 phút | ~$0,05–0,6 |
| 8 train | Modal A10G | 10–30 phút | ~$0,1 |
| **Tổng** | | **1–2 giờ** | **< $1** |

Số tiền tính theo `sec_per_step` ĐO được ở bước 5; bảng này là khoảng, không
phải báo giá.

---

## Khi có gì sai

| Triệu chứng | Nguyên nhân thật | Sửa |
|---|---|---|
| `thiếu /raw/ckpt/adm_lsun.pt` | đẩy nhầm vào `sources/ckpt/` | `modal volume put dire-raw <file> ckpt/adm_lsun.pt` |
| `[CHẶN] còn cổng đỏ` | G8 chưa đo, hoặc G0/G1 | xem `--stage gates` |
| `tar.gz rỗng (0 member)` | tar 130 B của dataset (biết trước) | không phải lỗi của bạn |
| `nội dung là TRANG HTML` | phiên Baidu hết hạn | tải lại, đừng đổi tên |
| eval ra ACC ~50% | nhãn đảo (`0_real`→0) hoặc thiếu `real.tar.gz` | kiểm `label_of`, `pairs_from_raw` |
| NFE = 1020 | lỗi F2 (chạy trọn 1000 bước) | dùng `ddim.Schedule`, không tự viết vòng lặp |

## Chưa kiểm được (nói trước)

Phần torch (`ddim.py`, `scratch_gen.py`, `scratch_train.py`, `scratch_verify.py`)
**chưa chạy trên GPU thật** — sandbox soạn code không có CUDA. Phần quyết định
đúng/sai và tiền (lịch, mốc 950, NFE, seed, cổng, tải tay) thì có 572 test chạy
mỗi commit. Lần chạy đầu nên để ý: tên tham số U-Net khớp checkpoint,
`invert` trả đúng shape, VRAM ở batch 1.

> **Ba lỗi đã sửa** trong `ddim.py` (xem `docs/from-zero-clean-room.md` §3):
> mốc 0 lấy `ᾱ = 1,0` ⇒ map toàn `NaN`; `invert`/`reconstruct` chạy thiếu một
> lượt mỗi bên ⇒ NFE 38 thay vì 40; bước cuối tái tạo để lại ~0,5 % nhiễu.
> Chúng **không** làm script dừng — đó là lý do nên chạy `--stage gates` trước
> và kiểm `[DIRE]` có in ra `nfe_total = 2·20·N` hay không.
