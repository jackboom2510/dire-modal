# Dùng dataset riêng trên ổ `F:` (cây DiffusionForensics)

> Dành cho: bạn đã có `F:\…\DiffusionForensics` và **không** muốn tải lại LSUN
> từ Baidu. Mọi lệnh dưới đây chạy trên máy bạn, không tốn GPU.
>
> Code: `src/dire_zero/dataset.py`. Test: `tests/test_zero_dataset.py` (25 test).

---

## 0. Đọc mục này trước: số liệu trong tài liệu không phải số đo

Mọi con số trong tài liệu này (kích thước archive, danh sách generator) là **bản
kê bạn dán vào**, không phải kết quả đo. Chúng **lệch** với inventory đã audit
trong `src/dire_modal/manifest.py` ở ít nhất ba chỗ:

| Mục | Bản kê bạn dán | `manifest.py` (audit trước đó) |
|---|---|---|
| số archive ở `images/test/` | 15 | 25 |
| `images/test/imagenet/sdv1.tar.gz` | có, 4,22 GB | ghi "**không** có `sdv1`, chỉ `sdv1_new`/`sdv1_new2`" |
| `dire/train/lsun_bedroom/` | 2 file (`real`, `adm`) | 5 file, 20,3 GB |

Không bên nào được coi là chân lý. Cả hai đều là **bản kê**, và bản kê thì sai
được. Vì vậy:

- **không suy số ảnh từ số MB** — tỉ lệ nén của PNG trong tar.gz thay đổi 2–3×
  giữa các generator;
- **không kết luận "không tồn tại"** từ một danh sách, kể cả danh sách này;
- lệnh `dataset` (mục 2) đếm **thật** trong archive được chọn, và đó là con số
  duy nhất được dùng để lập kế hoạch.

Một hệ quả kỹ thuật: `sdv1.tar.gz` 4,22 GB **không nằm được trên FAT32** (chặn
file > 4 GB). Nếu `ls` thấy file đó thì ổ là NTFS — nhưng nếu một archive > 4 GB
báo "No such file" sau khi chép, nghi vấn đầu tiên là định dạng ổ, không phải
đường dẫn.

## 1. Cây dữ liệu — cái gì là gì

```text
DiffusionForensics/
├── images/
│   ├── train/lsun_bedroom/{real,adm,iddpm,pndm,stylegan}.tar.gz
│   ├── val/lsun_bedroom/{real,adm,iddpm,pndm,stylegan}.tar.gz
│   ├── test/lsun_bedroom/{real,adm,dalle2,ddpm,diff-projectedgan,diff-stylegan,
│   │                        iddpm,if,ldm,midjourney,pndm,projectedgan,sdv2,
│   │                        stylegan_official,vqdiffusion}.tar.gz
│   ├── train/imagenet/{real,adm}.tar.gz      test/imagenet/{real,adm,sdv1}.tar.gz
│   └── train/celebahq/real.tar.gz
├── dire/                                        ← DIRE map của TÁC GIẢ (G0)
│   ├── train/lsun_bedroom/{real,adm}.tar.gz
│   ├── val/lsun_bedroom/{real,adm,iddpm,pndm,stylegan}.tar.gz
│   └── test/{lsun_bedroom,imagenet}/…
├── recons/                                      ← reconstruction của TÁC GIẢ
│   ├── train/lsun_bedroom/real.tar.gz
│   └── test/lsun_bedroom/lsun_bedroom/{real,ldm,pndm}.tar.gz
└── checkpoints/{lsun_adm,lsun_iddpm,lsun_pndm,lsun_stylegan,
                 imagenet_adm,celebahq_sdv2}.pth     ← detector của TÁC GIẢ
```

Thư mục | Là gì | Dùng được cho
---|---|---
`images/…/real.tar.gz` | ảnh **thật** (LSUN-Bedroom / ImageNet) | train + val + test, nhãn 0
`images/…/<gen>.tar.gz` | ảnh do **<gen>** sinh | train/val/test, nhãn 1
`dire/…` | **DIRE map** của tác giả | chấm lại R0 — cổng G0 chặn nếu không bật `--allow-published`
`recons/…` | reconstruction của tác giả | chỉ đối chiếu số học DDIM (việc B), **có ghi chú** khi bị bỏ qua
`checkpoints/…` | detector của tác giả | chấm lại R0

Nhãn lấy từ **tên file**, không từ tên thư mục: `real` ⇒ thật, mọi tên khác ⇒
giả, và tên đó là tên generator. Tên được chuẩn hoá hai lần:

- bỏ hậu tố đóng gói `_test` / `_val` / `_train` ⇒ `sdv1_test.tar.gz` = `sdv1`;
- áp bảng `GENERATOR_ALIASES` ⇒ `stylegan_official` = `stylegan`.

Vì sao cần bảng alias: `stylegan` ở train/val nhưng `stylegan_official` ở test.
Không gộp thì bảng kết quả có hai dòng cho cùng một generator, và "train trên
`stylegan`, test trên `stylegan_official`" trông như hai thứ khác nhau. Ngược
lại, `diff-stylegan` và `diff-projectedgan` **cố ý không** nằm trong bảng — đó
là Diffusion-styleGAN, một generator khác hẳn.

## 2. Hai điều đính chính bản trước của tài liệu này

Bản trước viết hai câu **sai**, và cả hai đều sai theo cùng một kiểu: kết luận
"không tồn tại" từ một danh sách bị cắt.

| Câu sai (bản trước) | Sự thật |
|---|---|
| "Ảnh ADM dùng cho test **chưa từng được phát hành**" | `images/test/lsun_bedroom/adm.tar.gz` **có** |
| "Cây này **không có** `images/val/`" | `images/val/lsun_bedroom/` có 5 archive |

Hai hệ quả khoa học, và chúng đổi hẳn kế hoạch:

1. **Dòng headline ADM làm được bằng dữ liệu phát hành.** Không cần `::gen` để
   có dòng ADM in-distribution; `images/test/lsun_bedroom/{adm,real}.tar.gz` là
   đủ. Nhãn trung thực là **`R1+R2`** (map và detector của mình, ảnh của tác
   giả) — không phải `R0`, cũng không phải `R3`.
2. **Giao thức train/val/test đúng như paper.** Có val riêng 5 archive ⇒
   **không** cắt 15 % từ train. Việc cắt chỉ còn là đường lui khi thiếu val, và
   `train()` ghi rõ điều đó vào `train.json`.

## 3. Ba lệnh, theo thứ tự

### 3.1 `peek` — xem bên trong tar, không giải nén (0 đồng, vài giây)

```bash
python -m dire_zero peek --tar "F:/du-lieu/.../dire/test/lsun_bedroom/lsun_bedroom/adm.tar.gz"
```

In ra số member, dung lượng, **đuôi file** và vài tên mẫu. Đây là lệnh phải chạy
**trước** khi viết bất cứ pipeline nào cho `dire/` hay `recons/`: tài liệu không
ghi bên trong là `.npy` float hay `.png` 3 kênh, và đoán sai định dạng là cách
nhanh nhất để có một pipeline chạy được mà vô nghĩa.

### 3.2 `dataset` — kiểm cây (0 đồng)

```bash
python -m dire_zero dataset --root F:/du-lieu --dataset lsun_bedroom \
                            --license research-only --splits train val test
```

In bảng `split × generator` (đã gộp alias), cảnh báo bất đối xứng, rồi **đếm
thật** số ảnh của đúng phần được chọn — `split/dataset/file: N ảnh / M MB`.
`scan_tree` cố ý **không** đếm khi quét: cây đầy đủ ~12 GB, bung hết mỗi lần
chạy là vài phút vô ích.

Cổng **đỏ ⇒ exit 3**:

Cổng | Vì sao
---|---
thiếu `--license` | repo có đường mirror công khai; `data_card.json` ghi thẳng giá trị này
`--dataset imagenet` mà không `--ack-ood` | ADM 256 học LSUN-Bedroom; ảnh thật ngoài phân bố ⇒ khoảng cách DIRE co lại ⇒ ACC về mức đoán mò
bộ lọc rỗng | gõ sai `--splits`/`--dataset` mà vẫn chạy tiếp là tiêu GPU cho 0 ảnh
chưa có `sec_per_step` | **không in tiền**. Khoảng thực tế 0,05–2,5 s/step ⇒ chênh 50×

`--root` nhận `F:/du-lieu`, `F:\du-lieu`, `/f/du-lieu`; trỏ thư mục cha cũng
được (tự lùi vào `DiffusionForensics/` nếu thấy `images/`).

### 3.3 `ingest` — chuẩn hoá (0 đồng, vài phút)

```bash
python -m dire_zero ingest --root F:/du-lieu \
                           --dataset lsun_bedroom --license research-only \
                           --splits train val test --limit 500
# mặc định ghi vào ./data/dire-zero-norm — muốn chỗ khác thì thêm --out
```

- giải nén **trực tiếp** từ `.tar.gz`, không cần giải ra đĩa trước;
- RGB, cạnh ngắn 256, cắt giữa 256×256 — đúng phép biến đổi của DIRE;
- tên file = 16 ký tự đầu sha256 của pixel: hai ảnh trùng nội dung không thể
  thành hai mẫu;
- ghi `manifest.tsv` (10 cột, có `source_tar` + `member` để truy ngược) và
  `data_card.json`;
- **chặn rò rỉ**: cùng pixel ở cả hai lớp ⇒ `IntegrityError`, dừng, không ghi gì;
- in sẵn lệnh `modal volume put`.

`--limit N` cắt **N ảnh đầu** mỗi archive (thứ tự trong tar ổn định ⇒ tái lập
được), không cắt ngẫu nhiên. Thêm hai cờ:

- `--strict` — archive hỏng ⇒ dừng, **không ghi gì** (mặc định: giữ phần đọc được
  và ghi chú; xem 3.4);
- `--no-resume` — ghi lại cả PNG đã có (mặc định bỏ qua file đã đúng).

#### Chỗ ghi: `./data/dire-zero-norm` trước, `F:/dire-zero-norm` khi thiếu chỗ

Thứ tự, và lý do của từng bậc:

| Bậc | Đường dẫn | Khi nào dùng |
|---|---|---|
| 1 | `--out` bạn gõ | luôn thắng, **trừ khi** không đủ chỗ (xem dưới) |
| 2 | `./data/dire-zero-norm` | mặc định, **tương đối** ⇒ không khoá vào ổ nào |
| 3 | `F:/dire-zero-norm` | chỉ khi bậc 1/2 **không đủ dung lượng** |

`F:` là **dự phòng dung lượng**, không phải chỗ mặc định. Không bao giờ nhảy sang
`F:` chỉ vì thư mục đích chưa tồn tại — thư mục ra thường chưa có, đó là chuyện
bình thường; nhảy ổ vì lý do đó là làm dữ liệu rơi vào chỗ bạn không ngờ.

Trước khi ghi, `ingest` trả lời ba câu và **in cả ba**:

```text
Ước lượng đầu ra: ≤ 6.1 GB (1,25 × dung lượng nguồn (chưa biết số ảnh))
  chỗ ghi: data/dire-zero-norm
  lý do:   ổ chính đủ chỗ (còn 45.0 GB, cần ≤ 6.1 GB)
```

Nếu phải dùng `F:`:

```text
  chỗ ghi: F:/dire-zero-norm
  lý do:   ổ chính không dùng được (chỉ còn 0.4 GB, cần ≤ 6.1 GB + 0.5 GB dự trữ)
           — dùng ổ dự phòng #1
```
và cùng dòng đó vào `data_card.json` (`out_root`, `out_reason`,
`warnings: ["[dự phòng dung lượng] …"]`) lẫn `stderr`.

Bốn chi tiết:

- **Ước lượng là trần, không phải trung bình.** Không `--limit`: `1,25 × dung
  lượng nguồn` (chỉ `stat`, tức thì) — PNG 256² đã cắt giữa nhỏ hơn hoặc xấp xỉ
  ảnh nguồn. Có `--limit`: đếm thật `n` ảnh (tar dừng ngay sau ảnh thứ `n`) rồi
  nhân `200 448 B`/ảnh. Đoán thấp rồi chết giữa đường đúng là lỗi mục này sinh ra
  để tránh.
- **Chừa 0,5 GB** cho hệ điều hành (`--min-free-gb`). Ghi kín ổ là cách làm hỏng
  cả máy lẫn Windows, và Windows còn tệ hơn khi đầy.
- **"Không truy cập được" ≠ "hết chỗ".** Ổ `F:` chưa cắm hoặc bị chặn quyền được
  báo riêng, không gộp thành "hết chỗ" — hai chuyện cần hai cách xử lý khác nhau.
- **Không có ứng viên nào đủ ⇒ dừng trước khi ghi** (`OutOfSpaceError`, exit 3), in
  số của từng ứng viên. Chết giữa đường ở 40 % thì đã để lại thư mục dở dang.

Tắt dự phòng: `--no-out-fallback` (thiếu chỗ thì dừng). Đổi ứng viên:
`--out-fallback D:/khac E:/khac2`.

#### `manifest.tsv` ghi đường dẫn **tương đối**

Có ổ dự phòng nghĩa là cùng một bộ dữ liệu có thể nằm ở `F:/…` hôm nay, `./data/…`
hôm sau, và `/zero/data` trong container. Đường dẫn tuyệt đối làm
`verify_manifest` chỉ đúng trên đúng một máy — và **hỏng im lặng**: nó báo "lệch
sha256" trong khi ảnh vẫn tốt, chỉ là tìm sai chỗ. Nên cột `png` là
`test/<gen>/<sha16>.png`, `verify_manifest` giải theo thư mục chứa manifest (hoặc
`root=` bạn truyền). Dòng kiểu cũ (tuyệt đối) vẫn kiểm được.

### 3.4 Khi ingest gặp lỗi — bốn đường dự phòng, và một điều cấm

> Đừng lẫn với mục 3.3: ở đó là dự phòng **chỗ ghi** (thiếu dung lượng), ở đây là
> dự phòng **dữ liệu** (hỏng giữa đường). Hai chuyện khác nhau, hai cơ chế khác
> nhau, và cả hai đều phải khai báo chứ không được im lặng.

Cây 12 GB đọc từ ổ ngoài, tải tay từ Baidu. Hỏng giữa đường là **chuyện thường**:
tar thiếu đuôi, một JPEG cụt, ổ đầy đúng lúc ghi. Trước đây một trong số đó làm
chết cả lượt chạy và để lại thư mục đầu ra nửa vời, không manifest, không
`data_card`.

Nay mỗi tầng có đường lui riêng:

| Lỗi | Đường dự phòng | Ghi lại ở đâu |
|---|---|---|
| tar không mở được `r:*` | đọc tuần tự `r\|*`, giữ phần đọc được | `stats["partial_archives"]` + `data_card.warnings` |
| ảnh cụt / header hỏng | đọc lại với `LOAD_TRUNCATED_IMAGES` | `stats["fallbacks"]` |
| ghi file lỗi (ổ đầy, file bị khoá) | thử lại sau 50 ms, rồi bỏ qua | `stats["rejects"]` |
| chạy lại sau khi hỏng | bỏ qua PNG đã đúng, **vẫn** kiểm rò rỉ | `data_card.warnings` (`[resume]`) |

Ba chi tiết quyết định đường dự phòng này có dùng được hay không:

**a. `r|*` là đường lui thật cho tar tải thiếu.** `r:*` cần bảng chỉ mục ở **cuối**
file; thiếu 3 % đuôi là `tarfile.open` ném lỗi và mất trắng archive dù 97 % nội
dung nằm ở đầu. `r|*` đọc tuần tự, không cần bảng chỉ mục, nên cứu được đúng
phần còn nguyên. Member đã đưa ra ở lần đọc trước được bỏ qua (theo tên + size)
để không đếm hai lần.

**b. Resume phải kiểm rò rỉ qua ranh giới hai lần chạy.** Đây là chỗ dễ sai
nhất. Nếu chỉ "bỏ qua file đã có", lần chạy trước để lại `real/abc.png`, lần này
`adm` có **đúng pixel đó** ⇒ không thấy trùng ⇒ ACC ~100 % và **không có lỗi
nào báo**. Vì vậy trước khi ghi, `ingest` quét lại `<out>/**/*.png` để dựng bảng
`sha → nhãn`. Rò rỉ qua nhiều lần chạy bị bắt như rò rỉ trong một lần.

**c. `--strict` kiểm **trước** khi ghi.** Nếu chỉ dừng lúc phát hiện hỏng ở cuối
archive thì ảnh đã nằm trên đĩa rồi — vẫn là thư mục dở dang, chỉ khác là có
thêm thông báo. Nên `--strict` đi hết archive một lượt chỉ-đọc-header trước
(đọc đúp; chỉ ở chế độ này), rồi mới ghi. Đổi lại: hoặc ghi đủ, hoặc không ghi gì.

**Điều cấm: dự phòng im lặng.** Một dataset thiếu 3 % mà `data_card.json` nói đủ
là kiểu sai lệch tệ hơn hẳn việc chương trình dừng — vì nó đi theo dữ liệu lên
Volume, vào báo cáo, vào con số cuối cùng, và không ai kiểm lại. Nên:

- có archive đọc dở ⇒ lệnh `ingest` **exit 3**, in `[KHÔNG ĐẦY ĐỦ]`, và không
  khuyến khích đẩy lên Volume;
- mọi lần dùng đường lui đều nằm trong `data_card.warnings` với tiền tố
  `[dự phòng]`;
- `stats["rejects"]` ghi **tên member** + lý do, không chỉ đếm. "Bỏ 7 ảnh" thì
  không sửa được gì; "`test/adm:000417.png` — UnidentifiedImageError" thì sửa được.

## 4. Thư mục ảnh nằm cạnh tar — phải cảnh báo, không được im lặng

Cây thật có chỗ **vừa** có `real.tar.gz` **vừa** có thư mục `real/` đã giải nén
dở:

```text
images/test/lsun_bedroom/real.tar.gz
images/test/lsun_bedroom/real/            ← cùng nội dung, đã giải nén
dire/test/imagenet/imagenet/real/
recons/test/lsun_bedroom/{imagenet,lsun_bedroom}/
```

Đây là bẫy hai chiều. Nếu tool **im lặng đọc cả hai**, mỗi ảnh thật vào hai lần
⇒ lớp 0 phình lên, tỉ lệ lớp lệch, và nếu bản giải nén nằm ở split khác thì đó
là **rò rỉ train/test** — loại lỗi cho ACC 100 % và không bao giờ tự báo. Nếu
tool **im lặng bỏ qua**, bạn tưởng dữ liệu đã có mặt trong khi nó không được
dùng. Nên: tool chỉ đọc `*.tar.gz`, và **in cảnh báo** cho mọi thư mục ảnh/rỗng
nằm cạnh tar (`_warn_extracted_dirs`). Cảnh báo đó không phải lỗi — nó là nhắc
bạn xoá bản giải nén hoặc chuyển nó ra khỏi cây.

## 5. Ba đường chạy, chọn theo nhãn bạn được phép nói

### 5.1 R0 — chấm lại bằng dữ liệu + checkpoint của tác giả (CPU, ~$0)

Không đụng GPU. Không phải "tái lập" — đây là **chấm lại**, và artefact phải tự
khai `R0`.

```bash
# map tác giả: dire/<split>/lsun_bedroom/<gen>.tar.gz  →  maps/<split>/<gen>/*.npy
for f in .../dire/train/lsun_bedroom/*.tar.gz; do
  gen=$(basename "$f" .tar.gz)
  mkdir -p "F:/dire-zero-r0/maps/train/$gen"
  tar -xzf "$f" -C "F:/dire-zero-r0/maps/train/$gen" --strip-components=0
done
# lặp lại cho val/ và test/ — ba split, đừng trộn
```

Sau đó `pipeline.collect_maps("F:/dire-zero-r0/maps")` đọc nhãn từ **tên thư mục
generator** (`real` ⇒ 0) và `pipeline.train(...)` chạy được trên CPU.

Hai điều phải nói rõ, vì chúng quyết định nhãn:

- **chạy bằng 6 `checkpoints/*.pth`** mới đúng là R0. Loader cho checkpoint tác
  giả **chưa có trong repo** ⇒ việc này còn là TODO, không phải đã xong.
- lấy map tác giả rồi **tự train detector** thì không phải R0: nhãn phải ghi
  "map: tác giả · detector: của mình". Gọi nó R0 là tự khen một thành phần chưa
  làm.
- `--allow-published` chỉ mở cổng G0 cho `dire/`; nó **không** đổi nhãn. Nhãn
  R0 phải khai bằng `--fake-source published`.

### 5.2 R1+R2 — map + detector của mình, ảnh của tác giả (GPU)

Đường rẻ nhất có ý nghĩa khoa học, và giờ **đã làm được dòng ADM**:

```bash
python -m dire_zero ingest --root F:/du-lieu \
        --dataset lsun_bedroom --license research-only \
        --splits train val test --fake-source author --limit 500
# → ./data/dire-zero-norm (tự sang F:/dire-zero-norm nếu ổ hiện tại không đủ chỗ)
modal volume put dire-zero ./data/dire-zero-norm data -r
modal run -m dire_zero.modal_app::checkdata       # sha256 + hai lớp + tỉ lệ ≤5:1
modal run -m dire_zero.modal_app::bench           # ĐO sec/step trước khi tin bảng giá
modal run -m dire_zero.modal_app::dire --splits train,val,test --steps 20
modal run -m dire_zero.modal_app::train --epochs 40
modal run -m dire_zero.modal_app::report
```

### 5.3 R1+R2+R3 — tự sinh ảnh giả

Chỉ khi cần một generator **không có** trong cây, hoặc khi muốn nói `R3`. Đắt
hơn ~2,7 USD cho 500 ảnh, và không còn cần cho dòng ADM nữa.

## 6. Chi phí

**Không có bảng tiền trong tài liệu này.** Bảng giá cũ (0,35 s/step giả định)
đã bị bỏ: nó là một con số chưa đo, và mọi kết luận "rẻ/đắt" dựa trên nó đều là
số bịa có đơn vị. Trình tự đúng:

1. `bench` đo `sec/step` thật trên A10G;
2. `dataset --sec-per-step <số đo>` in kế hoạch NFE → GPU-h → USD;
3. chỉ khi đó mới quyết định cắt bao nhiêu ảnh.

Bậc độ lớn để biết trước: DIRE tốn **2·S NFE/ảnh** (S = số bước DDIM, mặc định
20 ⇒ 40 NFE), sinh tốn `S`. Với `--limit 500` mỗi archive, toàn bộ đường 5.2
nằm trong khoảng "một buổi tối GPU"; toàn bộ train split (hàng chục nghìn ảnh ×
nhiều generator) thì không nên chạy — hãy cắt tiền tố và nói rõ đã cắt.

## 7. Hạn chế của đường này

- **Bản kê vs inventory lệch nhau** (mục 0) ⇒ mọi kế hoạch phải dựa trên số do
  `dataset` đếm, không dựa trên tài liệu, kể cả tài liệu này.
- **Dòng `imagenet/` không so được với paper**: ADM 256 vô điều kiện học
  LSUN-Bedroom; ảnh ImageNet "thật" cũng bị tái tạo kém ⇒ khoảng cách DIRE co
  lại. Kết quả chỉ có nghĩa "đo bằng mô hình sai phân bố".
- **`recons/test/` thiếu `adm`** (chỉ `real`, `ldm`, `pndm`) ⇒ đối chiếu số học
  DDIM trên ADM phải dùng map ở `dire/`, không dùng `recons/`.
- **`recons/` không dùng để train**, chỉ để kiểm phép tái tạo.
- **Ảnh test của tác giả không do mình sinh** ⇒ trần nhãn của đường 5.2 là
  `R1+R2`.
- **`--limit` cắt tiền tố** trong tar ⇒ nếu tar đóng gói theo thứ tự lớp con
  (vd. toàn ảnh sáng trước), mẫu bị lệch. Kiểm phân bố độ sáng của mẫu chưa có
  trong repo.
- **Chưa chạy trên dữ liệu thật của bạn.** Toàn bộ test hiện tại dựng cây giả
  trong `tmp_path`; chưa lần nào đọc 12 GB thật, chưa lần nào chạy trên GPU,
  chưa lần nào chạy trên Modal thật.
