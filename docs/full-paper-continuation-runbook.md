# DIRE: kế hoạch tiếp tục từ dữ liệu và kết quả hiện có

> **Mục tiêu:** hoàn thành thực nghiệm trên cây `DiffusionForensics` người dùng
> đã kiểm kê, tiếp tục từ artefact hợp lệ thay vì xoá hoặc chạy lại toàn bộ.
> Runbook này phân biệt việc **đã có bằng chứng**, việc **cần kiểm tra**, và việc
> **chưa được code hỗ trợ đầy đủ**.
>
> **Chưa thực hiện:** các con số kiểm kê dưới đây được chép từ log người dùng gửi;
> sandbox này chưa đọc được ổ `F:` hay Modal Volume để xác minh checksum/trạng
> thái. Runbook không khởi chạy GPU, upload dữ liệu hay tạo chi phí.

## 1. Phạm vi đã biết và cách hiểu đúng

Lệnh `dire_zero dataset` đã đếm 25 archive trong
`F:\DiffusionForensics/images/`, bố cục `diffusionforensics`:

| Split | Ảnh thật | Ảnh giả | Thành phần ảnh giả |
|---|---:|---:|---|
| `train` | 40.000 | 160.000 | ADM, iDDPM, PNDM, StyleGAN: mỗi loại 40.000 |
| `val` | 1.000 | 4.000 | ADM, iDDPM, PNDM, StyleGAN: mỗi loại 1.000 |
| `test` | 1.000 | 12.368 | ADM 1.000; DALL·E 2 500; DDPM 768; Diff-ProjectedGAN 1.000; Diff-StyleGAN 1.000; iDDPM 1.000; IF 1.000; LDM 1.000; Midjourney 100; PNDM 1.000; ProjectedGAN 1.000; SDv2 1.000; StyleGAN 1.000; VQ-Diffusion 1.000 |
| **Tổng** | **42.000** | **176.368** | **218.368 ảnh** |

`stylegan_official.tar.gz` được chuẩn hoá thành generator `stylegan`. `dire/` và
`recons/` bị bỏ qua có chủ ý: clean-room không lấy map/reconstruction đã xuất
bản để tự nhận R1. Ảnh giả trong cây là của tác giả, nên trần provenance hiện là
**R1+R2**, không phải R3.

Ở `S=20`, một map cần `2S = 40` lượt forward. Nếu tính đủ 218.368 ảnh thì tổng
là **8.734.720 NFE**. Mỗi map float16 `(3,256,256)` chiếm 393.216 byte trước
metadata/filesystem; toàn bộ map cần xấp xỉ **85,9 GB thập phân (80,0 GiB)**.
Đây là dung lượng map lý thuyết theo shape, không phải đo dung lượng thực tế của
Volume. Cần cộng thêm ảnh PNG, checkpoint, metadata và phần dự phòng.

**100% cây local không tự động đồng nghĩa 100% tái hiện paper.** Log có test
archive với 500, 768 và 100 ảnh. Phải so count, prompt, split và protocol với
Table 1/repo đã ghim. Không padding, nhân bản hay sinh bù ảnh để làm đẹp số.
Các generator chỉ có test giữ nguyên vai trò test; không tự chia chúng thành
train/val rồi gọi là split của paper.

## 2. Nguyên tắc không mất dữ liệu

1. `F:\DiffusionForensics` là bản nguồn; không tải lại nếu bytes đã có và không
   xoá hay đổi tên archive gốc.
2. Không đưa 218.368 ảnh vào RAM, một container hay một request upload. Đọc tar,
   giải mã, tính map và upload theo archive/lô; giới hạn bộ nhớ bằng batch nhỏ.
3. Giữ `.staging` cho tới khi đã đối chiếu file trên Volume. Chỉ xoá staging
   sau khi manifest/hash phía Volume xác nhận bản đích đầy đủ.
4. Chạy có thể resume; file có cùng tên nhưng khác hash/config không được coi là
   bản hợp lệ chỉ vì tồn tại.
5. Mọi đợt sinh ra `run.json`/manifest/log riêng, ghi sha checkpoint, input,
   split, generator, `S`, commit code, precision, phần đã xong và phần còn thiếu.
6. Các lần chạy trước được giữ nguyên. Kết quả thăm dò không bị sửa nhãn thành
   kết quả paper; artefact lỗi chỉ bị loại khỏi phân tích với lý do truy vết.

## 3. Kế hoạch từng bước

Mỗi bước có đầu ra mong đợi, lỗi dễ gặp và cổng quyết định. Chỉ chuyển bước khi
cổng trước đã xanh.

### Bước 0 — Đóng băng trạng thái hiện có, chưa chạy lại gì

**Thao tác**

- Ghi commit code, cấu hình, tên Modal Volume, đường dẫn `DIRE_LOCAL_ROOT`, ngày
  lấy inventory và vị trí của các log/checkpoint/map đã có.
- Lưu bản sao metadata (`data_card.json`, `manifest.tsv`, `dire.json`,
  `train.json`, `eval.json`, `scores.json`, `bench.json`) ra một thư mục archive
  riêng; không sửa artefact cũ tại chỗ.
- Trong workspace hiện đã thấy `tmp/eval/eval.json` và `tmp/eval/scores.json`
  cho 1.000 mẫu (500 `real`, 500 `adm`), ngưỡng nguồn `val`. `eval.json` ghi
  `acc=0,781` nhưng `acc_ci95=[0,47; 0,529]`, không chứa điểm ước lượng. Đây là
  lỗi summary CI đã biết; giữ nguyên artefact làm bằng chứng lịch sử và chấm lại
  từ điểm chính xác hoặc chạy lại evaluate trên map/checkpoint tương ứng. Không
  vì lỗi này mà mặc định tính lại map hoặc train lại detector.
- Không commit dữ liệu/checkpoint lớn vào Git.

**Kết quả mong đợi:** một ảnh chụp trạng thái có thể khôi phục/đối chiếu và danh
sách “đã có / chưa biết / chưa có”. Không có GPU, network data transfer hay file
nào bị xoá.

**Lỗi/thiếu sót có thể có:** không truy cập được Volume, thiếu `dire.json` hoặc
log của run cũ, tên volume không đúng. Ghi là `unknown`, không suy ra rằng dữ
liệu không tồn tại. Lấy log/metadata từ chủ Volume trước khi quyết định tính lại.

**Cổng:** các artefact cũ được giữ nguyên và có đường dẫn/ID rõ.

### Bước 1 — Đối chiếu protocol paper với chính inventory này

**Thao tác**

- Tạo ma trận `split × domain × generator × kind` từ 25 archive và so với
  paper Table 1, bảng/hình cần chạy, repo tác giả tại commit đã ghim.
- Ghi riêng count thực tế của DALL·E 2 (500), DDPM (768), Midjourney (100) và
  mọi khác biệt khác; ghi `expected_count: null` + lý do cho chi tiết chưa xác
  minh. Kiểm alias `stylegan_official → stylegan`.
- Chốt trước cấu hình cho từng paper table: detector train row, test columns,
  threshold, preprocess, số bước và metric. Không mặc định rằng mọi cấu hình ở
  tài liệu nội bộ khớp paper.

**Kết quả mong đợi:** ma trận có trạng thái `reuse / recompute / missing /
not-applicable / unresolved`; nêu được phần nào có thể so trực tiếp, phần nào
chỉ là mở rộng trên dữ liệu hiện có.

**Lỗi/thiếu sót có thể có:** paper không công bố seed/count/prompt đầy đủ; code và
paper bất đồng; archive mang tên khác hoặc split không tương ứng. Không đoán để
lấp chỗ trống. Đánh dấu hạn chế và xin quyết định khoa học trước khi chạy phần
chịu ảnh hưởng.

**Cổng:** không còn “100% paper” dựa trên tên archive; có bảng truy nguyên và
các khoảng trống được khai báo.

### Bước 2 — Kiểm tra run, map và checkpoint đã có để quyết định tái sử dụng

**Thao tác**

1. Trên Modal, đọc `data_card.json`/manifest và các run; xác nhận hash ADM.
2. Kiểm inventory map bằng lệnh không cần GPU:

   ```bash
   modal run -m dire_zero.modal_app::maps --n 0 --generator auto
   ```

   `--n 0` ở `::maps` có nghĩa xem cả cây. Dùng `--verify 20` cho kiểm tra mẫu
   ban đầu; bước audit đầy đủ ở Bước 8.
3. Với mỗi map, đối chiếu nguồn ảnh, split/generator, checkpoint SHA, sampler,
   `steps=20`, dtype/scale/shape, code version và run index. Với detector, đối
   chiếu config, init, số epoch, split, hash map và test nào đã từng được dùng.
4. Xếp artefact thành: **tái sử dụng cho kết quả chính**, **chỉ làm pilot**,
   **cần chấm lại**, **cần tính/train lại**, **không dùng**. Không gộp các loại.

**Kết quả mong đợi:** danh sách map/checkpoint được bảo toàn, phần thiếu theo
`split/generator`, và không chạy lại các map khớp chính xác cấu hình.

**Lỗi/thiếu sót có thể có:** map file hiện hữu nhưng bị ghi dở; metadata thiếu
`steps`/checkpoint hash; map trong `fake/` làm mất nhãn generator; checkpoint
được train theo cấu hình khác. `::maps` mặc định đếm file không rỗng; kiểm file
ngẫu nhiên không chứng minh toàn bộ file đúng.

**Thiếu sót code cần xử lý trước resume quy mô lớn:** `--skip-existing` hiện chỉ
kiểm đường dẫn và kích thước khác 0, không xác nhận hash/config của map. Cần
thêm cổng so fingerprint từ manifest/run trước khi cho tái sử dụng. Map không
khớp phải nằm ở namespace/run riêng; không xoá map cũ.

**Cổng:** tái sử dụng chỉ khi map và checkpoint có metadata đủ khớp; nếu không,
được xem là pilot cho tới khi xác minh.

### Bước 3 — Ingest toàn bộ ảnh thành PNG chuẩn hoá, đọc tuần tự

**Lệnh dự kiến trên máy chứa `F:`** (chỉ chạy sau khi xác nhận thư mục đích và
chỗ trống; không ghi đè một cây khác):

```bash
export DIRE_LOCAL_ROOT='F:/DiffusionForensics'
python -m dire_zero dataset --root "$DIRE_LOCAL_ROOT" --dataset lsun_bedroom \
  --license research-only --splits train val test

python -m dire_zero ingest --root "$DIRE_LOCAL_ROOT" --dataset lsun_bedroom \
  --license research-only --fake-source author --splits train val test \
  --out 'F:/dire-zero-norm' --strict
```

Nếu `F:/dire-zero-norm` đã có dữ liệu hợp lệ, dùng lại đúng thư mục đó theo
resume của ingest sau khi kiểm manifest; không chọn đường mới chỉ vì đường cũ
đã tồn tại. Không thêm `--allow-published`: đường clean-room bỏ `dire/`.

**Kết quả mong đợi:** cây PNG `<split>/<generator>`, `manifest.tsv`,
`data_card.json`, số ảnh mới/dùng lại/bị từ chối, kết quả verify SHA. Mục tiêu
đếm theo log hiện tại là 218.368 ảnh (42.000 thật + 176.368 giả), nếu tất cả ảnh
đọc/chuẩn hoá hợp lệ. `ingest` xử lý archive tuần tự; không giữ toàn bộ ảnh trong
RAM. Công cụ phải in ước lượng dung lượng trước khi ghi.

**Lỗi có thể xảy ra:** thiếu chỗ; tar lỗi; member ảnh hỏng; ảnh bị reject; Windows
đường dẫn/MSYS2 đổi tham số; `--strict` dừng khi archive không sạch. Khi có lỗi,
không đẩy cây thiếu lên Volume như thể đầy đủ. Giữ archive nguyên vẹn, đọc danh
sách member lỗi và sửa nguồn/đích rồi chạy resume.

**Thiếu sót cần hiểu:** PNG được resize/chuẩn hoá theo ingest; đây là input
pipeline, không thay archive gốc. So lại 256×256, RGB, range và thứ tự tên. Nếu
số ghi ra không khớp inventory, tìm nguyên nhân trước khi chạy DIRE.

**Cổng:** `manifest.tsv` SHA khớp; không có reject/partial archive chưa được giải
quyết; label của ảnh giả là `author`, `data_card` ghi `R1+R2`.

### Bước 4 — Upload dữ liệu theo lô có thể resume

**Thao tác**

```bash
python tools/put_resume.py --local 'F:/dire-zero-norm' --remote data --dry-run
python tools/put_resume.py --local 'F:/dire-zero-norm' --remote data --chunk 500
python tools/put_resume.py --local 'F:/dire-zero-norm' --remote data --verify
modal run -m dire_zero.modal_app::checkdata
```

`put_resume.py` gửi theo batch nhỏ, so tên/kích thước remote để chỉ gửi phần
thiếu và đẩy manifest/data card cuối cùng. `checkdata` chạy trên Volume và kiểm
lại dữ liệu/manifest bằng SHA. Không dùng một `modal volume put` cho cả cây
218.368 file: một request khổng lồ có thể không commit được gì dù tiến trình đã
chuyển gần hết byte.

**Kết quả mong đợi:** số file mỗi nhánh local/remote khớp; `checkdata` không có
hash mismatch, split leakage, thiếu manifest hoặc bất thường trong tỉ lệ lớp.
`checkdata` chỉ kiểm hash/cấu trúc, không giải mã pixel. Trước khi tính map, quét
ảnh có thể đọc được. Nếu nghi ngờ file 0 byte, dùng lượt quét metadata nhanh
trước; lượt này không mở/giải mã pixel:

```bash
modal run -m dire_zero.modal_app::pruneemptyimages --splits train,val,test \\
  --report-id empty-images-2026-10-14
```

Lệnh trên là dry-run, chỉ báo cáo. Sau khi xem TSV, nếu đúng là chỉ muốn xóa file
0 byte, chạy lại với ID mới và `--apply`:

```bash
modal run -m dire_zero.modal_app::pruneemptyimages --splits train,val,test \\
  --apply --report-id empty-images-delete-2026-10-14
```

Nó chỉ xóa image file vẫn có size 0 khi kiểm tra lần cuối; không đụng file rỗng
ngoài các thư mục split, nonzero file, hoặc DIRE map. Báo cáo JSON/TSV lưu dưới
`/zero/runs/image-audits/`, kèm tree path và byte size. Sau khi xóa, đồng bộ
manifest bằng đúng report ID: chạy trước ở dry-run, rồi `--apply` để lưu backup
và loại đúng các row đã xóa khỏi `manifest.tsv`:

```bash
modal run -m dire_zero.modal_app::reconcilemanifest --report-id empty-images-delete-2026-10-14
modal run -m dire_zero.modal_app::reconcilemanifest --report-id empty-images-delete-2026-10-14 --apply
```

Backup được giữ ở `/zero/data/manifest.tsv.before-empty-images-delete-2026-10-14.bak`.
Để tìm file không giải mã được hoặc sai kích thước 256×256, dùng audit sâu hơn:

```bash
modal run -m dire_zero.modal_app::auditimages --splits train,val,test \\
  --expected-size 256 --report-id image-audit-2026-10-14
```

Audit sâu mở tất cả ảnh và giải mã pixel để xác thực, nên lâu hơn; nó chỉ đánh dấu
trong báo cáo, không tự xóa hoặc tự lọc danh sách đầu vào của `::dire`.

**Lỗi có thể xảy ra:** token/volume sai; timeout/mạng; upload lô lỗi; remote file
cùng tên nhưng khác size; quota/disk Volume thiếu. Chạy lại đúng lệnh resume,
hoặc thu nhỏ `--chunk`/dùng `--only train/pndm`; không tạo bản remote mới rồi bỏ
bản dở mà chưa kiểm kê.

**Thiếu sót:** uploader so size để quyết định truyền lại; size khớp chưa chứng
minh bytes khớp. `checkdata`/manifest SHA là bước xác minh bắt buộc. Không xoá
`.staging` ở local trước khi bước này xanh.

**Cổng:** dữ liệu trên Volume khớp manifest SHA; data card/manifest đã được upload
sau ảnh; staging vẫn còn cho tới khi có biên nhận xác minh.

### Bước 5 — Nạp ADM và đo đúng cấu hình thiết bị

**Thao tác**

```bash
modal volume put dire-zero 'F:/<duong-dan-checkpoint-adm>' ckpt/adm_lsun.pt
modal run -m dire_zero.modal_app::selfcheck
modal run -m dire_zero.modal_app::bench
```

Thay đường dẫn checkpoint bằng file thực có trên máy; không tải lại nếu đã có.
So SHA của file nguồn với SHA mà `selfcheck` in sau khi nạp. `bench` phải chạy
trên cùng loại GPU/dtype/batch sẽ dùng cho map.

**Kết quả mong đợi:** nạp `strict=True`, 566 tensor/shape đúng, map hữu hạn,
NFE thực tế đúng `2S`; `bench.json` ghi GPU, batch, dtype, checkpoint SHA và
`sec_per_step` đo tại chỗ.

**Lỗi có thể xảy ra:** file chưa ở đúng path, checkpoint sai/không đủ byte,
SHA khác, CUDA/driver incompatibility, VRAM OOM, container startup fail. Dừng
trước full run; không hạ strictness hoặc thay checkpoint không ghi provenance.

**Thiếu sót:** số `0,0917 s/step` ở log cũ chỉ là benchmark của lần/GPU đó, không
thay cho bench mới. Nếu đổi GPU/precision/batch thì đo lại. Cổng tiền/time của
mỗi job vẫn là an toàn vận hành; không được cắt protocol để lách cổng.

**Cổng:** selfcheck và bench thành công, output JSON có thể truy nguyên; người
chủ tài khoản duyệt resource và dự toán theo số đo mới.

### Bước 6 — Chuẩn bị map hiện có và chạy phần còn thiếu theo đợt

**Trước hết:** Bước 2 phải sửa/giải quyết fingerprint. Không dùng `--skip-existing`
như bằng chứng rằng map cũ cùng `checkpoint/S/config`.

`::dire` hiện giới hạn `--n` là số ảnh đầu tiên **mỗi split/generator** theo thứ
tự tên file; nó chưa có offset độc lập. Có thể mở rộng dần tiền tố, giữ dữ liệu
không đổi và dùng map khớp đã xác minh:

```bash
# Đợt đầu: tối đa 2.000 ảnh mỗi split/generator
modal run -m dire_zero.modal_app::dire --splits train,val,test \
  --generator auto --n 2000 --steps 20 --skip-existing --max-usd <muc-da-duyet>

# Các đợt kế: mở rộng tiền tố; giữ cùng thứ tự input, config và checkpoint
modal run -m dire_zero.modal_app::dire --splits train,val,test \
  --generator auto --n 4000 --steps 20 --skip-existing --max-usd <muc-da-duyet>
# tiếp tục 6000, 8000, ... tới 40000; dừng ở ngưỡng cần thiết cho từng cell
```

`<muc-da-duyet>` là giá trị số sau khi xem estimate của chính đợt; dấu ngoặc nhọn
không phải cú pháp để copy nguyên văn. Nếu lệnh tự in nhiều nhóm do timeout,
chạy đúng các nhóm đã in, thêm cap được duyệt nếu công cụ không mang theo cap.
Không dùng `--generator all` ở `::dire`: entrypoint nhận `auto` hoặc danh sách
thật, `all` không phải alias ở hàm này.

**Kết quả mong đợi:** từng run ghi số ảnh mới/tái sử dụng, NFE và checkpoint
Volume định kỳ mỗi 200 ảnh; các đợt kế không tính lại ảnh đã xác minh. Đủ dữ
liệu thì `maps --n 0` báo `missing=0` cho các cell mục tiêu. Cây full local cần
218.368 map nếu chính sách là map mọi ảnh.

**Lỗi có thể xảy ra:** hết timeout 6 giờ, lỗi mạng/commit, OOM, mismatch shape,
không có `bench.json`, cap USD/job bị vượt, hoặc lệnh lặp lại prefix mà map có
sai metadata. Lệnh dừng không có nghĩa toàn bộ phần trước mất: kiểm `::maps`
trước khi tiếp tục; các map commit trước đó vẫn phải được kiểm hash/config.

**Thiếu sót hiện tại:** `::dire` tự chia nhóm generator theo thời gian nhưng
không chia một cell 40.000 ảnh thành các đoạn có offset; tăng `--n` theo tiền tố
là workaround hợp lệ chỉ khi danh sách ảnh bất biến. Với vận hành lâu dài nên
thêm `--offset/--limit` hoặc shard manifest stable, kèm ledger shard. Cổng
`--skip-existing` phải so fingerprint như Bước 2. Nếu bộ input đổi thứ tự hoặc
nội dung giữa các đợt, không tiếp tục theo prefix cũ.

**Cổng:** đủ map cho đúng ma trận đã chốt; mọi map có fingerprint/metadata đúng;
không còn file map rỗng/cụt.

### Bước 7 — Audit toàn bộ map và nguồn

**Lệnh dự kiến**

```bash
modal run -m dire_zero.modal_app::maps --n 0 --generator auto --verify 40000
modal run -m dire_zero.modal_app::checkdata
```

`--verify 40000` yêu cầu đọc thử tối đa 40.000 map mỗi cell (mọi cell hiện có tối
đa 40.000); `maps` dùng lấy mẫu không lặp trong từng cell. Đồng thời đối chiếu
`dire.json`/manifest: ảnh nguồn, generator, split, NFE, hash ADM, `steps=20`,
shape `(3,256,256)`, finite values và range `[0,2]`.

**Kết quả mong đợi:** inventory map bằng ảnh được chọn, `missing=0`,
`bad_files=[]`, không có map thừa không rõ nguồn, không có nhãn `fake/` lẫn với
generator thật.

**Lỗi/thiếu sót:** chỉ kiểm shape không phát hiện map tính sai về số; lấy mẫu
không kiểm từng file. Nếu cần bảo đảm từng file, thêm full scan/read/hash theo
manifest (có thể I/O lâu). Không dùng PNG visualization làm phép kiểm số học.

**Cổng:** map inventory khớp chính xác với ma trận đã chốt và mọi khác biệt được
giải thích.

### Bước 8 — Train detector theo protocol, không lấy run thăm dò làm kết quả cuối

**Dữ liệu train thật sự có** cho LSUN-Bedroom: ADM, iDDPM, PNDM, StyleGAN; mỗi
generator có train/val. Train riêng từng detector với `real` và đúng generator.
Các generator chỉ có test không được train; không dùng `split-mode hash` để giả
lập split paper.

Với map đã sinh trong namespace rebuild, truyền cùng `--run-id` để train đọc
đúng `/zero/rebuild/maps/<id>` và ghi checkpoint/ledger riêng dưới
`/zero/runs/rebuild/<id>/` (không đụng `/zero/maps` hoặc kết quả train cũ):

```bash
modal run -m dire_zero.modal_app::train --run-id rebuild-2026-10-14 --generator stylegan --epochs 40
modal run -m dire_zero.modal_app::evaluate --run-id rebuild-2026-10-14 --generator stylegan --split test
modal run -m dire_zero.modal_app::report --run-id rebuild-2026-10-14
```

Trong inventory hiện tại, chỉ `stylegan` có train/val fake maps; `ddpm` và
`midjourney` là test-only holdout. Không train chúng. Lệnh `::evaluate` hiện chấm
real + generator đã train; nó **chưa** sinh ma trận đầy đủ StyleGAN-detector ×
(DDPM, Midjourney, StyleGAN). Cần evaluator ma trận riêng trước khi tuyên bố
cross-generator Table 2 hoàn chỉnh.

**Kết quả mong đợi:** một checkpoint + history + `train.json` cho từng hàng
train, gồm best epoch, val AP, threshold từ val, số mẫu và hash map. Không test
trong lúc chọn epoch/hyperparameter.

**Thiếu sót triển khai phải giải quyết trước khi gọi là paper reproduction:**

- entrypoint `::train` hiện mặc định 40 epochs và pipeline ghi `init="scratch"`;
  tài liệu paper/repo cần đối chiếu cho thấy protocol tác giả dùng ImageNet
  initialization và tới 400 epoch/early stopping. Không được gắn nhãn kết quả
  40-epoch-from-scratch là exact.
- `pipeline.train` hiện không lưu optimizer/scheduler/RNG state để resume chính
  xác giữa epoch; `train` của Modal chỉ lưu `detector.pth` và `train.json` sau
  lượt. Timeout có thể làm mất tiến độ của detector đang train.
- cần ghi cấu hình khởi tạo, augment, optimizer, batch, early-stop và chọn
  checkpoint đúng nguồn paper; test chỉ chấm sau khi model/threshold đã đóng
  băng.

Các detector đã train bằng config khác vẫn được giữ làm pilot/so sánh phụ. Nếu
cần đúng protocol, chỉ train lại detector tương ứng trên map đã có — không cần
chạy lại `::dire`.

**Lỗi có thể xảy ra:** split một lớp, map thiếu, OOM batch, timeout, train không
resume, val thiếu một lớp hoặc AP không xác định. Giữ `null` và lý do khi metric
không định nghĩa; không thay bằng 0 hay threshold trên test.

**Cổng:** 4 hàng LSUN có train split được tạo theo config đã audit; mọi detector
có checkpoint hash và val selection trace.

### Bước 9 — Chấm ma trận train-row × test-generator-column

**Điều kiện:** test set và checkpoint đã đóng băng; không chỉnh threshold theo
cột test. Test-only generator là hold-out/cross-generator.

**Thiếu sót code quan trọng:** `::evaluate` hiện lọc map test theo generator đã
train trong `train_meta.generators_used`. Vì vậy nó có thể chấm đúng cell/đường
demo nhưng **không tự tạo đầy đủ ma trận cross-generator** cho Table 2. Cần bổ
sung matrix evaluator: mỗi detector row chấm từng test generator column, ghi
scores/labels/sample IDs đầy đủ và tách output theo model × test generator.

**Kết quả mong đợi:** ma trận số đo có đủ cell, count từng cell và cờ
`ran/missing/not-applicable`; ACC theo quy ước paper, AP/AUC, confusion matrix,
CI phù hợp với sample unit. Ghi cả metric ở ngưỡng paper và threshold đã chọn
trên validation nếu protocol cần cả hai; không chọn lại theo test.

**Lỗi/thiếu sót có thể có:** số mẫu test không cân bằng (Midjourney 100; DDPM
768; DALL·E 2 500), prompt tương quan làm CI quá hẹp, sample ID va chạm giữa
generator, model hash không khớp scores, threshold nhầm giữa logit và xác suất.
Báo `n`/CI từng cell và hạn chế; không diễn giải chênh nhỏ thành thứ hạng chắc
chắn.

**Đặc biệt:** `scores.json` hiện lưu xác suất làm tròn 6 chữ số ở đường
`::evaluate`; số đã làm tròn có thể tạo ties/gần threshold. Để tính lại metric/CI
chính xác, lưu full-precision logits/probabilities trong artefact đánh giá, đồng
thời có file visualization riêng đã làm tròn. Lỗi CI cũ phải sửa bằng full score
hoặc chấm lại trên map/checkpoint; không tính lại map/train chỉ để sửa CI.

**Cổng:** mọi paper table cell hợp lệ có scores truy nguyên; test không dùng để
chọn threshold/model.

### Bước 10 — Ablation và robustness trong phạm vi paper

Đã thêm runner tách riêng cho các protocol được yêu cầu:

- `::table4 --steps {5,10,20,50}`: mỗi S có input namespace riêng, map được tính lại, detector ImageNet khởi tạo riêng, năm generator test và ledger `missing_steps`.
- `::table5_prepare` + `::table5_train`: REC, RGB, RGB_DIRE và DIRE từ cùng triplet/split tại S=20; train/eval từng representation riêng.
- `::figure4_robustness`: clean, Gaussian blur σ=1,2,3 và JPEG q=65,30; lưu scores/labels/sample IDs/AP theo từng generator. `tools/figures.py f4-robustness` dựng hình từ ledger.

Lệnh chạy, yêu cầu checkpoint và vị trí artefact ở `docs/table4-table5-figure4-runbook.md`. Các runner không được thực thi trong lần cập nhật code này; cho tới khi chạy đủ và kiểm ledger, trạng thái vẫn là `not run`.

**Giới hạn:** Figure 4 hiện đối chiếu RGB ResNet-50 của dự án với DIRE, chưa có CNNDetection*/PatchForensics* của paper. Manuscript không nêu đủ chi tiết cho fusion RGB+DIRE và thời điểm perturbation; code khóa lựa chọn đang dùng trong metadata, không tuyên bố bitwise/exact reproduction. Optimizer/RNG chưa resume sau timeout.

**Cổng:** đủ 4 giá trị S, 4 representations và 6 điều kiện robustness; mọi cell có count, provenance và artifact scores. Các baseline Figure 4 còn thiếu phải được đánh dấu riêng, không nội suy hay nhập số paper.

### Bước 11 — Generator mở rộng, chỉ sau khi protocol gốc đã khép

Các generator test-only trong cây có thể dùng để đánh giá tổng quát hoá mà không
train trên chúng. Huấn luyện model generator mới chỉ là mở rộng riêng khi có
câu hỏi cụ thể, checkpoint/license/config/seed được lưu. `ours_adm` nếu dùng
checkpoint ADM công khai vẫn là **in-distribution**. Không cần train ADM từ đầu
để hoàn tất protocol dùng checkpoint ADM công khai.

**Kết quả mong đợi:** bảng bổ sung tách riêng khỏi bảng tái hiện; provenance
không nhập nhằng `author` với `gen`.

**Lỗi/thiếu sót:** model API/weights có thể không còn tải được, license hạn chế,
checkpoint không tương thích. Không tải một thứ chỉ để “đủ generator”; đánh dấu
blocked và giải thích.

**Cổng:** M0–M7 gốc được đóng/audit trước X1 mở rộng.

### Bước 12 — Report, audit độc lập và dọn staging có điều kiện

- Sinh report/bảng/hình tự động từ manifests, configs và score-level artifacts.
- So sánh với paper cạnh nhau; câu về mọi số paper ghi nguyên văn:
  *"số của paper (ICCV 2023), không phải của demo này"*.
- Rà lại nhãn tối đa R1+R2 cho ảnh giả tác giả; `ours_adm` in-distribution;
  không ghi “chính xác 99 %”; không hiện `p_fake` khi `gate != passed`.
- Auditor đối chiếu từng sample count, source hash, checkpoint hash, train
  history, threshold source và matrix completeness. Xuất PDF LaTeX có trích dẫn
  và mục **Hạn chế**.
- Chỉ sau khi Volume `checkdata`, uploader `--verify`, manifest và đầu ra đã
  đối chiếu xanh mới cân nhắc xoá `.staging`; giữ `F:\DiffusionForensics` gốc.

**Lỗi có thể xảy ra:** report gộp khác provenance, scores của checkpoint cũ,
thiếu cell nhưng bảng trông đầy đủ, license không cho công bố dữ liệu. Khi ấy
báo `incomplete`/`blocked`, không che bằng tổng trung bình.

**Kết quả cuối:** bộ artefact có thể tái lập và bảng sai khác trung thực, hoặc
một danh sách rõ ô nào chưa thể tái hiện và vì sao.

## 4. Những hạng mục code phải làm trước full run

Đây là các thiếu sót thực tế quan sát được trong implementation hiện tại; chúng
không phải lý do để xoá dữ liệu hay bỏ các kết quả thăm dò:

1. **Fingerprint map trước reuse:** không chỉ `exists && size > 0`; ràng map với
   hash ảnh, hash checkpoint, sampler, `S`, precision và version code.
2. **Shard ledger bền:** `::dire` có `--n` tiền tố và commit mỗi 200 ảnh, nhưng
   chưa có offset/shard ID chính thức. Tăng dần `--n` là workaround, không phải
   giao thức shard hoàn thiện. Job ledger phải bảo đảm retry không đếm đôi.
3. **Train resume:** checkpoint model + optimizer/RNG/scheduler state định kỳ,
   atomic write/Volume commit, resume đúng epoch/step sau timeout.
4. **Protocol exact:** expose `init=ImageNet` (nếu đúng theo paper), đầy đủ
   hyperparameters/early-stop/augmentation đã audit; giữ default scratch riêng
   cho clean-room pilot.
5. **Matrix evaluator:** mọi train row × test column; threshold chỉ từ validation
   hoặc threshold paper, không test tuning.
6. **Score precision + group-aware CI:** giữ full score và sample/prompt group;
   visualization được làm tròn tách riêng.
7. **Ablation runners:** representations, S, signed/absolute, blur/JPEG theo
   paper, mỗi config có cache key riêng.
8. **Volume capacity/inventory:** trước khi chạy map, xác nhận đủ chỗ cho khoảng
   80 GiB map float16 theo số đếm hiện tại cộng dữ liệu/checkpoint/dự phòng. Nếu
   thiếu, tăng Volume hoặc lưu shard/archive ở storage hỗ trợ truy xuất; không
   hạ số mẫu để vừa dung lượng rồi gọi là full.

## 5. Tóm tắt thứ tự và điểm dừng

```text
0. Giữ nguyên, snapshot mọi thứ đang có
1. Đối chiếu protocol paper ↔ inventory 25 archive
2. Audit map/checkpoint/run cũ; phân loại reuse/pilot/recompute
3. Ingest toàn bộ cây local ra PNG chuẩn + manifest (resume)
4. Upload theo batch → checkdata SHA trên Volume
5. ADM checkpoint → selfcheck → bench thật
6. Fingerprint + tính map thiếu theo đợt, 2k→4k→…→40k khi cần
7. Audit map/count/shape/source
8. Train riêng các hàng có train split, đúng config paper + resume
9. Matrix evaluator train-row × test-column, giữ score chính xác
10. Ablation/robustness theo protocol đã khóa
11. Generator mở rộng tùy chọn, không lẫn bảng chính
12. Audit độc lập → report/PDF → chỉ sau đó mới dọn staging
```

Một lỗi ở bước nào thì dừng tại bước đó, lưu artefact/log và xử lý đúng shard.
Không bắt đầu lại từ 0, không xoá dữ liệu đã có, và không dùng một kết quả thiếu
metadata làm bằng chứng cho cell đã hoàn tất.
