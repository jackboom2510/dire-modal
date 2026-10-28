# Runbook: từ 0 ₫ → có model + demo công khai

> Kèm theo `docs/iaas-options.md` (chọn nền tảng) và `docs/storage-comparison.md` (chọn chỗ lưu).
> Mọi bước đều ghi **chi phí dự kiến** và **cổng kiểm tra** (không được bỏ qua cổng).

## Bảng tổng — nên tốn bao nhiêu

| Bước | Nền tảng | Chi phí | Cổng kiểm tra (pass mới đi tiếp) |
|---|---|---|---|
| 1. Chuẩn bị | — | $0 | `modal secret list` thấy 4 secret |
| 2. Tải dữ liệu | Modal CPU | ~$1–2 credit | `download::verify` in "All OK." |
| 3. Manifest sha256 | Modal CPU | ~$0.5 | Có `_manifest_sha256.tsv`, số file đúng |
| 4. Pack shards | Modal CPU | ~$1–2 | `pack::stats` đếm đủ ảnh mỗi generator |
| 5. Mirror công khai | Modal CPU + egress | ~$0–1 | `verify_hf` → `mismatch=0, missing=0` |
| 6. Smoke test | **Kaggle** | **$0** | Notebook chạy 3 step train, loss giảm |
| 7. **KHÔNG train** — lấy `lsun_adm.pth` | Modal CPU | ~$0.1 | `sha256` ckpt khớp manifest |
| 8. Eval 16 generator | **Modal CPU** (ResNet50 inference) | ~$0.1 | Bảng ACC 16 dòng, **CỔNG S2 `ACC(adm) ≥ 0.99`** |
| 9. Gallery + demo | Modal A100 + HF Spaces | ~$0.5 + $0 | Space public trả kết quả thật |
| **Tổng** | | **≈ $3–6** (bỏ hẳn $8–20 train) | |

> Đường ngắn nhất, không cần chờ dữ liệu lớn: `download --tiers e2min --mode fetch`
> (3 file ≈ 400 MB) → `eval::main --tiers e2min` (< 2 phút CPU). Đó là toàn bộ
> "tái lập" theo nghĩa của dự án này: **nạp checkpoint tác giả rồi đo**, không train.

> **Đo trước khi cam kết:** `bash tools/measure_dire.sh` cho biết số giờ GPU thật của stage-2.
> Nếu ước lượng > 30 A100-giờ → **không làm stage-2 toàn bộ**; chỉ làm 2–5k ảnh.

---

## 1. Chuẩn bị (một lần, ~30 phút, $0)

```bash
git clone <repo> && cd dire-modal
cp .env.example .env        # rồi điền các token bên dưới
pip install -e .

# Đăng nhập Modal (mở trình duyệt)
modal token new

# Điền token vào .env:
#   BAIDU_BDUSS/BAIDU_STOKEN  — cookie Baidu Netdisk (xem hướng dẫn dưới)
#   R2_*                      — Cloudflare R2 → API Tokens
#   HF_TOKEN                  — https://huggingface.co/settings/tokens (write)
#   KAGGLE_USERNAME/KAGGLE_KEY— https://www.kaggle.com/settings → Create New Token

bash scripts/setup.sh
modal secret list                     # ✅ CỔNG 1: thấy baidu-cred, r2-cred, hf-cred, kaggle-cred
```

**Lấy BDUSS/STOKEN:** đăng nhập pan.baidu.com → DevTools → Application → Cookies → sao chép `BDUSS` và `STOKEN`.

### CỔNG 0 — repo import được bằng `modal` THẬT (0 credit, ~1 phút)

Chạy **trước** mọi lệnh `modal run`:

```bash
bash scripts/check.sh
```

Script tạo `.venv`, cài `modal` **thật** (không stub) từ `requirements-dev.txt`,
rồi import toàn bộ module và chạy cả bộ test.

**Vì sao cần bước riêng này:** bộ test dùng stub tự viết, mà stub không thể bắt
mọi thứ. Ví dụ thật (2026-10-21) — `compute_dire.py` từng có
`from __future__ import annotations` (PEP 563). Modal thật đọc `__annotations__`
**thô**, thấy `'str'` (chuỗi) thay vì `str`, rồi chết ngay lúc **import** module:

```
KeyError: 'str'
→ InvalidError("No class parameter encoder implemented for type `...`")
→ AttributeError: 'str' object has no attribute '__name__'
```

`AttributeError` cuối cùng che mất nguyên nhân, và vì lỗi ở import nên **mọi**
lệnh của module đều chết — kể cả `dire_batch --dry-run`, trước khi guard chi phí
kịp in ra con số nào.

Quy tắc rút ra:

- File nào có `@app.cls` + `modal.parameter` thì **không** được có
  `from __future__ import annotations`. Có test quét toàn `src/` để chặn.
- Stub phải mô phỏng **hành vi gây lỗi**, không được bỏ qua nó: stub
  `modal.parameter` giờ kiểm annotation và báo lỗi nếu thấy chuỗi.
- Nhưng stub vẫn là code của mình — nên vẫn cần một test dùng `modal` thật
  (`tests/test_real_modal_import.py`), chạy trong subprocess để có
  `sys.modules` sạch.

---

## 2. Tải dữ liệu về Volume (Modal CPU, ~$1–2)

Mặc định tải **22 file = 3.39 GB** (tier `e2` + `e4`), so với **61.40 GB / 104
file** của toàn cây đã audit. Detector DIRE chỉ đọc nhánh `dire/` (ImageFolder
`0_real`/`1_fake`), nên `recons/` và phần lớn `images/` là thừa và bị `NEVER`
chặn cứng.

Mọi con số GB trong mục này là **số đo thật** từ `download::audit` ngày
2026-10-21 (15 thư mục), không phải ước lượng.

```bash
# (a) Policy hiện hành — chạy LOCAL, không tốn credit
modal run -m dire_modal.download::local_status

# (b) Dò cây remote 1 lần để biết đường dẫn thật
modal run -m dire_modal.download::ls_remote --path /apps
# → chép kết quả vào src/dire_modal/manifest.py::BAIDU_ROOT

# (c) Đối chiếu cây remote ↔ manifest — KHÔNG tải gì
modal run -m dire_modal.download::audit

# (d) Kế hoạch + SIZE THẬT từ Baidu — KHÔNG tải gì
modal run -m dire_modal.download::plan --tiers e2,ckpts

# (e) Tải thật (mặc định của `main` là dry-run, phải nói rõ --mode fetch)
#     Gói (a)→(f): bash scripts/01_repro.sh --stage data
modal run -m dire_modal.download::main --tiers e2,ckpts --mode fetch
# đường ngắn nhất tới con số đầu tiên (3 file / ~400 MB / ~1.5 h):
modal run -m dire_modal.download::main --tiers e2min --mode fetch

# (f) Verify toàn bộ — archive (.tar.gz) VÀ checkpoint (.pth/.pt)
modal run -m dire_modal.download::verify    # ✅ CỔNG 2 (+ CỔNG 7): "All OK."
```

**Phải chạy (c) và (d) trước khi tải.** Hai bước đó gần như miễn phí, và đã
chứng minh giá trị: audit 2026-10-21 bắt được tier `e4` trỏ vào đường dẫn hai
tầng `images/test/lsun_bedroom/lsun_bedroom/` **không tồn tại** (cây thật chỉ có
một tầng), kèm hai tên file sai — `midjouney` thay vì `midjourney`, và `sdv1`
trong khi chỉ có `sdv1_new` / `sdv1_new2`. Nếu chạy `--mode fetch` trước thì cả
4 file đã fail và tốn một vòng chờ Baidu.

Con số "Tier A ≈ 40–70 GB" từng ghi trong docs là **sai 5.4–9.4 lần**: đó là số
của cả cây gán nhầm cho Tier A. Tier A cũ (20 file) thực tế **7.42 GB**; mặc
định mới `e2,e4` là **3.39 GB**. Toàn cây đã audit **61.40 GB**, trong đó
**47 GB (77%) là dữ liệu train** — bỏ được gần hết vì ràng buộc là *chỉ cải
thiện, không train lại*.

Chi phí thực tế chủ yếu là **thời gian chờ Baidu** (server chậm); CPU 0.25 core
≈ $0.012/giờ → rẻ. `fetch_one` là resumable và idempotent: chạy lại sẽ skip file
đã có và `gzip -t` pass.

Trần dung lượng (`max_gb`, mặc định 25) nằm trong `src/dire_modal/tiers.yaml`.
Vượt trần ⇒ `main` từ chối tải và in rõ tổng ước lượng. Ước lượng này **bảo thủ**:
file nào `ls -l` không đọc được size thì bị tính bằng size của file lớn nhất đã
biết, để guard không vô hiệu đúng lúc cần nó nhất.

### 2b. Tải ở MÁY LOCAL — không dùng Modal, không dùng Volume

Modal có ba trần mà **không đường nào ở trên tránh được**: mỗi attempt 1800 s,
mỗi lần gọi `fetch_one` 7200 s, mỗi app 20 h ngân sách. Ở 76.6 KiB/s — trần của
**tài khoản Baidu**, không phải của Modal — thì `e2` (1.94 GB) cần 7.4 h, nên
file 269 MB *luôn* bị cắt ít nhất một lần.

Máy local không có ba trần đó. Bật terminal, để chạy, Ctrl+C lúc nào cũng được.

```bash
# Kế hoạch (KHÔNG tải gì) — in size từng file + tổng, đối chiếu trần
bash tools/local_download.sh

# Tải thật
DEST=./data/raw bash tools/local_download.sh --mode fetch

# Đường ngắn nhất tới con số đầu tiên (3 file / ~0.40 GB / ~1.5 h)
TIERS=e2min DEST=./data/raw bash tools/local_download.sh --mode fetch

# Nhiều tier: e2 + checkpoint tác giả (22 file / 3.25 GB / ~12 h)
TIERS=e2,ckpts DEST=D:/dire-raw bash tools/local_download.sh --mode fetch

# Chỉ vài file cụ thể
bash tools/local_download.sh --paths e2/lsun_bedroom/lsun_bedroom/iddpm.tar.gz --mode fetch
```

Tương đương (không cần git checkout `tools/`): `python -m dire_modal.local_fetch`
với cùng tham số, khi `src/` đã có trong `PYTHONPATH`.

**Cần một lần:**

1. `BaiduPCS-Go` trong `PATH` — https://github.com/qjfoidnh/BaiduPCS-Go/releases
   ⚠ user free **phải** đặt `max_parallel=1`, `max_download_load=1` trong config.
   Tăng thread ⇒ Baidu throttle 0 B/s hàng giờ (đo được: k=1 → 76.6 KiB/s,
   k=2 → 75.0 KiB/s, tức thêm phiên không nhanh hơn).
2. `BaiduPCS-Go login -bduss=<BDUSS>` (BDUSS lấy từ cookie trình duyệt).

**Khác biệt so với đường Modal — có chủ đích:**

| | Modal | Local |
|---|---|---|
| thời gian mỗi attempt | 1800 s (trần nền tảng) | `--stall-s` (mặc định 900 s **không ghi thêm byte nào** thì bỏ) |
| số attempt | 3 | `--attempts` (mặc định 5) |
| ngân sách | 20 h/app | `--max-hours` (mặc định 0 = không giới hạn) |
| song song | 1 container | tuần tự, đúng 1 phiên Baidu |
| Ctrl+C | SIGTERM → commit Volume | `finally` giữ nguyên staging |
| kiểm tra file | `fetchcore` | **cùng** `fetchcore` — không có bản sao thứ hai |

Phép kiểm "file đã đúng chưa" dùng chung module `src/dire_modal/fetchcore.py`
(`_validate_ex`, `SizeEst` + `slack`, `gzip -t` trước size, `_hold` chuyển chứ
không xoá). Đây là bài học của sự cố 2026-10-25: hai bên trả lời khác nhau cho
cùng một câu hỏi là đúng thứ sinh ra lỗi.

**Resume**: mảnh dở nằm ở `<DEST>/.staging/baidu_<hash ổn định theo đường dẫn
remote>/`, cùng nơi BaiduPCS-Go để `.BaiduPCS-Go-downloading`. Chạy lại đúng
lệnh là tải tiếp từ byte đã có; file đã xong in `[SKIP]`.

**Kiểm lại không cần Baidu**: `_manifest.tsv` ở `<DEST>/` ghi mỗi file đã tải
(timestamp, remote, local, size, sha256, status) và `.state/validated.json` giữ
cache `(size, mtime_ns, sha256)`.

### 2c. Đã có dữ liệu ở máy: kiểm chứng, rồi đưa phần cần thiết lên Modal

```bash
# Đã tải đúng chưa? (KHÔNG tải byte nào; thiếu/sai thì in rõ từng file)
bash tools/local_download.sh --mode verify
bash tools/local_download.sh --mode verify --no-remote    # không cần BaiduPCS-Go
TIERS=e2 bash tools/local_download.sh --mode verify       # chỉ kiểm tier e2

# sha256 toàn bộ — CÙNG schema với archive::make_manifest (/raw/_manifest_sha256.tsv)
bash tools/local_download.sh --mode manifest
# → <DEST>/_manifest_sha256.tsv : source, path, size_bytes, sha256
```

`verify` trả `0` chỉ khi **đủ và đúng**; hỏng/tải dở thì in `gzip -t` nói gì và
gợi ý chạy lại `--mode fetch` (file đã đúng sẽ `[SKIP]`, **không xoá gì**). File
tải bằng GUI vào cây hơi khác vẫn được nhận nếu tên khớp **duy nhất** — hai ứng
viên cùng tên thì báo thiếu chứ không đoán, vì đoán sai là báo "đủ" cho file sai.

**Đưa lên Modal.** Eval chỉ đọc nhánh `dire/test/` (tier `e2`) + checkpoint tác
giả, nên chỉ cần đẩy phần đó. `upload.py` lo việc đó — nó **hỏi Volume trước**
để không đẩy lại file đã có, và đẩy **từng file** để resume được:

```bash
# (1) Kế hoạch: đẩy gì, bỏ qua gì, `.staging` trên Volume đang chiếm bao nhiêu
modal run -m dire_modal.upload::main --tiers e2min

# (2) Đẩy thật (chỉ file thiếu / sai size), rồi tự kiểm lại size từng file
modal run -m dire_modal.upload::main --tiers e2min --mode push

# (3) Dọn `.staging` cũ trên Volume — CHỈ sau khi (2) báo "Volume khớp local"
modal run -m dire_modal.upload::main --tiers e2min --mode clean --yes

# Hoặc gộp (2)+(3): chỉ dọn khi kiểm lại thấy khớp từng byte
modal run -m dire_modal.upload::main --tiers e2 --mode push --clean-after
```

Ba chi tiết mà `upload.py` xử lý thay bạn (đều thành test):

| Bẫy | Hậu quả | Cách xử lý |
|---|---|---|
| `modal volume put` nhận path **tính từ gốc Volume** (`dire/test/…`), không phải `/raw/...` | dữ liệu nằm ở `/raw/raw/...` ⇒ eval báo "không tìm thấy cặp real/generator nào" | `remote_path()` bỏ tiền tố `/raw` |
| `modal volume put <dir> <dest>` đặt thư mục **vào trong** đích | `/raw/dire/dire/...` | đẩy **từng file** tới đúng đường dẫn đích |
| xoá `.staging` khi dữ liệu chưa lên Volume | mất bản duy nhất còn lại trên cloud | `--mode clean` **từ chối** nếu còn file của tier chưa có trên Volume; `--clean-after` chỉ chạy sau khi kiểm lại khớp |

`--mode clean` **mặc định là dry-run** (in dung lượng từng vùng); chỉ xoá khi có
`--yes`. Nó chỉ nhận đúng ba tên vùng nội bộ (`.staging`/`.bench`/`.state`) —
gõ `--areas dire` sẽ bị từ chối, để một lần gõ sai không xoá dữ liệu thật.
Muốn giữ lại phần đã tải xong của đường Modal cũ thì dọn `.bench`, `.state`
trước và để `.staging` lại: `--mode clean --areas .bench,.state --yes`.

Sau đó xác nhận trên cloud — CỔNG 2 **và** CỔNG 7 (verify kiểm cả `.pth`):

```bash
modal run -m dire_modal.download::verify
bash tools/tree.sh                      # cây thật trên Volume, nếu nghi lệch tầng
```

`verify` kiểm `/raw` **và** `/ckpt`. Nếu mới upload và chưa sinh manifest, chạy
`modal run -m dire_modal.archive::make_manifest` trước để có sha256 mà đối chiếu
— không có nó thì checkpoint chỉ được kiểm "tồn tại + size > 0", và một file cụt
sẽ chỉ nổ ở `torch.load` giữa lúc eval đã tiêu tiền GPU.

Muốn kiểm sha256 khớp giữa máy và Modal thay vì tin `modal volume put`:

```bash
modal run -m dire_modal.archive::make_manifest          # /raw/_manifest_sha256.tsv
# rồi so với `local_fetch --mode manifest` (cùng cột, cùng thứ tự) bằng diff
```

Lưu ý dung lượng: Volume tính tiền theo **peak** đã dùng, nên đẩy 3.39 GB rồi
xoá không làm hoá đơn nhỏ đi trong tháng đó. Chỉ đẩy `e2` + checkpoint là đủ cho
Table 2 headline; `e4`/`e5` thêm khi thật cần ô đó.

### Tốc độ tải và resume xuyên container

Ba thay đổi, mỗi cái nhắm một chi phí khác nhau:

**1. Staging nằm trong Volume (`/raw/.staging`), không phải `/tmp`.**

`/tmp` là ephemeral: container chết giữa chừng là mất sạch tiến độ, vì
BaiduPCS-Go resume dựa trên file dở còn nằm trong thư mục đích. Ở 67 KB/s, file
692 MB cần ~2.9 giờ — mất tiến độ ở phút 170 là mất gần 3 giờ.

Giờ staging ở trong Volume và được **commit định kỳ** (`staging_commit_s`, mặc
định 45 s), nên lần chạy sau — kể cả ở container khác — resume được từ đúng chỗ
đã tải. Commit cũng chạy ngay sau mỗi attempt thất bại.

Đổi lại mỗi `write()` của BaiduPCS-Go đi qua FUSE. Ở băng thông Baidu thì đây là
nhiễu: mạng chậm hơn đĩa khoảng ba bậc độ lớn. Muốn quay lại `/tmp` (nhanh hơn
một chút, mất resume xuyên container): `DIRE_STAGING_DIR=/tmp/baidu_staging`.

**2. Bước công bố là `os.replace`, không phải copy.**

Staging cùng filesystem với `/raw` ⇒ rename nguyên tử, O(1). Bản trước tải vào
`/tmp` rồi `shutil.move()` sang `/raw`, tức là **copy toàn bộ file** — 692 MB
cho file lớn nhất. Nhánh dự phòng `EXDEV` vẫn còn cho trường hợp staging bị trỏ
ra ngoài Volume.

**3. Kiểm tra chạy trong staging, chỉ công bố khi hợp lệ.**

Nếu move sang `/raw` rồi mới kiểm thì bước kiểm hỏng phải xoá file khỏi `/raw`,
và **mảnh dở biến mất** — đúng thứ dùng để resume. Giữ nó trong staging thì
attempt sau (hoặc container sau) resume được.

**4. Cache kiểm tra (`.state/validated.json`).**

`gzip -t` phải đọc toàn bộ file. Với Tier A mặc định, mỗi lần chạy lại đều đọc
**3.39 GB** chỉ để trả lời "file này vẫn tốt chứ", rồi `sha256` đọc thêm 3.39 GB
nữa. Cache theo `(size, mtime_ns)` bỏ được cả hai: file bị ghi đè thì `mtime`
đổi nên cache tự vô hiệu. **Chỉ cache kết quả TỐT** — cache âm mà sai thì một
file lành sẽ bị tải lại mãi mãi.

**5. `--concurrency` KHÔNG giúp gì — đã đo, đừng thử lại.**

```bash
modal run -m dire_modal.download::bench_concurrency --tiers e2 --limit 2 --ks 1,2
```

| k (container) | thời gian | tổng | mỗi file |
|---|---|---|---|
| 1 | 5183.8 s | **76.6 KiB/s** | 37.5 KiB/s |
| 2 | 5293.3 s | **75.0 KiB/s** | 37.5 KiB/s |

387.7 MB/lượt, đo 2026-10-25. `k=2` **không nhanh hơn** (chậm 2.1 %, trong sai
số), và mỗi container chỉ được một nửa số file ⇒ file nào cũng chậm gấp đôi.
Trần ~77 KiB/s là của **tài khoản**, không phải của cấu hình. Vì thế
`max_containers: 1`.

`baidu_parallel: 1` là số kết nối song song **cho một file** (`-p N`), truyền
thành `-p 1`. Upstream BaiduPCS-Go nói rõ: tài khoản miễn phí phải để
`max_parallel=1`; tăng lên chỉ nhanh trong vài phút rồi bị throttle, có thể về
gần 0 tốc độ trên **mọi** client trong nhiều giờ tới nhiều ngày. `--mode` của
client (`locate|pcs|stream`) đặt được qua `baidu_mode` trong `tiers.yaml`; đo
bằng `bash tools/bench_baidu_mode.sh` (chưa đo).

**6. Một lần gọi tự chạy nhiều vòng (ngân sách 20 h).**

Ở 76.6 KiB/s, `e2,e4` cần 12.9 h — dài hơn `FETCH_CALL_TIMEOUT_S` (7200 s), nên
trước đây phải gõ lại lệnh ~9 lần. Giờ `_execute` có vòng lặp: sau mỗi wave nó
tính lại file còn thiếu **từ đĩa** rồi chạy tiếp, tối đa `_MAX_WAVES = 40` vòng
và `ORCHESTRATOR_BUDGET_S = 20 h` tổng. Nó dừng khi: hết file · hết ngân sách ·
hoặc **một vòng không xong thêm file nào** (chốt chống đốt 20 h khi BDUSS hết
hạn — đó là triệu chứng thường gặp, không phải file hỏng).

**7. `--paths`: tải đúng vài file lẻ, không cần tier.**

```bash
modal run -m dire_modal.download::main \
  --paths dire/test/lsun_bedroom/lsun_bedroom/real.tar.gz,checkpoints/lsun_adm.pth \
  --mode fetch
```

Vẫn đi qua `NEVER` và guard dung lượng. Dùng khi cần đúng một file để chẩn đoán
hoặc để đủ điều kiện chạy eval; đừng tạo tier mới cho danh sách dùng một lần.

**8. Nhanh hơn hẳn thì phải đổi NGUỒN, không phải đổi tham số.**

```bash
modal run -m dire_modal.archive::push_hf --repo-id <user>/<repo> --src /raw
modal run -m dire_modal.download::from_hf --repo-id <user>/<repo> --tiers e2
```

HF Hub thường cho vài MB/s, tức nhanh hơn Baidu 1–2 bậc. Đây là lối thoát duy
nhất khỏi 77 KiB/s: mọi mẹo trong Baidu đều đã đo và đều không tăng tổng băng
thông.

### Khi tải xong mà vẫn `gzip -t` fail

Đã xảy ra thật (2026-10-21): `real.tar.gz` tải **đủ đúng 27 863 040 byte** — khớp
100% con số Baidu báo — mà `gzip -t` vẫn FAIL. Vì size khớp, đây **không phải**
lỗi mạng; byte đến đủ nhưng *nội dung* không phải gzip. Trước đây thông báo chỉ
nói "file cụt/hỏng" rồi thử lại 3 lần × ~6 phút = **19 phút vô ích**.

Giờ `_fetch_one_impl` đọc magic bytes và chia lỗi làm hai loại:

| Thông báo | Nghĩa | Nên làm |
|---|---|---|
| `size 1234 ≠ remote 5678 → tải dở dang` | lỗi mạng/throttle | chạy lại, `fetch_one` resume |
| `ok — **tar KHÔNG nén**` | dữ liệu **hợp lệ**, chỉ sai tên đuôi | không làm gì: file được giữ, vẫn dùng được |
| `KHÔNG phải gzip mà là **rar**` | nguồn sai định dạng/tên | **đừng tải lại** — xem lại tên file gốc |
| `nội dung là HTML/JSON …` | Baidu trả trang lỗi | **đừng tải lại** — kiểm tra BDUSS/link |
| `hai lần tải cho ra byte GIỐNG HỆT` | byte ổn định ⇒ tất định | **đừng tải lại** |
| `gzip -t FAIL — không rõ …` | chưa phân loại được | thử lại 1 lần, rồi đọc chẩn đoán |

Lỗi tất định in `[STOP]` và **dừng ngay sau lần thử đầu**, không đốt thêm attempt.
`download::verify` cũng in chẩn đoán định dạng cho từng file hỏng, và tách danh
sách "tải lại được" khỏi "tải lại vô ích" trước khi `SystemExit(1)`.

`probe_pair` thì **không** đòi file phải là gzip (`expect_archive=False`), vì mục
đích của nó là so **sha256** — phép so đó đúng trên mọi chuỗi byte. Gặp archive
lạ, nó vẫn in sha256 + định dạng thật và ghi `ảnh: KHÔNG đọc được — …` thay vì
chết giữa đường.

### Dữ liệu thật là **tar không nén**, không phải `.tar.gz`

Chạy `probe_pair` trên dữ liệu thật (2026-10-21) cho `định dạng: tar (không nén)`
ở **cả hai** file: tên đuôi `.tar.gz` nói dối, dữ liệu vẫn nguyên vẹn. Hệ quả:

- Code nào hardcode `tarfile.open(path, "r:gz")` sẽ chết với mọi file thật. Đã
  sửa ở `_tar_profile` và `pack.py` → `"r:*"` (tự nhận nén **và** không nén).
  Riêng `pack.py` là chỗ nguy hiểm nhất: không có guard, member đầu tiên ném
  `ReadError: not a gzip file` là cả stage pack chết.
- `_validate_ex` **chấp nhận** tar không nén (`_tar_ok` kiểm tra EOF marker 1024
  byte cuối = 2 block 0). Bản trước xếp nó vào lỗi tất định rồi **`unlink` file
  vừa tải** — tức là xoá dữ liệu hợp lệ. Chỉ tar **cụt** mới bị coi là lỗi, và
  khi đó `retryable=True` (tải lại có ý nghĩa).

### `sha256` của container KHÔNG nói được tập ảnh có khác nhau hay không

Tar chứa metadata: `mtime`, owner, thứ tự member, padding. Cùng một tập ảnh,
đóng gói lại với `mtime` khác ⇒ `sha256` **khác** mà `size` giữ nguyên tuyệt đối.
Vì vậy `probe_pair` giờ so **từng member** (`_tar_content` + `_content_compare`):

| Kết luận in ra | Nghĩa |
|---|---|
| `CÙNG HOÀN TOÀN` | trùng cả tên lẫn byte từng ảnh ⇒ cùng một tập, chỉ khác đóng gói |
| `CÙNG ảnh, KHÁC tên/thứ tự` | cùng byte ảnh, khác đường dẫn bên trong |
| `KHÁC HOÀN TOÀN` | 0 ảnh trùng byte ⇒ thật sự hai tập khác nhau |
| `LỆCH MỘT PHẦN` | giao nhau nhưng không trùng khít ⇒ báo cáo riêng từng nhánh |
| `KHÔNG so được` | archive không đọc được ⇒ **không kết luận gì** |

Bản trước so `sha256` container rồi so `files == files` với giá trị `-1` khi
không đếm được. `-1 == -1` là phép so rỗng, và nó đã in ra "Số ảnh bằng nhau
(-1) ⇒ test domain HỢP LÝ" — kết luận không có cơ sở.

### Chọn tier theo mục tiêu

| Mục tiêu | Lệnh | File | Size |
|---|---|---|---|
| Chỉ code/docs | — | 0 | — |
| Demo AEROBLADE zero-shot | — (VAE kéo từ HF Hub trong Space) | 0 | — |
| **Eval headline DIRE trên LSUN** | `--tiers e2` | 17 | 1.94 GB |
| **Eval AEROBLADE trên ảnh gốc** | `--tiers e4` | 5 | 1.45 GB |
| Cả hai (**mặc định**) | `--tiers e2,e4` | **22** | **3.39 GB** |
| + cross-domain ImageNet/CelebA-HQ | `--tiers e2,e3,e4` | 46 | 8.24 GB |
| + ensemble & hiệu chỉnh ngưỡng | `--tiers e2,e3,e4,e5` | 57 | 10.35 GB |
| + AEROBLADE mọi generator | `--tiers e2,e3,e4,e4x,e5` | 69 | 12.37 GB |
| Train lại detector từ đầu | `--tiers train` | 4 | 5.74 GB |
| Train cả ensemble (3 detector nữa) | `--tiers train,trainens` | 7 | 20.34 GB |

`e4 ⊂ e4x` và `expand()` khử trùng lặp theo đường dẫn remote, nên
`--tiers e4,e4x` cho 17 file / 3.47 GB chứ không nhân đôi thành 22.

Kiểm tra đã tải thừa gì: `modal run -m dire_modal.download::prune --tiers e2,e4`
(chỉ in; thêm `--mode apply` mới xoá).

### Trước khi ghi số vào bảng eval

`dire/test/<A>/<B>/` có hai tầng, nhưng `train/` và `val/` thì phẳng. Dấu hiệu
đáng ngờ ban đầu: `real.tar.gz` ở `.../lsun_bedroom/lsun_bedroom/` và
`.../lsun_bedroom/imagenet/` có size **bằng nhau y hệt** (26.6 MB).

```bash
modal run -m dire_modal.download::probe_pair    # ~53 MB, so TỪNG ẢNH trong tar
```

### Kết quả (đã chạy `probe_pair` trên dữ liệu thật, 2026-10-21)

```
  sha256 container : KHÁC
  nội dung ảnh     : CÙNG HOÀN TOÀN (1000/1000 ảnh trùng tên + trùng byte)
```

`real.tar.gz` của hai nhánh là **cùng 1000 ảnh**; sha256 container khác chỉ vì
tar chứa `mtime`/owner khác. Nghĩa là:

- Tầng 1 = nguồn ảnh **real** (và do đó là domain của detector).
- Tầng 2 = nguồn của nhánh ảnh **giả**.
- ⇒ `dire/test/lsun_bedroom/imagenet/` là **cross-generator**, KHÔNG phải
  cross-domain.

**Hệ quả khi báo cáo số:** KHÔNG cộng nhánh này vào bảng eval như một "test
domain" riêng — làm vậy là đếm cùng 1000 ảnh real hai lần và kéo mean AP về
phía LSUN. Báo cáo riêng với nhãn "cross-generator".


---

## 3. Manifest sha256 — nguồn sự thật cho mọi mirror (~$0.5)

```bash
modal run -m dire_modal.archive::make_manifest
# → /raw/_manifest_sha256.tsv  (source, path, size_bytes, sha256)

modal volume get dire-raw /_manifest_sha256.tsv manifests/   # ✅ CỔNG 3
```

Giữ file này trong git. Mọi bản sao (HF, Kaggle, R2) sẽ được đối chiếu với nó.

---

## 4. Pack WebDataset shards (~$1–2)

```bash
bash tools/pack_shards.sh
# ✅ CỔNG 4: pack::stats in số ảnh từng generator, tổng khớp với manifest
```

Thiết kế đã tính tới giới hạn Modal Volume v1 (≤50k file khuyến nghị): shard 2000 ảnh → 100k ảnh chỉ tốn ~50 inode.

---

## 5. Mirror công khai (HF + Kaggle) — phần "chia sẻ cho người khác"

### 5a. Hugging Face (mirror chính, ~300 GB/repo)

```bash
# Tạo repo dataset (public)
huggingface-cli repo create diffusionforensics-dire --type dataset --public

modal run -m dire_modal.archive::push_hf --repo-id <user>/diffusionforensics-dire

# Đối chiếu sha256 Volume ↔ Hub
modal run -m dire_modal.archive::verify_hf --repo-id <user>/diffusionforensics-dire
# ✅ CỔNG 5a: mismatch=0, missing=0
```

### 5b. Kaggle (mirror phụ, 200 GB/dataset, **≤50 file top-level**)

Đẩy theo từng thư mục nhỏ (mỗi thư mục có ít file `.tar.gz`):

```bash
modal run -m dire_modal.archive::push_kaggle \
  --dataset-id <user>/dire-tier-a-train-lsun \
  --src-subdir dire/train/lsun_bedroom
```

### 5c. R2 (chỉ checkpoint + manifest — 10 GB free)

```bash
modal run -m dire_modal.archive::push_r2 --prefix diffusion-forensics/v1
# mặc định KHÔNG đẩy .tar.gz (vì R2 free chỉ 10 GB); muốn đẩy thêm: --include-tar
```

### ⚠ License — đọc trước khi mirror

Dataset DiffusionForensics là **dataset nghiên cứu**; repo gốc **đã archive 2025-07-08** nên khó xin xác nhận.
Khi mirror toàn bộ, bạn tự chịu trách nhiệm. Bắt buộc làm:

1. Ghi rõ nguồn + bài báo + link Baidu/Rec gốc trong mô tả dataset.
2. Ghi điều kiện: **nghiên cứu / phi thương mại**, không bán, không dùng cho sản phẩm.
3. Nêu rõ dataset mirror **để chống link rot**, và bạn sẽ gỡ nếu tác giả yêu cầu.
4. Không mirror bất cứ thứ gì bạn không có quyền phân phối (ảnh gốc LSUN/ImageNet/CelebA-HQ thuộc bên thứ ba).

---

## 6. Smoke test trên Kaggle ($0, không tốn credit)

1. Upload `platforms/kaggle/notebook.ipynb` lên Kaggle → *Settings → Accelerator → GPU P100*.
2. Nếu muốn notebook đọc dữ liệu trực tiếp: tạo Kaggle Dataset từ mirror ở bước 5b rồi *Add Data* vào notebook.
3. Chạy hết notebook.
   ✅ **CỔNG 6**: cell cuối in `[ok] đường đi dữ liệu hoạt động — 3 bước trên cuda`

Kiểm tra quota trước khi chạy:

```bash
python3 tools/free_quota.py --used kaggle=0
```

---

## 7. KHÔNG train — dùng checkpoint tác giả

Ràng buộc dự án (spec.md N1): **chỉ cải thiện, không train lại**. Bước train của
runbook cũ (`dire_modal.train::main`, ~$8–20, `train.py` vẫn là TODO) **không
còn nằm trên đường chạy**. Nếu vì lý do mới mà cần train lại thì phải viết ADR
riêng; xem `src/dire_modal/vendor/` (`train.py`, `utils/trainer.py` của DIRE gốc).

Thay vào đó chỉ cần đưa `lsun_adm.pth` vào Volume — tier `e2min` đã gồm nó:

```bash
modal run -m dire_modal.download::main --tiers e2min --mode fetch
# → /raw/checkpoints/lsun_adm.pth  (269.4 MB)
```

✅ **CỔNG 7** (đổi nghĩa): `/raw/checkpoints/lsun_adm.pth` tồn tại và
`sha256` khớp manifest — **giờ do `download::verify` kiểm** (trước đây nó chỉ
quét `*.tar.gz` nên 6 checkpoint 269.4 MB nằm ngoài mọi phép kiểm):

```bash
modal run -m dire_modal.download::verify
# [OK  ]     269.4MB  raw/checkpoints/lsun_adm.pth
#         ↳ ok — sha256 khớp manifest (269.4MB)
# checkpoint: 6 file .pth/.pt · 6 có tham chiếu manifest · hỏng 0
# [CỔNG 7] 6/6 checkpoint OK (đối chiếu sha256: 6)
```

`.pth` không kiểm được bằng `gzip -t`, nên bằng chứng là **size + sha256 đối
chiếu `_manifest_sha256.tsv`** (CỔNG 3). Chưa có manifest ⇒ `verify` vẫn chạy
nhưng **nói rõ** "CHƯA đối chiếu sha256" chứ không gọi đó là "khớp". sha256 được
cache theo `(size, mtime_ns)` trong `/raw/.state` nên không băm lại 1.6 GB mỗi
lần chạy.

---

## 8. Eval 16 generator (~$0 compute, chỉ CPU)

```bash
bash scripts/01_repro.sh --quick          # e2min: 3 file → eval → in CỔNG S2
# hoặc đầy đủ 32 ô của Table 2 (4 detector × 8 generator):
bash scripts/01_repro.sh                  # e2,ckpts → eval → report Δ so với paper
# ✅ CỔNG 8: bảng ACC cho 16 generator; so với paper (~99% trên ADM-LSUN)
```

`eval::main` tự tìm checkpoint (`/raw/checkpoints/lsun_adm.pth` rồi
`/ckpt/pretrained/lsun_adm.pth`) và in **CỔNG S2**: `ACC(adm) ≥ 0.99`. S2 KHÔNG
ĐẠT thì đừng đọc bảng AP — kiểm đúng file ckpt, `n_real`/`n_fake`, và tiền xử lý
(trường `preproc` trong `run.json`).

ResNet50 inference trên CPU: **~17k ảnh** ở 224² (1k real dùng chung + 16 generator
× 1k) → dưới 10 phút, không cần GPU. Chạy song song bằng `.map()` — mỗi cặp 1
container, scale-to-zero khi xong.

### Tier nào phủ bảng nào của paper

| tiers | file / GB | ~giờ @76.6 KiB/s | phủ |
|---|---|---|---|
| `e2min` | 3 / 0.40 | 1.5 | 1 ô — cổng S2 |
| `e2` | 17 / 1.94 | 7.4 | hàng ADM (8 cột của Table 2) |
| **`e2,ckpts`** | **22 / 3.25** | **12.4** | **32 ô của Table 2** (4 detector × 8 generator) |
| `+ 3 file qua --paths` | +0.14 | +0.5 | Table 3 (cross-dataset) |
| Table 4/5/6 + baseline `CNNDet*`/`Patchfor*`/`F3Net*` | — | — | cần **train lại** ⇒ ngoài phạm vi |

Số tham chiếu: `src/dire_modal/paper_table2.tsv` (arXiv:2303.09295v1, Table 2,
trang 7) — `eval::report` in Δ từng ô và đếm số ô khớp.

⚠ `checkpoints/lsun_adm.pth` là **detector ResNet-50**, KHÔNG phải weight
diffusion ADM (`lsun_bedroom.pt`, 2.5 GB, trong `NEVER`).

---

## 9. Gallery + demo công khai (~$0.5 + $0)

```bash
# 9a. Sinh gallery trên GPU (cần checkpoint ADM)
modal run -m dire_modal.compute_dire::make_gallery --n 12 --steps 20
modal volume get dire-shards /_gallery ./platforms/hf-spaces/gallery

# 9b. Đẩy checkpoint lên HF để Space tải được
#     (dùng checkpoint TÁC GIẢ, không phải bản train lại)
modal volume get dire-raw /checkpoints/lsun_adm.pth ./
huggingface-cli upload <user>/dire-detector ./lsun_adm.pth dire_resnet50.pth

# 9c. Tạo Space (xem platforms/hf-spaces/README.md)
```

✅ **CỔNG 9**: Space public mở được, upload 1 ảnh → ra P(real)/P(fake); tab viz hiện ảnh với chú thích.

---

## 9b. Guard chi phí GPU — đọc trước khi chạy bất cứ thứ gì trên A100

`compute_dire.py` là chỗ duy nhất trong pipeline **tiêu tiền thật** theo giờ.
Bản trước để cứng `max_containers=40` + `timeout=24*3600`:

```
40 container × 24 h × ~$2.50/h ≈ $2 400   ← trần xấu nhất của MỘT lần chạy hỏng
```

Điểm chết người là **timeout**, không phải số container: một ảnh 256² × 20 step
mất vài giây, nên timeout 24 h chỉ có tác dụng khi container đã treo — đúng lúc
nó tốn tiền nhất.

Giờ mọi thứ lấy từ `src/dire_modal/tiers.yaml`, và **clamp cứng trong code** để
một dòng config sai không thể bật lại 40 container:

| Khoá | Mặc định | Clamp cứng trong code |
|---|---|---|
| `dire_max_containers` | 4 | `[1, 16]` |
| `dire_timeout_s` | 1800 (30 phút) | `[60, 3600]` |
| `dire_max_usd` | 5.0 | — (so với ước tính) |
| `dire_max_images` | 500 | — |
| `dire_sec_per_image` | 5.0 (bảo thủ) | — |

Trần xấu nhất với mặc định: `4 × 0.5 h × $2.50 = $5`.

```bash
# Xem tiền TRƯỚC, không bật container nào:
modal run -m dire_modal.compute_dire::dire_batch --n 200 --dry-run

# Chạy thật — preflight chặn trước khi .map() được gọi:
modal run -m dire_modal.compute_dire::dire_batch --n 200
```

### Preflight từ chối ba trường hợp

`preflight_cost()` là việc **đầu tiên** trong entrypoint, trước cả bước liệt kê
file. Vượt trần ⇒ `SystemExit`, chưa GPU nào được cấp phát:

1. `n` ảnh > `dire_max_images` — chặn trước cả khi tính tiền.
2. Chi phí kỳ vọng > `dire_max_usd`.
3. **Trần xấu nhất** > `dire_max_usd` (mọi container treo tới timeout). Điều
   kiện này chặn cả trường hợp `n` rất nhỏ nhưng timeout/containers quá lớn —
   đúng loại cấu hình đã tạo ra hoá đơn $2 400.

`--force` bỏ qua cả ba nhưng vẫn in đủ số tiền và nhắc theo dõi bằng
`modal app list` / `modal billing report --resolution h`.

Ngoài code, vẫn nên đặt **spend limit $30** ở `modal.com/settings/usage` — guard
trong code không cứu được nếu ai đó chạy `modal run` từ thư mục khác.

---

## 10. Vòng lặp hàng tháng

```bash
python3 tools/free_quota.py --used modal=12 kaggle=20 lightning=0     # còn gì?
python3 tools/storage_cost.py --from-manifest manifests/raw.tsv       # lưu trữ tốn gì?
```

---

## Khi hết credit: phao cứu sinh

| Tình huống | Làm gì |
|---|---|
| Hết $30 giữa lúc train | Resume trên **Lightning free** (A100, restart mỗi 4 h — checkpoint mỗi epoch đã lo được) |
| Cần chạy nhiều giờ liên tục | **Thunder $1.09/h** → Vast ~$0.8–1.5/h → RunPod $1.79/h (A100 80GB), kèm mount R2/HF để lấy dữ liệu |
| Không có thẻ quốc tế | VNSO A100 40GB 29.000₫/h (~$1.12) |
| Cần TPU miễn phí | **TRC**: apply, phải open-source nghiên cứu, hàng đợi có thể dài |
| Chỉ có thời gian, không có tiền | Kaggle 30 h/tuần + Colab (tương tác) + Lightning → đủ để dev, **không** đủ cho stage-2 |

## Windows / Git Bash: `/raw` biến thành `C:/Program Files/Git/raw`

Triệu chứng (đã xảy ra thật):

```bash
$ ./tools/tree.sh
SystemExit: [!] C:/Program Files/Git/raw không tồn tại. Volume mount: ['/ckpt', '/raw', '/shards']
```

MSYS2 "chuyển đổi tham số": **mọi argv bắt đầu bằng `/`** bị đổi thành đường dẫn
Windows **trước khi** tiến trình Python nhìn thấy. Container Linux không có ổ `C:`
nên lệnh chết — và thông báo lỗi còn in ra đường dẫn đã bị đổi, rất khó đoán.

Đã chặn ở hai lớp:

| Lớp | Việc |
|---|---|
| shell | `export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'` trong `scripts/lib.sh`, `scripts/check.sh`, mọi `tools/*.sh` (Linux/macOS bỏ qua hai biến này) |
| Python | `_demangle()` trong `download.py` sửa `C:/…/Git/raw/…` → `/raw/…` cho `tree`, `ls_remote`, `audit`, `probe_pair`, `--paths` |

Gõ tay thì thêm biến vào trước lệnh:

```bash
MSYS_NO_PATHCONV=1 modal volume ls dire-raw /raw/.staging
MSYS_NO_PATHCONV=1 modal run -m dire_modal.download::tree --path /raw/checkpoints
```

Đường dẫn **tương đối** (`checkpoints/lsun_adm.pth`) không bị đổi, nên đó cũng là
một cách tránh.

## Có nên "reset toàn bộ volume" không?

**Không.** Ở 76.6 KiB/s, 1 GB trong Volume đáng giá **~3.8 giờ** băng thông Baidu
— reset là ném đi chừng đó, và lần chạy sau vẫn dùng đúng code đó.

Cũng không có dữ liệu "bẩn" cần dọn: phép kiểm cũ sai theo hướng **quá khắt
khe** (loại file tốt), chứ chưa bao giờ nhận file hỏng vào `/raw`. Mọi file đã
công bố đều đã qua `gzip -t` + `sha256`.

```bash
bash tools/tree.sh --depth 2
MSYS_NO_PATHCONV=1 modal run -m dire_modal.download::prune --tiers e2   # dry: chỉ in
```

`prune` in ra bốn vùng nội bộ, **bốn kết luận khác nhau**:

| Vùng | Là gì | Xoá? |
|---|---|---|
| `.staging/` | mảnh dở — **tiến độ tải** | ⛔ **KHÔNG** — lần sau resume từ đây |
| `.staging/_held/` | file không qua kiểm | ⚠️ kiểm tay (`gzip -t`) trước |
| `.bench/` | rác của `bench_concurrency` | ✅ xoá được (~775 MB từng nằm im ở đây) |
| `.state/` | cache `(size, mtime_ns, sha256)` | ✅ xoá được, chỉ khiến lần sau kiểm lại |

Chỉ reset khi **thật sự** cần: (a) muốn chứng minh pipeline chạy từ zero, (b)
nghi Volume hỏng ở tầng filesystem, (c) muốn đo lại tốc độ trên trạng thái sạch.
Còn muốn tải lại một file cụ thể thì dùng `--force`, không cần reset.

## Dừng giữa chừng mà không mất dữ liệu

Không cần "tạm dừng" riêng: **staging trong Volume đã được commit định kỳ**
(`staging_commit_s: 45` trong `configs/tiers.yaml`), nên mọi cách dừng đều chỉ
mất tối đa một nhịp commit — không bao giờ mất cả file.

| Cách dừng | Chuyện xảy ra | Mất |
|---|---|---|
| `Ctrl+C` ở máy local | client thoát → app ephemeral bị dọn → container nhận **SIGTERM** | ≤ 45 s |
| `modal app stop -y <app-id>` | SIGTERM tới từng container worker | ≤ 45 s |
| `DETACH=1` rồi `Ctrl+C` | app **không** bị dọn, vẫn tải trên cloud; xem bằng `modal app logs` | 0 |
| SIGKILL / `FunctionTimeoutError` | không chặn được | ≤ 45 s |

**SIGTERM giờ là "dừng êm"** (`_request_stop` trong `download.py`): giết tiến
trình `BaiduPCS-Go` con, `commit()` Volume, trả `status="partial"` và in

```
[stop] nhận tín hiệu dừng (15) → ngừng tải, commit Volume, giữ nguyên phần đã tải trong staging
⏸  3 file dừng giữa chừng theo yêu cầu (SIGTERM/SIGINT).
   Phần đã tải nằm trong /raw/.staging và đã được commit — chạy lại đúng lệnh là resume.
```

Trước đây SIGTERM làm Python chết ngay, **không chạy `finally`**, nên chỉ còn
lớp bảo hiểm 45 s. Muốn chắc hơn nữa (ví dụ mạng chập chờn): hạ
`staging_commit_s` xuống `15` — đổi lại là nhiều lệnh commit hơn.

**Resume:** chạy lại **đúng lệnh cũ**. File đã xong → `[SKIP]`; file dở →
BaiduPCS-Go thấy file `.BaiduPCS-Go-downloading` trong staging và tải tiếp từ
đúng byte đã có, không phải từ 0.

```bash
modal volume ls dire-raw /.staging          # còn gì trong staging
modal app list                              # app còn chạy?
modal app stop -y <app-id>                  # dừng êm
DETACH=1 bash scripts/01_repro.sh --quick   # chạy không phụ thuộc terminal
```

⛔ **Đừng xoá `/raw/.staging`** — đó không phải rác, đó là tiến độ. Chỉ
`/raw/.staging/_held` (file không qua kiểm) và `/raw/.bench` mới là thứ xoá được.

## Sự cố 2026-10-25 — "tải dở dang" trên file đã tải đủ

Ba file `iddpm.tar.gz`, `pndm.tar.gz`, `ldm.tar.gz` tải xong 100 % rồi bị loại,
lặp lại rồi bị xoá. Nguyên nhân **không** phải mạng, cũng không phải Baidu:

```
[RETRY 1/3] .../iddpm.tar.gz: size 125015131 ≠ remote 125011230 → tải dở dang
```

`ls -l` của BaiduPCS-Go in size **đã làm tròn 2 chữ số thập phân MB**
(`119.22MB`). Code cũ quy đổi ngược: `119.22 × 2²⁰ = 125 011 230`, rồi đòi khớp
từng byte với file thật `125 015 131`. Số "remote" trong log chính là
`119.22MiB` — sai 3 901 B, tức 0.003 %.

Sau đó: retry → Baidu thấy file đã có nên in `文件已经存在 … 跳过` (0 byte) →
code cũ gọi đó là "hai lần cho ra byte GIỐNG HỆT ⇒ lỗi tất định" → **`unlink()`**
→ mất 119 MB và lần sau tải lại từ 0 B.

Đã sửa (`download.py`):

| Trước | Sau |
|---|---|
| `int(float("119.22") × 2²⁰)` rồi đòi khớp tuyệt đối | `SizeEst(bytes, slack)` mang theo sai số làm tròn (±5 244 B cho `x.xxMB`) |
| kiểm size TRƯỚC | `gzip -t`/tar (bằng chứng nội dung) kiểm TRƯỚC size |
| nghi ngờ ⇒ kết luận hỏng | nghi ngờ ⇒ hỏi `meta` (size chính xác + md5) trước khi kết luận |
| file không qua kiểm ⇒ `unlink()` | `_hold()` → `/raw/.staging/_held/`, **không xoá byte nào** |
| Baidu bỏ qua vì file đã có ⇒ "lỗi tất định" | nhận diện riêng, tải lại từ 0 B đúng **một** lần |

Đọc kết quả:

```bash
modal volume ls dire-raw / .staging/_held     # file bị giữ lại (nếu có)
modal volume rm dire-raw /raw/.staging/_held -r   # xoá khi đã kiểm tay xong
```

Nếu `_held` có file: nó **không** bị xoá vì code cho là hỏng, mà vì code chưa
chứng minh được nó tốt. Kiểm tay bằng `gzip -t`; nếu qua thì chép về đúng đường
dẫn trong `/raw/` và chạy `::verify`.

## Những việc **không** làm (để không mất tài khoản / mất dữ liệu)

1. ❌ Dùng nhiều tài khoản Kaggle/Colab để lấy thêm quota (vi phạm ToS → ban).
2. ❌ Dùng Colab free làm worker SSH/remote desktop (ToS cấm rõ ràng).
3. ❌ Dựng pipeline phụ thuộc Kaggle (không SSH, mất máy khi hết phiên).
4. ❌ Coi Modal Volume là backup (không versioning; xoá là mất; vẫn tính phí 4 ngày).
5. ❌ Đẩy >10 GB lên R2 mà không kiểm tra (`push_r2` sẽ cảnh báo).
6. ❌ Cam kết stage-2 khi chưa chạy `tools/measure_dire.sh` (con số 30–60 A100-h mới là ước lượng).
