# skills.md — công thức cho từng loại việc

> Mỗi mục là một "skill": khi nào dùng, làm theo bước nào, cạm bẫy, cách kiểm.
> Quy tắc chung ở [instructions.md](instructions.md). Việc đang làm ở
> [plan.md](plan.md).

---

## Skill 0 — Tái lập thử nghiệm nhanh nhất (KHÔNG train)

**Khi nào:** cần có con số đầu tiên trên dữ liệu thật, hoặc ai đó hỏi "tái lập
DIRE thế nào". Đây là đường chính của dự án.

**Bước:**

```bash
modal run -m dire_modal.download::local_status          # 0 credit, xem policy
modal run -m dire_modal.download::main --tiers e2min --mode fetch   # 3 file, ~400 MB
modal run -m dire_modal.download::verify
modal run -m dire_modal.eval::main --tiers e2min        # in CỔNG S2
```

Gói cả 4 bước: `bash scripts/08_repro_minimal.sh`.

**Vì sao `e2min`:** chỉ cần `lsun_adm.pth` + test `real` + test `adm` để kiểm
`ACC(adm) ≥ 0.99` (cổng S2 của SPEC-001). 3 file ≈ 400 MB ≈ **1.5 h** ở 76.6
KiB/s, so với `e2` (17 file / 1.94 GB / 7.4 h).

**Cạm bẫy:**

- **Đừng "tối ưu" bằng cách tăng container hay `-p`.** Đo rồi: `k=1` 76.6 KiB/s
  vs `k=2` 75.0 KiB/s — trần là của tài khoản. Tăng thread còn bị throttle.
- **Đừng gõ lại lệnh tải nhiều lần.** `download::main` tự lặp vòng tới 20 h và
  tính lại file còn thiếu từ đĩa; nó chỉ dừng sớm khi một vòng không xong thêm
  file nào (dấu hiệu BDUSS hết hạn).
- `eval::main` **tự tìm** ckpt ở `/raw/checkpoints/lsun_adm.pth` rồi
  `/ckpt/pretrained/lsun_adm.pth`. Không thấy thì nó in lệnh cần chạy — đừng tự
  đoán đường dẫn.
- Cổng S2 KHÔNG ĐẠT thì đừng đọc bảng AP. Kiểm: đúng file ckpt chưa, `n_real`
  và `n_fake` có bằng nhau và > 0 không, tiền xử lý có đúng resize256 →
  centercrop224 → Normalize ImageNet không (xem `run.json`).

**Đường nhanh hơn Baidu (khi đã có mirror):**

```bash
modal run -m dire_modal.download::from_hf --repo-id <user>/<repo> --tiers e2
```

**Cần đúng vài file lẻ, không theo tier:**

```bash
modal run -m dire_modal.download::main \
  --paths dire/test/lsun_bedroom/lsun_bedroom/real.tar.gz,checkpoints/lsun_adm.pth \
  --mode fetch
```

`--paths` vẫn đi qua danh sách `NEVER` và guard dung lượng như mọi đường khác.

**Kiểm:** `pytest tests/ -q` xanh; `download::plan` in đúng số file/GB của tier.

---

## Skill 1 — Chẩn đoán một lần tải hỏng

**Khi nào:** `[STOP]`, `gzip -t FAIL`, file 0 byte, tải xong mà vẫn lỗi.

**Bước:**

1. Đọc dòng `[OK]/[SKIP]/[REDO]/[STOP]` — chúng nói loại lỗi, đừng đoán từ
   traceback.
2. Kiểm định dạng **thật** của file (không đoán theo tên đuôi):

   ```python
   from dire_modal.download import _diagnose
   print(_diagnose(Path("/raw/dire/test/lsun_bedroom/lsun_bedroom/real.tar.gz")))
   ```

3. Phân loại theo bảng ở `instructions.md` §7 (tất định hay tạm thời).
4. Nếu tất định → **đừng chạy lại**. Đối chiếu tên file với nguồn gốc của tác
   giả. Retry chỉ đốt 6 phút/lần.

**Cạm bẫy:**

- Size khớp remote mà vẫn hỏng ⇒ **không phải** lỗi mạng. Đừng tăng retry.
- tar không nén vẫn là dữ liệu hợp lệ — đừng xoá.
- `gzip -t` fail 3 lần trên cùng file = tất định, kể cả khi chưa biết lý do.

**Kiểm:** `modal run -m dire_modal.download::verify` (chỉ đọc `/raw`, không tải
byte nào) phải in định dạng thật cho từng file hỏng.

**Xem Volume có gì:** `modal volume ls <volume> [path]` — nhưng **chỉ một tầng**,
và khi stdout không phải TTY thì nó chỉ in đường dẫn (mất cột size). Cần cả cây
thì `modal run -m dire_modal.download::tree --path /raw` (thêm `--flat` để ra
`<size>\t<path>`). Đừng lặp `modal volume ls` bằng bash: mỗi lần là một round-trip
gRPC, mà cây `dire-raw` hơn trăm thư mục.

---

## Skill 2 — Thêm hoặc sửa một tier

**Khi nào:** cần tải thêm/bớt dữ liệu, hoặc đổi tier mặc định.

**Bước:**

1. Sửa `src/dire_modal/manifest.py` (`TIERS`) — đây là inventory.
2. Chạy `modal run -m dire_modal.download::audit` → đối chiếu tên thật trên
   Baidu, không đoán.
3. `modal run -m dire_modal.download::plan --tiers <tier>` → xem size **thật**.
4. Cập nhật bảng tra cứu `configs/tiers.yaml` (chỉ để đọc, nhưng phải khớp).
5. Nếu tier mới vượt `max_gb` → sửa `src/dire_modal/tiers.yaml` có ý thức, ghi
   lý do vào PR.

**Cạm bẫy:**

- Sửa `configs/tiers.yaml` mà quên `src/dire_modal/tiers.yaml` = không có tác
  dụng (Modal chỉ đóng gói package).
- Guard dùng `known + unknown × largest`: file chưa rõ size bị tính bằng file
  lớn nhất. Đừng "sửa" cho guard dễ thở hơn.
- `DirSpec` để liệt kê động — nếu tên file đã biết chắc thì chuyển sang `File`.
- Cần tải **một lần, vài file lẻ** thì đừng tạo tier mới: dùng `--paths`. Tier là
  để chia sẻ và tái lập, không phải để chứa danh sách dùng một lần.

**Kiểm:** `plan` in đúng số file/GB; `pytest tests/ -q` xanh.

---

## Skill 3 — Guard chi phí cho một job GPU

**Khi nào:** thêm/sửa hàm hoặc class có `gpu=...`.

**Bước:**

1. Mọi tham số lấy từ `POLICY` (`src/dire_modal/tiers.yaml`), **không** số cứng.
2. Thêm hàm clamp trong code, không chỉ trong YAML:

   ```python
   _CONTAINER_HARD_CAP = 16
   def _clamp_containers(raw, n_images=None): ...
   ```

3. `preflight_cost(n)` phải gọi **trước** `.map()`/`.remote()`.
4. In **hai** con số: kỳ vọng và trần xấu nhất (containers × timeout × giá).
5. Có `--dry-run` chỉ in tiền.
6. Test phải chứng minh thứ tự: vượt trần ⇒ `.map()` **và** `.remote()` không
   được gọi (đếm bằng sổ ghi, xem `tests/test_compute_dire_guard.py`).

**Cạm bẫy:**

- `timeout` là trần chi phí thật. `max_containers=40` + `timeout=24h` = $2 400.
- Clamp container mà quên clamp timeout vẫn để lọt trần lớn.
- `@app.cls` validate **lúc decorate** — config sai làm hỏng import của cả module.

**Kiểm:**

```bash
python3 -m pytest tests/test_compute_dire_guard.py -q
modal run -m dire_modal.compute_dire::dire_batch --n 200 --dry-run
```

---

## Skill 4 — Viết một test thật sự bắt lỗi

**Khi nào:** luôn luôn, khi sửa bug.

**Bước:**

1. Viết test tái hiện **triệu chứng người dùng thấy**, không phải triệu chứng
   bạn muốn.
2. Kiểm harness: fixture có đúng kịch bản không? có rò state giữa test không?
3. Chạy test trên code **cũ**:

   ```bash
   git archive origin/main | tar -x -C /tmp/old
   cp tests/<file> /tmp/old/tests/ && cd /tmp/old && pytest tests/ -q
   ```

4. Chỉ khi test đỏ trên code cũ và xanh trên code mới thì mới coi là xong.
5. Ghi số "X test mới, Y fail trên main cũ" vào PR.

**Cạm bẫy:** xem bảng ở `instructions.md` §2 — sáu kiểu harness tự phá mình.

**Kiểm:** `python3 -m pytest tests/ -q` và chạy file test trực tiếp
(`python3 tests/test_xxx.py`) — runner tự viết có thể bỏ sót test khai báo sau
khối `__main__`.

---

## Skill 5 — Chốt ngữ nghĩa dữ liệu trước khi ghi số

**Khi nào:** trước khi đưa một nhánh dữ liệu vào bảng eval.

**Bước:**

1. `modal run -m dire_modal.download::probe_pair` — so **từng ảnh** trong tar.
2. Đọc kết luận: `CÙNG HOÀN TOÀN` / `CÙNG ảnh, KHÁC tên` / `KHÁC HOÀN TOÀN` /
   `LỆCH MỘT PHẦN` / `KHÔNG so được`.
3. `KHÔNG so được` ⇒ **không kết luận gì**. Đừng suy ra từ sha256 container.
4. Nếu hai nhánh dùng chung ảnh real ⇒ nhãn phải là `cross-generator`, và
   **không** cộng vào mean AP như một domain riêng.

**Cạm bẫy:**

- `sha256` container khác nhau **không** chứng minh tập ảnh khác nhau.
- `-1` (không đếm được) không phải một con số để so.
- Hai file cùng size 26.6 MB là dấu hiệu mạnh của "cùng ảnh, khác đóng gói".

**Kiểm:** kết luận in ra phải kèm số ảnh đếm được, không phải `-1`.

---

## Skill 6 — Kiểm bằng Modal THẬT (không phải stub)

**Khi nào:** sửa chỗ chạm `@app.function`, `@app.cls`, `modal.parameter`, hoặc
import module.

**Bước:**

```bash
./scripts/07_check_imports.sh
```

Script tạo `.venv`, cài `modal` thật, chạy `tests/test_real_modal_import.py`,
toàn bộ test, rồi pyflakes.

**Cạm bẫy:**

- Kiểm "có modal thật" bằng `import modal` tại chỗ sẽ thấy stub nếu test khác đã
  cài vào `sys.modules` ⇒ **skip âm thầm** ⇒ cổng xanh mà chưa chạy.
- `from __future__ import annotations` + `modal.parameter` = chết lúc import.
  File có `@app.cls` **không** được có future import.

**Kiểm:** output phải là `2 passed` (không phải `skipped`) ở bước 1.

---

## Skill 7 — Đọc/ghi dữ liệu tar trong repo này

**Khi nào:** mọi chỗ mở `.tar.gz`.

```python
with tarfile.open(path, "r:*") as t:   # "r:*" — KHÔNG "r:gz"
    ...
```

- Đếm/duyệt member: `_tar_profile` (nhanh, không giải nén ra đĩa).
- Băm từng ảnh để so nội dung: `_tar_content` + `_content_compare`.
- Kiểm tar cụt: `_tar_ok` (đọc 1024 byte cuối, kiểm EOF marker 2 block 0).

**Cạm bẫy:** `"r:gz"` ném `ReadError: not a gzip file` ở member đầu tiên. Ở
`pack.py` lỗi đó không có guard và giết cả stage.

---

## Skill 8 — Cập nhật tài liệu sau khi merge

**Khi nào:** cuối mỗi việc.

1. `plan.md`: chuyển mục sang **Đã xong**, thêm số đo thật.
2. `README.md`: nếu có lệnh/tham số mới.
3. `docs/runbook-free.md`: nếu hành vi người dùng thấy được thay đổi.
4. `src/dire_modal/tiers.yaml` + `configs/tiers.yaml`: nếu đổi policy.
5. ADR mới nếu đổi quyết định kiến trúc.

**Cạm bẫy:** đừng tạo bản sao thứ hai của cùng một con số. Nếu số đã có trong
`tiers.yaml` thì tài liệu nên trỏ về đó thay vì chép lại.
