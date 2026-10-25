# instructions.md — cách làm việc trong repo này

> Dành cho agent. `AGENT.md` nói repo là gì; file này nói **được làm gì và
> không được làm gì**. Mỗi quy tắc dưới đây đều gắn với một lỗi đã xảy ra thật.

## 1. Vòng làm việc

```
đọc plan.md → chọn 1 việc → viết test trước (hoặc cùng lúc)
→ code → bash scripts/check.sh → commit → PR
```

Không gộp nhiều việc không liên quan vào một PR. Repo này đã merge 8 PR; mỗi PR
một chủ đề, mô tả nêu rõ **triệu chứng → nguyên nhân → cách sửa → cách kiểm**.

Trước khi commit, luôn chạy:

```bash
python3 -m pytest tests/ -q
python3 -m pyflakes <file vừa sửa> tests/
bash -n scripts/*.sh          # nếu có sửa shell
```

Nếu sửa chỗ chạm `modal.parameter`, `@app.cls` hoặc import: chạy thêm
`bash scripts/check.sh` (hoặc `pytest tests/test_real_modal_import.py`).

## 2. Test phải "cắn"

Một test chỉ có giá trị nếu nó **fail trên code cũ**. Cách kiểm:

```bash
git archive origin/main src | tar -x -C /tmp/old && \
  cp tests/<file> /tmp/old/tests/ && cd /tmp/old && pytest tests/ -q
```

Quy ước đã dùng trong repo: mỗi PR nêu rõ *"N test mới, X fail trên `main` cũ"*.

**Nhưng kiểm harness TRƯỚC khi kiểm code.** Trong lịch sử repo này, harness tự
viết đã cho kết quả sai **mười lần**. Trước khi tin một FAIL, tự hỏi: fixture có
đúng kịch bản không, stub có mô phỏng đúng không, biến môi trường có rò từ test
khác không.

Các bẫy đã gặp thật:

| Bẫy | Triệu chứng | Cách tránh |
|---|---|---|
| `download.time.sleep = None` | Vô hiệu hoá `sleep` của **cả process** (vì `download.time` là module stdlib) | Patch móc riêng `download._sleep` |
| Stub để lại trong `sys.modules` | Test file khác dùng nhầm stub, đỏ như bug thật | Fixture autouse dọn module; hoặc chạy trong subprocess |
| Kiểm "có modal thật" bằng `import modal` tại chỗ | Thấy stub ⇒ **skip âm thầm**, cổng báo xanh mà chưa chạy | Kiểm trong subprocess |
| `tar -czf` trong fixture | gzip header ghi `mtime` hiện tại ⇒ sha256 đổi giữa 2 lần chạy ⇒ test flaky | `--mtime='@0' --numeric-owner` + `gzip -n` |
| Fixture hai bên trùng byte | Rơi vào nhánh "sha256 TRÙNG" chứ không phải nhánh cần kiểm | Cố ý cho hai bên khác nhau |
| `grep` chuỗi để kiểm code | Bắt cả comment giải thích | Dùng `ast.parse` |

## 3. Stub Modal phải đúng ngữ nghĩa

Bài học gốc của repo: **stub không được dễ dãi hơn thứ nó thay thế.**

- `app.function` phải trả object có `.map/.remote/.local` và **`__call__` raise**
  (`TypeError: 'Function' object is not callable`). Nếu trả lại hàm thường, nó
  xoá đúng cái bug cần bắt.
- `modal.parameter` phải kiểm annotation **thô** như `modal.cls`, raise nếu thấy
  chuỗi (PEP 563), raise nếu thiếu annotation, raise nếu kiểu không hỗ trợ.
- `@app.cls` phải validate **lúc decorate**, giống Modal thật (lỗi ở import).

Khi nghi ngờ: viết một test dùng package thật. `tests/test_real_modal_import.py`
làm việc đó trong subprocess để có `sys.modules` sạch.

## 4. Tiền: quy tắc cứng

Ngân sách là **0đ**. Mọi thứ có thể chạm GPU phải:

1. Đi qua `preflight_cost()` **trước** khi gọi `.map()`/`.remote()`.
2. Có `max_containers` **và** `timeout` lấy từ policy, đã clamp trong code.
3. In ra hai con số: chi phí kỳ vọng **và** trần xấu nhất (mọi container treo
   tới timeout). Trần xấu nhất mới là thứ phải chặn.
4. Có `--dry-run` chỉ in tiền, không bật container.

Không bao giờ để số cứng trong decorator. Bài học: `max_containers=40` +
`timeout=24*3600` trên A100 ⇒ trần **~$2 400** cho một lần chạy hỏng.

Muốn chạy thật thì vẫn phải đặt spend limit ở `modal.com/settings/usage` —
guard trong code không cứu được ai chạy `modal run` từ thư mục khác.

## 5. Dữ liệu: quy tắc cứng

- **Không bao giờ xoá dữ liệu trong `/raw`** khi chưa in danh sách và chưa có
  `--mode apply`. Mất một file nghĩa là tải lại từ Baidu ở 67 KB/s.
- **Chỉ công bố file vào `/raw` sau khi đã kiểm.** File dở ở `.staging`; rời
  staging chỉ khi `_validate_ex` đồng ý. Đừng đảo thứ tự này — đảo là mất mảnh
  dở, tức là mất khả năng resume.
- **Không xoá `.staging`**: đó là tiến độ tải, không phải rác.
- `_manifest_sha256.tsv` được commit vào git vì nó là nguồn sự thật để verify
  mirror. `manifests/*.tsv` khác bị ignore.
- Không tải file không cần thiết. Ràng buộc dự án là **chỉ cải thiện, không
  train lại** ⇒ 47 GB dữ liệu train nằm ngoài phạm vi, và đường chạy là
  `download --tiers e2min --mode fetch` → `eval::main` trên `lsun_adm.pth` của
  tác giả. Cần vài file lẻ thì dùng `--paths`, đừng tạo tier mới.
- **Đừng đề xuất tăng container/`-p` để tải nhanh hơn.** Đo rồi: `k=1` 76.6 KiB/s,
  `k=2` 75.0 KiB/s ⇒ trần là của tài khoản. Nhanh hơn thật thì phải đổi nguồn
  (`download::from_hf`).

## 6. Đọc dữ liệu: ba cái bẫy

```python
tarfile.open(p, "r:*")     # KHÔNG "r:gz" — dữ liệu thật là tar không nén
```

- Muốn biết hai archive có cùng tập ảnh: so **từng member** (`_tar_content` +
  `_content_compare`). `sha256` container là bằng chứng yếu: đổi `mtime` là
  sha256 đổi, `size` giữ nguyên tuyệt đối.
- Không bao giờ so `-1 == -1` rồi kết luận "số ảnh bằng nhau". `-1` là *không
  đếm được*.
- `gzip -t` fail **không** đồng nghĩa file hỏng. Đọc magic bytes trước
  (`_diagnose`). Và nếu size đã khớp remote thì lỗi là **tất định** — đừng retry.

## 7. Chẩn đoán lỗi: phân biệt tất định và tạm thời

| Triệu chứng | Loại | Hành động |
|---|---|---|
| `size ≠ remote` | tạm thời | retry, resume |
| 0 byte | tạm thời | giảm `max_containers`, kiểm BDUSS |
| Nội dung là HTML/JSON | **tất định** | dừng, kiểm BDUSS/link |
| Archive sai định dạng (rar/zip) | **tất định** | dừng, đối chiếu tên file gốc |
| Hai lần tải cho sha256 giống hệt | **tất định** | dừng, không đốt thêm attempt |
| tar không nén | **không phải lỗi** | chấp nhận, chỉ ghi nhận định dạng |

Lỗi tất định phải dừng **sau lần thử đầu**. Bài học: 3 attempt × 6 phút = 19
phút vô ích cho một file nguồn sai.

## 8. Git & PR

- Branch hiện tại do Arena quản lý. **Không** tạo/đổi branch khác, không push
  sang branch khác.
- Commit subject: `<phạm vi>: <việc đã làm>` (tiếng Việt hoặc Anh đều được).
  Body nêu triệu chứng → nguyên nhân → sửa → test. Ghi cả **số đo** nếu có.
- PR: nêu rõ test nào fail trên `main` cũ, và phần nào chưa làm.
- Nếu phát hiện harness của mình sai, **nói ra trong PR** — không im lặng sửa
  test cho xanh.

## 9. Tài liệu phải khớp code

Khi merge xong một việc:

- Cập nhật `plan.md` (mục Done / Next).
- Nếu đổi hành vi người dùng thấy được: `README.md` + `docs/runbook-free.md`.
- Nếu đổi policy: `src/dire_modal/tiers.yaml` **và** bản tra cứu
  `configs/tiers.yaml` (ghi rõ nó chỉ để đọc).
- Nếu đổi quyết định kiến trúc: thêm/sửa ADR, đừng sửa lặng lẽ.

## 10. Không được làm

- Không chạy train/eval thật khi chưa có preflight chi phí và chưa có ngưỡng
  thành công ghi trước.
- Không `modal app stop` / xoá Volume khi chưa hỏi người dùng.
- Không commit `.env`, token, hay bất kỳ bí mật nào (`.env.example` mới được).
- Không commit artifact lớn (`*.pth`, shards, gallery PNG) — xem `.gitignore`.
- Không thêm `from __future__ import annotations` vào file có `modal.parameter`.
- Không dùng `git push --force` lên branch của phiên.
- Không bịa số liệu. Không kiểm chứng được thì ghi "chưa xác minh".
