# ADR-0005 — Tách `dire_core` (lõi thuần) · `dire_local` (CLI/API) · `dire_modal` (wrapper)

- **Trạng thái:** Accepted
- **Ngày:** 2026-10-14
- **Liên quan:** `src/dire_core/`, `src/dire_local/`, `src/dire_modal/api.py`,
  `src/dire_modal/eval.py`, `src/dire_modal/common.py`, `spec.md` FR-9,
  `tests/test_core_parity.py`, `docs/adr/adr-0002-iaas.md`

## Bối cảnh

Trước ADR này, **mọi** module đều bắt đầu bằng `from .common import app, ...`.
`common.py` import `modal` ở dòng đầu. Hệ quả đo được trong sandbox:

| Thử | Kết quả |
|---|---|
| `import dire_modal.common` khi **có** `modal` (không cần auth) | ✅ chạy — `Volume.from_name` là lazy |
| `import dire_modal.common` khi **không** cài `modal` | ❌ `ModuleNotFoundError: modal` |

Nghĩa là *không cài `modal`, không import được cả phần logic*. Người chỉ muốn
chạy `dire detect anh.jpg` trên máy mình vẫn phải `pip install modal`, vẫn phải
có một `modal.App`, và vẫn phải tin rằng đường cloud là đường duy nhất.

Điều này mâu thuẫn với thực tế dự án: `docs/iaas-options.md` §4 đã chỉ ra
**30–60 A100-h** cho stage-2 là rào cản tiền, và câu trả lời tự nhiên là *chạy
ở máy nhà*. Nhưng máy nhà không chạy được vì kiến trúc import.

Đồng thời có một rủi ro khoa học: nếu bản local được viết **mới** (thay vì dùng
lại), nó sẽ có pipeline tiền xử lý thứ hai. Repo đã gặp đúng lỗi này một lần:
`demo.py` của tác giả thiếu `Normalize` nên điểm demo lệch điểm paper, và không
có exception nào báo.

## Quyết định

Tách ba lớp, **một chiều phụ thuộc**:

```
dire_core   ← thuần: chỉ Python + (torch/PIL truyền vào)
   ↑                    ↑
dire_local           dire_modal
(CLI + FastAPI)      (@app.function + @fastapi_endpoint)
```

### 1. `dire_core` — nguồn sự thật duy nhất cho ba thứ quyết định con số

| Việc | File | Vì sao phải là một chỗ |
|---|---|---|
| Tiền xử lý | `dire_core/preprocess.py` | `demo.py` vs `datasets.py` của tác giả lệch nhau ⇒ điểm lệch |
| Đọc lớp cuối checkpoint | `dire_core/detector.py` | `fc.weight [1,2048]` vs `[2,2048]` ⇒ `load_state_dict` nổ cả eval |
| Logit → `P(fake)` | `dire_core/score.py` | `sigmoid` vs `softmax` ⇒ đảo dấu không exception |
| Nhãn real/fake | `dire_core/labels.py` | Nhãn sai làm AP vẫn "trông hợp lý" |
| Vòng lặp batch | `dire_core/batch.py` | Ba nơi gọi (eval, CLI, API) — chép là trôi |

Quy tắc cứng: **không** `import modal`, **không** `import torch` ở tầng module.
`torch`/`transforms` được truyền vào hàm. Nhờ vậy `import dire_core` chạy được
trên máy chỉ có Python, và test không cần torch.

### 2. `dire_local` — chạy ở máy nhà

```bash
dire info                                   # không cần torch
dire detect anh.jpg --weights lsun_adm.pth  # P(fake) = 0.982
dire batch ./thu_muc --out scores.csv       # CPU, tuần tự
dire serve --port 8000                      # FastAPI + uvicorn
```

### 3. `dire_modal` — wrapper, không chứa logic

`src/dire_modal/api.py` chỉ:

* nạp model một lần cho mỗi container (`_DET_CACHE`),
* quyết định vận hành (`max_containers=2`, `scaledown_window=120`,
  weight nhúng vào image),
* kiểm API key (`check_key`, hàm thuần nên test được),
* **gọi `dire_local.server.score_payload`** — tức là dùng chung một đường decode
  ảnh và một schema response với bản local.

`eval.py` giữ nguyên tên công khai (`PREPROC`, `label_of`, `preprocess`,
`prob_fake`, `fc_out_features`, `load_detector`) nhưng **uỷ quyền** cho core.
Nhờ vậy không phải sửa chỗ gọi và không có hai bản logic.

## Vì sao không chọn các cách khác

| Phương án | Vì sao loại |
|---|---|
| Viết `dire_local` như bản sao của `eval.py` | Hai pipeline sẽ trôi; đúng lỗi `demo.py` của tác giả |
| Bọc `dire_modal` bằng Modal cho mọi thứ, CLI chỉ gọi API | Chạy local vẫn phải có mạng + tài khoản = không phải "local" |
| Đưa `dire_core` vào một file duy nhất | Test AST không tách được "không import torch"; và file sẽ phình >600 dòng |
| Giữ `common.py` làm nơi chứa logic | `common.py` import `modal` ở dòng đầu ⇒ chặn toàn bộ đường local |

## Chi phí — phần phải nói rõ

| Việc | Chi phí |
|---|---|
| Tách core + CLI + wrapper (PR này) | **$0** — không chạy GPU, không tải dữ liệu |
| `dire detect` trên CPU | ~0,15–0,4 s/ảnh ⇒ 16k ảnh ≈ **40 phút CPU** |
| `dire serve` tại máy | $0 (điện) |
| Modal endpoint, scale-to-zero, 10k ảnh/tháng | **~$0,01** compute |
| Modal endpoint, `keep_warm` 24/7 | **~$8,6/tháng** = 29 % credit $30 ⇒ **không bật mặc định** |
| Endpoint public không auth | bị abuse thì **mất $30 trong vài giờ** ⇒ `check_key` fail-closed |

## Hệ quả

**Được:**
- `pip install -e .` + `dire detect` chạy được mà **không cần** Modal.
- Sửa `compute_dire` / `train.py` trên máy nhà: vòng lặp 20 giây thay vì 5 phút
  đẩy image — đây là điều kiện thực tế để gỡ 3 điểm chưa kiểm của `compute_dire`.
- Cùng một lõi ⇒ số của CLI, API local và eval cloud **khớp** (có test).
- Weight nhúng image ⇒ cold start ~1 s thay vì 2–5 s.

**Không được:**
- Thêm logic vào `dire_modal/api.py` (chỉ được là wrapper).
- `import modal` trong `dire_core` (test đỏ).
- Viết lại pipeline tiền xử lý ở bất kỳ đâu khác.
- Chạy `dire batch` song song nhiều tiến trình rồi coi đó là "nhanh hơn cloud" —
  một máy vẫn là một máy.

## Cách kiểm (để quy tắc không thành lời nói)

| Kiểm | Test |
|---|---|
| `dire_core` import được khi chặn `modal`, `torch`, `PIL`, `fastapi` | `test_core_imports_without_modal` |
| `dire_core` không có volume/`modal.App` | `test_core_has_no_volume_or_app_references` |
| `eval.py` không còn `Resize/Normalize` trong `preprocess` | `test_eval_has_no_second_preprocess_pipeline` |
| `eval.py` không còn viết tay `exp()` | `test_eval_has_no_hand_written_sigmoid` |
| Hằng số, `label_of`, `prob_fake`, `fc_out_features` khớp nhau | 5 test `*_parity*` |
| Image mount đủ 3 package | `test_common_mounts_all_three_packages` |
| CLI chạy được không torch | `test_cli_info_runs_without_torch` |
| API key fail-closed | `test_api_key_check_is_pure_and_fails_closed` |
| HTTP API tại máy trả đúng schema / mã lỗi | `tests/test_local_server.py` (10 test, detector giả) |
| CLI `detect`/`batch` dùng chung `dire_core.Detector` | `test_cli_detect_uses_the_shared_detector`, `test_cli_batch_writes_csv` |

### Bẫy đã gặp khi viết wrapper

`server.py` có `from __future__ import annotations`; khai báo body bằng một
`BaseModel` định nghĩa **bên trong** `build_app` làm FastAPI không resolve được
annotation (chuỗi) ⇒ nó coi `req` là **query param** và trả `422 field required`
dù client gửi JSON đúng. Cách chữa: nhận `dict` trần rồi validate trong
`score_payload` — local và cloud dùng **một** đường kiểm tra. `tests/test_local_server.py`
khoá lại hành vi này (200 khi hợp lệ, 400 khi base64 hỏng, 401 khi sai key).
