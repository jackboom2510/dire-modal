# Modal làm backend cho demo — kiểm được hay không, và tốn bao nhiêu

- **Trạng thái:** đã kiểm bằng số đo + endpoint đã viết (chưa deploy) · **Ngày:** 2026-11-17
- **Liên quan:** `docs/plans/demos/demo-hf-spaces-plan.md` (§13 chế độ tính tại chỗ),
  `docs/plans/demos/demo-web-plan.md` §3 chế độ C, `src/dire_zero/demo_api.py`,
  `platforms/hf-spaces/demolib/remote.py`
- **Câu hỏi:** *dùng Modal làm nơi chạy demo được không, vì Modal cũng có container?*

---

## 0. Trả lời ngắn

**Được, và đúng chỗ nhất là làm phần GPU.** Không phải vì Modal "có container" —
HF Spaces cũng có — mà vì ba thứ Modal có mà Space không:

| Thứ | Ý nghĩa cho demo |
|---|---|
| **GPU theo giây** (A10 = 1,1016 USD/h) | 3,67 s/ảnh ở `S=20` = **0,0011 USD/ảnh**; Space là 101 s/ảnh và **không có GPU** |
| **Kiểm soát cold start** (`scaledown_window`, `min_containers`, memory snapshot) | chọn giữa "chờ ~15 s" và "trả 0,02 USD/phút" — đo được, không phải đoán |
| **Checkpoint đã nằm trên Volume** (`/zero/ckpt/adm_lsun.pt`) | không phải upload 2,5 GB đi đâu nữa |

Điều **không** nên làm: dùng Modal làm *trang chủ* của demo công khai. Lý do ở §4 —
không phải kỹ thuật mà là chi phí và "ai trả tiền cho người xem thứ nhất".

## 1. Ba cách đặt Modal, và cách đã chọn

```
A. Space (miễn phí) ──HTTP──▶ Modal A10G          ← CHỌN
   trình duyệt → Space → (token) → endpoint /dire → 0,9 s

B. Modal chạy luôn Streamlit (web_server 8000) + GPU function
   trình duyệt → modal.run → Streamlit → GPU → 0,9 s

C. Modal chạy Streamlit trên CPU, gọi function GPU nội bộ
   như B, nhưng phần UI không chiếm GPU
```

**Chọn A.** Space là cửa vào miễn phí, luôn thức, không cần ai trả tiền khi rảnh;
Modal chỉ thức khi có người bấm. B và C đều làm được (Modal có ví dụ chính thức
"Run and share Streamlit apps"), nhưng đặt toàn bộ demo sau một URL của Modal nghĩa
là **mỗi người xem đầu tiên phải chờ container của họ khởi động**, và mọi lượt xem
đều tính tiền — kể cả lượt chỉ mở trang rồi đóng.

## 2. Endpoint đã viết (và những gì nó tự chặn)

`src/dire_zero/demo_api.py` (nối Modal) + `src/dire_zero/demo_service.py` (luật, không
import modal) + `POST /dire`:

| Luật | Mặc định | Vì sao |
|---|---|---|
| Token | **không có token ⇒ 503**, không phải mở | URL công khai + GPU = hoá đơn do người khác bấm |
| Kích thước | 10 MB | cùng giới hạn với `demo-web-plan.md` §6 |
| `S` | 1–20 | 20 là cấu hình của bài (Table 4); nhận `S=500` là tự phá hoá đơn |
| Hạn mức | 6 ảnh / 60 s, burst 2, theo client | đủ cho một buổi nói, không đủ cho bot |
| CORS | **không** mở trừ khi đặt `DEMO_ORIGINS` | mặc định đóng là mặc định đúng |
| Map NaN | 500, không trả số | "map toàn NaN" là lỗi số học, không phải "kết quả thấp" |
| Ảnh sai cỡ | resize về 256² và **ghi lại** `resized: true` | ADM chỉ học ở 256² |
| Trả về | `line` bắt buộc dạng `DIRE: … · detector: … · nhãn: … · S=…` | cùng định dạng với gallery và demo |

Trang 3 của Space **không gọi Modal từ trình duyệt**. `demolib/remote.py` chạy ở
server của Space: trình duyệt → Space (không token, cùng origin) → Modal (token
trong `st.secrets`). Nếu token nằm trong JavaScript thì nó không còn là bí mật —
chỉ là một chuỗi ai cũng đọc được.

## 3. Chi phí — tính từ bảng giá chính thức, không phải cảm giác

Giá niêm yết trên `modal.com/pricing` (2026-11): GPU A10 **0,000306 USD/s** =
**1,1016 USD/h** (khớp `budget.PRICES["A10G"] = 1,10`), CPU **0,0000131 USD/core/s**
(0,047 USD/core/h), RAM **0,00000222 USD/GiB/s** (0,008 USD/GiB/h), Volume 0,09
USD/GiB/tháng (1 TiB đầu miễn phí).

| Tình huống | Tính | Tiền |
|---|---|---|
| 1 ảnh, `S = 5` (0,92 s GPU) | 0,92 s × 1,1016/3600 | **0,00028 USD** |
| 1 ảnh, `S = 20` (3,67 s) | 3,67 s × 1,1016/3600 | **0,0011 USD** |
| 20 ảnh trong buổi nói (`S=5`) | 18,4 s GPU | **0,006 USD** |
| Container ấm, không ai gọi, mỗi phút | GPU 1,10 + CPU ~0,05 + RAM 4 GiB 0,03 | **~0,020 USD/phút** |
| Giữ ấm 30 phút quanh buổi nói | 30 × 0,020 | **~0,59 USD** |
| Giữ ấm cả ngày (8 h) | 480 × 0,020 | **~9,4 USD** (đừng) |
| Cold start (nạp checkpoint 10,7 s đo được) | 10,7 s × 1,1016/3600 | **0,0033 USD** |

Free tier Starter: **30 USD/tháng** ⇒ ~25 giờ A10 ấm, hoặc ~27 000 ảnh `S=20`.
Một buổi nói 30 phút nằm gọn trong hạn mức, và **không** cần thẻ.

So sánh trực tiếp với chế độ tính tại chỗ trên Space:

| | Space CPU free | Modal A10G |
|---|---|---|
| 1 ảnh `S=5` | 101 s | **0,92 s** |
| 1 ảnh `S=20` | 404 s | **3,67 s** |
| Tiền | 0 | 0,0003–0,0011 USD/ảnh |
| Ai trả khi rảnh | không ai | không ai (scale-to-zero) |

## 4. Vì sao không đặt cả demo lên Modal

Kỹ thuật thì được: `@modal.web_server(8000)` + WebSocket (Streamlit cần WebSocket;
Modal hỗ trợ đầy đủ RFC 6455), và ví dụ chính thức của Modal tắt CORS/XSRF vì đi
qua proxy của họ. Nhưng:

| Vấn đề | Hệ quả |
|---|---|
| Mỗi lượt xem đều tính tiền (kể cả mở rồi đóng) | một demo công khai có 100 lượt xem lẻ = 100 cold start, mỗi cái ~0,003 USD + thời gian |
| Không có "trang luôn thức" miễn phí | người xem đầu tiên luôn chờ container; Space thì đã thức |
| Streamlit trong container của Modal phải tắt CORS/XSRF | đúng theo ví dụ của họ, nhưng đó là một tầng cấu hình nữa để sai |
| Token/URL của Modal lộ ra ngoài | ai có URL là gọi được `/` của Streamlit, không qua hàng rào nào |

Nên: **Space làm cửa, Modal làm GPU.** Nếu sau này cần một URL riêng (không phụ
thuộc HF), cách B là phương án dự phòng đã kiểm về khả thi, không phải phương án đầu.

## 5. Cold start — bao nhiêu, và cách giảm

| Giai đoạn | Thời gian | Nguồn |
|---|---|---|
| Container boot | ~1 s | tài liệu Modal ("containers boot in about one second") |
| `import torch` + `import dire_zero` | ~2–4 s | **chưa đo trên Modal** |
| Nạp ADM 2,5 GB từ Volume | **10,7 s** | đo thật, log `::gen` |
| **Tổng, ước tính** | **~15 s** | cộng ba dòng trên |
| Với memory snapshot (`@modal.enter(snap=True)`) | **chưa đo** — mục tiêu < 5 s | tài liệu Modal: snapshot bỏ qua phần init |

Code đã bật `enable_memory_snapshot=True` và nạp model **bên trong** factory của
ASGI app, tức là trước khi chụp snapshot. Nhưng theo tài liệu, snapshot chỉ có tác
dụng với app **đã `deploy`** (app ephemeral của `modal run` tắt snapshot) — nên con
số "< 5 s" là **giả thuyết cần đo**, không phải kết quả.

Ba mức chọn, tuỳ buổi nói:

| Cấu hình | Cold start | Tiền khi rảnh | Dùng khi |
|---|---|---|---|
| `min_containers=None` (mặc định trong code) | ~15 s | 0 | demo tự phục vụ, người xem kiên nhẫn |
| `min_containers=1` + `scaledown_window=600` | 0 | ~0,020 USD/phút | **buổi nói**: bật trước 5 phút |
| `scaledown_window=60` (mặc định Modal) | ~15 s sau 1 phút rảnh | 0 | thử nghiệm |

## 6. Quy trình chạy buổi nói (đã kiểm từng bước trừ bước cuối)

```bash
# Đặt biến trước: trong bash, `<user>`/`<ws>` không phải chỗ trống mà là chuyển
# hướng stdin — lệnh sẽ chết với "No such file or directory".
export HF_USER=jackboom2510
export WS="$(modal profile current 2>/dev/null || echo '$HF_USER')"   # workspace Modal

# 1. một lần: secret (token do mình sinh, không dùng lại ở đâu)
modal secret create dire-demo \
  DEMO_TOKEN="$(python -c 'import secrets;print(secrets.token_urlsafe(24))')" \
  DEMO_ORIGINS="https://$HF_USER-dire-demo.hf.space"

# 2. deploy endpoint
modal deploy -m dire_zero.demo_api

# 3. smoke test: /health không cần token, /dire cần
curl -s "https://$WS--dire-demo-api.modal.run/health"
# {"ok":true,"engine":"adm (sha256 …)","s_range":[1,20],"max_bytes":10485760,"auth":"bật",…}

python - <<'PY'
import base64, json, os, pathlib, urllib.request
url = f"https://{os.environ['WS']}--dire-demo-api.modal.run/dire"
body = json.dumps({"image_b64": base64.b64encode(pathlib.Path("a.png").read_bytes()).decode(),
                   "steps": 5}).encode()
req = urllib.request.Request(url, body, {"Content-Type": "application/json",
                                         "X-Demo-Token": os.environ["DEMO_TOKEN"]})
r = json.loads(urllib.request.urlopen(req).read())
print(r["line"], r["seconds"], r["stats"])
PY

# 4. Space → Settings → Secrets: DEMO_URL=…  DEMO_TOKEN=…
# 5. trước giờ nói 5 phút: gọi 1 ảnh để container ấm (và để snapshot được tạo)
```

Bước 5 không phải mẹo vặt: lần gọi đầu tiên sau `deploy` là lần **tạo snapshot**,
và theo tài liệu Modal phải qua vài lần cold start mới thấy cải thiện. Nên gọi thử
trước, đừng để khán giả làm người thử.

## 7. Đã kiểm bằng gì, chưa kiểm bằng gì

**Đã kiểm (chạy được ngay trong repo, không cần GPU, không cần tài khoản):**

| Kiểm | Ở đâu |
|---|---|
| Token thiếu/sai ⇒ 401; không cấu hình ⇒ **503** | `tests/test_zero_demo_api.py` (24 test) |
| Body > 10 MB ⇒ 413, trước khi giải mã | nt |
| `S` ngoài 1–20 ⇒ 400; `S` không phải số ⇒ 400 | nt |
| Hạn mức ⇒ 429 kèm số giây; nạp lại sau cửa sổ; theo từng client | nt |
| Map NaN ⇒ 500; engine lỗi ⇒ 422 | nt |
| `line` có đủ nguồn gốc; PNG hợp lệ; `resized` được khai | nt |
| CORS chỉ mở khi có `DEMO_ORIGINS` | nt |
| Space → Modal: token không lọt vào thông báo lỗi | `tests/test_hf_spaces_remote.py` (16 test) |
| Endpoint trả JSON thiếu `line`/PNG rỗng ⇒ **từ chối** | nt |
| Trang 3 dùng endpoint khi có secret, không nạp ADM tại chỗ | nt (bấm nút thật qua `AppTest`) |
| `demo_api.py` import được, `api` là một `modal.Function` | kiểm bằng `python -c` với `modal==1.6.4` |

**Chưa kiểm (và không được nói như đã kiểm):**

- cold start thật, và memory snapshot có tác dụng hay không;
- thời gian `TorchEngine` trên A10G trong endpoint (suy từ `::bench` 0,0917 s/step);
- WebSocket/proxy nếu chọn phương án B (Streamlit trên Modal);
- giới hạn thật của Modal về số container đồng thời trên gói Starter (tài liệu ghi 100).

Ba thứ đó chỉ biết sau `modal deploy` đầu tiên; mục 8 nói cách ghi lại.

## 8. Sau lần deploy đầu, ghi vào đây

| # | Đo | Ghi vào |
|---|---|---|
| 1 | cold start lần 1 (không snapshot) | bảng §5 |
| 2 | cold start lần 3–5 (có snapshot) | bảng §5 |
| 3 | `seconds` của `S=5` trên ảnh 256² thật | so với 0,92 s dự kiến |
| 4 | RAM/CPU thực tế của container | đối chiếu giả định 4 GiB ở §3 |
| 5 | hoá đơn 24 h đầu, từ dashboard | so với 0,020 USD/phút khi ấm |

Nếu (3) lệch quá 30 % so với 0,92 s thì `sec_per_step` trong `bench.json` đã cũ —
chạy lại `::bench` trước khi tin mọi con số tiền.

## 9. Rủi ro

| Rủi ro | Chặn bằng |
|---|---|
| URL công khai bị lạm dụng | token + 503 khi thiếu cấu hình + 6 ảnh/phút |
| Token lộ trong log/exception | `remote._redact` + test |
| Ảnh của người dùng bị lưu | endpoint **không ghi đĩa**, chỉ trả bytes; không log ảnh |
| Quên tắt `min_containers` sau buổi nói | mặc định `None` trong code; chỉ bật bằng tay khi cần |
| Endpoint trả "kết quả" không có nguồn gốc | `remote.call` từ chối nếu `line` không bắt đầu bằng `DIRE:` |
| Chi phí vượt dự tính | `budget` trong repo + dashboard; 30 USD/tháng free tier |

## 10. Việc còn lại

| # | Việc | Chặn bởi |
|---|---|---|
| 1 | `modal deploy -m dire_zero.demo_api` + smoke test | tài khoản + secret |
| 2 | Ghi cold start và `seconds` vào §8 | #1 |
| 3 | Nếu cần: `min_containers=1` cho buổi nói (đổi một dòng) | #1 |
| 4 | Tạo gallery từ `::dire` để trang 2 có dữ liệu | `::dire` xong |
| 5 | Chỉ khi cần URL riêng: phương án B (Streamlit trong Modal) | quyết định riêng |
