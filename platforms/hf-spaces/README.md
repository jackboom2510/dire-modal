---
title: DIRE — sai số tái tạo
emoji: 🔍
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
app_file: gradio_app.py
suggested_hardware: cpu-basic
pinned: false
license: mit
short_description: DIRE map, nguồn gốc từng con số, và một phép đo chạy thật
---

# DIRE — demo thuyết trình (HF Spaces, CPU free)

Demo cho **DIRE — Diffusion Reconstruction Error** (Wang et al., ICCV 2023). Repo
gốc: <https://github.com/ZhendongWang6/DIRE> (đã archive 2025-07-08).

## Hai giao diện, một thư viện

| Tệp | SDK | Dùng khi |
|---|---|---|
| **`gradio_app.py`** | `gradio` (mặc định của Space này) | bản chính: hàng đợi, trạng thái phiên và nút *Restart* do Gradio lo |
| `app.py` | `streamlit` | bản dự phòng, giữ nguyên và vẫn chạy được (`streamlit run app.py`) |

Hai bản dùng chung `demolib/` (`toy2d`, `provenance`, `charts`, `cost`, `remote`),
nên **không có con số nào khác nhau** giữa hai giao diện: chúng chỉ khác cách bày.
Test của hai bản nằm ở `tests/test_hf_spaces_demo.py` (Streamlit) và
`tests/test_hf_spaces_gradio.py` (Gradio).

Cả hai có **cùng một tab gallery** (ảnh gốc · DIRE map · điểm detector, kèm `.json`
nguồn gốc), tab thí nghiệm 2 chiều, tab lịch/chi phí, và tab Table 2.

## Demo này làm gì

| Tab (Gradio) | Nội dung | Chạy ở đâu | Nhãn |
|---|---|---|---|
| 1 | **Gallery**: ảnh gốc · DIRE map · điểm detector, kèm `.json` nguồn gốc | ảnh tính trước | `R0` / `R1+R2` |
| 2 | **Tính DIRE thật**: upload ảnh → **DIRE map** (không trả `P(fake)`), qua endpoint Modal | GPU Modal, **mặc định tắt** | `R1` |
| 3 | **Thí nghiệm 2 chiều**: mô hình khuếch tán có score chính xác, đo DIRE | ngay trên Space, 61 ms | minh hoạ lý thuyết |
| 4 | Lịch nhiễu `ᾱ_t`, NFE, và chi phí | ngay trên Space | công thức |
| 5 | Bảng 2 của bài + đường tái lập + hạn chế | tĩnh | số của paper |

## Demo này **không** làm gì

- **Không có ô upload ảnh.** Một ResNet-50 học trên DIRE map không có nghĩa gì khi
  nhận ảnh RGB thô. Bản Gradio trước đây trả `P(fake)` cho ảnh upload; đó là lỗi bị
  ghi ở `docs/demo-web-plan.md` §2 và nó đã bị bỏ, không được dựng lại.
- **Không** ghi "độ chính xác 99 %" ở bất kỳ đâu. Số của bài luôn kèm chữ
  *"số của paper (ICCV 2023), không phải của demo này"*.
- **Không** hiện `p_fake` khi chưa qua cổng kiểm: `detector.gate` phải là `passed`
  và ngưỡng phải chọn trên `val`.
- **Không** tính DIRE lúc chạy **theo mặc định**: một lượt forward ADM 256² là
  **10,1 s** trên 2 vCPU (đo), nên `S = 5` là **101 s**/ảnh và `S = 20` là 404 s.
  Trang 3 làm được nhưng phải bật thủ công (`DIRE_CKPT` + torch) và phải tick xác
  nhận thời gian chờ; lý do và số đo ở `docs/demo-hf-spaces-plan.md` §13.

## Cấu trúc

```
gradio_app.py          # UI Gradio, 5 tab  (mặc định của Space)
app.py                 # UI Streamlit, 6 trang (bản dự phòng)
demolib/toy2d.py       # mô hình 2 chiều, score chính xác (thuần math)
demolib/provenance.py  # đọc .json của gallery, quyết định khi nào hiện điểm
demolib/charts.py      # SVG thuần, không matplotlib
gallery/               # <tên>_orig.png  <tên>_diff.png  [<tên>_recon.png]  <tên>.json
requirements.txt       # gradio (UI chính) — streamlit nằm ở requirements-streamlit.txt
```

`gallery/` đang **rỗng** (chỉ `.gitkeep`). Đó là trạng thái hợp lệ: trang 2 hiện
lệnh dựng gallery chứ không hiện ba khung xám giả.

## Dựng gallery (khi đã có DIRE map)

**Cách 1 — dựng ngay trên Volume (không cần đường dẫn nào):**

```bash
modal run -m dire_zero.modal_app::gallery --n 8 --steps 20
modal volume get dire-zero gallery ./platforms/hf-spaces/gallery
```

**Cách 2 — chạy ở máy, khi map đã tải về.** Đặt đường dẫn vào biến, đừng gõ
`<volume>`: trong bash, `<...>` là **chuyển hướng stdin**, không phải chỗ trống.

```bash
VOL=/c/Users/ban/dire-vol          # thư mục đã `modal volume get dire-zero maps .`
python tools/make_gallery.py \
    --maps  "$VOL/maps/test" \
    --orig  "$VOL/data/test" \
    --out   platforms/hf-spaces/gallery \
    --n 8 --steps 20 --label R1+R2 \
    --map-source "maps/test/*.npy · ::dire --steps 20"
```

Cách 1 mount **hai** thứ vào container: `tools/make_gallery.py` và
`platforms/hf-spaces/demolib/` (đặt ở `/root/platforms/hf-spaces/demolib` — đúng
chỗ `make_gallery.py` thêm vào `sys.path`). Mount thiếu thư viện thì container
chết ngay ở dòng `from demolib import provenance`; `tests/test_make_gallery.py`
dựng lại bố cục đó và chạy thật để chặn tái phát.

`--orig` trỏ tới `data/<split>/<generator>/`, **không** phải `images/…`: ảnh gốc
được `::dire` đọc từ `data/` (cây chuẩn hoá), và đó là nơi duy nhất có tên tệp
khớp với tên `.npy`.

Map do `::dire` ghi có shape **`(3, 256, 256)`** — ba kênh của `|x − x̂|`, không
phải ảnh xám. Tool vẽ nguyên ba kênh thành PNG RGB (`<tên>_diff.png`), không trộn
thành xám: mỗi kênh là một phép đo riêng, trộn lại là bịa thêm một đại lượng không
có trong `.npy`. Đổi lại, `(256, 256)` và `(1, 256, 256)` vẫn được nhận (map một
kênh), và con số trong `.json` (`display.channels`) nói rõ ảnh đang xem có 1 hay 3
kênh.

Thêm `--scores scores.json --gate passed` chỉ khi cổng 2 (`docs/demo-web-plan.md`
§5) đã chạy; mặc định `--gate failed` nên **không** điểm nào được hiện.

## Deploy (0 ₫)

```bash
# Đặt ba biến này trước — đừng gõ `<user>`/`<repo>`: bash hiểu `<...>` là chuyển
# hướng stdin và báo "No such file or directory", lệnh thật không bao giờ chạy.
export HF_USER=jackboom2510            # tài khoản Hugging Face của bạn
export SPACE=dire-demo                 # tên Space (sẽ tạo)
export REPO="$HOME/dire-modal"         # thư mục repo này

# 0. Kiểm máy + hardware của Space (1 giây) — bắt đúng hai lỗi bên dưới
python "$REPO/tools/hf_space_preflight.py" --space "$HF_USER/$SPACE"

# 1. Space mới (SDK gradio, hardware CPU basic)
hf repos create "$HF_USER/$SPACE" --repo-type space --space-sdk gradio
# `hf repo` (số ít) vẫn chạy nhưng đã deprecated — dùng `hf repos`.

# 2. Đẩy nội dung thư mục này lên
git clone "https://huggingface.co/spaces/$HF_USER/$SPACE" /tmp/space && cd /tmp/space
cp -r "$REPO/platforms/hf-spaces/." .
git add -A && git commit -m "DIRE demo (gradio, CPU free)" && git push

# 3. Kiểm trước khi push (2 giây, tránh đẩy nhầm lên GitHub)
git remote -v      # phải là huggingface.co/spaces/$HF_USER/$SPACE
```

Lệnh cuối không thừa: nếu bạn từng gõ `cp -r "$REPO/." .` (thiếu
`platforms/hf-spaces`), `origin` trong `/tmp/space` sẽ là
`github.com/jackboom2510/dire-modal` và `git push` đẩy lên **repo gốc**. Thấy vậy
thì `rm -rf /tmp/space` rồi clone lại — đừng `git remote set-url`, vì `.git` đó
còn mang toàn bộ lịch sử của `dire-modal`.

### `hf` chết với `FileNotFoundError: [Errno 2]` — không phải lỗi token

Traceback kết thúc ở `ssl.py` / `httpx/_config.py`, **không có tên tệp**:

```text
File "...\httpx\_config.py", line 35, in create_ssl_context
    ctx = ssl.create_default_context(cafile=os.environ["SSL_CERT_FILE"])
FileNotFoundError: [Errno 2] No such file or directory
```

`httpx` đọc biến `SSL_CERT_FILE` khi dựng SSL context. Biến **có** nhưng trỏ tới
tệp không mở được — thường là `/mingw64/ssl/certs/ca-bundle.crt` do Git Bash đặt
(Python bản Windows đọc thành `C:\mingw64\...`), hoặc một `cacert.pem` đã mất khi
nâng cấp conda. Không liên quan tới token, quyền, hay `huggingface_hub`.

```bash
# Git Bash
unset SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE SSL_CERT_DIR
# cmd.exe
set SSL_CERT_FILE=
# PowerShell
Remove-Item Env:SSL_CERT_FILE,Env:REQUESTS_CA_BUNDLE,Env:CURL_CA_BUNDLE -ErrorAction SilentlyContinue
```

Hoặc trỏ cả ba biến vào một bundle có thật:
`python -c "import certifi; print(certifi.where())"`. Nếu biến nằm trong `.bashrc`
hay biến hệ thống thì sửa ở đó — không thì mở terminal mới là lỗi lại.

Không muốn sửa gì: tạo Space bằng web UI **không đi qua httpx của máy** —
<https://huggingface.co/new-space> (SDK *Gradio*, hardware *CPU basic*), rồi
`git clone`/`git push` như bước 2 (git dùng CA riêng của nó, không đọc biến trên).

### `No @spaces.GPU function detected during startup` — không phải lỗi app

Space báo:

```text
Runtime Error
No @spaces.GPU function detected during startup
```

ZeroGPU (hardware `zero-a10g`) **từ chối khởi động** khi trong Space không có hàm
nào trang trí `@spaces.GPU`. Nó kiểm tra điều đó *trước khi* import
`gradio_app.py`, nên log container không có traceback, không nhắc tới code — chỉ
có đúng một dòng này.

App này chạy trên CPU; GPU chỉ được dùng **qua Modal** (tab 2, mặc định tắt). Vì
vậy cấu hình đúng là CPU basic, không phải ZeroGPU:

```text
Space → Settings → Space hardware → CPU basic (free) → Factory reboot
```

Hoặc bằng CLI (cần token):

```bash
python "$REPO/tools/hf_space_preflight.py" --space "$HF_USER/$SPACE" --fix-hardware
```

Thêm `@spaces.GPU` cho có là **sai**: nó khiến ZeroGPU cấp GPU mà app không dùng,
và che mất sự thật rằng Space này không cần GPU. `suggested_hardware: cpu-basic`
trong front-matter chỉ là gợi ý khi ai đó duplicate Space; hardware thật vẫn phải
đặt trong *Settings*.

`python tools/hf_space_preflight.py --space "$HF_USER/$SPACE"` kiểm bốn thứ —
biến CA, token, `git`/`hf`, và hardware của Space — rồi in lệnh sửa cho từng
cái. Bỏ `--space` thì chỉ kiểm máy.

`README.md` của thư mục này là front-matter của Space: `sdk: gradio`,
`app_file: gradio_app.py`. Nếu HF báo `sdk_version 6.32.0` không hợp lệ, chọn bản
6.x mới nhất trong *Settings → Gradio* — app chỉ dùng API ổn định từ 5.x
(`Blocks`, `Tab`, `Gallery`, `Image(sources=…)`, `.release/.change`, `gr.JSON`),
nên không phụ thuộc một bản vá cụ thể.

Muốn Space chạy **bản Streamlit** thay thế: sửa front-matter thành
`sdk: streamlit` + `app_file: app.py`, và thêm `streamlit==1.67.0` vào
`requirements.txt`.

Chi tiết, số đo cold start, cổng kiểm và phương án rollback:
**`docs/demo-hf-spaces-plan.md`**.

## Đọc log của Space: dòng nào là bình thường

Một lần khởi động **thành công** in đúng bốn dòng này (không có gì khác):

```text
WARNING: Running pip as the 'root' user can result in broken permissions ...
[demo] 65 component · 64 mẫu gallery · sẵn sàng cho launcher
* Running on local URL:  http://0.0.0.0:7860
* To create a public link, set `share=True` in `launch()`.
```

| Dòng | Nghĩa | Phải làm gì |
|---|---|---|
| `WARNING: Running pip as the 'root' user` | HF cài `requirements.txt` bằng pip **trong container**, nơi user là root. Đây là cảnh báo của HF, không phải của app | không gì cả. Không tắt được từ repo: `pip` **từ chối** `--root-user-action` khi nó nằm trong `requirements.txt` (`no such option`) |
| `[demo] … sẵn sàng cho launcher` | app dựng xong (65 component), `demo` ở cấp module đã sẵn cho launcher | — |
| `* Running on local URL: http://0.0.0.0:7860` | uvicorn đã nghe trên cổng 7860, cổng HF chờ. Chữ *local* là cách Gradio gọi URL của chính nó, **không** có nghĩa "chỉ chạy ở máy" | — |
| `* To create a public link, set share=True` | Gradio in câu này mỗi khi `share=False`. Trên Space **đừng** đặt `share=True`: nó mở thêm tunnel `gradio.live`, chậm và có hạn một tuần | không gì cả |

Chỉ những dòng sau là lỗi thật, và mỗi dòng có một nguyên nhân riêng:

| Dòng | Nguyên nhân |
|---|---|
| `GRADIO_HOT_RELOAD: Launching demo not found` | thiếu `demo` ở cấp module — xem mục dưới |
| `ModuleNotFoundError` | `requirements.txt` thiếu gói, hoặc import một tệp không được đóng gói |
| `Traceback (most recent call last)` | app ném lỗi lúc import; đọc dòng cuối, đó là nơi sửa |
| `Stopping Node.js server...` | SSR bật mà app không phục vụ; app đã tắt SSR (`GRADIO_SSR_MODE=False`) |

Kiểm Space đang sống, không cần mở trình duyệt — `$SPACE` đặt như ở §Deploy:

```bash
curl -s -o /dev/null -w "%{http_code}\n" "https://$HF_USER-$SPACE.hf.space/config"
```

`200` và body là JSON của app nghĩa là đang chạy; `503` là đang build/ngủ; `404` là sai tên.

## Nếu Space bật lên rồi tắt ngay

Log kiểu này **không** phải app hỏng:

```text
GRADIO_HOT_RELOAD: Launching demo not found in __main__. Using 'demo'
* Running on local URL: http://0.0.0.0:7860, with SSR ⚡ (Node proxy -> Python :7861)
Stopping Node.js server...
```

HF (SDK gradio) **import** `app_file`, rồi tìm một object tên **`demo`** ở cấp
module để tự gọi `launch()`. Nếu tệp chỉ dựng app bên trong
`if __name__ == "__main__":`, launcher không có gì để phục vụ: nó rơi vào nhánh
dự phòng, bật SSR proxy, rồi tiến trình Python kết thúc và proxy chết theo. Ứng
dụng *không* hỏng — nó chưa bao giờ được giao cho launcher.

Điều kiện bắt buộc, đã có trong `gradio_app.py`:

```python
demo = build_app()          # cấp module, KHÔNG nằm trong `if __name__`
if __name__ == "__main__":  # chỉ dùng khi chạy tay ở máy
    demo.launch(...)
```

Nếu vẫn thấy `with SSR ⚡` và muốn bỏ hẳn Node proxy, đặt biến `GRADIO_SSR_MODE`
= `false` trong *Settings → Variables* (app đã `setdefault` sẵn giá trị này, biến
của Space chỉ cần khi bạn muốn bật lại).

## Cấu hình

| Biến | Ý nghĩa | Mặc định |
|---|---|---|
| `DIRE_GALLERY_DIR` | thư mục gallery, tuyệt đối hoặc tương đối với `app.py` | `gallery` |
| `DIRE_CKPT` | đường dẫn checkpoint ADM (bật tab DIRE **tại chỗ** của bản Streamlit); cần thêm `torch==2.4.0` | `ckpt/adm_lsun.pt` |
| `DEMO_URL` | địa chỉ endpoint GPU trên Modal, dạng `https://$WS--dire-demo-api.modal.run` (`$WS` = workspace Modal của bạn) — bật tab DIRE **qua GPU** | *(trống = tắt)* |
| `DEMO_TOKEN` | token của endpoint; **chỉ đặt trong Secrets**, không bao giờ gửi xuống trình duyệt | *(trống)* |

**Không có secret nào là bắt buộc.** Thiếu `DEMO_URL`/`DEMO_TOKEN` thì Space chạy
y như trước: tab *Tính DIRE thật* hiện lời giải thích thay vì nút chạy, 0 đồng,
không gọi mạng ngoài.

Đặt `DEMO_URL` + `DEMO_TOKEN` (Settings → Secrets) thì trang 3 gọi endpoint GPU:
`S = 5` còn **0,92 s** thay vì 101 s. Trình duyệt **không** thấy token — nó gọi
server của Space, server gọi Modal. Dựng endpoint: `docs/demo-modal-plan.md`
(`modal secret create dire-demo …` rồi `modal deploy -m dire_zero.demo_api`).

Ảnh người dùng vẫn không bị lưu ở đâu: Space chỉ chuyển bytes sang Modal, endpoint
tính rồi trả PNG, không ghi đĩa.

## Nguồn & bản quyền

- Bài báo / code gốc: <https://github.com/ZhendongWang6/DIRE>
- Ảnh trong gallery (khi có): DiffusionForensics (Wang et al., ICCV 2023),
  chỉ dùng cho mục đích nghiên cứu; xem `docs/runbook-free.md` §License.
