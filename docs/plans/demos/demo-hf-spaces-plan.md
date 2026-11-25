# Kế hoạch chạy demo trên HF Spaces (Gradio)

- **Trạng thái:** kế hoạch + app đã viết và chạy thử · **Ngày:** 2026-11-17 (cập nhật 2026-10-14: quay lại Gradio)
- **UI hiện tại:** `platforms/hf-spaces/gradio_app.py` (Gradio, 5 tab) — bản
  Streamlit cũ vẫn còn ở `app.py` và vẫn chạy được; xem §0b.
- **Liên quan:** `docs/plans/demos/demo-web-plan.md` (§2 lỗi phải sửa, §3 ba chế độ, §5 cổng
  kiểm), `platforms/hf-spaces/` (mã), `tools/make_gallery.py` (dựng gallery),
  `docs/from-zero-clean-room.md` (§3.6 về `--n` và ước lượng tiền), ADR-0003
- **Câu hỏi của mục này:** *đưa demo lên HF Spaces bằng Streamlit, chạy được, 0 ₫,
  và không nói dối — cần những gì, theo thứ tự nào, kiểm bằng gì.*

---

## 0b. Đổi UI về Gradio (2026-10) — cái gì đổi, cái gì **không**

Yêu cầu: quay lại demo dùng **Gradio** thay cho Streamlit. Đã làm, và giới hạn
quan trọng là: **chỉ đổi cách bày, không đổi một con số nào**.

| Đổi | Cụ thể |
|---|---|
| UI | `gradio_app.py`, 5 tab; front-matter của Space là `sdk: gradio`, `app_file: gradio_app.py` |
| Phụ thuộc | `requirements.txt` = `gradio==6.32.0`; Streamlit chuyển sang `requirements-streamlit.txt` |
| Lý do chọn Gradio | hàng đợi, trạng thái phiên và nút *Restart* do SDK lo; bản Streamlit phải tự xử ba thứ đó |
| Bản Streamlit | **giữ**, ở `app.py`; đổi một dòng front-matter là quay lại được |

**Không** đổi:

- **Tab upload ảnh → `P(fake)` vẫn bị bỏ.** Đây là lỗi của bản Gradio *đầu tiên*,
  và là lý do nó từng bị thay (`demo-web-plan.md` §2). Tab *Tính DIRE thật* của bản
  mới nhận ảnh nhưng chỉ trả **DIRE map**, kèm câu "đây là map, không phải kết luận
  thật/giả". Có test chặn: `test_the_upload_path_returns_a_map_never_a_probability`.
- Cổng hiện điểm vẫn là `provenance.Sample.score()`, không phải `if` viết trong
  giao diện — test `test_gallery_page_uses_the_shared_gate_not_its_own_if` so hai
  trạng thái cổng trên cùng một `.json`.
- Hai UI dùng chung `demolib/`, nên thí nghiệm 2 chiều, lịch/chi phí và gallery
  cho ra **cùng** số.

Số đo trên máy này (Python 3.11, `gradio==6.32.0`):

| Phép đo | Kết quả |
|---|---|
| Kích thước `site-packages/gradio` | **82 MB** (66 MB là `templates/`) — vẫn nhỏ hơn torch ~200 MB |
| `import gradio` | **2,17 s** (Streamlit: 0,27 s) |
| `build_app()` | **0,48 s** |
| Test của bản Gradio | **9 test** (`tests/test_hf_spaces_gradio.py`) |

Cold start thật của Space **chưa đo** — như §4, chỉ biết sau lần deploy đầu; đừng
trích dẫn con số nào khác.

## 0. Kết luận trước, chi tiết sau

| Quyết định | Chốt | Vì sao |
|---|---|---|
| Framework | **Streamlit 1.67.0**, một phụ thuộc duy nhất | Space CPU free; UI là 5 trang tĩnh + vài widget; Gradio cũ không cho kiểm bằng `pytest` mà không dựng app |
| Có `torch` trên Space? | **Không** | Space không chấm ảnh và không tính DIRE; bỏ torch cắt ~800 MB cài đặt và ~40 s cold start |
| Có ô upload ảnh? | **Không** | đường "ảnh RGB → `P(fake)`" là lỗi bị ghi ở `demo-web-plan.md` §2, đã bỏ hẳn |
| Điểm detector lấy từ đâu? | `.json` của từng mẫu, chỉ hiện khi `gate == "passed"` và ngưỡng chọn trên `val` | để "chưa qua cổng" hiện thành chữ, không thành một con số |
| Phép đo chạy thật trên Space | thí nghiệm khuếch tán 2 chiều có score chính xác | 61 ms/lượt, không cần GPU, và là chỗ giải thích *cơ chế* |
| Tính DIRE thật trên CPU | **có, nhưng mặc định tắt** — `S=5` là **101 s**/ảnh | đo được 10,1 s/forward trên 2 vCPU; `demo-web-plan.md` §3 chốt > 45 s thì tắt (mục 13) |
| Tính DIRE thật trên GPU | **có, qua Modal** — `S=5` là **0,92 s**, 0,0003 USD/ảnh | token ở server Space, không xuống trình duyệt; `docs/plans/demos/demo-modal-plan.md` |
| Chi phí | **0 ₫** (hardware `cpu-basic`) | không có GPU, không gọi mạng ngoài |

Ba con số quyết định kế hoạch này, **đo trên máy** (mục 4): cold start 0,57 s ·
RAM 60 MB · toàn bộ 5 trang render dưới 1,5 s.

## 1. Cái gì thay đổi so với app Gradio cũ

| | Bản Gradio (`app.py` cũ, 216 dòng) | Bản Streamlit (hiện tại) |
|---|---|---|
| Tab 1 | upload ảnh → `P(real)/P(fake)` bằng ResNet-50 học trên DIRE | **bỏ** |
| Tab 2 | 3 panel từ `gallery/` + hướng dẫn nếu rỗng | giữ, thêm dòng nguồn gốc và cổng kiểm |
| Nạp model | `hf_hub_download` + `torch.load` lúc khởi động | không có model (trừ trang 3, mặc định tắt) |
| Phụ thuộc | gradio, torch, torchvision, pillow, numpy | `streamlit` |
| Kiểm bằng test | đọc mã nguồn (chuỗi) | đọc mã nguồn **và** chạy thật bằng `AppTest` |
| Số đo trên Space | không có | 6 trang · 4 kịch bản gallery · 1 phép đo CPU thật (mục 13) |

Điều **không** đổi: không có đường nào trả xác suất từ ảnh RGB; gallery rỗng thì
hiện lệnh, không hiện khung trống; mọi số kèm nguồn.

## 2. Kiến trúc

```
platforms/hf-spaces/                 ← đúng thư mục này được đẩy lên Space
├── app.py              giao diện: 5 trang, không quyết định con số nào
├── demolib/
│   ├── toy2d.py        mô hình 2 chiều, score chính xác, 2 vòng lặp DIRE
│   ├── provenance.py   đọc .json, quyết định khi nào được hiện p_fake
│   ├── charts.py       SVG thuần (histogram, scatter, sweep)
│   └── cost.py         công thức chi phí, khoá bằng test với dire_zero.budget
├── gallery/            <tên>_orig.png · <tên>_diff.png · [<tên>_recon.png] · <tên>.json
├── requirements.txt    streamlit==1.67.0
└── README.md           YAML front-matter của Space + hướng dẫn
```

Ranh giới quan trọng nhất: **Space không tính DIRE**. DIRE map được tính trên
Modal (`::dire`), rồi `tools/make_gallery.py` biến `.npy` thành PNG chỉ để xem và
ghi `.json` nguồn gốc. Lý do: 40 lượt U-Net 552 M tham số trên 2 vCPU là hàng
phút mỗi ảnh — không phải "chậm", mà là "không dùng được".

```
::dire (A10G)                tools/make_gallery.py            HF Space (CPU)
maps/test/{real,adm}/*.npy → orig.png + diff.png + .json  →  app.py đọc gallery
   float16 [0, 2]              sha256, S, nhãn R, nguồn        chỉ hiện, không tính
```

Hợp đồng giữa hai nửa là lược đồ `.json`; nó được kiểm bằng cách chạy chính bộ
đọc của app lên kết quả của tool (`tests/test_make_gallery.py`), không phải bằng
đọc mắt.

## 3. Ba chế độ của `demo-web-plan.md` §3, nay ánh xạ thế nào

| | A — Gallery | B — Upload tự tính | C — GPU backend |
|---|---|---|---|
| Trạng thái | **bật** (trang 2) | **tắt** | **không làm** |
| Vì sao | không cần tính gì lúc chạy | p95 trên CPU chưa đo; `demo-web-plan.md` §3 đòi > 45 s thì tắt | cần Modal + token + rate limit, ngoài phạm vi 0 ₫ |
| Cần gì để bật | `make_gallery.py` chạy trên volume có map | số đo `S = 5/10/20` trên 2 vCPU | bạn đồng ý trả tiền |
| Nhãn trên UI | `R0` / `R1+R2` + `S` | — | — |

Cổng quyết định chế độ B giữ nguyên như đã chốt: `< 15 s` bật mặc định `S = 5`;
`15–45 s` chỉ sau nút bấm; `> 45 s` tắt. **Chưa đo**, nên chưa bật.

## 4. Số đo (máy này, không phải suy đoán)

Thiết bị: container Linux, Python 3.11.2, `streamlit==1.67.0`, không GPU.

| Phép đo | Kết quả | Cách đo |
|---|---|---|
| Cài đặt sạch (`pip install streamlit==1.67.0`) | **19,0 s** · **472 MB** trong `site-packages` | venv trắng, `time pip install` |
| `import streamlit` | **0,27 s** · RSS **43 MB** | `time` + `ru_maxrss` |
| Script chạy lần đầu (boot) | **0,57 s** | `AppTest.run()` |
| Trang 1 · 2 · 4 · 5 | **0,02 · 0,02 · 0,03 · 0,03 s** | render lại sau khi đổi radio |
| Trang 3 (thí nghiệm) | **1,33 s** lần đầu, ~0 s khi đã cache | 7 lượt quét `S`, `n = 200`/lớp |
| `toy2d.run(S=20, n=200)` | **61 ms** | một lượt đo đầy đủ |
| Tiến trình server khi rỗng | RSS **60 MB** | `ps` sau 2 request |
| Bộ test demo | **68 test** (48 giao diện/số học + 11 chế độ tính thật + 9 gallery) | `pytest tests/test_hf_spaces_{demo,live}.py tests/test_make_gallery.py` |
| Một mẫu gallery | **~150–250 KB** (orig 100–190 KB + diff 30–65 KB) | PNG 256², nén tối ưu |

**Chưa đo (và không được nói như đã đo):** thời gian build image trên HF, cold
start của Space thật (phụ thuộc cache của HF), RAM đỉnh khi 5 người xem cùng lúc.
Ba thứ đó chỉ biết sau lần deploy đầu; mục 8 nói cách ghi lại.

Vì sao vẫn đáng tin: phần lớn cold start của một Space Streamlit là cài đặt phụ
thuộc, và ở đây chỉ có một phụ thuộc. Bản Gradio cũ kéo theo torch — riêng wheel
torch CPU đã ~200 MB vài chục giây cài.

## 5. Ngân sách tài nguyên của Space

| Tài nguyên | Gói CPU basic (theo trang cài đặt Space) | Demo dùng |
|---|---|---|
| vCPU | 2 | 1 (Streamlit đơn luồng khi render) |
| RAM | 16 GB | ~60 MB thường trực, ~120 MB đỉnh khi render trang 3 |
| Đĩa | 50 GB tạm | 472 MB cài đặt + vài MB gallery |
| Thời gian ngủ | sau 48 h không có truy cập (gói free) | chấp nhận; người xem đầu tiên chờ cold start |

Con số của gói có thể đổi; kiểm lại ở *Settings → Hardware* của Space chứ đừng
tin bảng này. Cái demo **tự bảo đảm** là dùng ít: không model, không GPU, không
ghi đĩa.

## 6. Cấu hình Space

`README.md` của thư mục này chính là front-matter của Space:

```yaml
sdk: streamlit
sdk_version: 1.67.0
app_file: app.py
```

| Cần đặt | Giá trị | Ghi chú |
|---|---|---|
| SDK | Streamlit | đừng chọn Gradio: `app.py` không còn `gr.Blocks` |
| Hardware | CPU basic | GPU không cần và không được bật |
| Biến môi trường | không bắt buộc | `DIRE_GALLERY_DIR` chỉ để trỏ chỗ khác khi thử |
| Secrets | **không có** | Space không gọi dịch vụ nào; không có gì để rò |
| Persistent storage | **không** | gallery nằm trong git, đúng như nó phải thế |

## 7. Triển khai, theo thứ tự

```bash
# Đặt biến trước. Trong bash, `<volume>`/`<user>` KHÔNG phải chỗ trống: `<` là
# chuyển hướng stdin, nên lệnh chết với "volume: No such file or directory" và
# python không bao giờ chạy.
export REPO="$HOME/dire-modal"
export HF_USER=jackboom2510
export SPACE=dire-demo
export VOL=/c/Users/ban/dire-vol    # chỉ cần nếu dựng gallery ở máy (cách 2)

# 1. Dựng gallery — cách 1: ngay trên Volume, không cần đường dẫn
modal run -m dire_zero.modal_app::gallery --n 8 --steps 20
modal volume get dire-zero gallery "$REPO/platforms/hf-spaces/gallery"

#    cách 2: ở máy, khi map + ảnh đã tải về ($VOL/maps, $VOL/data)
python "$REPO/tools/make_gallery.py" \
    --maps "$VOL/maps/test" --orig "$VOL/data/test" \
    --out "$REPO/platforms/hf-spaces/gallery" \
    --n 8 --steps 20 --label R1+R2 \
    --map-source "maps/test/*.npy · ::dire --steps 20"
# không thêm --scores cho tới khi cổng 2 (demo-web-plan.md §5) đã chạy

# 2. Kiểm tại chỗ, chưa cần mạng
cd "$REPO/platforms/hf-spaces" && python gradio_app.py     # hoặc: streamlit run app.py
python -m pytest "$REPO/tests/test_hf_spaces_gradio.py" "$REPO/tests/test_make_gallery.py" -q

# 3. Tạo Space (SDK gradio, hardware CPU basic)
python "$REPO/tools/hf_space_preflight.py" --space "$HF_USER/$SPACE"   # máy + hardware
#    Space đang ở ZeroGPU ⇒ tool báo ĐỎ và in lệnh `--fix-hardware`
hf repos create "$HF_USER/$SPACE" --repo-type space --space-sdk gradio

# 4. Đẩy nội dung
git clone "https://huggingface.co/spaces/$HF_USER/$SPACE" /tmp/space
cp -r "$REPO/platforms/hf-spaces/." /tmp/space/
cd /tmp/space && git remote -v && git add -A && git commit -m "DIRE demo (gradio)" && git push

# 5. Chờ build; xem log ở tab *Logs* của Space
```

**Nếu bước 3 chết với `FileNotFoundError: [Errno 2]`** ở `ssl.py` /
`httpx/_config.py`: đó là `SSL_CERT_FILE` trỏ tệp không tồn tại (Git Bash đặt
`/mingw64/ssl/...`, hoặc conda đổi đường). Không phải token. Sửa:

```bash
unset SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE SSL_CERT_DIR   # Git Bash
```

rồi chạy lại, hoặc tạo Space bằng web UI (<https://huggingface.co/new-space>,
SDK Gradio, CPU basic) — đường đó không đi qua `httpx` của máy. Chi tiết và ba
cách sửa: `platforms/hf-spaces/README.md` §Deploy.

**Nếu Space chết với `No @spaces.GPU function detected during startup`**: Space
đang đặt hardware `zero-a10g`. ZeroGPU từ chối khởi động khi không có hàm
`@spaces.GPU`, và nó từ chối *trước khi* import `gradio_app.py` — nên log không
có traceback. Đây không phải lỗi code: app chạy CPU và gọi GPU qua Modal. Sửa
bằng *Settings → Space hardware → CPU basic (free) → Factory reboot*, hoặc:

```bash
python "$REPO/tools/hf_space_preflight.py" --space "$HF_USER/$SPACE" --fix-hardware
```

Một chi tiết CLI: `hf repo` (số ít) đã deprecated, `hf repos` là lệnh hiện hành.

Thứ tự này có một điểm không được đảo: **gallery trước, Space sau**. Đẩy một
Space có `gallery/` rỗng vẫn hợp lệ (trang 2 hiện lệnh), nhưng nếu đẩy gallery
sau thì mỗi lần cập nhật là một lần rebuild, và rebuild là lúc dễ hỏng nhất.

## 8. Kiểm sau deploy (làm ngay, đừng để "hôm sau")

| # | Việc | Đạt khi |
|---|---|---|
| 1 | Mở URL, chờ cold start | trong **60 s**; ghi lại số thật vào bảng mục 4 |
| 2 | Bấm đủ 5 trang | không trang nào hiện traceback |
| 3 | Trang 2 | số mẫu = số `.json` đã commit; **không** có "Gallery đang rỗng" |
| 4 | Trang 2, một mẫu | có dòng `DIRE: … · detector: … · nhãn: … · S=…` |
| 5 | Trang 3, kéo `S` 5 → 50 | tỉ lệ tăng; số trên hình khớp bảng dưới |
| 6 | Trang 4 | đổi "Số ảnh" 800 → 1000 thì NFE và USD đổi theo |
| 7 | Ctrl-F "99" | không có chỗ nào nói demo đạt 99 % |
| 8 | Tab Network | **không** có request ra ngoài `*.hf.space` |

Điểm 8 là điểm dễ bỏ sót nhất: Streamlit tự gọi telemetry. Nếu thấy request lạ,
thêm `[browser] gatherUsageStats = false` vào `.streamlit/config.toml`. (Bản
đang chạy thử ở đây đã tắt bằng cờ dòng lệnh.)

## 9. Vận hành

| Tình huống | Làm gì |
|---|---|
| Space ngủ sau 48 h | không sửa gì; người xem đầu tiên chờ cold start. Muốn tránh: gói trả tiền, **không** làm |
| Cập nhật gallery | commit vào Space repo, build lại; giữ nguyên tên mẫu để link cũ không chết |
| Space lỗi build | xem *Logs*; 90 % là `requirements.txt` bị thêm gói, hoặc `app.py` import thứ không có trong Space (ví dụ `dire_zero`) |
| Rollback | `git revert` commit gần nhất trong Space repo rồi push; hoặc *Factory reboot* để xoá cache build |
| Đổi `S` của gallery | phải dựng lại **toàn bộ** gallery: `S` là một phần của phép đo, không phải nhãn trang trí |

Nguyên tắc: Space là **bản chiếu** của dữ liệu trong git. Không sửa tay trên
Space, không upload trực tiếp qua UI — lần build sau sẽ xoá sạch.

## 10. Rủi ro, và cái chặn tương ứng

| Rủi ro | Dấu hiệu | Chặn bằng |
|---|---|---|
| Ai đó "sửa" lại thành upload ảnh | `st.file_uploader` xuất hiện | `test_app_has_no_upload_path`, `test_hf_space_has_no_classifier_to_misload` |
| `p_fake` hiện khi chưa qua cổng | metric khác `—` | `provenance.score()` + 4 test cho từng điều kiện từ chối |
| Số của paper bị đọc thành số của demo | "99 %" xuất hiện | `test_app_never_claims_paper_accuracy` (chặn cả chuỗi) |
| Gallery lệch với map | `.json` qua được nhưng ảnh khác | `sha256` của `.npy` và của PNG, ghi trong `.json` |
| Tool ghi `.json` sai lược đồ | Space hiện 0 mẫu dù có 16 tệp | tool tự gọi `provenance.discover()` sau khi ghi, exit 2 nếu lỗi |
| Chi phí hai nơi hai số | demo và pipeline lệch | `cost.estimate` so từng trường với `dire_zero.budget.Estimate` |
| Cold start dài | > 60 s | đã bỏ torch; nếu vẫn dài thì đo lại và ghi vào mục 4, không đoán |
| Dữ liệu không có quyền | ảnh lạ trong gallery | chỉ dùng DiffusionForensics, ghi nguồn trong caption |

## 11. Việc còn lại, xếp theo thứ tự làm

| # | Việc | Chặn bởi | Ai làm |
|---|---|---|---|
| 1 | Chạy `::dire` (đang chạy, ~50 phút) rồi `make_gallery.py` | GPU + volume | phiên này |
| 2 | Cổng 2: chấm `dire/val/lsun_bedroom/`, so Table 2 | cần checkpoint | D1 của kế hoạch cũ |
| 3 | `--scores` + `--gate passed` để trang 2 có điểm | #2 | sau #2 |
| 4 | Đo `sec/ảnh` trên 2 vCPU cho `S = 5/10/20` | máy thật | D4, quyết định chế độ B |
| 5 | Deploy Space thật, ghi cold start vào mục 4 | #1 | sau #1 |

## 13. Tính DIRE thật trong demo — đo rồi mới quyết

Câu hỏi đặt ra sau khi demo chạy: *DIRE có thật sự lâu vậy không, và không thể cho
người xem thấy quá trình tính sao.* Trả lời bằng số đo, không bằng cảm giác.

### 13.1 Số đo trên 2 vCPU (đúng cấu hình `cpu-basic`)

Máy đo: Xeon 2,6 GHz, **2 vCPU**, 3,9 GB RAM, AVX-512, `torch==2.4.0`, fp32,
batch 1, `torch.set_num_threads(2)`, kiến trúc ADM 256² **thật** (552,8 M tham số):

| Phép đo | Kết quả |
|---|---|
| 1 lượt forward 256², fp32 | **10,1 s** |
| 1 lượt forward 256², bf16 | 34,0 s (**chậm hơn 3,4×** — CPU không có AVX-512-BF16) |
| 1 lượt forward, `channels_last` | 10,9 s (không lợi) |
| 1 lượt forward ở 128² | 2,6 s (**không dùng được**: ADM chỉ học ở 256²) |
| RSS đỉnh khi chạy ADM fp32 | **3,3 GB** (Space có 16 GB — đủ) |
| DIRE `S = 1` (2 forward) | **20 s** |
| DIRE `S = 5` (10 forward) | **101 s** |
| DIRE `S = 20` (40 forward) | **404 s** |
| Cùng phép tính trên A10G (`::bench`) | 0,18 / 0,92 / 3,7 s ⇒ **CPU chậm ~110×** |

`torch.compile` **chưa đo được** ở máy đó (thiếu `Python.h` để inductor biên dịch);
không hứa gì về nó.

### 13.2 Quyết định: bật, nhưng mặc định tắt, và nói rõ giá

`demo-web-plan.md` §3 đã chốt trước khi đo: p95 > 45 s thì **tắt** chế độ tự tính.
Đo được **101 s** cho `S = 5` ⇒ vượt ngưỡng 2,2 lần. Nên:

* **mặc định tắt** trên Space (`demolib.live.status()` trả lý do + cách bật);
* bật được bằng `DIRE_CKPT` + `torch` trong `requirements.txt`, và khi bật thì
  trang **3 · Tính DIRE thật** hiện từng bước: ở mỗi mốc `t`, `x̂₀`, `ε̂`, thời
  gian từng bước, và `|x₀ − x̂₀|` lớn lên/nhỏ đi — quá trình thật, không phải thanh
  chạy giả;
* chặn cứng `S ≤ 20`, và `S > 5` phải tick "tôi chấp nhận chờ ~N giây".

Đổi lại khi bật: `requirements.txt` thêm torch (~800 MB, build lâu hơn ~2–4 phút),
RAM đỉnh 3,3 GB, và **mỗi lần bấm là 1,7 phút CPU của Space**. Trên gói free, một
người bấm 10 lần là 17 phút — đủ để Space bị coi là treo.

**Đường thứ ba, đã dựng xong:** nếu Space có `DEMO_URL` + `DEMO_TOKEN` trong
Secrets thì trang 3 **không nạp ADM tại chỗ** nữa mà gọi endpoint GPU trên Modal
(`demolib/remote.py`, token chỉ nằm ở server Space). `S = 5` còn **0,92 s**, vẫn
thấy map + ảnh tái tạo, và câu nguồn gốc ghi rõ "Modal A10G". Không có secret ⇒
hành vi y như mục 13.2 (mặc định tắt, hướng dẫn cách bật). Chi tiết và chi phí:
`docs/plans/demos/demo-modal-plan.md`.

### 13.3 Ba cách để *thấy* phép tính, xếp theo thời gian chờ

| Cách | Chờ bao lâu | Thấy được gì | Nhãn |
|---|---|---|---|
| Thí nghiệm 2 chiều (trang 4) | **0,06 s** | cùng hai vòng lặp, từng bước, trên mô hình có score chính xác | minh hoạ lý thuyết |
| Gallery (trang 2) | 0 s | map của ADM trên ảnh LSUN, cạnh điểm detector | `R0` / `R1+R2` |
| Tính tại chỗ trên CPU (trang 3) | **101 s** (`S=5`) | ADM thật, `x̂₀` từng bước, map cuối | không áp dụng (chỉ tính map) |
| **Modal A10G** (trang 3, có secret) | **0,92 s** (`S=5`) | như trên, nhanh hơn **110×**, 0,0003 USD/ảnh | `R1` (map tự tính) |

Kết luận cho buổi nói: dùng **trang 4** để giải thích cơ chế (nó chạy 61 ms), dùng
**trang 2** để nói về kết quả trên dữ liệu thật, và **một lần** bấm trang 3 để
người xem thấy ADM thật đang chạy — rồi nói thẳng "cùng phép tính này, trên 2 vCPU
là 101 giây, trên A10G là 0,92 giây". Nếu có endpoint Modal thì chính lần bấm đó
chạy trên GPU (`docs/plans/demos/demo-modal-plan.md`); muốn khán giả thấy giá của CPU thì tắt
secret `DEMO_URL` trước buổi nói.

### 13.4 Chỗ dễ tự lừa, và test chặn

| Cái bẫy | Vì sao nguy hiểm | Chặn bằng |
|---|---|---|
| chạy 128² cho nhanh 4× | ADM chỉ học ở 256² ⇒ map đẹp nhưng vô nghĩa | `live` luôn resize về 256; test `test_wrong_size_image_is_resized...` |
| viết lại vòng lặp rồi lệch số | demo sẽ "chứng minh" một phép tính khác bài báo | `test_stepwise_loop_matches_dire_zero`: map **bằng nhau** với `dire_map` |
| bật mặc định cho "tiện" | người xem đầu tiên chờ 101 s và bỏ đi | `test_live_page_is_off_by_default_and_says_why` |
| bf16 cho "nhanh" | trên CPU này **chậm hơn 3,4×** | ghi trong `MEASURED_CPU_NOTE` |
| đường UI chưa từng bấm thử | `run_steps` đúng mà trang vẫn trắng | `test_live_page_computes_and_renders_when_enabled`: bấm nút thật, đòi ảnh + dòng nguồn gốc |

## 12. Định nghĩa "xong" cho lần deploy này

- [x] app Streamlit chạy được, 5 trang, không traceback (`AppTest`)
- [x] không có đường nào nhận ảnh RGB rồi trả xác suất (3 test chặn)
- [x] `p_fake` chỉ hiện khi `gate == "passed"` và ngưỡng trên `val` (4 test)
- [x] gallery rỗng hiện lệnh dựng, không hiện panel trống
- [x] mọi kết quả có dòng `DIRE: … · detector: … · nhãn: … · S=…`
- [x] chi phí khớp `dire_zero.budget` từng trường
- [x] `requirements.txt` đúng một dòng; không `torch`
- [x] README có front-matter `sdk: streamlit`, hướng dẫn deploy và rollback
- [ ] gallery có mẫu thật (chờ `::dire` xong) — **điều kiện để trang 2 có nội dung**
- [ ] cold start đo trên Space thật, ghi vào mục 4
- [ ] cổng 2 chạy xong trước khi bật `--scores`
- [x] chế độ tính DIRE tại chỗ: có, **mặc định tắt**, số đo 10,1 s/forward và lý do (mục 13)
