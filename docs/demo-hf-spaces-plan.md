# Kế hoạch chạy demo trên HF Spaces (Streamlit)

- **Trạng thái:** kế hoạch + app đã viết và chạy thử · **Ngày:** 2026-11-17
- **Liên quan:** `docs/demo-web-plan.md` (§2 lỗi phải sửa, §3 ba chế độ, §5 cổng
  kiểm), `platforms/hf-spaces/` (mã), `tools/make_gallery.py` (dựng gallery),
  `docs/from-zero-clean-room.md` (§3.6 về `--n` và ước lượng tiền), ADR-0003
- **Câu hỏi của mục này:** *đưa demo lên HF Spaces bằng Streamlit, chạy được, 0 ₫,
  và không nói dối — cần những gì, theo thứ tự nào, kiểm bằng gì.*

---

## 0. Kết luận trước, chi tiết sau

| Quyết định | Chốt | Vì sao |
|---|---|---|
| Framework | **Streamlit 1.67.0**, một phụ thuộc duy nhất | Space CPU free; UI là 5 trang tĩnh + vài widget; Gradio cũ không cho kiểm bằng `pytest` mà không dựng app |
| Có `torch` trên Space? | **Không** | Space không chấm ảnh và không tính DIRE; bỏ torch cắt ~800 MB cài đặt và ~40 s cold start |
| Có ô upload ảnh? | **Không** | đường "ảnh RGB → `P(fake)`" là lỗi bị ghi ở `demo-web-plan.md` §2, đã bỏ hẳn |
| Điểm detector lấy từ đâu? | `.json` của từng mẫu, chỉ hiện khi `gate == "passed"` và ngưỡng chọn trên `val` | để "chưa qua cổng" hiện thành chữ, không thành một con số |
| Phép đo chạy thật trên Space | thí nghiệm khuếch tán 2 chiều có score chính xác | 61 ms/lượt, không cần GPU, và là chỗ giải thích *cơ chế* |
| Chi phí | **0 ₫** (hardware `cpu-basic`) | không có GPU, không gọi mạng ngoài |

Ba con số quyết định kế hoạch này, **đo trên máy** (mục 4): cold start 0,57 s ·
RAM 60 MB · toàn bộ 5 trang render dưới 1,5 s.

## 1. Cái gì thay đổi so với app Gradio cũ

| | Bản Gradio (`app.py` cũ, 216 dòng) | Bản Streamlit (hiện tại) |
|---|---|---|
| Tab 1 | upload ảnh → `P(real)/P(fake)` bằng ResNet-50 học trên DIRE | **bỏ** |
| Tab 2 | 3 panel từ `gallery/` + hướng dẫn nếu rỗng | giữ, thêm dòng nguồn gốc và cổng kiểm |
| Nạp model | `hf_hub_download` + `torch.load` lúc khởi động | không có model |
| Phụ thuộc | gradio, torch, torchvision, pillow, numpy | `streamlit` |
| Kiểm bằng test | đọc mã nguồn (chuỗi) | đọc mã nguồn **và** chạy thật bằng `AppTest` |
| Số đo trên Space | không có | 5 trang, 4 kịch bản gallery |

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
| Bộ test demo | **47 test · 2,84 s** | `pytest tests/test_hf_spaces_demo.py` |
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
# 1. Dựng gallery (một lần, trên máy có DIRE map)
python tools/make_gallery.py \
    --maps  <volume>/maps/test --orig <volume>/images/test \
    --out   platforms/hf-spaces/gallery \
    --n 8 --steps 20 --label R1+R2 \
    --map-source "maps/test/*.npy · ::dire --steps 20"
# không thêm --scores cho tới khi cổng 2 (demo-web-plan.md §5) đã chạy

# 2. Kiểm tại chỗ, chưa cần mạng
cd platforms/hf-spaces && streamlit run app.py --server.headless true
pytest <repo>/tests/test_hf_spaces_demo.py <repo>/tests/test_make_gallery.py -q

# 3. Tạo Space (SDK streamlit, hardware CPU basic)
hf repo create <user>/dire-demo --repo-type space --space-sdk streamlit

# 4. Đẩy nội dung
git clone https://huggingface.co/spaces/<user>/dire-demo /tmp/space
cp -r . /tmp/space/          # từ platforms/hf-spaces/
cd /tmp/space && git add -A && git commit -m "DIRE demo (streamlit)" && git push

# 5. Chờ build; xem log ở tab *Logs* của Space
```

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
