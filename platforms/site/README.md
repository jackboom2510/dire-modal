# `platforms/site/` — trang pitch deck (tĩnh, offline)

Trang một-file-để-thuyết-trình: 12 slide, chuyển bằng phím, **không CDN, không
mạng, không build step**. Mở là chạy; dữ liệu do `build.py` sinh từ repo.

Mục đích: đưa người nghe từ *vấn đề* → *ý tưởng DIRE* → *bằng chứng của paper*
→ *trạng thái THẬT của repo* → *việc còn thiếu* → *giới hạn*, trong ~7 phút.
Cùng mạch kể với `docs/manim-video-plan.md`, để nói miệng và xem video không
kể hai câu chuyện khác nhau.

## Chạy

```bash
python3 platforms/site/build.py                 # sinh data.json (bắt buộc, 1 lần)
python3 -m http.server 8000 --bind 0.0.0.0 --directory platforms/site
# mở http://localhost:8000
```

`fetch("data.json")` cần HTTP origin: mở thẳng `file://` sẽ **không** đọc được
dữ liệu (trình duyệt chặn), trang sẽ hiện cảnh báo đỏ thay vì số.

### Điều khiển

| Phím | Việc |
|---|---|
| `→` `Space` `PageDown` · lăn chuột xuống · vuốt lên | slide sau |
| `←` `PageUp` · lăn chuột lên · vuốt xuống | slide trước |
| `Home` / `End` | slide đầu / cuối |
| `?` | bảng phím tắt |
| `o` | xem TẤT CẢ slide (để in / chụp) |
| `Ctrl+P` | bản in (có `@media print`) |

URL có `#<số>` nên gửi link đúng slide được (`…/#4`).

## Ba tầng số — đừng trộn

Mỗi con số trên trang thuộc đúng một tầng, và tầng đó được **in ngay cạnh số**:

| Nhãn | Nghĩa | Ví dụ |
|---|---|---|
| *số của paper* | Table 2, ICCV 2023 — **không phải** số của mình | ACC 99.9 % của ADM |
| *số đo của repo* | đo thật, có ngày + nguồn | 76.6 KiB/s Baidu, bẫy `ls -l` lệch 3 901 B |
| *ước lượng demo* | do `demo_train --mode plan` tính, **chưa đo** | 160 step ≈ $0.013, `measured: false` |

Trang **không có** tầng thứ tư, và không có ô nào ghi "acc của dự án" — vì dự án
chưa chạy eval thật. Chỗ đó ghi thẳng *"chưa có"*.

## Kiến trúc

```
build.py     đọc repo → data.json     (nguồn số DUY NHẤT)
data.json    sinh ra, KHÔNG sửa tay
index.html   khung + tiêu đề; không chứa số, chỉ [data-bind]
deck.js      điều hướng + đổ số; không chứa số, chỉ logic
style.css    bảng màu lấy từ tools/manim/style.py (web và video cùng một bộ màu)
```

`build.py` lấy số từ:

| Nguồn | Dùng cho |
|---|---|
| `src/dire_modal/paper_table2.tsv` | bảng Table 2 + trung bình 4 detector |
| `src/dire_modal/manifest.py` (tier `e2`) | 15 cặp · 16 tar · 60 container |
| `src/dire_modal/demo_train.yaml` + `demo_train.estimate()` | tham số & chi phí demo |
| hằng `MEASURED` trong `build.py` | số đo tay (có nguồn ghi kèm) |
| hằng `STATUS`, `ROADMAP` | trạng thái thật + việc còn thiếu |

Sửa TSV/manifest/yaml ⇒ `data.json` **cũ** ⇒ `tests/test_site.py` đỏ cho tới khi
chạy lại `build.py`. Đó là chủ ý.

## Kiểm

```bash
python3 platforms/site/build.py --check   # exit 1 nếu data.json cũ
python3 -m pytest tests/test_site.py -q   # 24 test (+1 skip nếu không có jsdom)
```

Test kiểm: số trong `data.json` khớp TSV/manifest/`demo_train` · `data.json`
không cũ · HTML/CSS/JS **không** chứa số liệu viết tay · mọi khoá `data-bind`
tồn tại · không tài nguyên ngoài · bảng màu khớp `tools/manim/style.py` ·
thứ tự slide · có mục "chưa có" (trang không được tự khen). Nếu máy có
`node` + `jsdom` thì có thêm một test dựng DOM thật và đọc lại số; không có
thì `skip` (CI không cài `node_modules`).

## Giới hạn đã biết

- **Preview trong iframe không có network** ⇒ `fetch` fail ở đó. Phải serve qua
  HTTP như trên; trang vẫn bắt lỗi và hiện cảnh báo chứ không hiện số 0.
- Ảnh/mẫu thật của DIRE **chưa có** (chưa tải xong dữ liệu) ⇒ trang chỉ có bảng
  và số, không có hình minh hoạ ảnh thật. Không vẽ hình giả.
- Slide nói về AEROBLADE/Fusion là **kế hoạch**, không phải kết quả: `aero.py` và
  `fusion.py` chưa tồn tại.
