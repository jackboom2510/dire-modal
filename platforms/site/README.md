# `platforms/site/` — trang pitch deck (tĩnh, offline)

Trang deck tĩnh: 10 slide, chuyển bằng phím, **không CDN, không phụ thuộc dịch vụ ngoài**. Dữ liệu paper do `build.py` sinh từ TSV; build lại sau khi nguồn thay đổi.

Mục đích: giải thích DIRE → so sánh các phương pháp → trình bày bằng chứng paper
đã công bố → nêu trạng thái repo và giới hạn. Deck không trình bày số benchmark
repo chưa được xác minh.

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

## Quy tắc provenance

- Các số ACC/AP và hình robustness trong deck đều là **kết quả paper ICCV 2023**, gắn Table/Figure nguồn tại slide.
- Không có metric hiệu năng, chi phí hay thời gian chạy của repo trong deck.
- Các file run chưa được dùng làm bằng chứng hiệu năng repo; chỉ đưa vào sau khi ledger, thời điểm đo, mẫu số và protocol được xác minh.
- Trạng thái repo là ghi chú tiến độ, không phải kết quả benchmark.

## Kiến trúc

```
build.py     đọc repo → data.json     (nguồn số DUY NHẤT)
data.json    sinh ra, KHÔNG sửa tay
index.html   cấu trúc + nội dung giải thích; số liệu chỉ có trong data.json
              thông qua data-bind hoặc các bảng được render từ dữ liệu
deck.js      điều hướng + renderer an toàn cho bảng; không chứa metric
style.css    bảng màu lấy từ tools/manim/style.py (web và video cùng một bộ màu)
```

`build.py` lấy số từ:

| Nguồn | Dùng cho |
|---|---|
| `src/dire_modal/paper_table2.tsv` | kết quả DIRE theo generator trong Table 2 |
| `src/dire_modal/paper_detector_benchmark.tsv` | baseline Table 2: detector gốc, retrain và DIRE |
| `src/dire_modal/paper_table5.tsv` | ablation representation (REC / RGB / RGB + DIRE / DIRE) |
| hằng `STATUS` trong `build.py` | trạng thái repo, không có metric hiệu năng |

Sửa TSV ⇒ `data.json` **cũ** ⇒ `tests/test_site.py` đỏ cho tới khi chạy lại
`build.py`. Đó là chủ ý.

## Kiểm

```bash
python3 platforms/site/build.py --check   # exit 1 nếu data.json cũ
python3 -m pytest tests/test_site.py -q
```

Test kiểm: số trong `data.json` khớp TSV paper · không có run metrics repo chưa xác minh · `data.json`
không cũ · HTML/CSS/JS **không** chứa số liệu viết tay · mọi khoá `data-bind`
tồn tại · không tài nguyên ngoài · bảng màu khớp `tools/manim/style.py` ·
thứ tự slide · có mục giới hạn. Nếu máy có
`node` + `jsdom` thì có thêm một test dựng DOM thật và đọc lại số; không có
thì `skip` (CI không cài `node_modules`).

## Giới hạn đã biết

- **Preview trong iframe không có network** ⇒ `fetch` fail ở đó. Phải serve qua
  HTTP như trên; trang vẫn bắt lỗi và hiện cảnh báo chứ không hiện số 0.
- Ảnh/mẫu thật của DIRE **chưa có** (chưa tải xong dữ liệu) ⇒ trang chỉ có bảng
  và số, không có hình minh hoạ ảnh thật. Không vẽ hình giả.
- Slide nói về AEROBLADE/Fusion là **kế hoạch**, không phải kết quả: `aero.py` và
  `fusion.py` chưa tồn tại.
