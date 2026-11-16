# `tools/pdf` — dựng PDF giáo trình từ Markdown

Sandbox (và máy người dùng) không có `pandoc`/`xelatex`, cũng không tải được
Chromium để in HTML. Bộ dựng này chỉ dùng **thư viện Python**: `reportlab` (dàn
trang) + `matplotlib.mathtext` (kết xuất công thức LaTeX ra ảnh 600 dpi) +
`pygments` (tô sáng mã) + `fontTools` (kiểm phủ glyph, dùng khi kiểm phông).

## Cài đặt

```bash
python3 -m venv /tmp/pdfenv
/tmp/pdfenv/bin/pip install reportlab matplotlib pygments fonttools
```

`pypdfium2` chỉ cần khi muốn **kiểm tra bằng mắt**: render trang ra PNG.
Không bắt buộc để dựng PDF.

## Dựng

```bash
/tmp/pdfenv/bin/python tools/pdf/md2pdf.py \
    --src docs/book \
    --out docs/dire-tu-so-0.pdf
```

Script chạy **hai lượt**: lượt 1 dựng để lấy số trang của từng mục, lượt 2 dựng
lại với mục lục đã có số trang. Số hàng của mục lục không đổi giữa hai lượt nên
phân trang không xê dịch — nhờ vậy số trang in ra là đúng.

## Phông chữ

| Dùng cho | Phông | Nguồn |
|---|---|---|
| Thân bài | DejaVu Serif | đi kèm matplotlib |
| Tiêu đề | DejaVu Sans | đi kèm matplotlib |
| Công thức | STIX | đi kèm matplotlib |
| Mã nguồn | JetBrains Mono (OFL) | tải từ `codeload.github.com` vào `~/.cache/dire-pdf`, **không** commit |

Hai cái bẫy đã gặp và đã xử lý trong `md2pdf.py`:

- **`ᾱ` (U+1FB1) không có glyph** trong DejaVu lẫn JetBrains Mono ⇒ ra ô "tofu".
  Hàm `fix_marks` đổi thành `α` + U+0304 (combining macron), cả ba phông đều có.
- **JetBrains Mono thiếu một số dấu tiếng Việt dựng sẵn** (`ẫ ả ế ề ể ễ ở`) ⇒ mã
  nội dòng có những ký tự đó được chuyển sang phông thân bài.

## Cú pháp Markdown được hỗ trợ

Tập con có chủ đích, đủ cho tài liệu kỹ thuật:

```
# Phần        ## Mục        ### Mục con
đoạn văn, **đậm**, *nghiêng*, `mã nội dòng`, [chữ](https://url)
$\frac{a}{b}$ nội dòng      công thức nội dòng (cũng dùng được trong ô bảng)
\$ ... \$                   escape khi cần hiện ký tự $ thật
trích dẫn → [[1]]            (chỉ số trên, màu xanh)
- gạch đầu dòng             1. danh sách số
> trích dẫn khối
```python linenums         khối mã, có tô sáng + số dòng tuỳ chọn
$$[12] \frac{a}{b} $$       công thức hiển thị, [12] là số hiệu (tuỳ chọn)
| a | b |                   bảng; dùng \| cho dấu | bên trong ô
::: note Tiêu đề ... :::    hộp ghi chú (note / warn / key)
<!-- pagebreak -->          ngắt trang
```

## Giới hạn đã biết

- **Công thức** do `matplotlib.mathtext` dựng, tức tập con LaTeX của mathtext
  (`sanitize()` tự đổi `\ge→\geq`, `\le→\leq`, `\to→\rightarrow`, `\left|→\left\vert`,
  bỏ `\big`/`\Big`, `\text{}→\mathrm{}`):
  không có `\begin{align}`, không có `\text{}` (bộ dựng tự đổi thành `\mathrm{}`),
  không có `\big`/`\Big` (bộ dựng tự bỏ), `\left|` được đổi thành `\left\vert`.
  Công thức **không** đánh số tự động — số hiệu do người viết đặt trong `$$[n]`.
- **Ngắt trang** chỉ xảy ra ở tiêu đề `#`, ở `<!-- pagebreak -->`, và bên trong
  khối mã/bảng (Table được phép tách hàng). Khối mã **≤ 14 dòng** được bọc
  `KeepTogether` để không bị tách rời khỏi ngữ cảnh; khối dài hơn thì tách tự do.
- **Thụt lề trong khối mã** được giữ bằng cách đổi khoảng trắng đầu dòng và cụm
  ≥ 2 dấu cách thành `&#160;` — vì `Paragraph` gộp mọi khoảng trắng liên tiếp
  (đây chính là lỗi "mất indent" đã gặp ở bản trước).
- **Công thức nội dòng** được chèn dưới dạng `<img>` với chiều cao chặn ở 1,5 lần
  cỡ chữ (và bề rộng chặn theo ô bảng): `Paragraph` tính chiều cao dòng theo cỡ
  chữ, **không** theo ảnh, nên công thức cao (phân số, tích phân) sẽ đè lên dòng
  trên nếu không chặn.
- **Không có chỉ mục (index)** theo từ khoá; Phụ lục tra cứu viết tay.

## Kiểm tra kết quả

```bash
/tmp/pdfenv/bin/pip install pypdfium2
/tmp/pdfenv/bin/python - <<'PY'
import pypdfium2 as pdfium
pdf = pdfium.PdfDocument("docs/dire-tu-so-0.pdf")
print("pages:", len(pdf))
for i in range(len(pdf)):
    pdf[i].render(scale=1.8).to_pil().save(f"/tmp/page{i+1}.png")
PY
```
