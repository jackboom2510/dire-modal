# Mở đầu — đọc tài liệu này thế nào

## Vấn đề mà tài liệu này giải quyết

Một bức ảnh do mô hình sinh tạo ra, ở kích thước 256×256, hiện nay gần như không
phân biệt được bằng mắt với ảnh chụp. Nhưng "gần như" không phải "hoàn toàn": ảnh
sinh ra không đi qua cảm biến, không có nhiễu đọc, không có méo ống kính, và được
tạo ra bởi một quá trình tính toán có thể mô phỏng lại. Câu hỏi trung tâm của bài
báo DIRE [[1]] là:

> Có thể dùng **chính quá trình sinh ảnh** như một phép biến đổi đặc trưng để
> phân biệt ảnh thật với ảnh sinh hay không?

Trả lời của bài báo là có, và biểu diễn đó có dạng rất đơn giản:

$$[1] \mathrm{DIRE}(x_0) \;=\; \left|\, x_0 - R\big(I(x_0)\big) \,\right| $$

trong đó `I` là phép đảo ngược (inversion) đưa ảnh về nhiễu, `R` là phép tái tạo
(reconstruction) đưa nhiễu về ảnh, và phép trừ được hiểu theo từng điểm ảnh. Ảnh
**thật** khó được tái tạo nên hiệu lớn; ảnh **sinh** được tái tạo tốt nên hiệu nhỏ.

Bài báo đánh số công thức riêng: công thức trên là phương trình (9) của bài báo
[[1]], còn tài liệu này đánh số tuần tự theo thứ tự xuất hiện nên nó là (1).

Đó là toàn bộ ý tưởng. Phần còn lại của tài liệu là giải thích vì sao công thức
đó đúng, nó được cài đặt thế nào, và nó yếu ở đâu.

## Ba câu hỏi mà tài liệu trả lời

1. **Vì sao cần một biểu diễn mới?** Các detector cũ học dấu vết (artefact) của
   bộ sinh ảnh; dấu vết đó đổi khi kiến trúc đổi. DIRE thay dấu vết bằng một đại
   lượng có tính hệ thống hơn: sai số tái tạo của mô hình khuếch tán.
2. **Vì sao phép đảo ngược tồn tại?** Vì quá trình sinh của DDIM là một phương
   trình vi phân thường (ODE) tất định, nên đi được cả hai chiều. Không có tính
   chất này thì không có DIRE.
3. **Đọc code thế nào?** Mỗi bước toán ứng với một hàm; mỗi thang đo (uint8,
   `[−1,1]`, chuẩn hoá ImageNet) ứng với một chỗ trong mã. Ba thang này lẫn nhau
   là nguồn lỗi phổ biến nhất khi tái lập.

## Bản đồ tài liệu

```
Phần 0     Công cụ toán tối thiểu: véc-tơ · gradient & Adam · Gauss ·
           hợp lý cực đại → BCE · entropy/KL → ELBO · Nyquist · ROC/AUC · NFE
Phần 1–3   Nền tảng: ảnh số · gamma & JPEG · học máy · CNN/ResNet-50
Phần 4–6   Khuếch tán: ba họ mô hình sinh · quá trình thuận · lịch nhiễu · DDIM · ODE
           và phép đảo ngược · vì sao ảnh sinh dễ tái tạo
Phần 7–9   DIRE: định nghĩa và thiết kế · thực nghiệm của bài báo · phản biện
Phần 10–11 Đọc code: mã của tác giả · mã của kho này và các sai lệch
Phần 12    Từ đọc hiểu tới nghiên cứu: bài tập và thí nghiệm đề xuất
Phần 13–14 Bản đồ module (cả những file nhỏ) · hạ tầng: scripts, platforms, tools
Phụ lục A–D  Thuật ngữ · bảng tra công thức → mã · danh mục kiểm · tham khảo
```

## Ba tuyến đọc

- **Tuyến nhanh (2 giờ):** Mở đầu → Phần 7 → Phần 9. Đủ để hiểu ý tưởng và hạn
  chế, chưa đủ để tự cài đặt.
- **Tuyến "nền chắc" (nửa ngày):** Phần 0 → Phần 1 → Phần 2. Đây là phần trả lời
  "vì sao hàm mất mát có dạng đó", "vì sao phải lấy trị tuyệt đối", "vì sao AUC là
  xác suất xếp hạng đúng" — những chỗ hay bị bỏ qua nhưng quyết định việc hiểu
  đúng hay hiểu sai.
- **Tuyến đầy đủ (1–2 ngày):** đọc tuần tự Phần 1 → 12. Đây là tuyến dành cho
  người muốn nghiên cứu tiếp hoặc tái lập.
- **Tuyến theo code:** Phần 7 (định nghĩa) → Phần 5.6 (NFE) → Phần 10.9–10.10
  (`compute_dire.py`, `gaussian_diffusion.py`) → Phần 11 → Phần 13 (mọi module còn
  lại, mỗi module ba dòng). Dành cho người đã quen diffusion và chỉ muốn đối chiếu
  cài đặt.

## Quy ước ký hiệu

| Ký hiệu | Nghĩa | Ghi chú |
|---|---|---|
| `x₀` | ảnh sạch, đầu vào | tensor 3×256×256, giá trị trong `[−1,1]` |
| `x_t` | ảnh sau `t` bước thêm nhiễu | cùng kích thước `x₀` |
| $\beta_t$, $\alpha_t$ | hệ số thêm nhiễu bước $t$ | $\alpha_t = 1-\beta_t$ |
| $\bar{\alpha}_t$ | tích luỹ $\alpha_1\cdots\alpha_t$ | giảm từ ~1 về ~0 |
| $\varepsilon$, $\varepsilon_\theta$ | nhiễu thật / nhiễu mô hình dự đoán | cùng kích thước ảnh |
| $\hat{x}_0$ | ảnh sạch **dự đoán** từ $x_t$ | khác $x_0$ là ảnh thật |
| `T` | số bước huấn luyện | 1000 với ADM |
| `S` | số bước khi sinh / đảo ngược | 20 trong bài báo |
| NFE | số lượt chạy mạng | mỗi ảnh DIRE tốn $2S$ |
| `I(·)`, `R(·)` | phép đảo ngược, phép tái tạo | cả hai đều `S` bước |

Ba thang giá trị cần phân biệt tuyệt đối, vì lẫn chúng là hỏng kết quả mà không
có lỗi nào báo:

1. **uint8 `[0,255]`** — dạng lưu trên đĩa (PNG/JPEG, mảng `.npy`).
2. **float `[−1,1]`** — dạng mô hình khuếch tán nhận vào, thu được bằng
   `pixel/127.5 − 1`.
3. **Chuẩn hoá ImageNet** — dạng ResNet-50 nhận vào, thu được bằng
   `(x − μ)/σ` với `μ = (0.485, 0.456, 0.406)`, `σ = (0.229, 0.224, 0.225)`.

## Cách đọc con số trong tài liệu

Mọi số liệu thuộc một trong bốn loại, và tài liệu ghi rõ loại nào:

- **Số của bài báo** — trích từ bảng/hình, luôn kèm số hiệu bảng.
- **Số đọc từ code** — ví dụ "`fc.weight` có shape `[1, 2048]`".
- **Số đo trong kho này** — hiện tại **chưa có** kết quả thực nghiệm nào; mọi chỗ
  cần số đo đều ghi rõ là chưa có.
- **Ước lượng** — luôn kèm công thức và giả định (ví dụ `sec_per_step = 0,35` là
  giả định chưa đo).

::: warn
Không có kết quả thực nghiệm nào trong kho này tính đến thời điểm viết. Nếu bạn
đọc một con số không kèm một trong bốn nhãn trên, hãy coi đó là lỗi tài liệu.
:::

## Mức tái lập: ba trục độc lập, không phải ba cấp bậc

Dùng lại DIRE map và trọng số của tác giả để chấm điểm **không** phải là tái lập
thí nghiệm. Đó là **chấm lại** (re-scoring): kiểm tra đường đo của mình có khớp
đường đo của họ. Ba trục:

| Trục | Câu hỏi | Nhãn |
|---|---|---|
| DIRE map | tự tính được map từ ảnh chưa? | `R1` = tự tính, `R0` = dùng map phát hành |
| Detector | tự train lại được detector chưa? | `R2` |
| Ảnh | tự sinh lại được ảnh chưa? | `R3` |

Nhãn hợp (ví dụ `R1+R2`) mô tả **đúng những gì đã làm**. Nhãn `R0` **cấm** dùng
cụm "tái lập DIRE"; phải nói "chấm lại trên artefact của tác giả". Quy định này
nằm trong `docs/adr-0006-reproduction-levels.md` của kho.

## Kiến thức cần có trước

- **Bắt buộc:** tích chập, pooling, hàm mất mát, lan truyền ngược, chia tập
  train/val/test, độ chính xác và precision/recall.
- **Nên có:** batch normalization, ResNet, transfer learning.
- **Không cần:** xác suất biến phân, lý thuyết score matching. Tài liệu chỉ dùng
  đúng hai ý xác suất: phân phối Gauss và ước lượng hợp lý cực đại.

## Bản PDF của tài liệu này

Tài liệu được viết bằng Markdown trong `docs/book/` (sáu phần, mỗi phần một file),
rồi dựng thành PDF bằng `tools/pdf/md2pdf.py`:

```
docs/dire-tu-so-0.pdf
```

Bản PDF có mục lục bấm được, bookmark theo từng mục, công thức kết xuất từ LaTeX,
và mã nguồn có tô sáng cùng số dòng — **giữ nguyên thụt lề** (khoảng trắng đầu dòng
được chuyển thành ký tự không ngắt, vì bộ dựng `Paragraph` của reportlab gộp mọi
khoảng trắng liên tiếp).

**Cú pháp công thức:** \$...\$ cho công thức nội dòng (chèn trong câu, trong ô bảng),
\$\$[n] ... \$\$ cho công thức hiển thị có số hiệu (n). Cả hai đều là LaTeX, kết
xuất bằng `matplotlib.mathtext` (tập con LaTeX: không có `\begin{align}`, `\text`
tự đổi thành `\mathrm`, `\big` tự bỏ). Cách dựng lại ghi trong `tools/pdf/README.md`.

## Đọc thêm

Bài báo gốc [[1]] nên đọc song song: Phần 3 (phương pháp) và Phần 4 (thí nghiệm)
của bài tương ứng với Phần 8–9 ở đây. Phụ lục A của bài liệt kê nguồn của từng
mô hình sinh; Phụ lục B là lập luận lý thuyết mà Phần 6.2 của tài liệu này phân
tích kỹ vì nó là chỗ yếu nhất của bài.
