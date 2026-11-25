---
name: dire-academic-slides
description: "Tạo hoặc chỉnh sửa bài thuyết trình học thuật tiếng Việt về DIRE trong repo này; luôn đọc docs/slide-agent-brief.md trước khi lập dàn ý, viết slide hoặc kiểm chứng claim."
---

# Slide học thuật DIRE (tiếng Việt)

Skill này điều chỉnh quy trình tạo slide học thuật cho dự án DIRE. Nó bổ sung yêu cầu ngôn ngữ và ranh giới khoa học của repo; không thay thế skill kỹ thuật tạo PPTX.

## Trước khi làm

1. Đọc `docs/slide-agent-brief.md` toàn bộ; brief là nguồn sự thật về luận điểm, trạng thái, protocol, asset và nguồn trích. Đọc tiếp source packet mà brief chỉ ra khi claim cần kiểm chứng. Không suy trạng thái job hoặc dữ liệu ngoài workspace thành kết quả.
2. Nếu tạo/sửa `.pptx`, tìm và đọc skill PPTX đang có trong môi trường; làm theo hướng dẫn kỹ thuật và QA của skill đó. Skill này không cài đặt hay ngụ ý đã có skill PPTX.
3. Xác định người nghe, thời lượng và mục tiêu. Mặc định là lập luận học thuật có cấu trúc, 16:9, tiếng Việt. Lập dàn ý trước; với deck trên 10 slide hoặc nội dung phức tạp, xin người dùng duyệt dàn ý trước khi dựng.

## Quy tắc nội dung và tiếng Việt

- Viết tiếng Việt tự nhiên, chuẩn dấu và Unicode; giữ tên riêng, tên mô hình, mã protocol và thuật ngữ chuẩn bằng tiếng Anh khi giúp tránh nhập nhằng. Giải nghĩa acronym tiếng Anh ở lần xuất hiện đầu tiên, rồi dùng nhất quán.
- Mỗi slide nội dung có một kết luận làm tiêu đề (action title), không dùng nhãn chủ đề chung. Đọc riêng chuỗi tiêu đề phải thấy được mạch lập luận hoàn chỉnh.
- Một ý chính và một exhibit chủ đạo mỗi slide. Chú thích ngắn gọn; phần giải thích chi tiết chuyển sang speaker notes hoặc appendix. Giữ nội dung đủ thoáng, dễ đọc khi trình chiếu.
- Giọng khoa học, thận trọng, không quảng cáo. Tách rõ giả thuyết/trực giác, protocol đã chốt, kết quả paper-reported, kết quả repo-measured và việc chưa chạy.
- Mọi số liệu, hình mượn và claim không phải kết quả mới của dự án phải có trích dẫn tại slide; đưa tham chiếu đầy đủ vào slide References hoặc notes. Không tự đặt tên tác giả, affiliation, venue hay ngày nếu brief không cung cấp.

## Ranh giới khoa học bắt buộc

Áp dụng đầy đủ các quy tắc và trạng thái trong `docs/slide-agent-brief.md`, đặc biệt:

- Ảnh nguồn `images/` và author DIRE map `dire/` là hai loại dữ liệu khác nhau; so implementation bằng cách tính lại map từ ảnh nguồn rồi đối chiếu map-với-map.
- Không đoán encoding/scale từ đuôi PNG hoặc ảnh xem trước. Không tạo dữ liệu/ảnh/chart/metric giả; SVG trong brief là sơ đồ minh họa, phải giữ nhãn schematic/not measured.
- Gắn nhãn trực tiếp số Table 2 là kết quả tác giả trong paper, không phải kết quả dự án. Không trình bày Table 4/5/Figure 4 của subset như đã chạy nếu chưa có bằng chứng.
- Giữ đúng generator test SDv2 và giới hạn Figure 4 implementation như brief mô tả. Không nâng provenance R1+R2 thành R3.
- Pilot subset không đại diện full-data; phân biệt stage/model/ledger, sample count thực tế và upper bound. Không khẳng định author DIRE cache đáp ứng representation REC nếu thiếu reconstruction.

## Hình ảnh và bố cục

- Tái dùng các SVG truy nguyên được trong `docs/slide-assets/` khi phù hợp; tham khảo `tools/build_slide_assets.py` để sinh lại asset/bảng protocol, không tự vẽ lại số liệu.
- Khi chưa có cặp ảnh thật đã xác minh và cấp phép, dùng sơ đồ trong repo thay cho ảnh photorealistic giả lập.
- Ưu tiên nền sáng, typography sans-serif có hỗ trợ tiếng Việt đầy đủ, độ tương phản cao và tối đa ba màu nhấn. Kiểm tra dấu tiếng Việt, xuống dòng, cắt chữ và khả năng đọc ở kích thước trình chiếu.

## Kiểm tra trước khi bàn giao

- Soát từng slide với brief và nguồn: mọi claim/số có provenance; mọi trạng thái chưa chạy nói đúng là chưa chạy.
- Chạy ghost-deck test trên action titles; kiểm nhịp kể, một thông điệp/slide, citation, References và giới hạn.
- Nếu có PPTX/PDF, dùng quy trình QA của skill PPTX: trích xuất nội dung, render slide thành ảnh, kiểm toàn bộ trang về font, dấu tiếng Việt, overflow, asset SVG và citation; sửa rồi kiểm lại đến khi không còn lỗi.
- Báo rõ định dạng bàn giao, kiểm tra đã chạy, và phần chưa xác minh/chưa làm.
