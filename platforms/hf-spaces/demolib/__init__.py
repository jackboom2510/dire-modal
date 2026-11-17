"""Thư viện của demo thuyết trình — cố ý **không** nằm trong `src/`.

Bốn module đầu **không** dùng torch, không đọc Volume, không gọi mạng. Module
`live` là ngoại lệ có chủ ý (nó cần torch + checkpoint của ADM), nên nó tự khai
trạng thái và mặc định không được bật.

* `toy2d` — mô hình khuếch tán 2 chiều có score chính xác, đo DIRE;
* `provenance` — đọc `.json` của gallery và quyết định khi nào được hiện điểm;
* `charts` — SVG thuần cho ba hình của buổi nói;
* `cost` — công thức chi phí, khoá bằng test với `dire_zero.budget`;
* `live` — **tính DIRE thật** tại chỗ (chậm: 10,1 s/forward trên 2 vCPU), mặc định tắt.

Tách khỏi `src/dire_zero/` vì đây là thứ *trình diễn*, không phải thứ *đo trên dữ
liệu thật*; trộn hai loại vào một package là cách chắc chắn nhất để một hôm nào đó
có người import nhầm `toy2d` rồi tưởng đó là kết quả LSUN.
"""

__all__ = ["toy2d", "provenance", "charts", "cost", "live"]
