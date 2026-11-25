"""Thư viện của demo thuyết trình — cố ý **không** nằm trong `src/`.

Bốn module đầu **không** dùng torch, không đọc Volume, không gọi mạng. Module
`live` là ngoại lệ có chủ ý (nó cần torch + checkpoint ADM). Backend local được ưu tiên;
nếu checkpoint thiếu, demo vẫn phát chuỗi DDIM tổng hợp đã render và cache sẵn.

* `toy2d` — mô hình khuếch tán 2 chiều có score chính xác, đo DIRE;
* `provenance` — đọc `.json` của gallery và quyết định khi nào được hiện điểm;
* `charts` — SVG thuần cho ba hình của buổi nói;
* `cost` — công thức chi phí, khoá bằng test với `dire_zero.budget`;
* `live` — **tính DIRE thật** tại chỗ (chậm: 10,1 s/forward trên 2 vCPU); có cache theo ảnh/model/S.
* `frame_cache` — cache giới hạn 8 lượt, lưu riêng x_t/x̂₀ từng bước và DIRE map.
* `cached_illustration` — đọc mô phỏng DDIM tổng hợp đã render trước, không cần torch/GPU.

Tách khỏi `src/dire_zero/` vì đây là thứ *trình diễn*, không phải thứ *đo trên dữ
liệu thật*; trộn hai loại vào một package là cách chắc chắn nhất để một hôm nào đó
có người import nhầm `toy2d` rồi tưởng đó là kết quả LSUN.
"""

__all__ = ["toy2d", "provenance", "charts", "cost", "live", "frame_cache", "cached_illustration"]
