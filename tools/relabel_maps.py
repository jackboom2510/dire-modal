#!/usr/bin/env python3
"""Sửa nhãn generator cho map đã tính — **không** tính lại, không tốn GPU.

Vì sao cần
----------
Bản `::dire` cũ dựng item 3 phần tử `(path, label, split)`; `_as_item` thấy thiếu
generator thì gán `"fake"`. Nên map của **mọi** generator (tác giả và tự sinh) rơi
vào `maps/<split>/fake/`, và `::train --generator adm` báo "chưa có map" dù map đã
nằm trên đĩa, đã trả tiền.

Tin tốt: `dire.json` của lần chạy đó ghi `source` cho từng map — đường dẫn ảnh gốc,
tức là đã mang tên generator (`data/<split>/adm/<sha>.png`). Nhờ vậy việc sửa nhãn
là **đổi tên thư mục**, không phải chạy lại 2S NFE/ảnh: 0 USD thay vì ~1 USD cho
1 000 ảnh.

Nguyên tắc
----------
* **Mặc định chỉ in kế hoạch** (`--apply` mới đổi tên). Tool sửa dữ liệu đã trả
  tiền để tạo, nên mặc định phải là "cho tôi xem trước".
* Nguồn sự thật để đoán generator là `source` trong `dire.json`, **không** phải
  tên file: tên `.npy` là sha của ảnh, không mang thông tin.
* Đích đã tồn tại ⇒ **không ghi đè**, báo xung đột. Ghi đè im lặng một map đúng
  bằng một map khác là cách nhanh nhất để làm hỏng tập train mà không ai biết.
* Item không có `source`, hoặc `source` không suy ra được generator ⇒ để nguyên
  và báo. Đoán bừa ở đây là gắn nhãn sai cho dữ liệu.

Ví dụ
-----
::

    # xem trước
    python tools/relabel_maps.py --maps /zero/maps
    # đổi tên thật
    python tools/relabel_maps.py --maps /zero/maps --apply
    # trên Modal (volume đã mount ở /zero)
    modal run -m dire_zero.modal_app::relabel --apply
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dire_zero.maps_index import main  # noqa: E402

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
