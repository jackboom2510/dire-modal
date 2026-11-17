"""Matplotlib giả — ĐỦ để chạy `tools/figures.py` trong test, không vẽ gì.

Vì sao là stub chứ không phải dependency thật
---------------------------------------------
Test cần chứng minh *code vẽ chạy được và ghi đủ file*, không cần chứng minh
matplotlib vẽ đẹp. Cài matplotlib thật vào CI làm mỗi lần chạy test nặng thêm
~40 MB và kéo theo numpy version pin — trong khi thứ cần kiểm chỉ là: hàm nào
được gọi, file nào được ghi, sidecar có đủ khoá.

Stub này **không** được dùng ở chỗ nào khác: nếu `matplotlib` thật có trong môi
trường thì test dùng bản thật (xem `tests/test_figures_tool.py`).
"""

__version__ = "0.0-stub"


def use(backend: str | None = None) -> None:   # noqa: ARG001 - giữ đúng API
    return None
