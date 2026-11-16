"""``dire_local`` — chạy DIRE **trên máy mình**: không Modal, không tài khoản cloud.

Ba lệnh, cùng một lõi với đường cloud (`dire_core`):

    dire info                                  # in cấu hình + kiểm môi trường
    dire detect anh.jpg --weights lsun_adm.pth # 1 ảnh → P(fake)
    dire batch ./thu_muc --out scores.csv      # nhiều ảnh → CSV
    dire serve --port 8000                     # HTTP API tại máy (cần fastapi)

Mặc định **không** cần torch để chạy ``info``; ``detect``/``batch``/``serve``
mới cần ``torch`` + ``torchvision`` + ``Pillow``.

Quan hệ với ``dire_modal``: cùng gọi ``dire_core``, nên điểm số phải **khớp**.
``modal run -m dire_modal.eval`` không phải bản sao của CLI — nó là cùng một
pipeline, khác hạ tầng. ``tests/test_core_parity.py`` khoá điều đó.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
