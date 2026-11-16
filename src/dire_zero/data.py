"""Chia tập và băm — không phụ thuộc torch, không phụ thuộc thư viện ảnh.

Một thứ ở đây quyết định kết quả có nghĩa hay không: **chia tất định theo hash
nội dung**, không theo thứ tự file. `ls` trên Windows và trên Linux sắp xếp
khác nhau (dấu, hoa/thường, Unicode NFC/NFD); chia theo thứ tự thì "test set"
trên máy bạn khác test set trong container — cùng một con số ACC mang hai nghĩa,
và không ai phát hiện vì cả hai lần đều in ra "0,93".
"""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["sha256_file", "stable_key", "split_indices", "list_images", "IMAGE_SUFFIXES"]

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def stable_key(name: str, seed: int) -> int:
    """Số nguyên ổn định cho một tên file — cùng tên, cùng seed ⇒ cùng số.

    Dùng `blake2b` chứ không dùng `hash()`: `hash()` bị làm nhiễu theo tiến trình
    (PYTHONHASHSEED), nên lần chạy sau ra tập khác — đúng loại lỗi khiến ảnh test
    bị nhìn thấy lúc train mà không có cảnh báo nào.
    """
    return int.from_bytes(hashlib.blake2b(f"{seed}:{name}".encode("utf-8"), digest_size=8).digest(), "big")


def split_indices(names: list[str], seed: int = 0, frac_train: float = 0.8) -> dict[str, list[int]]:
    """Chia index theo thứ tự khoá băm: rời nhau, phủ kín, tái lập được.

    Khoá là `(hash, tên)` — thêm tên làm khoá phụ để hai tên trùng hash (xác
    suất ~0 nhưng không bằng 0) vẫn có thứ tự xác định.
    """
    if not 0.0 < frac_train < 1.0:
        raise ValueError("frac_train phải trong (0, 1)")
    if len(set(names)) != len(names):
        raise ValueError("danh sách tên có phần tử trùng — không chia tập được")
    order = sorted(range(len(names)), key=lambda i: (stable_key(names[i], seed), names[i]))
    cut = round(len(order) * frac_train)
    return {"train": sorted(order[:cut]), "test": sorted(order[cut:])}


def list_images(root: str | Path) -> list[Path]:
    """Ảnh trong thư mục, sắp xếp **theo tên** (chỉ để in log, không dùng để chia)."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"không thấy thư mục ảnh: {root}")
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())
