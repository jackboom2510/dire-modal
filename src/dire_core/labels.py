"""Nhãn real/fake — đọc từ **cấu trúc member**, không đoán theo cảm giác.

Dataset DIRE dùng hai lớp ``0_real`` / ``1_fake`` (convention ImageFolder).
Nhưng một tar một-generator thường chỉ có một lớp, nên khi member không có
thư mục thì phải dựa vào **tên archive** (``real.tar.gz`` ⇒ real).

Chỗ này từng là nguồn của một lỗi im lặng: nhãn sai không làm code nổ, nó chỉ
làm AP trông vẫn "hợp lý". Vì thế hàm này nằm trong ``dire_core`` — nếu CLI
local và eval trên Modal dùng hai bản khác nhau thì hai bên sẽ cho hai con số
khác nhau trên **cùng một tar**.
"""

from __future__ import annotations

from pathlib import Path

REAL, FAKE = 0, 1


def label_of(member_name: str, archive_name: str = "") -> int:
    """Nhãn của một member trong tar.

    Ưu tiên cấu trúc ``0_real``/``1_fake`` bên trong member; không thấy thì đoán
    theo **tên archive**.
    """
    low = member_name.replace("\\", "/").lower()
    if "0_real" in low or "/real/" in low or low.startswith("real/"):
        return REAL
    if "1_fake" in low or "/fake/" in low or low.startswith("fake/"):
        return FAKE
    base = Path(archive_name).name.lower()
    return REAL if base.startswith("real") else FAKE
