"""Tiền xử lý — **một** pipeline duy nhất, chép từ ``utils/datasets.py`` của tác giả.

::

    Resize(256) → CenterCrop(224) → ToTensor() → Normalize(ImageNet)

``demo.py`` của tác giả **thiếu** ``Normalize``; đó là lý do điểm trong demo
không khớp điểm trong paper. Mọi đường của dự án (eval trên Modal, CLI ở máy
nhà, endpoint API) phải đi qua đúng hàm :func:`preprocess` dưới đây, và chuỗi
``PREPROC`` được ghi vào CSV/``run.json``/response để một lần chạy không thể
lẫn với lần chạy khác.

Không import ``torch`` ở tầng module: ``torch``/``transforms`` được truyền vào.
Nhờ vậy file này test được bằng Python thuần, và container chỉ đọc CSV không
phải cài torch.
"""

from __future__ import annotations

#: Định danh pipeline — ghi vào CSV + ``run.json`` + mọi response của API.
PREPROC = "resize256_centercrop224_normalize"

RESIZE = 256
CROP = 224
# Hằng số ImageNet, đúng như `utils/datasets.py` gốc.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_transform(transforms):
    """Dựng ``Compose`` một lần cho cả lần chạy.

    Tách khỏi :func:`preprocess` để vòng lặp 32k ảnh không dựng lại ``Compose``
    mỗi ảnh (không sai, chỉ chậm), và để test AST kiểm được đủ 4 bước ở **một**
    chỗ.
    """
    return transforms.Compose(
        [
            transforms.Resize(RESIZE),
            transforms.CenterCrop(CROP),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def preprocess(img, torch, transforms):
    """PIL ảnh → tensor CHW đã chuẩn hoá ImageNet.

    Chữ ký giữ nguyên ``(img, torch, transforms)`` như bản cũ trong ``eval.py``
    để không phải sửa chỗ gọi, và để ``torch`` tiếp tục là tham số chứ không
    phải import tầng module.

    ``torch`` hiện không dùng trực tiếp; giữ lại vì (a) chữ ký cũ, (b) vài nơi
    truyền ``torch`` để đảm bảo đúng bản torch đang chạy trong container.
    """
    del torch  # không cần: mọi thứ nằm trong transforms
    return build_transform(transforms)(img)
