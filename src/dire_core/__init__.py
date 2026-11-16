"""``dire_core`` — lõi DIRE **thuần**: không cần Modal, không cần tài khoản cloud.

Vì sao tách khỏi ``dire_modal``
--------------------------------
Trước đây mọi thứ nằm trong ``dire_modal`` và mọi module đều
``from .common import app, ...``. Hệ quả đo được: **không cài ``modal`` thì
không import được cả phần logic** — người chỉ muốn chạy ``dire detect anh.jpg``
trên máy mình vẫn bị buộc ``pip install modal`` và có một ``modal.App``.

Module này là **nguồn sự thật duy nhất** cho ba thứ quyết định con số:

1. tiền xử lý (``Resize 256 → CenterCrop 224 → ToTensor → Normalize``),
2. cách đọc lớp cuối của checkpoint (``fc.weight`` 1 logit hay 2 lớp),
3. cách đổi logit → ``P(fake)`` (``sigmoid`` hay ``softmax``).

Cả ``dire_modal.eval`` (cloud) và ``dire_local`` (máy nhà) đều **import từ đây**;
không bên nào được có bản sao thứ hai. Lý do không phải cho gọn: repo này đã
từng có hai pipeline lệch nhau (``demo.py`` của tác giả thiếu ``Normalize``) và
nó làm điểm demo lệch điểm paper mà không có exception nào.

Quy tắc cứng (có test)
----------------------
    import dire_core          # ✅ chỉ cần Python thuần
    import dire_core          # ❌ không được kéo modal, torch ở tầng module

``torch`` được truyền **vào** các hàm (``torch=...``, ``transforms=...``) chứ
không import ở cấp module — nhờ vậy test chạy được mà không cần torch, và
container không cần cài torch cho các job chỉ đọc CSV.
"""

from .batch import Detector, batch_logits
from .detector import fc_out_features, load_detector, unwrap_state
from .labels import FAKE, REAL, label_of
from .preprocess import CROP, MEAN, PREPROC, RESIZE, STD, build_transform, preprocess
from .score import prob_fake, sigmoid

__all__ = [
    "CROP",
    "Detector",
    "FAKE",
    "MEAN",
    "PREPROC",
    "REAL",
    "RESIZE",
    "STD",
    "batch_logits",
    "build_transform",
    "fc_out_features",
    "label_of",
    "load_detector",
    "preprocess",
    "prob_fake",
    "sigmoid",
    "unwrap_state",
]
