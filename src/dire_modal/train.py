# src/dire_modal/train.py
"""Train lại ResNet-50 của DIRE để **ĐỐI CHIẾU** trọng số tác giả (FR-8).

Đây KHÔNG phải đường báo cáo chính. Mọi câu "DIRE đạt X%" chỉ được nói bằng
checkpoint tác giả (``eval::main`` trên ``lsun_adm.pth``). File này trả lời một
câu khác: *trọng số mà tác giả phát hành có tái tạo được không, và nếu lệch thì
lệch ở đâu* — xem ``docs/adr/adr-0004-retrain-for-reconciliation.md``.

Hợp đồng (spec.md §7 FR-8):
  * Đầu vào là **DIRE map đã phát hành** (tar ``/raw/dire/train/...``), nên
    KHÔNG cần ``compute_dire`` và KHÔNG cần weight ADM ``lsun_bedroom.pt``.
    Docstring bản cũ giả định ``/shards`` — sai: ``demo_train`` đọc tar ``/raw``
    trực tiếp, và ``pack`` không nằm trên đường bắt buộc.
  * Siêu tham số chép từ tác giả (``vendor/utils/config.py``): 1 logit +
    ``BCEWithLogitsLoss``, batch 64, Adam lr 1e-4 β1 0.9, 256 → crop 224,
    hflip 0,5, seed 3407, 400 epoch, earlystop 5 kèm ``lr/10``.
  * ``--mode plan`` là **mặc định**: in cấu hình + ước tính chi phí, 0 credit.
  * Chọn epoch/ngưỡng **chỉ trên val**; test không được chạm.
  * Ghi ``/ckpt/results/<exp>/`` với ``detector=dire_retrained`` +
    ``provenance: retrained`` + ``loss_log.json`` + ``run.json``.
  * Resume được từng epoch (container chết không mất tiến độ).

Trạng thái: **chưa port** — vẫn ``NotImplementedError``. Việc này thuộc mốc M6
của spec.md, và chỉ nên làm sau khi đã đo ``sec/epoch`` thật bằng một epoch
(~$0.1) rồi mới cam kết 400 epoch.
"""

import modal
from .common import app, gpu_image, vol_shards, vol_ckpt


@app.function(
    image=gpu_image,
    volumes={"/shards": vol_shards, "/ckpt": vol_ckpt},
    gpu="A100",
    timeout=24 * 3600,
    retries=modal.Retries(max_retries=3, initial_delay=60.0),
)
def main(exp: str = "lsun_adm_repro", resume: bool = True) -> None:
    """modal run -m dire_modal.train::main --exp lsun_adm_repro"""
    # TODO (FR-8): port vòng train của tác giả
    # 1. build DataLoader từ tar /raw/dire/train/lsun_bedroom/{real,adm}
    #    (KHÔNG cần /shards; giữ volume đó chỉ vì pack có thể đã sinh ra)
    # 2. nếu resume & /ckpt/results/{exp}/last.pt tồn tại → load
    # 3. train loop: lưu epoch_{i}.pt + last.pt mỗi epoch, ghi loss_log.json
    # 4. chọn epoch tốt nhất trên VAL, rồi mới eval trên TEST
    # 5. ghi run.json với detector=dire_retrained + provenance=retrained
    raise NotImplementedError("FR-8: port vòng train của Wang (xem docs/adr/adr-0004)")
