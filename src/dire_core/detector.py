"""Nạp checkpoint DIRE — đọc quy ước lớp cuối **từ chính file**, không đoán.

Sự cố đã xảy ra thật (2026-10-28): chọn ``num_classes=2`` cho checkpoint 1 logit
làm ``load_state_dict`` nổ ``size mismatch for fc.weight``, và vì lỗi nằm ở bước
nạp nên **toàn bộ** eval chết sau khi đã trả tiền tải 3.25 GB.

Hai quy ước cùng tồn tại trong thực tế:

=====================  =====================  =====================
checkpoint             ``fc.weight`` shape     hàm lấy ``P(fake)``
=====================  =====================  =====================
DIRE gốc (tác giả)     ``[1, 2048]``          ``sigmoid``
train kiểu CE          ``[2, 2048]``          ``softmax``
=====================  =====================  =====================

Nên ``n_out`` luôn được **đọc** (:func:`fc_out_features`), không có chỗ nào
truyền cứng.
"""

from __future__ import annotations


def unwrap_state(state):
    """Bóc checkpoint về dict tensor thuần.

    File gốc lưu cả ``optimizer``/``epoch`` (269.4 MB so với 97.8 MB của
    ResNet50 fp32), nên phải bóc ``model``/``state_dict``/``net``, rồi bỏ tiền tố
    ``module.`` do ``DataParallel``.
    """
    if isinstance(state, dict):
        for key in ("model", "state_dict", "net", "model_state_dict"):
            value = state.get(key)
            if isinstance(value, dict):
                state = value
                break
    if not isinstance(state, dict):
        raise SystemExit(
            "[!] Checkpoint không phải dict state_dict.\n"
            f"    Nhận được: {type(state).__name__}"
        )
    return {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
    }


def fc_out_features(state: dict) -> int:
    """Số unit của lớp cuối, đọc từ **chính checkpoint** — không đoán theo tên.

    Tìm ``fc.weight`` (bản vendor của tác giả) hoặc ``fc.0.weight``/
    ``classifier.*`` (vài bản timm); trả ``0`` nếu không nhận ra, để chỗ gọi
    quyết định báo lỗi.
    """
    # Bỏ tiền tố `module.` tại đây để hàm dùng được với state dict THÔ (chưa qua
    # `unwrap_state`) — checkpoint train bằng DataParallel có `module.fc.weight`.
    state = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
    }
    for key in ("fc.weight", "fc.0.weight", "classifier.weight", "last_linear.weight"):
        w = state.get(key)
        shape = getattr(w, "shape", None)
        if shape is None:
            continue
        try:
            dims = [int(x) for x in shape]
        except (TypeError, ValueError):
            continue
        if len(dims) == 2:
            return dims[0]
    return 0


def load_detector(ckpt, torch, resnet50, device="cpu"):
    """ResNet50 nhị phân của DIRE → ``(model, n_out)``.

    ``torch``/``resnet50`` được truyền vào để module này **không** import torch
    ở tầng module (test chạy được không cần torch; container đọc CSV không cần
    cài torch).
    """
    raw = torch.load(ckpt, map_location="cpu")
    state = unwrap_state(raw)

    n_out = fc_out_features(state)
    if n_out not in (1, 2):
        raise SystemExit(
            f"[!] Không đọc được lớp cuối của checkpoint: {ckpt}\n"
            f"    Có {len(state)} tensor, không thấy `fc.weight` shape [n, 2048].\n"
            f"    Khoá đầu tiên: {list(state)[:5]}\n"
            "    Kiểm file có đúng là ResNet50 của DIRE không "
            "(`modal run -m dire_modal.download::verify` in sha256 để đối chiếu)."
        )

    model = resnet50(num_classes=n_out)
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise SystemExit(
            f"[!] Checkpoint không khớp kiến trúc ResNet50: {ckpt}\n"
            f"    n_out đọc từ checkpoint = {n_out}\n"
            f"    {exc}\n"
            "    Gợi ý: `fc.weight` [1,2048] = DIRE gốc (BCE, 1 logit); [2,2048] = 2 lớp."
        ) from exc
    model.to(device).eval()
    return model, n_out
