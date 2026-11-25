"""Optional ResNet scoring for a *computed* DIRE map (never raw RGB).

Torch and detector weights are loaded only after the user explicitly chooses a
checkpoint and runs inference. Author weights and project-trained weights have
separate loaders and provenance/threshold handling.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

__all__ = ["AUTHORED_PATH", "PROJECT_TRAIN_ROOT", "suggested_path", "threshold_from_train_json", "predict_map"]

AUTHORED_PATH = r"F:\DiffusionForensics\checkpoints\lsun_adm.pth"
PROJECT_TRAIN_ROOT = r"D:\Documents\Code\dire-modal\modal-runs\runs\rebuild\rebuild-2026-10-14\train"


def suggested_path(source: str, generator: str = "gen-adm") -> str:
    if source == "author":
        return AUTHORED_PATH
    if source == "project":
        return f"{PROJECT_TRAIN_ROOT}\\{generator}\\detector.pth"
    return ""


def threshold_from_train_json(weights_path: str | Path, source: str) -> tuple[float, str]:
    """Read the project's recorded validation cutoff; author weights use 0.5."""
    if source == "author":
        return 0.5, "0.5 mặc định · detector gốc tác giả"
    meta_path = Path(weights_path).parent / "train.json"
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        value = float(metadata["threshold_from_val"])
        threshold_space = metadata.get("threshold_space", "logit")
        threshold_source = metadata.get("threshold_source", "")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return 0.5, "0.5 dự phòng · không có train.json/ngưỡng hợp lệ"
    if threshold_source != "val" or not math.isfinite(value):
        return 0.5, "0.5 dự phòng · ngưỡng train.json không được chọn trên val"
    if threshold_space not in ("logit", "probability"):
        return 0.5, f"0.5 dự phòng · thang ngưỡng không hỗ trợ ({threshold_space})"
    # This repo's saved detector forward returns sigmoid probabilities and both
    # val selection and later metric evaluation operate on those values. Older
    # train.json files label the field "logit"; preserve the numeric cutoff to
    # remain compatible with their recorded scoring semantics instead of
    # sigmoid-transforming it a second time.
    if not 0.0 <= value <= 1.0:
        return 0.5, "0.5 dự phòng · ngưỡng ngoài [0, 1]"
    note = "probability" if threshold_space == "probability" else "legacy logit label · cutoff dùng nguyên trạng"
    return value, f"ngưỡng chọn trên val ({note})"


@lru_cache(maxsize=4)
def _load_detector_cached(path_text: str, source: str, mtime_ns: int, size: int):
    del mtime_ns, size  # part of the cache key: reload when an operator replaces weights
    import torch  # noqa: PLC0415
    from dire_zero.detector import ResNetDetector, load_dire_detector  # noqa: PLC0415

    if source == "author":
        model = load_dire_detector(path_text)
    elif source == "project":
        raw = torch.load(path_text, map_location="cpu", weights_only=True)
        if isinstance(raw, dict):
            for wrapper in ("state_dict", "model", "net", "ema"):
                if isinstance(raw.get(wrapper), dict):
                    raw = raw[wrapper]
                    break
        if not isinstance(raw, dict):
            raise ValueError("checkpoint detector phải là state_dict tensor")
        state = {
            (key.removeprefix("module.") if isinstance(key, str) else key): value
            for key, value in raw.items()
        }
        model = ResNetDetector(in_channels=3)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                "checkpoint tự train không khớp ResNetDetector 3-kênh của DIRE; "
                "hãy chọn đúng detector.pth từ thư mục train/gen-*/"
            ) from exc
        model.eval()
    else:
        raise ValueError("nguồn detector không hợp lệ; chọn author hoặc project")
    model.to(device="cpu").eval()
    threshold, threshold_note = threshold_from_train_json(path_text, source)
    return model, threshold, threshold_note


def predict_map(dire_map, weights_path: str | Path, source: str) -> dict:
    """Return P(fake) from a raw CHW `[0,2]` DIRE map, using the chosen weights."""
    import numpy as np  # noqa: PLC0415

    path = Path(weights_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"không thấy detector checkpoint: {path}")
    arr = np.asarray(dire_map, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"DIRE map cần shape (3,H,W), nhận {arr.shape}")
    if not np.isfinite(arr).all() or arr.min() < -1e-5 or arr.max() > 2.0001:
        raise ValueError("DIRE map phải hữu hạn và nằm trên thang [0,2]")
    height, width = arr.shape[1:]
    if height < 224 or width < 224:
        raise ValueError(f"DIRE map tối thiểu 224×224, nhận {width}×{height}")
    top, left = (height - 224) // 2, (width - 224) // 2
    cropped = arr[:, top:top + 224, left:left + 224]

    import torch  # noqa: PLC0415

    st = path.stat()
    model, threshold, threshold_note = _load_detector_cached(
        str(path), source, st.st_mtime_ns, st.st_size
    )
    tensor = torch.from_numpy(cropped.copy()).unsqueeze(0)
    with torch.inference_mode():
        logit = float(model.logits(tensor).reshape(-1)[0].item())
        probability = float(torch.sigmoid(torch.tensor(logit)).item())
    if not math.isfinite(logit):
        raise ValueError(f"detector trả logit không hữu hạn: {logit}")
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"detector trả xác suất ngoài [0,1]: {probability}")
    return {
        "logit": logit,
        "p_fake": probability,
        "threshold": threshold,
        "prediction": "ẢNH SINH" if probability >= threshold else "ẢNH THẬT",
        "threshold_note": threshold_note,
        "weights": path.name,
        "source": source,
    }
