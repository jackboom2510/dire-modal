"""Chạy model theo batch và trả ``P(fake)`` — dùng chung cho CLI, API, eval.

Trước đây vòng lặp ``stack → to(device) → model → tolist → prob_fake`` nằm rải
trong ``eval.score_pair`` và (nếu viết thêm CLI) sẽ bị chép sang ``dire_local``.
Đó là đúng kiểu trôi đã làm ``demo.py`` của tác giả lệch ``datasets.py``: hai vòng
lặp "gần giống nhau" nhưng khác một bước tiền xử lý hoặc khác thứ tự nhãn.

Module này gom **một** vòng lặp. Nó nhận tensor đã tiền xử lý, không biết ảnh
đến từ tar hay từ thư mục — nhờ vậy eval (tar) và CLI (file) dùng chung.
"""

from __future__ import annotations

from .score import prob_fake


def batch_logits(model, tensors, torch, device, batch_size: int = 64) -> list:
    """``list[tensor]`` → ``list[logit(s)]`` (list Python, không phải tensor).

    Trả raw logit chứ không trả xác suất: nơi gọi biết ``n_out`` và phải dùng
    đúng ``prob_fake``. Giữ hai bước tách rời là cố ý — nó cho phép test
    ``prob_fake`` bằng giá trị đã biết mà không cần model.
    """
    out: list = []
    if not tensors:
        return out
    full = [tensors[i:i + batch_size] for i in range(0, len(tensors), batch_size)]
    with torch.inference_mode():
        for chunk in full:
            logits = model(torch.stack(chunk).to(device))
            out.extend(logits.detach().cpu().tolist())
    return out


class Detector:
    """Bọc ``(model, n_out, transform)`` để nơi gọi không phải nhớ 3 thứ.

    Dùng:

        det = Detector.load("checkpoints/lsun_adm.pth")
        det.score_pils([img1, img2])      # → [0.98, 0.02]

    Torch/PIL được import **trong** ``load``/``score_pils`` để ``import
    dire_core`` vẫn chạy được trên máy chỉ có Python.
    """

    def __init__(self, model, n_out, torch, transforms, device) -> None:
        self.model = model
        self.n_out = n_out
        self.torch = torch
        self.transforms = transforms
        self.device = device

    # -- nạp ---------------------------------------------------------------
    @classmethod
    def load(cls, weights, device: str | None = None, *, torch=None, torchvision=None):
        """Nạp checkpoint; tự chọn ``cuda`` nếu có."""
        if torch is None or torchvision is None:  # pragma: no cover - cần torch thật
            import torch as _torch
            import torchvision as _torchvision

            torch = torch or _torch
            torchvision = torchvision or _torchvision
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        from .detector import load_detector

        model, n_out = load_detector(
            str(weights), torch, torchvision.models.resnet50, device
        )
        return cls(model, n_out, torch, torchvision.transforms, device)

    # -- chấm --------------------------------------------------------------
    def logits_of(self, pils, batch_size: int = 64) -> list:
        from .preprocess import preprocess

        tensors = [preprocess(img, self.torch, self.transforms) for img in pils]
        return batch_logits(self.model, tensors, self.torch, self.device, batch_size)

    def score_pils(self, pils, batch_size: int = 64) -> list[float]:
        """``list[PIL]`` → ``list[P(fake)]``."""
        return prob_fake(self.logits_of(pils, batch_size), self.n_out)

    # -- metadata ----------------------------------------------------------
    def info(self) -> dict:
        from .preprocess import CROP, MEAN, PREPROC, RESIZE, STD

        return {
            "preproc": PREPROC,
            "resize": RESIZE,
            "crop": CROP,
            "mean": list(MEAN),
            "std": list(STD),
            "n_out": self.n_out,
            "loss": "bce_with_logits" if self.n_out == 1 else "cross_entropy",
            "prob": "sigmoid" if self.n_out == 1 else "softmax",
            "device": str(self.device),
        }
