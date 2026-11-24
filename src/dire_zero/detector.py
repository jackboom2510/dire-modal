"""ResNet-50 cho DIRE — viết lại, không dùng `torchvision` cũng không dùng bản chép của tác giả.

Khác biệt **có chủ ý** so với bài báo: bài báo khởi tạo từ trọng số ImageNet.
Bản này khởi tạo ngẫu nhiên, vì "từ số 0" mà còn nạp 25 triệu tham số của
ImageNet thì chữ "from scratch" chỉ còn đúng một nửa. Hệ quả phải nói rõ:

* với 1 000 ảnh train, ResNet-50 ngẫu nhiên **sẽ** kém hơn bản ImageNet;
* ACC vì thế không so được với Table 2 của bài báo, và không được trình bày như
  thể so được.

Nạp ImageNet là tuỳ chọn (`init="imagenet"`), có ghi vào `train.json` để người
đọc biết con số đến từ đâu. Mặc định: `"scratch"`.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["Bottleneck", "ResNet", "resnet50", "ResNetDetector", "load_dire_detector"]


class Bottleneck(nn.Module):
    """Khối thắt nút: 1×1 → 3×3 → 1×1, cộng tắt, ReLU sau mỗi BN."""

    expansion = 4

    def __init__(self, in_ch: int, width: int, stride: int = 1, groups: int = 1, base: int = 64) -> None:
        super().__init__()
        out = width * self.expansion
        self.conv1 = nn.Conv2d(in_ch, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out)
        self.relu = nn.ReLU(inplace=True)

        self.downsample: nn.Module = nn.Identity()
        if stride != 1 or in_ch != out:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out),
            )

    def forward(self, x: Tensor) -> Tensor:
        idt = self.downsample(x)
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        return self.relu(h + idt)


class ResNet(nn.Module):
    """ResNet chuẩn, đầu ra 1 logit (thật/giả) — không phải 1000 lớp ImageNet."""

    def __init__(self, blocks: tuple[int, ...] = (3, 4, 6, 3), widths: tuple[int, ...] = (64, 128, 256, 512),
                 in_channels: int = 3) -> None:
        super().__init__()
        if in_channels not in (3, 6):
            raise ValueError("ResNet DIRE chỉ hỗ trợ 3 kênh hoặc RGB+DIRE 6 kênh")
        self.in_channels = in_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        stages = []
        in_ch = 64
        for i, (n, w) in enumerate(zip(blocks, widths)):
            for j in range(n):
                stride = 2 if (i > 0 and j == 0) else 1
                stages.append(Bottleneck(in_ch, w, stride))
                in_ch = w * Bottleneck.expansion
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).flatten(1)
        return self.fc(x).squeeze(1)   # logit thô, chưa sigmoid

    def imagenet_alias(self) -> dict[str, str]:
        """Bảng ``tên của mình → tên torchvision`` cho trọng số ImageNet.

        `stages` là một `Sequential` phẳng 16 khối; torchvision gom thành
        `layer1..4`. Bảng này là chỗ duy nhất biết cách gom, và có test vòng
        tròn (đổi tên → nạp lại → tham số không đổi) để nó không lệch âm thầm.
        """
        alias: dict[str, str] = {"stem.0": "conv1", "stem.1": "bn1"}
        leaves = ("conv1", "bn1", "conv2", "bn2", "conv3", "bn3", "downsample.0", "downsample.1")
        sizes = (3, 4, 6, 3)
        i = 0
        for layer, n in enumerate(sizes, start=1):
            for b in range(n):
                for leaf in leaves:
                    alias[f"stages.{i}.{leaf}"] = f"layer{layer}.{b}.{leaf}"
                i += 1
        return alias

    def load_authored(self, state: dict) -> None:
        """Nạp detector **của tác giả** (torchvision naming, `fc` 1 logit).

        Khác `load_imagenet`: bản ImageNet có 1000 lớp nên `fc` bị bỏ và giữ
        khởi tạo ngẫu nhiên. Detector tác giả thì `fc` **chính là** đầu ra
        thật/giả — bỏ nó đi là nạp 23,5 M tham số rồi vẫn đoán ngẫu nhiên, và
        không có lỗi nào báo (mọi shape đều khớp).

        `strict=True` ở cuối để lệch một tensor là nổ ngay tại đây.
        """
        table = self.imagenet_alias()
        own = self.state_dict()
        renamed: dict[str, "Tensor"] = {}
        for my_key, ref_prefix in table.items():
            for suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                full, my_full = f"{ref_prefix}.{suffix}", f"{my_key}.{suffix}"
                if full in state and my_full in own and state[full].shape == own[my_full].shape:
                    renamed[my_full] = state[full]
        fc_w = state.get("fc.weight")
        if fc_w is None:
            raise KeyError(
                "checkpoint không có `fc.weight` — không phải detector của tác giả "
                f"(có {len(state)} key, vd {sorted(state)[:3]})"
            )
        if tuple(fc_w.shape) != (1, own["fc.weight"].shape[1]):
            raise ValueError(
                f"`fc.weight` có shape {tuple(fc_w.shape)}, cần (1, {own['fc.weight'].shape[1]}). "
                "Shape (1000, 2048) là ResNet-50 ImageNet, không phải detector DIRE."
            )
        for suffix in ("weight", "bias"):
            full = f"fc.{suffix}"
            if full in state:
                renamed[f"fc.{suffix}"] = state[full]
        own.update(renamed)
        self.load_state_dict(own, strict=True)

    def load_imagenet(self, state: dict) -> list[str]:
        """Nạp trọng số ImageNet (bỏ `fc` vì 1000 lớp ≠ 1 logit). Trả key bị bỏ qua."""
        table = self.imagenet_alias()
        own = self.state_dict()
        renamed, skipped = {}, []
        for my_key, ref_prefix in table.items():
            for suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                full = f"{ref_prefix}.{suffix}"
                my_full = f"{my_key}.{suffix}"
                if full in state and my_full in own and state[full].shape == own[my_full].shape:
                    renamed[my_full] = state[full]
        if self.in_channels == 6 and "conv1.weight" in state:
            # RGB+DIRE starts from ImageNet ResNet-50: initialize both input
            # groups from the RGB conv and divide by two so equal branches match
            # the response scale of the original 3-channel stem.
            w = state["conv1.weight"]
            if tuple(w.shape) == (64, 3, 7, 7):
                renamed["stem.0.weight"] = torch.cat((w * 0.5, w * 0.5), dim=1)
        skipped = [k for k in own if k not in renamed]
        own.update(renamed)          # thay tại chỗ: giữ nguyên fc và các buffer khác
        self.load_state_dict(own, strict=True)
        return skipped


#: Tổng tham số của ResNet-50 với `fc` 1 logit — con số mà checkpoint tác giả có.
AUTHOR_DETECTOR_PARAMS = 23_510_081


def load_dire_detector(path: str) -> "ResNetDetector":
    """Nạp detector tác giả từ `.pth` (chấm lại R0 — **không** dùng cho R2).

    Vì sao cần cổng riêng thay vì `load_state_dict` trần: trên volume thật,
    ``checkpoints/lsun_adm.pth`` (283 MB, 23,5 M tham số) trông y hệt một
    checkpoint hợp lệ nhưng **không phải** trọng số ADM. Hàm này chỉ nhận đúng
    kiến trúc ResNet-50-1-logit, sai là ném lỗi kèm lý do.
    """
    import torch  # noqa: PLC0415

    raw = torch.load(str(path), map_location="cpu", weights_only=True)
    for key in ("state_dict", "model", "net", "ema"):
        if isinstance(raw, dict) and key in raw and isinstance(raw[key], dict):
            raw = raw[key]
            break
    if not isinstance(raw, dict) or "conv1.weight" not in raw:
        raise KeyError(f"{path}: không thấy `conv1.weight` — không phải detector ResNet")

    # Chỉ đếm **tham số**, không đếm buffer BN: checkpoint thật có tổng `numel`
    # 23.563.254, trong đó 53.173 là running_mean/var/num_batches_tracked.
    n_params = sum(
        int(v.numel()) for k, v in raw.items()
        if hasattr(v, "numel") and not k.endswith(("running_mean", "running_var", "num_batches_tracked"))
    )
    if n_params != AUTHOR_DETECTOR_PARAMS:
        hint = " (đúng cỡ ResNet-50 ImageNet 1000 lớp)" if 25_000_000 < n_params < 26_000_000 else ""
        raise ValueError(
            f"{path}: {n_params:,} tham số, cần {AUTHOR_DETECTOR_PARAMS:,} của detector DIRE{hint}"
        )

    det = ResNetDetector()
    det.backbone.load_authored(raw)
    det.eval()
    return det


def resnet50() -> ResNet:
    return ResNet()


class ResNetDetector(nn.Module):
    """Bọc thêm chuẩn hoá đầu vào + sigmoid, để pipeline không phải nhớ."""

    #: Trung bình/độ lệch ImageNet — detector gốc cũng dùng đúng bộ này.
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, backbone: nn.Module | None = None, in_channels: int = 3) -> None:
        super().__init__()
        if in_channels not in (3, 6):
            raise ValueError("detector chỉ hỗ trợ 3 kênh hoặc RGB+DIRE 6 kênh")
        self.backbone = backbone or ResNet(in_channels=in_channels)
        if int(getattr(self.backbone, "in_channels", in_channels)) != in_channels:
            raise ValueError("in_channels của detector không khớp backbone")
        self.in_channels = in_channels
        base_mean = torch.tensor(self.MEAN)
        base_std = torch.tensor(self.STD)
        mean = base_mean.repeat(in_channels // 3).view(1, in_channels, 1, 1)
        std = base_std.repeat(in_channels // 3).view(1, in_channels, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def logits(self, x: Tensor) -> Tensor:
        return self.backbone((x - self.mean) / self.std)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.logits(x))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
