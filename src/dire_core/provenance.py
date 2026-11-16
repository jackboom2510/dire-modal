"""Mức tái lập — ba thành phần nào là của tác giả, thành phần nào của mình.

Vì sao có file này
------------------
Repo này nói "tái lập DIRE", nhưng đường eval chính đang chạy **dùng lại cả
hai thứ quyết định phép đo**:

* ``dire/**``  — DIRE map do tác giả tính (ADM-LSUN + DDIM S=20), và
* ``checkpoints/*.pth`` — detector do tác giả train.

Chạy ResNet-50 của họ trên map của họ rồi so với Table 2 của họ **không** phải
tái lập thí nghiệm. Nó là *chấm lại* (re-scoring): bằng chứng rằng **đường đo
của mình** (Resize→CenterCrop→Normalize, cách đọc ``fc.weight``, cách đổi
logit → ``P(fake)``, cách tính AP) khớp với đường đo của họ. Đó là việc cần
làm, nhưng hẹp hơn nhiều so với câu "tái lập được DIRE".

Cụ thể cái **không** được suy ra từ một lần chấm lại:

* DIRE map có đúng không — nếu tác giả tính sai, kết quả của mình vẫn đẹp,
  vì mình dùng đúng map sai đó.
* Detector có train lại được không — FR-8 mới trả lời câu đó, và nó **độc
  lập** với việc tính map.
* Ảnh sinh ra có tái tạo được không — ảnh gốc cũng là của tác giả; sinh lại
  cần generator + seed, và ba generator API đóng (DALL·E 2, IF, Midjourney)
  thì **không bao giờ** tái lập được.

Ba thành phần, không phải một cái thang
---------------------------------------
``dire_maps``, ``detector``, ``images`` là **trục độc lập**. Làm được
``detector`` mà không làm được ``dire_maps`` là chuyện bình thường (FR-8 dùng
map phát hành). Vì vậy hàm :func:`claim` nhận ba cờ "mình tự làm" chứ không
nhận một con số cấp độ — một con số sẽ che mất việc mình đang tự làm phần nào.

Quy ước đặt tên: ``R0`` = không tự làm thành phần nào, ``R1`` = tự tính map,
``R2`` = tự train detector, ``R3`` = tự sinh ảnh. Tên chỉ để đọc log; **nhãn
đầy đủ luôn liệt kê thành phần**, vì ``R2`` một mình và ``R2`` cộng ``R1`` là
hai mức bằng chứng rất khác nhau.

Chi phí để lên mức (số đã có trong repo, không ước thêm ở đây):

* ``dire_maps``: ``lsun_bedroom.pt`` 2,5 GB (đang trong ``NEVER``) + GPU;
  tự tính 200k map = 8 M NFE ⇒ **$22–150** tuỳ ms/NFE.
* ``detector``: FR-8 ⇒ **$4–56** tuỳ GPU và early-stop.
* ``images``: ``rawx`` 4,48 GB + generator; ba generator API đóng ⇒ phần đó
  ghi ``null`` kèm lý do, không điền 0.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

#: Thành phần của phép đo, theo đúng thứ tự trong chuỗi DIRE.
COMPONENTS = ("dire_maps", "detector", "images")

#: Nhãn ngắn cho từng thành phần khi in log/CSV.
COMPONENT_VI = {
    "dire_maps": "DIRE map",
    "detector": "detector (trọng số)",
    "images": "ảnh gốc real+fake",
}

#: Việc phải làm để tự tạo thành phần đó (in ra khi còn thiếu).
NEEDS = {
    "dire_maps": "ADM-LSUN 2.5 GB + DDIM S=20 (tier đang NEVER) ⇒ $22–150 GPU",
    "detector": "FR-8: train lại trên split của tác giả ⇒ $4–56 GPU",
    "images": "generator + seed; DALL·E 2 / IF / Midjourney không tái lập được",
}


@dataclass(frozen=True)
class Claim:
    """Câu được phép nói, suy ra từ việc thành phần nào do mình tạo.

    Không dựng tay object này trong code — dùng :func:`claim`, để nhãn và câu
    "chứng minh / không chứng minh" luôn đi cùng nhau và không ai sửa một nửa.
    """

    label: str
    title: str
    self_made: tuple[str, ...]
    author_made: tuple[str, ...]
    proves: tuple[str, ...]
    does_not_prove: tuple[str, ...]
    caveat: str

    @property
    def data_source(self) -> str:
        """Chuỗi ngắn để ghi vào cột ``data_source`` của CSV."""
        parts = [
            f"self:{c}" if c in self.self_made else f"authors:{c}" for c in COMPONENTS
        ]
        return "+".join(parts)

    @property
    def is_full(self) -> bool:
        return not self.author_made

    def as_dict(self) -> dict:
        d = asdict(self)
        # `asdict` giữ nguyên tuple; JSON biến chúng thành list ⇒ nếu để nguyên
        # thì `json.loads(write(doc)) != doc` và test round-trip đỏ (đã xảy ra).
        for k in ("self_made", "author_made", "proves", "does_not_prove"):
            d[k] = list(d[k])
        d["data_source"] = self.data_source
        d["full"] = self.is_full
        d["needs"] = {c: NEEDS[c] for c in self.author_made}
        return d


#: Nhãn ngắn cho từng thành phần khi mình TỰ làm. Nhãn của một lần chạy là
#: phép HỢP các nhãn này (``"R1+R2"``), không phải một cấp bậc — vì ba thành
#: phần độc lập: train lại detector được mà không cần tính lại map (FR-8).
TOKEN = {"dire_maps": "R1", "detector": "R2", "images": "R3"}


def claim(dire_maps: bool = False, detector: bool = False, images: bool = False) -> Claim:
    """Suy ra câu được phép nói từ ba cờ "mình tự làm thành phần này".

    Mặc định **toàn bộ là của tác giả** — đúng với đường ``01_repro.sh`` hiện
    tại. Phải bật cờ một cách có ý thức, và khi bật thì ``run.json`` ghi lại
    ``git_sha`` + ``tiers`` để người đọc kiểm được mình đã làm gì.
    """
    flags = (bool(dire_maps), bool(detector), bool(images))
    self_made = tuple(c for c, f in zip(COMPONENTS, flags) if f)
    author_made = tuple(c for c, f in zip(COMPONENTS, flags) if not f)
    label = "+".join(TOKEN[c] for c in COMPONENTS if c in self_made) or "R0"
    title = ("chấm lại trên đặc trưng của tác giả" if not self_made
             else "tự làm: " + ", ".join(COMPONENT_VI[c] for c in self_made))

    proves: list[str] = []
    if "dire_maps" not in self_made:
        proves.append("đường ĐO của mình khớp đường đo của tác giả (tiền xử lý, "
                      "đọc checkpoint, logit→P(fake), ACC/AP)")
    if "detector" in self_made:
        proves.append("công thức train (split, siêu tham số, epoch) tái tạo được "
                      "checkpoint đã phát hành")
    if "dire_maps" in self_made:
        proves.append("pipeline DIRE (inversion + reconstruction + |x−x̂|) tái tạo "
                      "được map đã phát hành")
    if "images" in self_made:
        proves.append("chuỗi sinh ảnh tái tạo được")

    not_proves: list[str] = []
    if "dire_maps" not in self_made:
        not_proves.append("DIRE map có ĐÚNG không — map sai thì kết quả vẫn đẹp, "
                          "vì mình dùng đúng map sai đó")
    if "detector" not in self_made:
        not_proves.append("detector có train lại được không (cần FR-8)")
    if "images" not in self_made:
        not_proves.append("ảnh real/fake có sinh lại được không")

    if not author_made:
        caveat = "Tự tạo cả 3 thành phần: đây là tái lập đầy đủ, được gọi là " \
                 "\"tái lập thí nghiệm\"."
    elif len(author_made) == len(COMPONENTS):
        caveat = ("0/3 thành phần do mình tạo — đây là CHẤM LẠI, không phải tái "
                  "lập: cả DIRE map và trọng số đều của tác giả. Chỉ được nói "
                  "\"khớp số paper\", không được nói \"tái lập DIRE\".")
    else:
        missing = ", ".join(COMPONENT_VI[c] for c in author_made)
        caveat = (f"tự làm {len(self_made)}/3 thành phần; còn dùng lại của tác giả: "
                  f"{missing}. Chỉ được nói \"khớp số\" cho phần chưa tự làm.")

    return Claim(
        label=label,
        title=title,
        self_made=self_made,
        author_made=author_made,
        proves=tuple(proves),
        does_not_prove=tuple(not_proves),
        caveat=caveat,
    )


def banner(c: Claim, width: int = 78) -> list[str]:
    """Khối 5–7 dòng in ở đầu mọi báo cáo — để bảng không bị đọc quá lời."""
    lines = [
        "=" * width,
        f"MỨC TÁI LẬP: {c.label} — {c.title}",
        f"  của tác giả: {', '.join(COMPONENT_VI[x] for x in c.author_made) or '(không)'}",
        f"  tự làm    : {', '.join(COMPONENT_VI[x] for x in c.self_made) or '(không)'}",
        f"  ⇒ {c.caveat}",
    ]
    if c.self_made:
        # Cờ là TỰ KHAI: code không tự kiểm được "map này do mình tính". Nói ra
        # để người đọc biết mức bằng chứng thật của nhãn, không phải tin tuyệt đối.
        lines.append(
            "  ⚠ phần 'tự làm' là CỜ TỰ KHAI, không có bằng chứng tự động trong "
            "lần chạy này"
        )
    lines.append("=" * width)
    return lines
