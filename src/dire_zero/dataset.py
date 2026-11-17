"""Đọc **cây DiffusionForensics** trên ổ của bạn — tar.gz, không phải thư mục ảnh.

Vì sao phải có lớp riêng
------------------------
Cây thật trên máy::

    DiffusionForensics/
    ├── images/{train,val,test}/{lsun_bedroom,imagenet,celebahq}/{real,adm,pndm,…}.tar.gz
    ├── checkpoints/{lsun_adm,…}.pth          (detector của tác giả — chỉ dùng cho R0)
    ├── recons/test/lsun_bedroom/lsun_bedroom/{real,ldm,pndm}.tar.gz
    └── dire/{train,val,test}/…

Bốn quyết định trong đó **không phải chi tiết kỹ thuật**, mà là quyết định về
ý nghĩa của phép đo:

* **`real.tar.gz` = ảnh thật. Mọi file khác = ảnh do máy sinh**, và tên file
  chính là tên generator. Đây là nguồn nhãn duy nhất; đoán theo tên thư mục sẽ
  sai ở `sdv1_test.tar.gz` và ở cặp `stylegan` (train/val) — `stylegan_official`
  (test): cùng một generator, hai tên, nên có `GENERATOR_ALIASES`.
* **`dire/` là DIRE map của tác giả**, không phải ảnh. Dùng nó cho đường
  from-scratch là vi phạm cổng G0 — ở đây nó bị **từ chối**, trừ khi bật
  `--allow-published` để chấm lại R0, và khi đó nhãn là `R0`.
* **`recons/` là reconstruction của tác giả** — dùng để đối chiếu số học DDIM
  (việc B), không dùng để train.
* **`imagenet/` không cùng phân bố với `lsun_bedroom/`**. ADM 256 vô điều kiện
  được train trên LSUN-Bedroom; ảnh ImageNet "thật" cũng bị tái tạo kém ⇒
  khoảng cách DIRE giữa hai lớp co lại. Trộn hai dataset vào một lần đo là tự
  làm hỏng kết quả mà không có lỗi nào báo.

Nhãn mức tái lập phụ thuộc **ảnh giả lấy từ đâu**, không phụ thuộc ảnh thật
(ảnh thật luôn là dữ liệu của người khác — LSUN):

===============  ==============================================
nguồn ảnh giả    nhãn được phép nói
===============  ==============================================
`gen` (tự sinh)  ``R1+R2+R3`` — map, detector, ảnh đều của mình
`author` (adm,…) ``R1+R2``   — ảnh là của tác giả, không phải R3
===============  ==============================================

Ghi sai một bậc ở đây là tự khen mình một thành phần chưa làm.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import re
import shutil
import tarfile
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from .data import sha256_file, stable_key

__all__ = [
    "Item",
    "Scan",
    "DataCard",
    "IntegrityError",
    "drive_candidates",
    "resolve_root",
    "scan_tree",
    "select",
    "ingest",
    "write_manifest_tsv",
    "read_manifest_tsv",
    "verify_manifest",
    "verify_rows",
    "write_data_card",
    "MANIFEST_COLUMNS",
    "IMAGE_SUFFIXES",
    "SPLITS",
    "generator_of",
    "label_for_generator",
    "choose_out_root",
    "estimate_output_bytes",
    "free_bytes",
    "OutChoice",
    "OutOfSpaceError",
    "DEFAULT_OUT_ROOT",
    "DEFAULT_OUT_FALLBACKS",
]

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
SPLITS = ("train", "val", "test")

#: Tên file trong cây được hiểu là ảnh thật.
REAL_STEM = "real"

#: Tên **khác nhau cho cùng một generator**. Cây thật có ``stylegan`` ở train/val
#: nhưng ``stylegan_official`` ở test; không gộp thì bảng kết quả có hai dòng cho
#: cùng một thứ, và "train trên stylegan, test trên stylegan_official" trông như
#: hai generator khác nhau. Đây là **quyết định**, không phải suy đoán:
#: ``diff-stylegan`` và ``diff-projectedgan`` là generator KHÁC (Diffusion-styleGAN)
#: nên cố ý không nằm trong bảng này.
GENERATOR_ALIASES = {
    "stylegan_official": "stylegan",
}

MANIFEST_COLUMNS = (
    "png", "label", "generator", "split", "dataset", "source_tar", "member", "sha256", "width", "height",
)


class IntegrityError(RuntimeError):
    """Dataset có vấn đề làm kết quả vô nghĩa. Không phải cảnh báo."""


# --------------------------------------------------------------------------- #
# Đường dẫn Windows / Git Bash
# --------------------------------------------------------------------------- #

def drive_candidates(path: str | Path) -> list[str]:
    """Các dạng tương đương của một đường dẫn ổ đĩa, theo thứ tự thử.

    ``F:/x`` → ``["F:/x", "/f/x"]``; ``/f/x`` → ``["/f/x", "F:/x"]``. Chỉ đổi
    **dạng**, không đoán ổ — đoán sai ổ là đọc nhầm dữ liệu của người khác.
    """
    raw = str(path).replace("\\", "/")
    m = re.match(r"^([A-Za-z]):/(.*)$", raw)
    if m:
        return [raw, f"/{m.group(1).lower()}/{m.group(2)}"]
    m = re.match(r"^/([A-Za-z])/(.*)$", raw)
    if m:
        return [raw, f"{m.group(1).upper()}:/{m.group(2)}"]
    return [raw]


def resolve_root(path: str | Path) -> Path:
    """Trả thư mục gốc **tồn tại**; tự lùi vào ``DiffusionForensics`` nếu có.

    Người dùng thường trỏ vào thư mục cha (``F:/du-lieu``) chứ không phải thư mục
    trong (``F:/du-lieu/DiffusionForensics``). Tự lùi một cấp là tiện, **không**
    phải đoán: chỉ lùi khi thấy đúng dấu hiệu (``images/`` nằm trong đó).
    """
    tried = drive_candidates(path)
    for cand in tried:
        p = Path(cand)
        if not p.exists():
            continue
        nested = p / "DiffusionForensics"
        if (p / "images").is_dir():
            return p
        if (nested / "images").is_dir():
            return nested
        return p
    raise FileNotFoundError(
        "không thấy thư mục dữ liệu. Đã thử:\n  "
        + "\n  ".join(tried)
        + "\nKiểm lại: ổ đã cắm chưa; trong Git Bash đường dẫn là /f/… chứ không "
        "phải F:/…; tên thư mục có dấu cách thì phải để trong ngoặc kép."
    )


# --------------------------------------------------------------------------- #
# Tar
# --------------------------------------------------------------------------- #

def _is_image_member(name: str) -> bool:
    return name.lower().endswith(IMAGE_SUFFIXES)


def peek_archive(path: str | Path, limit: int | None = None) -> dict:
    """Đếm ảnh + tổng byte **không ghi ra đĩa**.

    `tarfile` vẫn phải bung gzip để đi hết archive ⇒ ~1 giây/100 MB. Cây đầy đủ
    nặng 12 GB, nên `scan_tree` **không** đếm; chỉ đếm phần thật sự được chọn
    (:func:`count_items`). `limit` dừng sớm khi chỉ cần "ít nhất bao nhiêu".
    """
    n, total = 0, 0
    with tarfile.open(path, "r:*") as tf:
        for m in tf:
            if m.isfile() and _is_image_member(m.name):
                n += 1
                total += m.size
                if limit is not None and n >= limit:
                    break
    return {"path": str(path), "n_images": n, "bytes": total}


def generator_of(tar_path: str | Path) -> str:
    """Tên generator từ tên file: ``adm.tar.gz`` → ``adm``; ``sdv1_test`` → ``sdv1``.

    ``sdv1_test.tar.gz`` trong ``images/test/lsun_bedroom/`` là **cùng một
    generator** với ``sdv1`` ở chỗ khác, chỉ khác hậu tố. Giữ nguyên hậu tố thì
    bảng kết quả sẽ có hai dòng ``sdv1`` và ``sdv1_test`` cho cùng một thứ.
    """
    stem = Path(tar_path).name
    for suffix in (".tar.gz", ".tgz", ".tar"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    stem = re.sub(r"_(test|val|train)$", "", stem).lower()
    return GENERATOR_ALIASES.get(stem, stem)


def label_for_generator(gen: str) -> int:
    return 0 if gen == REAL_STEM else 1


# --------------------------------------------------------------------------- #
# Quét
# --------------------------------------------------------------------------- #

@dataclass
class Item:
    """Một **nguồn** dữ liệu: một file PNG, hoặc một archive + tên member."""

    source: Path                    # .tar.gz hoặc .png
    member: str | None = None       # None ⇒ source là file ảnh
    label: int = 0
    generator: str = REAL_STEM
    split: str = ""
    dataset: str = ""
    n_images: int | None = None     # None = chưa đếm (đếm sau, chỉ khi được chọn)
    raw_name: str = ""              # tên file gốc, trước khi áp GENERATOR_ALIASES
    sha256: str = ""                # sau ingest: sha của PNG chuẩn hoá
    width: int = 0
    height: int = 0
    png: str = ""                   # đường dẫn PNG sau ingest

    @property
    def key(self) -> str:
        return f"{self.split}/{self.dataset}/{self.generator}"

    def as_row(self) -> dict:
        return {
            "png": self.png,
            "label": self.label,
            "generator": self.generator,
            "split": self.split,
            "dataset": self.dataset,
            "source_tar": str(self.source),
            "member": self.member or "",
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class Scan:
    root: Path
    layout: str                     # "diffusionforensics" | "folder"
    items: list[Item] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def matrix(self) -> dict[str, dict[str, list[str]]]:
        """``dataset → split → [generator]`` — bảng dùng được để đọc nhanh."""
        out: dict[str, dict[str, list[str]]] = {}
        for it in self.items:
            out.setdefault(it.dataset, {}).setdefault(it.split, [])
            if it.generator not in out[it.dataset][it.split]:
                out[it.dataset][it.split].append(it.generator)
        for ds_, splits in out.items():
            for sp in splits:
                splits[sp].sort(key=lambda g: (g != REAL_STEM, g))
        return out

    def generators(self, dataset: str | None = None, split: str | None = None) -> list[str]:
        seen = {
            it.generator
            for it in self.items
            if (dataset is None or it.dataset == dataset) and (split is None or it.split == split)
        }
        return sorted(seen, key=lambda g: (g != REAL_STEM, g))


def scan_tree(
    root: str | Path,
    allow_published: bool = False,
) -> Scan:
    """Quét cây, **không** giải nén. Trả `Scan` kèm ghi chú/cảnh báo.

    ``dire/`` bị từ chối (trừ ``allow_published=True``, dùng cho chấm lại R0).
    ``recons/`` bị bỏ qua nhưng **có ghi chú**, vì người dùng thấy nó trên đĩa
    và sẽ thắc mắc tại sao nó không được dùng.
    """
    root = Path(root)
    scan = Scan(root=root, layout="unknown")

    images = root / "images"
    if images.is_dir():
        scan.layout = "diffusionforensics"
        for split_dir in sorted(p for p in images.iterdir() if p.is_dir()):
            split = split_dir.name.lower()
            for ds_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                for tar in sorted(ds_dir.glob("*.tar.gz")):
                    gen = generator_of(tar)
                    scan.items.append(
                        Item(
                            source=tar,
                            label=label_for_generator(gen),
                            generator=gen,
                            split=split,
                            dataset=ds_dir.name.lower(),
                            raw_name=tar.name,
                        )
                    )
                _warn_extracted_dirs(scan, ds_dir, root)
        if (root / "dire").is_dir():
            if allow_published:
                scan.notes.append(
                    "dire/ có mặt và --allow-published đã bật ⇒ chỉ dùng để CHẤM LẠI (R0), "
                    "không được trộn vào đường from-scratch (G0)"
                )
            else:
                scan.notes.append(
                    "dire/ (DIRE map của tác giả) bị BỎ QUA. Đường from-scratch không được đọc nó "
                    "(G0). Nếu muốn chấm lại R0, thêm --allow-published."
                )
        if (root / "recons").is_dir():
            scan.notes.append(
                "recons/ (reconstruction của tác giả) bị bỏ qua — chỉ dùng cho việc đối chiếu "
                "số học DDIM, không dùng để train"
            )
        if not scan.items:
            raise IntegrityError(f"{images} không có archive ảnh nào (*.tar.gz)")
        _cross_check(scan)
        return scan

    # ---- bố cục thư mục ảnh rời (real/ + fake/) ----
    subs = {p.name.lower(): p for p in root.iterdir() if p.is_dir()} if root.is_dir() else {}
    real = subs.get("real") or subs.get("0_real")
    fake = subs.get("fake") or subs.get("1_fake")
    if real or fake:
        scan.layout = "folder"
        for label, sub, gen in ((0, real, "real"), (1, fake, "fake")):
            if sub is None:
                continue
            for f in sorted(sub.rglob("*")):
                if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES:
                    scan.items.append(
                        Item(source=f, label=label, generator=gen, split="", dataset="custom", n_images=1)
                    )
        if not scan.items:
            raise IntegrityError(f"không có ảnh trong {root}")
        return scan

    raise IntegrityError(
        f"không nhận ra bố cục trong {root}.\n"
        "Cần một trong hai:\n"
        "  • cây DiffusionForensics: <root>/images/{train,val,test}/<dataset>/<gen>.tar.gz\n"
        "  • thư mục ảnh rời:       <root>/real/*.png + <root>/fake/*.png"
    )


def _warn_extracted_dirs(scan: Scan, ds_dir: Path, root: Path) -> None:
    """Cảnh báo khi có **thư mục ảnh nằm cạnh tar** — chúng bị bỏ qua, và phải nói ra.

    Cây thật có cả ``images/test/lsun_bedroom/real.tar.gz`` **và**
    ``images/test/lsun_bedroom/real/`` (bản giải nén dở), cùng
    ``dire/test/imagenet/imagenet/real/`` (lồng hai tầng). Im lặng bỏ qua thì số
    ảnh thật trong báo cáo khác số người dùng nhìn thấy trên đĩa, và họ không có
    cách nào biết tại sao.
    """
    for sub in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
        n = sum(1 for f in sub.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES)
        rel = sub.relative_to(root)
        if n:
            scan.warnings.append(
                f"{rel}/ là THƯ MỤC ảnh ({n} file) nằm cạnh tar — bị BỎ QUA, chỉ tar được đọc. "
                "Nếu là bản giải nén dở, xoá hoặc chuyển đi để không nhầm số ảnh."
            )
        else:
            scan.warnings.append(f"{rel}/ là thư mục rỗng (giải nén dở?) — bị bỏ qua")


def count_items(items: list[Item], progress=None) -> tuple[int, list[str]]:
    """Đếm ảnh trong **đúng** các archive được chọn. Trả ``(tổng, cảnh báo)``.

    Đếm ở đây thay vì lúc quét vì: (a) cây đầy đủ nặng 12 GB, đếm hết mất vài
    phút mỗi lần chạy; (b) con số chỉ có nghĩa với phần sẽ dùng — đếm
    ``imagenet/train`` khi chạy ``lsun_bedroom/test`` là lãng phí và gây nhiễu.
    """
    total = 0
    warnings: list[str] = []
    for it in items:
        if it.member is not None or not _is_archive(it.source):
            it.n_images = 1
            total += 1
            continue
        info = peek_archive(it.source)
        it.n_images = info["n_images"]
        total += info["n_images"]
        if info["n_images"] == 0:
            warnings.append(
                f"{it.split}/{it.dataset}/{it.source.name}: 0 ảnh (tar rỗng 130 B là lỗi đã biết của dataset)"
            )
        if progress:
            progress(it, info)
    return total, warnings


def _cross_check(scan: Scan) -> None:
    """Cảnh báo về cấu trúc bất đối xứng — thứ dễ làm hỏng phép so sánh."""
    real_splits = {(it.dataset, it.split) for it in scan.items if it.label == 0}
    fake_splits = {(it.dataset, it.split) for it in scan.items if it.label == 1}
    for pair in sorted(fake_splits - real_splits):
        scan.warnings.append(
            f"{pair[0]}/{pair[1]}: có ảnh giả nhưng KHÔNG có real.tar.gz ⇒ không ghép cặp được"
        )
    for ds_ in sorted({it.dataset for it in scan.items}):
        per_split = {it.split for it in scan.items if it.dataset == ds_}
        if "train" in per_split and "test" not in per_split:
            scan.warnings.append(f"{ds_}: có train nhưng không có test — không chấm được")
    gens = {(it.dataset, it.generator) for it in scan.items}
    for ds_, gen in sorted(gens):
        if gen == REAL_STEM:
            continue
        splits = {it.split for it in scan.items if it.dataset == ds_ and it.generator == gen}
        if splits != {"train", "val", "test"}:
            scan.warnings.append(
                f"{ds_}/{gen}: chỉ có ở {sorted(splits)} — bảng cross-model sẽ không đầy đủ"
            )


def select(
    scan: Scan,
    dataset: str | None = "lsun_bedroom",
    splits: tuple[str, ...] = ("train", "test"),
    generators: tuple[str, ...] | None = None,
    include_real: bool = True,
    max_images: int | None = None,
) -> list[Item]:
    """Lọc bộ archive cần dùng. **Không** cắt ảnh ở đây — cắt lúc ingest.

    `max_images` cắt theo **tổng số ảnh** của mỗi archive (giữ tỉ lệ hai lớp),
    không cắt ngẫu nhiên từng file, để còn tái lập được bằng tham số.
    """
    out = []
    for it in scan.items:
        if dataset and it.dataset != dataset:
            continue
        if it.split not in splits:
            continue
        if not include_real and it.label == 0:
            continue
        if generators is not None and it.label == 1 and it.generator not in generators:
            continue
        out.append(it)
    if max_images:
        for it in out:
            if it.n_images is not None:
                it.n_images = min(it.n_images, max_images)
    return out


# --------------------------------------------------------------------------- #
# Ingest: archive → PNG 256²
# --------------------------------------------------------------------------- #

def _norm_image(data: bytes, size: int, truncated_ok: bool = False):
    """bytes → (ảnh RGB 256², w_gốc, h_gốc). Resize cạnh ngắn + cắt giữa.

    ``truncated_ok=True`` là **đường dự phòng thứ hai**: JPEG/PNG bị cắt cụt
    (tải thiếu vài KB cuối, hoặc tar member bị ghi dở) vẫn đọc được phần đầu.
    Pillow mặc định từ chối; bật cờ này thì nó đọc tới chỗ có dữ liệu rồi lấp
    phần thiếu. Ảnh nhận được **không** giống ảnh gốc ở vài dòng cuối, nên chỉ
    dùng khi lần đọc sạch đã thất bại, và người gọi phải ghi lại là đã dùng.
    """
    from PIL import Image, ImageFile  # noqa: PLC0415

    prev = ImageFile.LOAD_TRUNCATED_IMAGES
    if truncated_ok:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
            im = im.convert("RGB")
            short = min(w, h)
            if short != size:
                scale = size / short
                im = im.resize((max(size, round(w * scale)), max(size, round(h * scale))), Image.BICUBIC)
            left = (im.size[0] - size) // 2
            top = (im.size[1] - size) // 2
            im = im.crop((left, top, left + size, top + size))
            return im.copy(), w, h
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = prev


def ingest(
    items: list[Item],
    out_root: str | Path,
    size: int = 256,
    limit_per_archive: int | None = None,
    progress=None,
    on_error: str = "record",
    resume: bool = True,
    rejects_max: int = 50,
) -> tuple[list[Item], dict]:
    """Đổ archive ra PNG ``out_root/<split>/<generator>/<sha16>.png``.

    Tên PNG = 16 ký tự đầu sha256 **của PNG đã chuẩn hoá** (tức của pixel mô
    hình nhìn thấy, không phải của byte trong tar). Nhờ vậy:

    * hai ảnh trùng nội dung ⇒ cùng tên ⇒ ghi đè, không thành hai mẫu;
    * tên gốc kiểu ``000123.jpg`` không mang thông tin, bỏ đi thì chia train/test
      theo hash vẫn ổn định qua các máy;
    * truy ngược được về ``source_tar`` + ``member`` khi cần kiểm.

    Ảnh **trùng giữa hai lớp** (rò rỉ) không thể phát hiện chắc chắn nếu chỉ so
    sha của byte gốc (JPEG khác encoder ra sha khác). Vì vậy ở đây so **sha của
    pixel đã chuẩn hoá** — đó mới là thứ mô hình nhìn thấy.

    Đường dự phòng (``on_error="record"``, mặc định)
    ------------------------------------------------
    Ingest chạy trên cây 12 GB đọc từ ổ ngoài, nên hỏng giữa đường là chuyện
    thường, không phải ngoại lệ hiếm: tar tải thiếu, một JPEG cụt, ổ đầy giữa
    lúc ghi. Trước đây **một** trong số đó làm chết cả lượt chạy và để lại thư
    mục đầu ra nửa vời không manifest. Nay mỗi tầng có đường lui riêng, và
    **mọi** đường lui đều được ghi lại:

    ==========================  ==========================================
    lỗi                         đường dự phòng
    ==========================  ==========================================
    tar không mở được ``r:*``   đọc tuần tự ``r|*``; giữ phần đọc được
    ảnh cụt / hỏng header       đọc lại với ``LOAD_TRUNCATED_IMAGES``
    ghi file lỗi (ổ/khoá file)  thử lại sau 50 ms, rồi ghi nhận vào ``rejects``
    chạy lại sau khi hỏng       bỏ qua PNG đã đúng, **vẫn** kiểm rò rỉ
    ==========================  ==========================================

    Nguyên tắc: **dự phòng thì được, im lặng thì không.** Mọi lần dùng đường lui
    nằm trong ``stats["fallbacks"]`` / ``stats["partial_archives"]``, và
    ``cmd_ingest`` đẩy chúng vào ``data_card.warnings``. Một dataset thiếu 3 %
    mà ``data_card`` nói đủ là loại lỗi tệ hơn hẳn việc chương trình dừng.

    ``on_error="strict"`` trả lại hành vi cũ: archive hỏng ⇒ ``IntegrityError``.
    """
    if on_error not in ("record", "strict"):
        raise ValueError(f"on_error phải là 'record' hoặc 'strict', không phải {on_error!r}")

    if on_error == "strict":
        # `strict` chỉ có nghĩa nếu **không ghi gì** trước khi biết archive đọc được.
        # Không kiểm trước thì ảnh đã ghi ra đĩa rồi mới phát hiện hỏng ở cuối ⇒
        # vẫn là thư mục dở dang, chỉ khác là có thêm thông báo.
        for it in items:
            if not _is_archive(it.source):
                continue
            ok, err = _archive_is_readable(it.source)
            if not ok:
                raise IntegrityError(
                    f"archive hỏng: {it.source}\n  {err}\n"
                    "Chạy lại không kèm --strict để giữ phần đọc được (có ghi chú)."
                )

    out_root = Path(out_root)
    made: list[Item] = []
    pix_seen: dict[str, tuple[int, str]] = _seed_pix_seen(out_root) if resume else {}
    leaks: list[str] = []
    rejects: list[dict] = []
    fallbacks: list[str] = []
    partial_archives: list[dict] = []
    per_archive: dict[str, int] = {}
    reused = 0

    for it in items:
        src_state: dict = {}
        if _is_archive(it.source):
            sources = _iter_tar_images(it.source, limit_per_archive, state=src_state)
        else:                                      # ảnh rời (bố cục real/ + fake/)
            sources = [(it.source.name, it.source.read_bytes())]

        try:
            for name, data in sources:
                reused += _ingest_one(
                    it, name, data, out_root, size, pix_seen, leaks,
                    made, rejects, fallbacks, per_archive, rejects_max, resume,
                )
        except (OSError, EOFError, tarfile.TarError, zlib.error) as exc:
            # Lỗi ngoài dự kiến của archive (vd. file biến mất giữa lúc đọc).
            src_state.setdefault("error", f"{type(exc).__name__}: {exc}")
            src_state["partial"] = True

        if src_state.get("partial"):
            rec = {
                "archive": str(it.source),
                "key": f"{it.split}/{it.dataset}/{it.generator}",
                "read": src_state.get("n", 0),
                "error": src_state.get("error", ""),
                "mode": src_state.get("mode", "r:*"),
            }
            partial_archives.append(rec)
            if on_error == "strict":
                raise IntegrityError(
                    f"đọc archive hỏng: {it.source}\n  {rec['error']}\n"
                    f"  đã đọc được {rec['read']} ảnh trước khi hỏng.\n"
                    "Chạy lại không kèm --strict để giữ phần đọc được (có ghi chú)."
                )
        if progress:
            progress(it)

    stats = {
        "written": len(made) - reused,
        "reused": reused,
        "skipped": len(rejects),
        "rejects": rejects[:rejects_max],
        "n_rejects": len(rejects),
        "leaks": leaks[:10],
        "n_leaks": len(leaks),
        "per_archive": per_archive,
        "partial_archives": partial_archives,
        "fallbacks": fallbacks[:rejects_max],
        "n_fallbacks": len(fallbacks),
    }
    if leaks:
        raise IntegrityError(
            f"{len(leaks)} ảnh trùng pixel giữa hai lớp (đã dừng, chưa ghi file trùng).\n"
            f"  vd: {leaks[:3]}\n"
            "Đây là rò rỉ: ACC sẽ ~100% và vô nghĩa. Loại một trong hai phía rồi chạy lại."
        )
    return made, stats


def _ingest_one(
    it: Item,
    name: str,
    data: bytes,
    out_root: Path,
    size: int,
    pix_seen: dict[str, tuple[int, str]],
    leaks: list[str],
    made: list[Item],
    rejects: list[dict],
    fallbacks: list[str],
    per_archive: dict[str, int],
    rejects_max: int,
    resume: bool = True,
) -> int:
    """Một member: chuẩn hoá → kiểm rò rỉ → ghi. Mọi lỗi thành bản ghi, không ném.

    Trả 1 nếu file đã có sẵn và được dùng lại (resume), 0 nếu vừa ghi.
    """
    try:
        img, _w0, _h0 = _norm_image(data, size)
    except Exception as exc0:              # noqa: BLE001 — ảnh hỏng là dữ liệu, không phải bug
        try:                               # đường lui 2: chấp nhận ảnh cụt
            img, _w0, _h0 = _norm_image(data, size, truncated_ok=True)
        except Exception as exc1:          # noqa: BLE001
            if len(rejects) < rejects_max:
                rejects.append(
                    {"member": name, "generator": it.generator, "split": it.split,
                     "reason": f"{type(exc0).__name__}: {exc0}",
                     "retry": f"{type(exc1).__name__}: {exc1}"}
                )
            return 0
        if len(fallbacks) < rejects_max:
            fallbacks.append(f"{it.split}/{it.generator}:{name} đọc ở chế độ ảnh cụt")

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    png_bytes = buf.getvalue()
    sha_full = hashlib.sha256(png_bytes).hexdigest()
    sha16 = sha_full[:16]          # tên file ngắn; cột manifest giữ sha đầy đủ

    prev = pix_seen.get(sha16)
    if prev and prev[0] != it.label:
        leaks.append(f"{it.generator}:{name} trùng pixel với {prev[1]} ở lớp khác")
        return 0
    pix_seen[sha16] = (it.label, f"{it.generator}:{name}")

    dst = out_root / it.split / it.generator / f"{sha16}.png"
    reused_here = False
    if resume and dst.exists() and dst.stat().st_size == len(png_bytes):
        reused_here = True                 # resume: file đã đúng, không ghi lại
    else:
        for attempt in (0, 1):             # đường lui 3: thử lại 1 lần
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(png_bytes)
                break
            except OSError as exc:
                if attempt:
                    if len(rejects) < rejects_max:
                        rejects.append(
                            {"member": name, "generator": it.generator, "split": it.split,
                             "reason": f"ghi file lỗi: {type(exc).__name__}: {exc}",
                             "retry": "2 lần"}
                        )
                    return 0
                time.sleep(0.05)           # NTFS/AV giữ file trong vài chục ms

    key = f"{it.split}/{it.generator}"
    per_archive[key] = per_archive.get(key, 0) + 1
    made.append(
        Item(
            source=it.source,
            member=name,
            label=it.label,
            generator=it.generator,
            split=it.split,
            dataset=it.dataset,
            sha256=sha_full,
            width=size,
            height=size,
            png=str(dst),
            n_images=1,
        )
    )
    return 1 if reused_here else 0


def tar_inventory(path: str | Path, n_sample: int = 3) -> dict:
    """``tar`` chứa gì: đếm theo đuôi file + vài tên mẫu. **Không** giải nén ra đĩa.

    Vì sao cần: ``dire/*.tar.gz`` và ``recons/*.tar.gz`` là map/reconstruction của
    tác giả, nhưng **định dạng bên trong không được ghi ở đâu cả** — có thể là
    ``.npy`` float, có thể là PNG đã chuẩn hoá, có thể là ``.png`` 3 kênh. Đoán
    sai định dạng là cách nhanh nhất để viết ra một pipeline chạy được mà vô
    nghĩa. Một lệnh đọc header trả lời dứt điểm, giá 0 đồng.
    """
    from collections import Counter  # noqa: PLC0415

    ext = Counter()
    sample: list[str] = []
    total = 0
    with tarfile.open(path, "r:*") as tf:
        for m in tf:
            if not m.isfile():
                continue
            suffix = Path(m.name).suffix.lower() or "(không đuôi)"
            ext[suffix] += 1
            total += m.size
            if len(sample) < n_sample:
                sample.append(f"{m.name} ({m.size} B)")
    return {
        "path": str(path),
        "n_members": sum(ext.values()),
        "bytes": total,
        "by_ext": dict(ext.most_common()),
        "sample": sample,
    }


def _is_archive(path: str | Path) -> bool:
    """`.tar.gz` / `.tgz` / `.tar` — mọi thứ khác coi là một file ảnh."""
    return str(path).lower().endswith((".tar.gz", ".tgz", ".tar"))


def _iter_tar_images(path: Path, limit: int | None = None, state: dict | None = None):
    """Sinh ``(tên member, bytes)`` cho tối đa ``limit`` ảnh, theo thứ tự trong tar.

    Thứ tự trong tar là **ổn định** (do người đóng gói), nên ``limit`` cắt tiền
    tố đầu — tái lập được bằng tham số, khác với cắt ngẫu nhiên.

    Hai chế độ đọc, đây chính là đường dự phòng ở tầng archive:

    * ``r:*`` — truy cập ngẫu nhiên, **cần đọc được bảng chỉ mục ở cuối file**.
      Nhanh, nhưng file tải thiếu vài MB cuối thì ``tarfile.open`` ném lỗi ngay
      và **mất trắng** archive dù 99 % nội dung nằm ở đầu.
    * ``r|*`` — đọc **tuần tự**, không cần bảng chỉ mục. Đọc được đúng phần dữ
      liệu còn nguyên và dừng ở chỗ hỏng.

    Nên: thử ``r:*`` trước; lỗi thì ghi lại và đọc lại bằng ``r|*``, bỏ qua
    member đã đưa ra (trùng tên+size) để không đếm hai lần. Lỗi được ghi vào
    ``state`` chứ **không** im lặng nuốt: người gọi phải báo cho người dùng biết
    dữ liệu này là một phần.
    """
    state = {} if state is None else state
    state.setdefault("error", "")
    state.setdefault("partial", False)
    state.setdefault("n", 0)
    state.setdefault("retried", False)
    state.setdefault("mode", "r:*")
    seen: set[tuple[str, int]] = set()

    def _walk(mode: str):
        with tarfile.open(path, mode) as tf:
            for m in tf:
                if not (m.isfile() and _is_image_member(m.name)):
                    continue
                if limit is not None and state["n"] >= limit:
                    return
                key = (m.name, m.size)
                if key in seen:           # đã đưa ra ở lần đọc trước
                    continue
                fh = tf.extractfile(m)
                if fh is None:
                    continue
                data = fh.read()
                seen.add(key)
                state["n"] += 1
                yield m.name, data

    try:
        yield from _walk("r:*")
    except (tarfile.TarError, OSError, EOFError, zlib.error) as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["partial"] = True
        if state["retried"] or (limit is not None and state["n"] >= limit):
            return
        state["retried"] = True
        state["mode"] = "r|*"
        try:
            yield from _walk("r|*")
        except (tarfile.TarError, OSError, EOFError, zlib.error) as exc2:
            state["error"] += f" · đọc tuần tự cũng hỏng: {type(exc2).__name__}: {exc2}"


def _archive_is_readable(path: Path) -> tuple[bool, str]:
    """Đi hết archive **chỉ đọc header** (không giải mã ảnh, không ghi) để biết nó nguyên vẹn.

    Dùng cho ``on_error="strict"``: đọc đúp, nhưng đổi lại được quyền nói "hoặc
    ghi đủ, hoặc không ghi gì". Với cây 12 GB, đọc đúp chỉ ở chế độ strict là
    đánh đổi chấp nhận được; chế độ mặc định chỉ đọc một lần.
    """
    try:
        with tarfile.open(path, "r:*") as tf:
            for _m in tf:
                pass
        return True, ""
    except (tarfile.TarError, OSError, EOFError, zlib.error) as exc:
        return False, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Chọn thư mục đầu ra: ổ chính trước, ổ dự phòng khi thiếu dung lượng
# --------------------------------------------------------------------------- #

#: Chỗ ghi mặc định — **tương đối**, để không khoá vào ổ nào.
DEFAULT_OUT_ROOT = "data/dire-zero-norm"

#: Ổ dự phòng mặc định. Chỉ dùng khi ổ chính **không đủ chỗ**, không bao giờ dùng
#: chỉ vì ổ chính chưa tồn tại (thư mục ra thường chưa có, đó là chuyện bình thường).
DEFAULT_OUT_FALLBACKS = ("F:/dire-zero-norm",)

#: Chừa lại cho hệ điều hành. Ghi kín ổ là cách làm hỏng cả máy lẫn Windows.
RESERVE_BYTES = 512 * 1024 * 1024

#: Trần byte cho một PNG 256² RGB: 256·256·3. PNG có thể nhích hơn vài phần nghìn
#: với ảnh nhiễu, nên dùng trần + 2 % thay vì con số lý thuyết.
BYTES_PER_PNG = int(256 * 256 * 3 * 1.02)


def _fmt_size(n: int | None) -> str:
    """Số byte → chuỗi đọc được. ``None`` = không biết, **không** phải 0."""
    if n is None:
        return "không rõ"
    if n >= 10**9:
        return f"{n / 1e9:.1f} GB"
    return f"{n / 1e6:.0f} MB"


class OutOfSpaceError(RuntimeError):
    """Không ứng viên nào đủ chỗ. Dừng **trước** khi ghi, không chết giữa đường."""


def _nearest_existing(path: Path) -> Path | None:
    """Đi lên tới thư mục/ổ gần nhất đang tồn tại. ``F:/`` chưa cắm ⇒ ``None``."""
    p = Path(path)
    for _ in range(64):
        try:
            if p.exists():
                return p
        except OSError:                     # ổ tháo giữa lúc kiểm
            return None
        if p.parent == p:
            return None
        p = p.parent
    return None


def free_bytes(path: str | Path) -> int | None:
    """Dung lượng còn trống của **ổ chứa** ``path``; ``None`` nếu không truy cập được.

    Đo trên thư mục gần nhất đang tồn tại: ``F:/dire-zero-norm`` chưa tạo vẫn đo
    được dung lượng của ``F:/``. Trả ``None`` (không trả 0) để phân biệt "không
    biết" với "hết chỗ" — hai chuyện khác nhau và cần thông báo khác nhau.
    """
    base = _nearest_existing(Path(path))
    if base is None:
        return None
    try:
        return shutil.disk_usage(base).free
    except OSError:
        return None


def estimate_output_bytes(items: list[Item], limit_per_archive: int | None = None) -> dict:
    """Ước lượng **trần** dung lượng PNG đầu ra — đo bằng ``stat``, không giải nén.

    Hai chế độ, vì hai thứ đã biết khác nhau:

    * **không ``--limit``** ⇒ chưa biết số ảnh, nên lấy trần theo dung lượng
      nguồn: PNG 256² đã cắt giữa *nhỏ hơn* hoặc xấp xỉ ảnh nguồn, nên
      ``1,25 × tổng byte archive`` là trần an toàn. Chỉ ``stat``, tức thì.
    * **có ``--limit``** ⇒ biết chính xác bao nhiêu ảnh sẽ ghi, và đếm được rẻ
      (``tarfile`` dừng ngay sau ảnh thứ ``limit``). Dùng ``n × BYTES_PER_PNG``.

    Ước lượng **trần** là lựa chọn có ý thức: đoán thấp rồi chết giữa đường đúng
    là loại lỗi mà cả mục này sinh ra để tránh.
    """
    src_bytes = 0
    n_unknown = 0
    for it in items:
        try:
            src_bytes += it.source.stat().st_size
        except OSError:
            continue
        if not _is_archive(it.source):
            n_unknown += 1                 # ảnh rời: 1 file = 1 ảnh
    archives = [it.source for it in items if _is_archive(it.source)]

    if limit_per_archive is None:
        n_images = None
        need = int(src_bytes * 1.25)
        basis = "1,25 × dung lượng nguồn (chưa biết số ảnh)"
    else:
        n_images = n_unknown
        for a in archives:
            try:
                n_images += min(peek_archive(a, limit=limit_per_archive)["n_images"], limit_per_archive)
            except (OSError, tarfile.TarError, EOFError, zlib.error):
                continue
        need = n_images * BYTES_PER_PNG
        basis = f"{n_images} ảnh × {BYTES_PER_PNG} B"
    return {"need": need, "src_bytes": src_bytes, "n_images": n_images,
            "n_archives": len(archives), "basis": basis}


@dataclass
class OutChoice:
    """Kết quả chọn chỗ ghi — kèm **lý do**, vì lý do phải đi theo dữ liệu."""

    path: Path
    reason: str
    needed: int
    free: int | None
    redirected: bool
    tried: list[dict] = field(default_factory=list)

    def as_warning(self) -> str:
        return f"[dự phòng dung lượng] ghi vào {self.path}: {self.reason}"


def _same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:                        # ổ chưa tồn tại ⇒ resolve vẫn chạy, nhưng chắc ăn
        return str(a).lower() == str(b).lower()


def choose_out_root(
    preferred: str | Path,
    fallbacks: tuple[str, ...] | list[str] = DEFAULT_OUT_FALLBACKS,
    needed: int = 0,
    reserve: int = RESERVE_BYTES,
) -> OutChoice:
    """Chọn chỗ ghi: ``preferred`` nếu đủ chỗ, ngược lại thử ``fallbacks`` theo thứ tự.

    Ba quyết định ở đây, mỗi cái đều là để tránh một kiểu hỏng cụ thể:

    * **Dự phòng chỉ vì dung lượng**, không vì "thư mục chưa tồn tại". Thư mục ra
      thường chưa có; coi đó là lý do để nhảy sang ổ khác thì dữ liệu rơi vào chỗ
      người dùng không ngờ.
    * **Đo trên ổ chứa**, không đo trên thư mục (``shutil.disk_usage`` đo ổ).
    * **Không biết dung lượng ⇒ không chọn.** Ổ chưa cắm, quyền bị chặn: bỏ ứng
      viên đó và ghi lý do, thay vì thử rồi chết giữa đường.
    """
    tried: list[dict] = []
    candidates: list[Path] = [Path(preferred)]
    for f in fallbacks:
        pf = Path(f)
        if not any(_same_path(pf, c) for c in candidates):
            candidates.append(pf)

    for idx, cand in enumerate(candidates):
        free = free_bytes(cand)
        if free is None:
            tried.append({"path": str(cand), "free": None, "ok": False,
                          "reason": "không truy cập được (ổ chưa cắm / không có quyền?)"})
            continue
        if free < needed + reserve:
            tried.append({"path": str(cand), "free": free, "ok": False,
                          "reason": f"chỉ còn {_fmt_size(free)}, cần ≤ {_fmt_size(needed)} "
                                    f"+ {_fmt_size(reserve)} dự trữ"})
            continue
        tried.append({"path": str(cand), "free": free, "ok": True, "reason": "đủ chỗ"})
        if idx == 0:
            reason = f"ổ chính đủ chỗ (còn {_fmt_size(free)}, cần ≤ {_fmt_size(needed)})"
        else:
            why = tried[0]["reason"]
            reason = f"ổ chính không dùng được ({why}) — dùng ổ dự phòng #{idx}"
        return OutChoice(path=cand, reason=reason, needed=needed, free=free,
                         redirected=idx != 0, tried=tried)

    lines = "\n".join(f"  - {t['path']}: {t['reason']}" for t in tried)
    raise OutOfSpaceError(
        f"không chỗ nào đủ để ghi (cần ≤ {_fmt_size(needed)} + "
        f"{_fmt_size(reserve)} dự trữ):\n{lines}\n"
        "Dọn ổ, hoặc giảm --limit, hoặc trỏ --out sang ổ khác."
    )


def _seed_pix_seen(out_root: Path) -> dict[str, tuple[int, str]]:
    """Nhặt ``sha16 → (nhãn, nguồn)`` từ PNG **đã có** trong ``out_root``.

    Cần cho hai việc khi chạy lại sau khi hỏng giữa đường:

    * không ghi lại file đã đúng (resume);
    * **vẫn bắt được rò rỉ qua ranh giới hai lần chạy**. Không có bước này, lần
      chạy trước để lại ``real/abc.png``, lần này ingest ``adm`` có đúng pixel đó
      ⇒ không thấy trùng, ACC ~100 % và không có lỗi nào báo.
    """
    seen: dict[str, tuple[int, str]] = {}
    if not out_root.is_dir():
        return seen
    for f in out_root.rglob("*.png"):
        rel = f.relative_to(out_root)
        if len(rel.parts) != 3:            # <split>/<generator>/<sha16>.png
            continue
        gen = rel.parts[1]
        seen.setdefault(f.stem, (label_for_generator(gen), f"{gen}/{f.name} (lần chạy trước)"))
    return seen


# --------------------------------------------------------------------------- #
# Manifest + data card
# --------------------------------------------------------------------------- #

def write_manifest_tsv(path: str | Path, items: list[Item], root: str | Path | None = None) -> Path:
    """Ghi manifest. ``root`` ⇒ cột ``png`` thành **tương đối** so với gốc đó.

    Vì sao phải tương đối: có ổ dự phòng nghĩa là cùng một bộ dữ liệu có thể được
    ghi ở ``F:/…`` hôm nay và ``./data/…`` hôm sau, và cùng một manifest có thể
    được đẩy lên Volume với gốc ``/zero/data``. Đường dẫn tuyệt đối làm
    ``verify_manifest`` chỉ đúng trên đúng một máy — hỏng ở mọi nơi khác, và hỏng
    **im lặng** (nó báo "lệch sha256" trong khi ảnh vẫn tốt, chỉ là tìm sai chỗ).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    base = Path(root) if root is not None else None
    lines = ["\t".join(MANIFEST_COLUMNS)]
    for it in sorted(items, key=lambda i: (i.split, i.generator, i.png)):
        row = it.as_row()
        if base is not None and row["png"]:
            try:
                row["png"] = Path(row["png"]).relative_to(base).as_posix()
            except ValueError:             # ngoài gốc (đường khác ổ) ⇒ giữ tuyệt đối
                pass
        lines.append("\t".join(str(row[c]) for c in MANIFEST_COLUMNS))
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def read_manifest_tsv(path: str | Path) -> list[dict]:
    rows = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if i == 0 or not line.strip():
            continue
        parts = line.split("\t")
        row = dict(zip(MANIFEST_COLUMNS, parts))
        row["label"] = int(row["label"])
        rows.append(row)
    return rows


def verify_rows(
    rows: list[dict],
    base: str | Path | None = None,
    workers: int = 1,
    progress=None,
    progress_every: int = 2000,
) -> tuple[int, list[str]]:
    """Băm song song các dòng manifest. Trả ``(số khớp, danh sách lỗi)``.

    Vì sao có `workers`
    ------------------
    217 585 PNG nhỏ (~103 KB) không phải bài toán **băng thông** mà là bài toán
    **độ trễ mở file**: 217 585 lần `open`, mỗi lần vài ms trên FUSE của Volume ⇒
    hơn 30 phút với một luồng, đúng bằng timeout 1800 s đã gặp. Băm là việc
    nhả GIL (C code + syscall), nên luồng thật sự chạy song song; 64 luồng trên
    8 core rút ngắn còn vài phút.

    `progress(done, total)` được gọi mỗi `progress_every` file — chỉ để thấy tiến
    trình trong log, không ảnh hưởng kết quả.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: PLC0415

    base_path = Path(base) if base is not None else None

    def job(r: dict) -> tuple[bool, str]:
        target = Path(r["png"])
        if base_path is not None and not target.is_absolute():
            target = base_path / target
        try:
            got = sha256_file(target)
        except OSError as exc:
            return False, f"{target}: {exc}"
        if got == r["sha256"]:
            return True, ""
        return False, f"{target}: sha256 lệch"

    results: list[tuple[bool, str]] = []
    total = len(rows)
    if workers and workers > 1 and total > 1:
        with ThreadPoolExecutor(max_workers=int(workers)) as ex:
            futures = [ex.submit(job, r) for r in rows]
            for done, fut in enumerate(as_completed(futures), start=1):
                results.append(fut.result())
                if progress and progress_every and done % progress_every == 0:
                    progress(done, total)
    else:
        for i, r in enumerate(rows, start=1):
            results.append(job(r))
            if progress and progress_every and i % progress_every == 0:
                progress(i, total)

    ok = sum(1 for good, _msg in results if good)
    bad = [msg for good, msg in results if not good]
    return ok, bad


def verify_manifest(
    path: str | Path,
    root: str | Path | None = None,
    workers: int = 1,
    progress=None,
    progress_every: int = 2000,
) -> tuple[int, list[str]]:
    """Kiểm sha256 từng PNG so với manifest — bắt file bị sửa/xoá sau ingest.

    Đường dẫn tương đối được hiểu là **so với thư mục chứa manifest** (mặc định)
    hoặc so với ``root``. Nhờ vậy cùng một manifest kiểm được ở ``F:/``, ở
    ``./data``, và ở ``/zero/data`` trong container — ba gốc khác nhau, một
    manifest. Dòng kiểu cũ (đường tuyệt đối) vẫn kiểm được.
    """
    p = Path(path)
    base = Path(root) if root is not None else p.parent
    return verify_rows(read_manifest_tsv(p), base, workers=workers,
                       progress=progress, progress_every=progress_every)


@dataclass
class DataCard:
    """Mọi thứ cần để người khác biết mình đo trên **cái gì**, và được nói gì."""

    root: str
    layout: str
    dataset: str
    license: str
    splits: list[str]
    generators: list[str]
    n_real: int
    n_fake: int
    seed: int
    fake_source: str                 # "gen" | "author"
    label: str                       # "R1+R2+R3" | "R1+R2" | "R0"
    comparable_to_paper: bool
    manifest_sha256: str = ""
    warnings: list[str] = field(default_factory=list)
    created: str = ""
    out_root: str = ""              # chỗ ghi thật (có thể là ổ dự phòng)
    out_reason: str = ""            # vì sao ghi ở đó — để người sau kiểm lại được

    def as_dict(self) -> dict:
        return {
            "root": self.root,
            "layout": self.layout,
            "dataset": self.dataset,
            "license": self.license,
            "splits": self.splits,
            "generators": self.generators,
            "n_real": self.n_real,
            "n_fake": self.n_fake,
            "seed": self.seed,
            "fake_source": self.fake_source,
            "label": self.label,
            "comparable_to_paper": self.comparable_to_paper,
            "manifest_sha256": self.manifest_sha256,
            "warnings": self.warnings,
            "created": self.created,
            "out_root": self.out_root,
            "out_reason": self.out_reason,
            "note": (
                "nếu fake_source='author', ảnh giả KHÔNG phải của mình ⇒ nhãn tối đa là R1+R2; "
                "không được nói R3"
            ),
        }


def write_data_card(path: str | Path, card: DataCard) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(card.as_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return p


def manifest_sha(items: list[Item]) -> str:
    """Sha của **nội dung** bộ dữ liệu: (sha256 ảnh, nhãn), đã sắp xếp.

    Không băm đường dẫn — hai người ở hai ổ đĩa khác nhau vẫn phải so được "có
    phải cùng một bộ dữ liệu không".
    """
    h = hashlib.sha256()
    for sha, label in sorted((i.sha256, i.label) for i in items):
        h.update(f"{sha}\t{label}\n".encode("utf-8"))
    return h.hexdigest()


def split_items(items: list[Item], seed: int = 3407, frac_train: float = 0.8) -> dict[str, list[Item]]:
    """Chia **theo split có sẵn** nếu archive đã có split; nếu không thì chia theo hash.

    Cây DiffusionForensics đã có ``train``/``val``/``test``; chia lại là bỏ đi
    quyết định của người đóng gói và làm kết quả không so được với paper.
    """
    have = {it.split for it in items}
    if {"train", "test"} <= have or {"train", "val", "test"} <= have:
        return {s: [it for it in items if it.split == s] for s in SPLITS}
    order = sorted(items, key=lambda i: (stable_key(i.png or str(i.source), seed), i.png))
    cut = round(len(order) * frac_train)
    return {"train": order[:cut], "val": [], "test": order[cut:]}


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
