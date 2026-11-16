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
import tarfile
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
    "write_data_card",
    "MANIFEST_COLUMNS",
    "IMAGE_SUFFIXES",
    "SPLITS",
    "generator_of",
    "label_for_generator",
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

def _norm_image(data: bytes, size: int):
    """bytes → (ảnh RGB 256², w_gốc, h_gốc). Resize cạnh ngắn + cắt giữa."""
    from PIL import Image  # noqa: PLC0415

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


def ingest(
    items: list[Item],
    out_root: str | Path,
    size: int = 256,
    limit_per_archive: int | None = None,
    progress=None,
) -> tuple[list[Item], dict]:
    """Đổ archive ra PNG ``out_root/<split>/<generator>/<sha16>.png``.

    Tên PNG = 16 ký tự đầu sha256 **của byte gốc trong tar**. Nhờ vậy:

    * hai ảnh trùng nội dung ⇒ cùng tên ⇒ ghi đè, không thành hai mẫu;
    * tên gốc kiểu ``000123.jpg`` không mang thông tin, bỏ đi thì chia train/test
      theo hash vẫn ổn định qua các máy;
    * truy ngược được về ``source_tar`` + ``member`` khi cần kiểm.

    Ảnh **trùng giữa hai lớp** (rò rỉ) không thể phát hiện chắc chắn nếu chỉ so
    sha của byte gốc (JPEG khác encoder ra sha khác). Vì vậy ở đây còn so
    **sha của pixel đã chuẩn hoá** — đó mới là thứ mô hình nhìn thấy.
    """
    out_root = Path(out_root)
    made: list[Item] = []
    pix_seen: dict[str, tuple[int, str]] = {}
    leaks: list[str] = []
    skipped = 0
    per_archive: dict[str, int] = {}

    for it in items:
        if _is_archive(it.source):
            sources = _iter_tar_images(it.source, limit_per_archive)
        else:                                      # ảnh rời (bố cục real/ + fake/)
            sources = [(it.source.name, it.source.read_bytes())]

        for name, data in sources:
            try:
                img, w0, h0 = _norm_image(data, size)
            except Exception:                      # noqa: BLE001 — ảnh hỏng trong dataset
                skipped += 1
                continue
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=False)
            png_bytes = buf.getvalue()
            sha = hashlib.sha256(png_bytes).hexdigest()
            prev = pix_seen.get(sha)
            if prev and prev[0] != it.label:
                leaks.append(f"{it.generator}:{name} trùng pixel với {prev[1]} ở lớp khác")
                continue
            pix_seen[sha] = (it.label, f"{it.generator}:{name}")

            key = f"{it.split}/{it.generator}"
            per_archive[key] = per_archive.get(key, 0) + 1
            dst = out_root / it.split / it.generator / f"{sha[:16]}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(png_bytes)
            made.append(
                Item(
                    source=it.source,
                    member=name,
                    label=it.label,
                    generator=it.generator,
                    split=it.split,
                    dataset=it.dataset,
                    sha256=sha,
                    width=size,
                    height=size,
                    png=str(dst),
                    n_images=1,
                )
            )
        if progress:
            progress(it)

    stats = {
        "written": len(made),
        "skipped": skipped,
        "leaks": leaks[:10],
        "n_leaks": len(leaks),
        "per_archive": per_archive,
    }
    if leaks:
        raise IntegrityError(
            f"{len(leaks)} ảnh trùng pixel giữa hai lớp (đã dừng, chưa ghi file trùng).\n"
            f"  vd: {leaks[:3]}\n"
            "Đây là rò rỉ: ACC sẽ ~100% và vô nghĩa. Loại một trong hai phía rồi chạy lại."
        )
    return made, stats


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


def _iter_tar_images(path: Path, limit: int | None):
    """Sinh ``(tên member, bytes)`` cho tối đa ``limit`` ảnh, theo thứ tự trong tar.

    Thứ tự trong tar là **ổn định** (do người đóng gói), nên ``limit`` cắt tiền
    tố đầu — tái lập được bằng tham số, khác với cắt ngẫu nhiên.
    """
    n = 0
    with tarfile.open(path, "r:*") as tf:
        for m in tf:
            if not (m.isfile() and _is_image_member(m.name)):
                continue
            fh = tf.extractfile(m)
            if fh is None:
                continue
            yield m.name, fh.read()
            n += 1
            if limit is not None and n >= limit:
                return


# --------------------------------------------------------------------------- #
# Manifest + data card
# --------------------------------------------------------------------------- #

def write_manifest_tsv(path: str | Path, items: list[Item]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(MANIFEST_COLUMNS)]
    for it in sorted(items, key=lambda i: (i.split, i.generator, i.png)):
        row = it.as_row()
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


def verify_manifest(path: str | Path) -> tuple[int, list[str]]:
    """Kiểm sha256 từng PNG so với manifest — bắt file bị sửa/xoá sau ingest."""
    bad, ok = [], 0
    for r in read_manifest_tsv(path):
        try:
            got = sha256_file(r["png"])
        except OSError as exc:
            bad.append(f"{r['png']}: {exc}")
            continue
        if got == r["sha256"]:
            ok += 1
        else:
            bad.append(f"{r['png']}: sha256 lệch")
    return ok, bad


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
