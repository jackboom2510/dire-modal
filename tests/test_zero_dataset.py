"""Đọc **cây DiffusionForensics thật** (tar.gz) — không cần dữ liệu thật, không cần GPU.

Fixture dựng đúng cây trên máy người dùng::

    DiffusionForensics/images/{train,val,test}/{lsun_bedroom,imagenet}/{gen}.tar.gz
    DiffusionForensics/dire/…          ← phải bị TỪ CHỐI (G0)
    DiffusionForensics/recons/…        ← phải bị bỏ qua nhưng có ghi chú

Điều đáng test không phải "đọc được file", mà là **không đọc nhầm thứ không được
đọc**: `dire/` là map của tác giả, dùng nó là mất quyền gọi "from scratch"; và
một ảnh nằm ở cả hai lớp làm ACC ~100 % mà vô nghĩa.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from dire_zero import dataset as ds  # noqa: E402
from dire_zero.cli import main as cli_main  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixture: cây giống hệt máy người dùng
# --------------------------------------------------------------------------- #

def _png_bytes(color: tuple[int, int, int], size: int = 64, seed: int = 0) -> bytes:
    img = Image.new("RGB", (size, size), color)
    for i in range(size):                       # thêm vài vạch để ảnh khác nhau
        img.putpixel(((i + seed) % size, i), (i * 3 % 256, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_tar(path: Path, images: list[bytes], names: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tf:
        for i, data in enumerate(images):
            name = (names[i] if names else f"{i:06d}.png")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def _tree(root: Path, leak: bool = False) -> Path:
    base = root / "DiffusionForensics"
    real_train = [_png_bytes((10, 10, 10), seed=i) for i in range(4)]
    _make_tar(base / "images/train/lsun_bedroom/real.tar.gz", real_train)
    _make_tar(base / "images/train/lsun_bedroom/adm.tar.gz", [_png_bytes((200, 30, 30), seed=i) for i in range(4)])
    _make_tar(base / "images/train/lsun_bedroom/stylegan.tar.gz", [_png_bytes((30, 200, 30), seed=i) for i in range(3)])
    _make_tar(base / "images/val/lsun_bedroom/real.tar.gz", [_png_bytes((10, 10, 10), seed=100 + i) for i in range(2)])
    _make_tar(base / "images/val/lsun_bedroom/pndm.tar.gz", [_png_bytes((30, 30, 200), seed=i) for i in range(2)])
    _make_tar(base / "images/test/lsun_bedroom/real.tar.gz", [_png_bytes((10, 10, 10), seed=200 + i) for i in range(3)])
    _make_tar(base / "images/test/lsun_bedroom/pndm.tar.gz", [_png_bytes((30, 30, 200), seed=50 + i) for i in range(3)])
    _make_tar(base / "images/test/lsun_bedroom/sdv1_test.tar.gz", [_png_bytes((90, 90, 20), seed=i) for i in range(2)])
    _make_tar(base / "images/train/imagenet/real.tar.gz", [_png_bytes((11, 11, 11), seed=i) for i in range(2)])
    _make_tar(base / "images/train/imagenet/adm.tar.gz", [_png_bytes((210, 40, 40), seed=i) for i in range(2)])

    # dire/ — map của tác giả (bị từ chối)
    _make_tar(base / "dire/test/lsun_bedroom/adm.tar.gz", [_png_bytes((1, 1, 1), seed=i) for i in range(2)])
    # recons/ — reconstruction của tác giả (bỏ qua, có ghi chú)
    _make_tar(base / "recons/test/lsun_bedroom/lsun_bedroom/real.tar.gz", [_png_bytes((2, 2, 2), seed=i) for i in range(2)])
    # tar rỗng 130 B đã biết trong dataset thật
    (base / "images/test/lsun_bedroom" ).mkdir(parents=True, exist_ok=True)

    if leak:
        # cùng một ảnh ở cả real và fake
        same = _png_bytes((77, 77, 77))
        _make_tar(base / "images/train/lsun_bedroom/real.tar.gz", [same] + real_train[:2])
        _make_tar(base / "images/train/lsun_bedroom/adm.tar.gz", [same])
    return base


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    return _tree(tmp_path)


# --------------------------------------------------------------------------- #
# Đường dẫn Windows / Git Bash
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("given", ["F:/du-lieu", "F:\\du-lieu", "/f/du-lieu"])
def test_drive_forms_are_equivalent(given):
    """Ba cách gõ của người dùng thật phải quy về **cùng một cặp** dạng thử."""
    forms = [f.replace("\\", "/") for f in ds.drive_candidates(given)]
    assert len(forms) == 2
    assert "F:/du-lieu" in forms and "/f/du-lieu" in forms


def test_resolve_root_accepts_parent_and_child(tree: Path, tmp_path: Path):
    assert ds.resolve_root(tree) == tree
    assert ds.resolve_root(tmp_path) == tree        # trỏ vào thư mục cha ⇒ tự lùi vào


def test_resolve_root_reports_what_it_tried(tmp_path: Path):
    with pytest.raises(FileNotFoundError) as exc:
        ds.resolve_root(tmp_path / "khong-co")
    assert "Đã thử" in str(exc.value)


# --------------------------------------------------------------------------- #
# Quét
# --------------------------------------------------------------------------- #

def test_scan_finds_the_matrix(tree: Path):
    scan = ds.scan_tree(tree)
    assert scan.layout == "diffusionforensics"
    m = scan.matrix()
    assert set(m) == {"lsun_bedroom", "imagenet"}
    assert m["lsun_bedroom"]["train"][0] == "real"          # real luôn đứng đầu
    assert set(m["lsun_bedroom"]["test"]) == {"real", "pndm", "sdv1"}
    assert set(scan.generators("lsun_bedroom", "train")) == {"real", "adm", "stylegan"}


def test_sdv1_test_is_the_same_generator_as_sdv1():
    assert ds.generator_of("/x/images/test/lsun_bedroom/sdv1_test.tar.gz") == "sdv1"
    assert ds.generator_of("adm.tar.gz") == "adm"
    assert ds.label_for_generator("real") == 0
    assert ds.label_for_generator("adm") == 1


def _parts_under(item, root: Path) -> tuple[str, ...]:
    """Thành phần đường dẫn **tương đối** — không thì tên thư mục tmp lẫn vào."""
    return item.source.relative_to(root).parts


def test_dire_is_refused_unless_explicitly_allowed(tree: Path):
    scan = ds.scan_tree(tree)
    assert all(p[0] != "dire" for p in (_parts_under(it, tree) for it in scan.items))
    assert any("G0" in n for n in scan.notes)

    scan2 = ds.scan_tree(tree, allow_published=True)
    assert any("R0" in n for n in scan2.notes)


def test_recons_is_skipped_with_a_note(tree: Path):
    scan = ds.scan_tree(tree)
    assert all(p[0] != "recons" for p in (_parts_under(it, tree) for it in scan.items))
    assert any("recons/" in n for n in scan.notes)


def test_asymmetry_warnings(tree: Path):
    """"imagenet chỉ có train" và "sdv1 chỉ có test" phải được nói ra."""
    scan = ds.scan_tree(tree)
    text = " ".join(scan.warnings)
    assert "imagenet" in text                       # chỉ có train
    assert "sdv1" in text and "'test'" in text      # chỉ có test
    assert scan.generators("lsun_bedroom", "test") == ["real", "pndm", "sdv1"]


def test_select_filters(tree: Path):
    scan = ds.scan_tree(tree)
    picked = ds.select(scan, dataset="lsun_bedroom", splits=("test",), generators=("pndm",))
    assert {i.generator for i in picked} == {"real", "pndm"}   # real luôn đi kèm
    fake = [i for i in picked if i.label == 1]
    assert len(fake) == 1 and fake[0].split == "test"

    with_real = ds.select(scan, dataset="lsun_bedroom", splits=("test",), generators=("pndm",), include_real=True)
    assert {i.generator for i in with_real} == {"real", "pndm"}


def test_folder_layout_still_works(tmp_path: Path):
    """Bố cục real/ + fake/ (ảnh rời) vẫn phải dùng được."""
    for sub in ("real", "fake"):
        d = tmp_path / "mine" / sub
        d.mkdir(parents=True)
        (d / "a.png").write_bytes(_png_bytes((5, 5, 5) if sub == "real" else (250, 5, 5)))
    scan = ds.scan_tree(tmp_path / "mine")
    assert scan.layout == "folder"
    assert {i.label for i in scan.items} == {0, 1}


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #

def test_ingest_writes_pngs_and_manifest(tree: Path, tmp_path: Path):
    scan = ds.scan_tree(tree)
    items = ds.select(scan, dataset="lsun_bedroom", splits=("train", "test"))
    made, stats = ds.ingest(items, tmp_path / "out", size=64)

    assert stats["written"] == sum(it.n_images for it in items)
    dirs = {(p.parts[-3], p.parts[-2]) for p in (tmp_path / "out").rglob("*.png")}
    assert ("train", "real") in dirs and ("test", "pndm") in dirs
    with Image.open(next((tmp_path / "out").rglob("*.png"))) as im:
        assert im.size == (64, 64) and im.mode == "RGB"

    manifest = ds.write_manifest_tsv(tmp_path / "out" / "manifest.tsv", made)
    ok, bad = ds.verify_manifest(manifest)
    assert ok == len(made) and not bad
    rows = ds.read_manifest_tsv(manifest)
    assert {r["label"] for r in rows} == {0, 1}
    assert all(len(r["sha256"]) == 64 for r in rows)


def test_ingest_names_files_by_content_hash(tree: Path, tmp_path: Path):
    """Hai ảnh giống nhau ⇒ cùng tên ⇒ không thể thành hai mẫu."""
    scan = ds.scan_tree(tree)
    items = ds.select(scan, dataset="lsun_bedroom", splits=("train",))
    made, _ = ds.ingest(items, tmp_path / "out", size=64)
    names = [Path(m.png).name for m in made]
    assert len(names) == len(set(names))


def test_ingest_blocks_pixel_leak(tmp_path: Path):
    """Cùng ảnh ở hai lớp ⇒ `IntegrityError`, không phải cảnh báo."""
    base = _tree(tmp_path, leak=True)
    scan = ds.scan_tree(base)
    items = ds.select(scan, dataset="lsun_bedroom", splits=("train",))
    with pytest.raises(ds.IntegrityError) as exc:
        ds.ingest(items, tmp_path / "out", size=64)
    assert "rò rỉ" in str(exc.value)


def test_ingest_normalises_to_square_center_crop(tmp_path: Path):
    """Ảnh 100×60 ⇒ cạnh ngắn thành 64, cắt giữa ⇒ 64×64 (không méo)."""
    d = tmp_path / "mine" / "real"
    d.mkdir(parents=True)
    img = Image.new("RGB", (100, 60), (20, 20, 20))
    img.paste((255, 0, 0), (40, 0, 60, 60))          # vạch đỏ dọc ở giữa
    img.save(d / "wide.png")
    scan = ds.scan_tree(tmp_path / "mine")
    made, _ = ds.ingest(scan.items, tmp_path / "out", size=64)
    with Image.open(made[0].png) as out:
        assert out.size == (64, 64)
        assert out.getpixel((32, 32))[0] > 200        # vạch đỏ vẫn ở giữa


def test_manifest_sha_ignores_paths(tree: Path, tmp_path: Path):
    """Hai bản ingest ở hai thư mục khác nhau phải cho cùng 'sha của bộ dữ liệu'."""
    scan = ds.scan_tree(tree)
    items = ds.select(scan, dataset="lsun_bedroom", splits=("test",))
    a, _ = ds.ingest(items, tmp_path / "a", size=64)
    b, _ = ds.ingest(items, tmp_path / "b", size=64)
    assert ds.manifest_sha(a) == ds.manifest_sha(b)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def test_cli_dataset_requires_license(tree: Path, capsys):
    rc = cli_main(["dataset", "--root", str(tree)])
    assert rc == 3
    assert "--license" in capsys.readouterr().err


def test_cli_dataset_requires_ood_ack_for_imagenet(tree: Path, capsys):
    rc = cli_main(["dataset", "--root", str(tree), "--dataset", "imagenet", "--license", "private"])
    assert rc == 3
    assert "phân bố" in capsys.readouterr().err
    rc2 = cli_main(["dataset", "--root", str(tree), "--dataset", "imagenet",
                    "--license", "private", "--ack-ood"])
    assert rc2 == 0


def test_cli_dataset_prints_label_and_cost_gate(tree: Path, capsys):
    card = tree.parent / "card.json"
    rc = cli_main(["dataset", "--root", str(tree), "--license", "research-only",
                   "--card", str(card), "--splits", "train", "test"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "R1+R2" in out                      # ảnh giả của tác giả ⇒ không được nói R3
    assert "KHÔNG báo" in out                  # chưa đo sec/step ⇒ không báo tiền
    doc = json.loads(card.read_text(encoding="utf-8"))
    assert doc["license"] == "research-only"
    assert doc["fake_source"] == "author"
    assert doc["comparable_to_paper"] is True
    assert "R3" in doc["note"]


def test_cli_dataset_labels_gen_when_no_fake(tree: Path, capsys):
    """--no-real + --fake-source gen ⇒ nhãn R1+R2+R3."""
    rc = cli_main(["dataset", "--root", str(tree), "--license", "private",
                   "--fake-source", "gen", "--splits", "test"])
    assert rc == 0
    assert "R1+R2+R3" in capsys.readouterr().out


def test_cli_ingest_end_to_end(tree: Path, tmp_path: Path, capsys):
    out = tmp_path / "norm"
    rc = cli_main(["ingest", "--root", str(tree), "--out", str(out),
                   "--license", "research-only", "--splits", "train", "test",
                   "--limit", "2"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "modal volume put dire-raw" in text
    assert "checkdata" in text
    assert (out / "manifest.tsv").exists() and (out / "data_card.json").exists()
    card = json.loads((out / "data_card.json").read_text(encoding="utf-8"))
    assert card["label"] == "R1+R2"
    assert card["n_real"] > 0 and card["n_fake"] > 0
    # giới hạn 2 ảnh/archive
    rows = ds.read_manifest_tsv(out / "manifest.tsv")
    per_archive = {}
    for r in rows:
        per_archive[r["source_tar"]] = per_archive.get(r["source_tar"], 0) + 1
    assert max(per_archive.values()) <= 2
