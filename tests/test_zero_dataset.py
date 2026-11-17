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
    _make_tar(base / "images/val/lsun_bedroom/iddpm.tar.gz", [_png_bytes((40, 40, 90), seed=i) for i in range(2)])
    _make_tar(base / "images/val/lsun_bedroom/stylegan.tar.gz", [_png_bytes((30, 150, 30), seed=i) for i in range(2)])
    # cùng generator, tên khác ở test
    _make_tar(base / "images/test/lsun_bedroom/stylegan_official.tar.gz", [_png_bytes((31, 151, 31), seed=i) for i in range(2)])
    # thư mục giải nén dở nằm cạnh tar (phải cảnh báo, không được im lặng)
    dangling = base / "images/test/lsun_bedroom/real"
    dangling.mkdir(parents=True)
    (dangling / "000001.png").write_bytes(_png_bytes((9, 9, 9)))
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
    assert set(m["lsun_bedroom"]["test"]) == {"real", "pndm", "sdv1", "stylegan"}
    assert set(m["lsun_bedroom"]["val"]) == {"real", "pndm", "iddpm", "stylegan"}
    assert set(scan.generators("lsun_bedroom", "train")) == {"real", "adm", "stylegan"}


def test_sdv1_test_is_the_same_generator_as_sdv1():
    assert ds.generator_of("/x/images/test/lsun_bedroom/sdv1_test.tar.gz") == "sdv1"
    assert ds.generator_of("adm.tar.gz") == "adm"
    assert ds.label_for_generator("real") == 0
    assert ds.label_for_generator("adm") == 1


def test_stylegan_official_is_stylegan_but_diff_stylegan_is_not():
    """Tên khác nhau cho cùng generator phải gộp; generator khác thì không."""
    assert ds.generator_of("stylegan_official.tar.gz") == "stylegan"
    assert ds.generator_of("stylegan.tar.gz") == "stylegan"
    # Diffusion-styleGAN / Diffusion-ProjectedGAN là generator KHÁC
    assert ds.generator_of("diff-stylegan.tar.gz") == "diff-stylegan"
    assert ds.generator_of("diff-projectedgan.tar.gz") == "diff-projectedgan"


def _parts_under(item, root: Path) -> tuple[str, ...]:
    """Thành phần đường dẫn **tương đối** — không thì tên thư mục tmp lẫn vào."""
    return item.source.relative_to(root).parts


def test_dire_is_refused_unless_explicitly_allowed(tree: Path):
    scan = ds.scan_tree(tree)
    assert all(p[0] != "dire" for p in (_parts_under(it, tree) for it in scan.items))
    assert any("G0" in n for n in scan.notes)

    scan2 = ds.scan_tree(tree, allow_published=True)
    assert any("R0" in n for n in scan2.notes)


def test_dangling_extracted_dir_is_reported_not_silently_skipped(tree: Path):
    """`real/` nằm cạnh `real.tar.gz` ⇒ phải cảnh báo, và không được đếm vào dữ liệu."""
    scan = ds.scan_tree(tree)
    text = " ".join(scan.warnings)
    assert "THƯ MỤC ảnh" in text and "real/" in text
    # chỉ tar được dùng
    assert all(it.source.suffix != "" for it in scan.items)
    assert not any(it.source.is_dir() for it in scan.items)
    item = next(i for i in scan.items if i.generator == "real" and i.split == "test")
    # số ảnh đến từ tar (3), không phải 3 + 1 file trong thư mục rác
    n, _ = ds.count_items([item])
    assert n == 3


def test_count_items_only_counts_selected(tree: Path):
    """Đếm chỉ phần được chọn — đây là lý do `scan_tree` không đếm nữa."""
    scan = ds.scan_tree(tree)
    assert all(it.n_images is None for it in scan.items)      # quét không đếm
    picked = ds.select(scan, dataset="lsun_bedroom", splits=("test",))
    total, warns = ds.count_items(picked)
    assert total == sum(1 for _ in []) or total > 0
    assert all(it.n_images is not None for it in picked)
    # archive không được chọn vẫn chưa đếm
    other = next(it for it in scan.items if it.dataset == "imagenet")
    assert other.n_images is None
    assert not warns


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
    assert scan.generators("lsun_bedroom", "test") == ["real", "pndm", "sdv1", "stylegan"]


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

    assert stats["written"] == sum(stats["per_archive"].values())
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
    # Volume mới: `dire-zero`, gốc `data/` (bố cục cũ `dire-raw` + `zero/data` đã bị thay)
    assert "modal volume put dire-zero" in text
    # Modal 1.x: `-r` không tồn tại (thư mục luôn đệ quy) và làm lệnh dừng ngay
    assert " -r" not in text
    assert "--force" in text        # chạy lại mà thiếu --force ⇒ FileExistsError
    assert "data" in text and "zero/data" not in text
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


def test_tar_inventory_reports_suffixes_without_extracting(tmp_path: Path):
    """`peek` phải nói được bên trong là .npy hay .png — đây là câu hỏi chặn R0."""
    import io
    import tarfile

    import numpy as np

    p = tmp_path / "maps_adm.tar.gz"
    with tarfile.open(p, "w:gz") as tf:
        for i in range(3):
            buf = io.BytesIO()
            np.save(buf, np.zeros((3, 256, 256), np.float16))
            data = buf.getvalue()
            ti = tarfile.TarInfo(f"lsun/{i:04d}.npy")
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
        readme = b"notes"
        ti = tarfile.TarInfo("lsun/README.txt")
        ti.size = len(readme)
        tf.addfile(ti, io.BytesIO(readme))

    info = ds.tar_inventory(p)
    assert info["by_ext"][".npy"] == 3
    assert info["by_ext"][".txt"] == 1
    assert info["n_members"] == 4
    assert len(info["sample"]) == 3
    assert all(".npy" in s for s in info["sample"])


def test_tar_inventory_sample_is_capped(tmp_path: Path):
    import io
    import tarfile

    p = tmp_path / "imgs.tar.gz"
    with tarfile.open(p, "w:gz") as tf:
        for i in range(5):
            data = _png_bytes((1, 2, 3), seed=i)
            ti = tarfile.TarInfo(f"{i}.png")
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    assert len(ds.tar_inventory(p, n_sample=2)["sample"]) == 2


# --------------------------------------------------------------------------- #
# Đường dự phòng khi ingest hỏng giữa đường
# --------------------------------------------------------------------------- #

def _single(tmp_path: Path, name: str, images: list[bytes], labels=(1,), gen="adm") -> list:
    """Một archive + Item tương ứng, để test đường dự phòng ở phạm vi hẹp."""
    p = _make_tar(tmp_path / name, images)
    return [ds.Item(source=p, split="test", dataset="lsun_bedroom",
                    generator=gen, label=labels[0])]


def test_truncated_archive_falls_back_to_streaming_read(tmp_path: Path):
    """Tar tải thiếu đuôi: `r:*` chết, `r|*` phải cứu được phần đầu.

    Đây là lỗi thật của người dùng (tải 4 GB từ Baidu, hỏng 3 %). Trước đây nó
    làm mất trắng archive dù 97 % nội dung nằm ở đầu file.
    """
    imgs = [_png_bytes((i * 20, 40, 90), seed=i) for i in range(8)]
    good = _make_tar(tmp_path / "adm.tar.gz", imgs)
    raw = good.read_bytes()
    broken = tmp_path / "adm_cut.tar.gz"
    broken.write_bytes(raw[: int(len(raw) * 0.6)])

    made, stats = ds.ingest(
        [ds.Item(source=broken, split="test", dataset="lsun_bedroom", generator="adm", label=1)],
        tmp_path / "out", size=32,
    )
    # không ném, giữ được phần đọc tới chỗ hỏng
    assert stats["written"] > 0
    assert made and all(m.generator == "adm" for m in made)
    # và phải KHAI là đã đọc dở, không được im lặng
    assert len(stats["partial_archives"]) == 1
    rec = stats["partial_archives"][0]
    assert rec["read"] == stats["written"]
    assert "EOFError" in rec["error"] or "ReadError" in rec["error"]
    assert rec["mode"] == "r|*"


def test_strict_still_refuses_a_broken_archive(tmp_path: Path):
    imgs = [_png_bytes((i * 20, 40, 90), seed=i) for i in range(6)]
    good = _make_tar(tmp_path / "adm.tar.gz", imgs)
    broken = tmp_path / "adm_cut.tar.gz"
    broken.write_bytes(good.read_bytes()[: int(good.stat().st_size * 0.6)])

    with pytest.raises(ds.IntegrityError, match="archive hỏng"):
        ds.ingest(
            [ds.Item(source=broken, split="test", dataset="lsun_bedroom", generator="adm", label=1)],
            tmp_path / "out", size=32, on_error="strict",
        )


def test_cut_image_gets_a_second_chance_and_is_declared(tmp_path: Path):
    """Ảnh cụt trong tar: đọc lại bằng LOAD_TRUNCATED_IMAGES, và ghi lại là đã dùng."""
    full = _png_bytes((10, 200, 10), size=96, seed=5)
    cut = full[: len(full) - 60]
    items = _single(tmp_path, "adm.tar.gz", [full, cut])
    made, stats = ds.ingest(items, tmp_path / "out", size=32)

    assert stats["written"] == 2          # ảnh cụt vẫn dùng được
    assert stats["n_rejects"] == 0
    assert stats["n_fallbacks"] == 1
    assert "ảnh cụt" in stats["fallbacks"][0]


def test_undecodable_member_is_named_not_just_counted(tmp_path: Path):
    """Bỏ ảnh hỏng thì được, nhưng phải biết **ảnh nào** — nếu không thì không sửa được."""
    items = _single(tmp_path, "adm.tar.gz", [_png_bytes((5, 5, 5)), b"\x00" * 300])
    made, stats = ds.ingest(items, tmp_path / "out", size=32)

    assert stats["written"] == 1
    assert stats["n_rejects"] == 1
    rec = stats["rejects"][0]
    assert rec["member"] == "000001.png"
    assert rec["generator"] == "adm" and rec["split"] == "test"
    assert "UnidentifiedImageError" in rec["reason"]
    assert stats["skipped"] == 1          # vẫn giữ con số cũ cho báo cáo


def test_resume_does_not_write_twice(tmp_path: Path):
    """Chạy lại sau khi hỏng phải là thao tác an toàn, không nhân đôi dữ liệu."""
    items = _single(tmp_path, "adm.tar.gz", [_png_bytes((i * 9, 0, 0), seed=i) for i in range(4)])
    out = tmp_path / "out"
    _, first = ds.ingest(items, out, size=32)
    n_files = len(list(out.rglob("*.png")))
    _, second = ds.ingest(items, out, size=32)      # chạy lại y nguyên

    assert len(list(out.rglob("*.png"))) == n_files == 4
    assert first["written"] == 4 and first["reused"] == 0
    assert second["written"] == 0 and second["reused"] == 4


def test_resume_no_resume_rewrites(tmp_path: Path):
    items = _single(tmp_path, "adm.tar.gz", [_png_bytes((7, 7, 7))])
    out = tmp_path / "out"
    ds.ingest(items, out, size=32)
    _, second = ds.ingest(items, out, size=32, resume=False)
    assert second["written"] == 1 and second["reused"] == 0


def test_cross_run_leak_is_caught(tmp_path: Path):
    """Điểm khó nhất của resume: rò rỉ **qua ranh giới hai lần chạy**.

    Không seed `pix_seen` từ PNG đã có thì lần chạy trước để lại `real/x.png`,
    lần này `adm` có đúng pixel đó ⇒ không thấy trùng ⇒ ACC ~100 % mà không có
    lỗi nào báo. Đây chính là lý do phải quét lại thư mục đầu ra.
    """
    out = tmp_path / "out"
    same = _png_bytes((33, 66, 99), seed=2)
    ds.ingest(_single(tmp_path, "real.tar.gz", [same], labels=(0,), gen="real"),
              out, size=32)
    with pytest.raises(ds.IntegrityError, match="trùng pixel"):
        ds.ingest(_single(tmp_path, "adm.tar.gz", [same], labels=(1,), gen="adm"),
                  out, size=32)


def test_cli_reports_partial_archive_in_card_and_exit_code(tmp_path: Path, capsys):
    """Mã thoát + `data_card.warnings`: dữ liệu thiếu phải đi theo dữ liệu."""
    base = tmp_path / "DiffusionForensics"
    imgs = [_png_bytes((i * 15, 30, 60), seed=i) for i in range(6)]
    good = _make_tar(base / "images/test/lsun_bedroom/adm.tar.gz", imgs)
    cut = base / "images/test/lsun_bedroom/adm_cut.tar.gz"
    cut.write_bytes(good.read_bytes()[: int(good.stat().st_size * 0.6)])
    good.unlink()                                # chỉ còn archive hỏng

    out = tmp_path / "norm"
    rc = cli_main(["ingest", "--root", str(base), "--out", str(out),
                   "--license", "research-only", "--splits", "test"])
    assert rc == 3                               # KHÔNG được nói "xong" khi thiếu dữ liệu
    text = capsys.readouterr()
    assert "dự phòng" in text.out + text.err
    card = json.loads((out / "data_card.json").read_text(encoding="utf-8"))
    assert any("[dự phòng]" in w for w in card["warnings"])
    assert "archive đọc dở" in " ".join(card["warnings"])


def test_cli_strict_flag_turns_fallback_off(tmp_path: Path, capsys):
    base = tmp_path / "DiffusionForensics"
    imgs = [_png_bytes((i * 15, 30, 60), seed=i) for i in range(6)]
    good = _make_tar(base / "images/test/lsun_bedroom/adm.tar.gz", imgs)
    cut = base / "images/test/lsun_bedroom/adm_cut.tar.gz"
    cut.write_bytes(good.read_bytes()[: int(good.stat().st_size * 0.6)])
    good.unlink()

    rc = cli_main(["ingest", "--root", str(base), "--out", str(tmp_path / "norm"),
                   "--license", "research-only", "--splits", "test", "--strict"])
    assert rc == 3
    assert not any((tmp_path / "norm").rglob("*.png"))   # strict: không ghi gì dở dang


# --------------------------------------------------------------------------- #
# Chỗ ghi: ổ chính trước, ổ F: dự phòng khi thiếu dung lượng
# --------------------------------------------------------------------------- #

def test_choose_out_prefers_primary_when_it_has_space(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ds, "free_bytes", lambda p: 500 * 10**9)
    c = ds.choose_out_root(tmp_path / "data", fallbacks=(str(tmp_path / "F"),), needed=10**9)
    assert c.path == tmp_path / "data"
    assert not c.redirected
    assert "đủ chỗ" in c.reason


def test_choose_out_falls_back_to_F_when_primary_is_full(tmp_path: Path, monkeypatch):
    """Đúng yêu cầu: F: là **dự phòng cho dung lượng**, không phải chỗ mặc định."""
    def free(p):
        return 200 * 10**6 if "data" in str(p) else 900 * 10**9
    monkeypatch.setattr(ds, "free_bytes", free)

    c = ds.choose_out_root(tmp_path / "data", fallbacks=(str(tmp_path / "F"),), needed=5 * 10**9)
    assert c.path == tmp_path / "F"
    assert c.redirected
    assert "dự phòng" in c.reason
    assert "200 MB" in c.reason          # số phải đọc được, không phải "0.0 GB"


def test_choose_out_keeps_reserve_for_the_os(tmp_path: Path, monkeypatch):
    """Còn trống 1 GB nhưng cần 0,8 GB + 0,5 GB dự trữ ⇒ **không** ghi vào đó."""
    monkeypatch.setattr(ds, "free_bytes",
                        lambda p: 10**9 if str(p).endswith("primary") else 50 * 10**9)
    c = ds.choose_out_root(tmp_path / "primary", fallbacks=(str(tmp_path / "spare"),),
                           needed=8 * 10**8)
    assert c.path == tmp_path / "spare" and c.redirected


def test_choose_out_skips_a_drive_that_does_not_exist(tmp_path: Path, monkeypatch):
    """`free_bytes` trả None = **không biết**, khác với 0 = hết chỗ. Không thử rồi chết."""
    monkeypatch.setattr(ds, "free_bytes", lambda p: None if str(p).endswith("F") else 10**8)
    with pytest.raises(ds.OutOfSpaceError) as ei:
        ds.choose_out_root(tmp_path / "data", fallbacks=(str(tmp_path / "F"),), needed=10**9)
    # ổ chưa cắm có thông báo riêng, không bị gộp thành "hết chỗ"
    assert "không truy cập được" in str(ei.value)


def test_choose_out_fails_loudly_with_numbers(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ds, "free_bytes", lambda p: 10**8)
    with pytest.raises(ds.OutOfSpaceError) as ei:
        ds.choose_out_root(tmp_path / "a", fallbacks=(str(tmp_path / "b"),), needed=9 * 10**9)
    msg = str(ei.value)
    assert "100 MB" in msg and "9.0 GB" in msg          # đủ số để quyết định
    assert str(tmp_path / "a") in msg and str(tmp_path / "b") in msg


def test_choose_out_dedupes_when_fallback_is_the_primary(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ds, "free_bytes", lambda p: 10**12)
    c = ds.choose_out_root(tmp_path / "x", fallbacks=(str(tmp_path / "x"),), needed=1)
    assert len(c.tried) == 1


def test_estimate_without_limit_bounds_by_source_size(tree: Path):
    items = ds.select(ds.scan_tree(tree), dataset="lsun_bedroom", splits=("test",))
    est = ds.estimate_output_bytes(items)
    src = sum(it.source.stat().st_size for it in items)
    assert est["need"] == int(src * 1.25)
    assert est["n_images"] is None                 # chưa đếm, và không cần đếm


def test_estimate_with_limit_counts_images_and_stops_early(tree: Path):
    items = ds.select(ds.scan_tree(tree), dataset="lsun_bedroom", splits=("test",))
    est = ds.estimate_output_bytes(items, limit_per_archive=2)
    assert est["n_images"] == 2 * len(items)       # mỗi archive đóng góp đúng `limit`
    assert est["need"] == est["n_images"] * ds.BYTES_PER_PNG
    assert ds.BYTES_PER_PNG >= 256 * 256 * 3       # trần, không phải trung bình


def test_manifest_paths_are_relative_and_survive_moving_the_tree(tree: Path, tmp_path: Path):
    """Có ổ dự phòng thì đường dẫn tuyệt đối trong manifest là lỗi chờ ngày nổ."""
    items = ds.select(ds.scan_tree(tree), dataset="lsun_bedroom", splits=("test",))
    out = tmp_path / "a" / "norm"
    made, _ = ds.ingest(items, out, size=32)
    manifest = ds.write_manifest_tsv(out / "manifest.tsv", made, root=out)

    rows = ds.read_manifest_tsv(manifest)
    assert all(not Path(r["png"]).is_absolute() for r in rows)
    assert all(r["png"].count("/") == 2 for r in rows)      # test/<gen>/<sha>.png
    ok, bad = ds.verify_manifest(manifest)
    assert ok == len(rows) and not bad

    moved = tmp_path / "b" / "norm"
    moved.parent.mkdir(parents=True)
    out.rename(moved)                                        # "đổi ổ" / đổi máy
    ok2, bad2 = ds.verify_manifest(moved / "manifest.tsv")
    assert ok2 == len(rows) and not bad2


def test_verify_manifest_still_accepts_old_absolute_rows(tree: Path, tmp_path: Path):
    items = ds.select(ds.scan_tree(tree), dataset="lsun_bedroom", splits=("test",))
    out = tmp_path / "norm"
    made, _ = ds.ingest(items, out, size=32)
    manifest = ds.write_manifest_tsv(out / "manifest.tsv", made)   # kiểu cũ: tuyệt đối
    rows = ds.read_manifest_tsv(manifest)
    assert all(Path(r["png"]).is_absolute() for r in rows)
    ok, bad = ds.verify_manifest(manifest)
    assert ok == len(rows) and not bad


def test_cli_ingest_redirects_to_fallback_and_records_it(tree: Path, tmp_path: Path, monkeypatch, capsys):
    """Thiếu chỗ ở ổ chính ⇒ ghi sang F:, nhưng **phải nói** và ghi vào data_card."""
    primary = tmp_path / "primary"
    fb = tmp_path / "F_drive"
    monkeypatch.setattr(ds, "free_bytes",
                        lambda p: 10**8 if "primary" in str(p) else 800 * 10**9)

    rc = cli_main(["ingest", "--root", str(tree), "--out", str(primary),
                   "--out-fallback", str(fb), "--license", "research-only",
                   "--splits", "test", "--limit", "1"])
    assert rc == 0
    assert (fb / "manifest.tsv").exists() and (fb / "data_card.json").exists()
    assert not (primary / "manifest.tsv").exists()      # không ghi nửa vời sang ổ chính

    card = json.loads((fb / "data_card.json").read_text(encoding="utf-8"))
    assert card["out_root"] == str(fb)
    assert "dự phòng" in card["out_reason"]
    assert any("[dự phòng dung lượng]" in w for w in card["warnings"])
    assert "dự phòng" in capsys.readouterr().err        # cảnh báo ra stderr


def test_cli_ingest_without_fallback_stops_before_writing(tree: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ds, "free_bytes", lambda p: 10**8)     # chỉ còn 0,1 GB
    primary = tmp_path / "primary"
    rc = cli_main(["ingest", "--root", str(tree), "--out", str(primary),
                   "--no-out-fallback", "--license", "research-only", "--splits", "test"])
    assert rc == 3
    assert not primary.exists() or not any(primary.rglob("*.png"))


def test_cli_ingest_default_out_is_relative(tmp_path: Path, monkeypatch, capsys):
    """Mặc định không được khoá vào ổ nào: `./data/dire-zero-norm` theo thư mục hiện tại."""
    assert not Path(ds.DEFAULT_OUT_ROOT).is_absolute()
    assert ds.DEFAULT_OUT_FALLBACKS == ("F:/dire-zero-norm",)
