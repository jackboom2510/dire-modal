"""Read-only inventory adapter for the extracted CelebA-HQ DIRE maps on Kaggle.

The Kaggle input reported by the user is an extracted directory tree, e.g.
``dire/train/celebahq/real/real/<image>``. This module resolves the repeated
leaf folder without copying or changing the mounted data. The inferred member
locator is explicitly not proof of the original tar member path; archive bytes
and provenance are unavailable from this extracted tree.

This is an inventory/path adapter only. It does not assign validation splits,
open/decode images, compute hashes, train, or modify input data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
REQUIRED_GROUPS: tuple[tuple[str, str], ...] = (
    ("train", "real"),
    ("train", "sdv2"),
    ("test", "real"),
    ("test", "dalle2"),
    ("test", "if"),
    ("test", "midjourney"),
    ("test", "stargan"),
    ("test", "sdv2"),
)


class CelebAHQTreeLayoutError(ValueError):
    """The extracted tree is missing a required group or has ambiguous layout."""


@dataclass(frozen=True)
class CelebAHQImage:
    """One image path and metadata; no image bytes are read."""

    split: str
    generator: str
    label: int
    path: Path
    relative_path: str
    member_locator: str
    locator_inferred: bool
    leaf_layout: str
    size_bytes: int
    mtime_ns: int


def _iter_image_files(
    directory: Path,
    *,
    progress_callback: Callable[[str], None] | None = None,
    progress_label: str = "",
):
    """Yield ``(path, stat)`` using scandir metadata without following symlinks.

    Keeping the ``DirEntry.stat`` result avoids later resolving and statting
    every file a second time when building the Stage 2 metadata fingerprint.
    """
    yielded = 0
    stack = [directory]
    while stack:
        current = stack.pop()
        with os.scandir(current) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
        child_directories: list[Path] = []
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                child_directories.append(Path(entry.path))
                continue
            if Path(entry.name).suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if not entry.is_file(follow_symlinks=False):
                continue
            stat = entry.stat(follow_symlinks=False)
            yielded += 1
            if progress_callback is not None and yielded % 5_000 == 0:
                progress_callback(
                    f"path scan {progress_label}: {yielded:,} image files indexed"
                )
            yield Path(entry.path), stat
        # Match os.walk's stable top-down traversal order.
        stack.extend(reversed(child_directories))


def _direct_images(directory: Path) -> list[tuple[Path, os.stat_result]]:
    with os.scandir(directory) as iterator:
        entries = sorted(iterator, key=lambda entry: entry.name)
    images: list[tuple[Path, os.stat_result]] = []
    for entry in entries:
        if Path(entry.name).suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if entry.is_file(follow_symlinks=False):
            images.append((Path(entry.path), entry.stat(follow_symlinks=False)))
    return images


def scan_celebahq_tree(
    root: str | Path,
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> list[CelebAHQImage]:
    """Index selected CelebA-HQ groups from an extracted tree.

    ``root`` is the mounted ``dire`` directory. For each generator folder, the
    scanner prefers its repeated same-name child (``real/real``) when it holds
    images and the outer folder has no direct images. If images exist both
    directly and below the repeated child, it raises instead of silently
    choosing one. A flat ``real/<image>`` layout is also accepted.

    ``member_locator`` approximates a member-style locator such as
    ``real/0001.png`` by prefixing the generator name to the path under the
    selected leaf. ``locator_inferred`` remains true because the dataset is
    extracted and original tar member paths were not verified in this scan.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"DIRE root is not a directory: {root_path}")

    records: list[CelebAHQImage] = []
    missing: list[str] = []
    for split, generator in REQUIRED_GROUPS:
        outer = root_path / split / "celebahq" / generator
        if not outer.is_dir():
            missing.append(f"{split}/celebahq/{generator}")
            continue

        direct = _direct_images(outer)
        repeated = outer / generator
        repeated_images = (
            list(_iter_image_files(
                repeated,
                progress_callback=progress_callback,
                progress_label=f"{split}/celebahq/{generator}",
            ))
            if repeated.is_dir() and not repeated.is_symlink()
            else []
        )
        other_nested_images: list[tuple[Path, os.stat_result]] = []
        for child in sorted(outer.iterdir()):
            if child == repeated or not child.is_dir() or child.is_symlink():
                continue
            other_nested_images.extend(_iter_image_files(
                child,
                progress_callback=progress_callback,
                progress_label=f"unexpected nested directory {child}",
            ))
        if other_nested_images:
            raise CelebAHQTreeLayoutError(
                f"unexpected nested image directories under {outer}: "
                f"{[path.relative_to(outer).as_posix() for path, _ in other_nested_images[:5]]}"
            )
        if direct and repeated_images:
            raise CelebAHQTreeLayoutError(
                f"ambiguous class directory {outer}: found direct images and images under {repeated}"
            )
        if repeated_images:
            leaf_root = repeated
            leaf_layout = "repeated-generator-child"
            files = repeated_images
        elif direct:
            leaf_root = outer
            leaf_layout = "flat-class-directory"
            files = direct
        else:
            missing.append(f"{split}/celebahq/{generator} (no image files in expected leaf)")
            continue

        if progress_callback is not None:
            progress_callback(
                f"path scan complete {split}/celebahq/{generator}: {len(files):,} images"
            )

        for path, stat in files:
            relative = path.relative_to(root_path).as_posix()
            member_inside_leaf = path.relative_to(leaf_root).as_posix()
            member_locator = (Path(generator) / member_inside_leaf).as_posix()
            records.append(CelebAHQImage(
                split=split,
                generator=generator,
                label=0 if generator == "real" else 1,
                path=path,
                relative_path=relative,
                member_locator=member_locator,
                locator_inferred=True,
                leaf_layout=leaf_layout,
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            ))

    if missing:
        raise CelebAHQTreeLayoutError(
            "missing or empty required CelebA-HQ groups: " + ", ".join(missing)
        )
    return records
