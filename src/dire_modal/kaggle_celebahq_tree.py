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
    """One image path and its class metadata; no image bytes are read."""

    split: str
    generator: str
    label: int
    path: Path
    relative_path: str
    member_locator: str
    locator_inferred: bool
    leaf_layout: str


def _iter_image_files(directory: Path):
    """Yield regular image files in stable order without following symlinks."""
    for current, dirnames, filenames in os.walk(directory, topdown=True, followlinks=False):
        current_path = Path(current)
        dirnames[:] = sorted(
            name for name in dirnames
            if not (current_path / name).is_symlink()
        )
        for filename in sorted(filenames):
            path = current_path / filename
            if path.is_symlink() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if path.is_file():
                yield path


def _direct_images(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file()
        and not path.is_symlink()
        and path.suffix.lower() in IMAGE_SUFFIXES
    )


def scan_celebahq_tree(root: str | Path) -> list[CelebAHQImage]:
    """Index the selected CelebA-HQ train/test classes from an extracted tree.

    ``root`` is the mounted ``dire`` directory. For each generator folder, the
    scanner prefers its repeated same-name child (``real/real``) when it holds
    images and the outer folder has no direct images. If images exist both
    directly and below the repeated child, it raises instead of silently
    choosing one. A flat ``real/<image>`` layout is also accepted.

    ``member_locator`` approximates a full member-style locator such as
    ``real/0001.png`` by prefixing the generator name to the path under the
    selected leaf. ``locator_inferred`` remains true because this dataset is
    extracted and the original tar archives were not verified in this scan.
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
            list(_iter_image_files(repeated))
            if repeated.is_dir() and not repeated.is_symlink()
            else []
        )
        other_nested_images = []
        for child in sorted(outer.iterdir()):
            if child == repeated or not child.is_dir() or child.is_symlink():
                continue
            other_nested_images.extend(_iter_image_files(child))
        if other_nested_images:
            raise CelebAHQTreeLayoutError(
                f"unexpected nested image directories under {outer}: "
                f"{[path.relative_to(outer).as_posix() for path in other_nested_images[:5]]}"
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

        for path in files:
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
            ))

    if missing:
        raise CelebAHQTreeLayoutError(
            "missing or empty required CelebA-HQ groups: " + ", ".join(missing)
        )
    return records
