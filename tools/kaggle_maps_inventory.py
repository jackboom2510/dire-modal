#!/usr/bin/env python3
"""Read-only inventory for the CelebA-HQ DIRE maps destined for Kaggle.

This tool does not extract, rewrite, upload, or train on the input archives.
It verifies the expected per-split/per-generator tar files, records archive
SHA-256s and member names/counts, and reports candidate train/test ID overlaps.
Member basenames are only candidate IDs; they are not proof of source-image
identity. ``--hash-members`` optionally hashes every regular tar member so an
exact byte match can be distinguished from a basename collision.

Example (run on the machine holding the local maps)::

    python tools/kaggle_maps_inventory.py \
      --root /path/to/dataset \
      --out /path/to/kaggle-map-manifest.json

Add ``--hash-members`` for a full-content overlap audit before publishing the
private Kaggle Dataset. No generated manifest or data is committed by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

DATASET_REL = Path("dire")
EXPECTED: tuple[tuple[str, str, str | None], ...] = (
    ("train", "real", None),
    ("train", "sdv2", "stable-diffusion"),
    ("test", "real", None),
    ("test", "dalle2", "dall-e"),
    ("test", "if", "imagen"),
    ("test", "midjourney", "midjourney"),
    ("test", "stargan", "stargan"),
    ("test", "sdv2", "stable-diffusion"),
)
IMAGE_SUFFIXES = {
    ".bmp", ".jpeg", ".jpg", ".npy", ".npz", ".png", ".webp",
}
CHUNK_SIZE = 1024 * 1024


class InventoryError(Exception):
    """An expected archive is missing, unreadable, or not a readable tar."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_member(tf: tarfile.TarFile, member: tarfile.TarInfo) -> str:
    stream = tf.extractfile(member)
    if stream is None:
        raise InventoryError(f"cannot read regular tar member {member.name!r}")
    digest = hashlib.sha256()
    with stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_id(member_name: str) -> str:
    basename = PurePosixPath(member_name).name
    return Path(basename).stem


def _archive_path(root: Path, split: str, generator: str) -> Path:
    return root / DATASET_REL / split / "celebahq" / f"{generator}.tar.gz"


def _inspect_archive(
    root: Path,
    split: str,
    generator: str,
    family: str | None,
    *,
    hash_members: bool,
) -> dict[str, Any]:
    path = _archive_path(root, split, generator)
    relative = path.relative_to(root).as_posix()
    if not path.is_file():
        raise InventoryError(f"missing required archive: {relative}")

    members: list[dict[str, Any]] = []
    suffix_counts: Counter[str] = Counter()
    seen_names: set[str] = set()
    duplicate_names: list[str] = []
    try:
        # r:* intentionally accepts a tar stream whether or not it is gzip'd.
        with tarfile.open(path, "r:*") as tf:
            for info in tf:
                if not info.isfile():
                    continue
                name = info.name
                if name in seen_names:
                    duplicate_names.append(name)
                seen_names.add(name)
                suffix = PurePosixPath(name).suffix.lower()
                suffix_counts[suffix or "<none>"] += 1
                row: dict[str, Any] = {
                    "path": name,
                    "size_bytes": info.size,
                    "candidate_id": _candidate_id(name),
                    "suffix": suffix,
                    "image_like": suffix in IMAGE_SUFFIXES,
                }
                if hash_members:
                    row["content_sha256"] = _sha256_member(tf, info)
                members.append(row)
    except (tarfile.TarError, OSError, EOFError, InventoryError) as exc:
        raise InventoryError(f"cannot inspect {relative}: {exc}") from exc

    if not members:
        raise InventoryError(f"archive has no regular file members: {relative}")
    candidate_counts = Counter(member["candidate_id"] for member in members)

    return {
        "path": relative,
        "split": split,
        "generator": generator,
        "generator_family": family,
        "archive_bytes": path.stat().st_size,
        "archive_sha256": _sha256_file(path),
        "regular_members": len(members),
        "image_members": sum(1 for member in members if member["image_like"]),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "duplicate_member_paths": sorted(set(duplicate_names)),
        "duplicate_candidate_ids": sorted(
            candidate for candidate, count in candidate_counts.items() if count > 1
        ),
        "members": members,
    }


def _overlaps(archives: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_generator: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for archive in archives:
        ids = by_generator[archive["generator"]][archive["split"]]
        for member in archive["members"]:
            candidate = member["candidate_id"]
            # Preserve all rows in the unusual case of repeated basenames.
            ids.setdefault(candidate, member)

    rows: list[dict[str, Any]] = []
    for generator, splits in sorted(by_generator.items()):
        train_ids = splits.get("train", {})
        test_ids = splits.get("test", {})
        for candidate in sorted(train_ids.keys() & test_ids.keys()):
            left = train_ids[candidate]
            right = test_ids[candidate]
            left_hash = left.get("content_sha256")
            right_hash = right.get("content_sha256")
            rows.append({
                "generator": generator,
                "candidate_id": candidate,
                "left_path": next(
                    a["path"] for a in archives
                    if a["split"] == "train" and a["generator"] == generator
                ),
                "right_path": next(
                    a["path"] for a in archives
                    if a["split"] == "test" and a["generator"] == generator
                ),
                "map_bytes_identical": (
                    left_hash == right_hash
                    if left_hash is not None and right_hash is not None
                    else None
                ),
                "content_sha256": left_hash if left_hash == right_hash else None,
            })
    return rows


def build_manifest(root: Path, *, hash_members: bool = False) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise InventoryError(f"input root is not a directory: {root}")

    archives = [
        _inspect_archive(root, split, generator, family, hash_members=hash_members)
        for split, generator, family in EXPECTED
    ]
    overlaps = _overlaps(archives)
    return {
        "schema_version": 1,
        "dataset": "dire-celebahq-maps",
        "inventory_mode": "read-only; no extraction or source modification",
        "root_name": root.name,
        "taxonomy_version": "adr-0007-celebahq-families-v1",
        "split_policy": {
            "validation": "not assigned; decide after inventory",
            "train": "source train maps only",
            "test": "source test maps only",
        },
        "id_policy": {
            "kind": "candidate basename, not verified identity",
            "hash_members": hash_members,
            "member_hash_scope": "uncompressed regular tar-member bytes" if hash_members else None,
        },
        "holdout_families": ["dall-e", "imagen", "midjourney", "stargan"],
        "seen_generator_control": "stable-diffusion",
        "archives": archives,
        "candidate_id_overlaps": overlaps,
        "warnings": [
            "Candidate IDs come from member basenames; inspect provenance before treating them as sample IDs.",
            "This manifest records local observations only; it does not establish source provenance or remote/Kaggle availability.",
            "Validation IDs and fraction are intentionally not assigned by this inventory tool.",
            "Member format is inferred from filename suffix only; this tool does not decode/validate image pixels.",
            "Without --hash-members, member payload checksums are not verified even though the whole archive hash is recorded.",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path,
                        help="dataset root containing dire/{train,test}/celebahq/*.tar.gz")
    parser.add_argument("--out", required=True, type=Path,
                        help="manifest JSON output path; source archives are not modified")
    parser.add_argument("--hash-members", action="store_true",
                        help="hash decompressed bytes of every regular tar member (CPU/I/O intensive)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = build_manifest(args.root, hash_members=args.hash_members)
        out = args.out.expanduser().resolve()
        root = args.root.expanduser().resolve()
        if out == root or root in out.parents:
            raise InventoryError("--out must be outside --root so the input dataset stays read-only")
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_name(out.name + ".tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, out)
    except (InventoryError, OSError) as exc:
        print(f"inventory failed: {exc}", file=sys.stderr)
        return 2

    total_bytes = sum(row["archive_bytes"] for row in manifest["archives"])
    total_members = sum(row["regular_members"] for row in manifest["archives"])
    print(f"read-only inventory: {len(manifest['archives'])} archives")
    print(f"archive bytes: {total_bytes}; regular members: {total_members}")
    print(f"candidate train/test ID overlaps: {len(manifest['candidate_id_overlaps'])}")
    print(f"manifest: {out}")
    print("Counts are archive-member counts, not yet verified unique source-image counts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
