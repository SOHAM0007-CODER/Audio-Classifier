#!/usr/bin/env python3
"""
scripts/package_for_cloud.py

Zips the code needed for cloud training (no datasets, checkpoints, caches or virtual environments) so it can
be uploaded to Google Drive or a cloud GPU machine. Datasets and manifests are regenerated there.

Usage (from the repository root):
    python scripts/package_for_cloud.py                         # -> dist/multitask_ast_code.zip
    python scripts/package_for_cloud.py --output my_code.zip
"""

import argparse
import zipfile
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIRS = ("src", "scripts", "configs", "notebooks", "tests")
INCLUDE_FILES = ("requirements.txt", "README.md", ".gitignore", ".gitattributes")
EXCLUDE_DIR_NAMES = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache"}


def collect_files(root: Path = REPO_ROOT) -> List[Path]:
    files = [root / name for name in INCLUDE_FILES if (root / name).is_file()]
    for dirname in INCLUDE_DIRS:
        for path in sorted((root / dirname).rglob("*")):
            relative_parts = path.relative_to(root).parts
            if path.is_file() and path.suffix != ".pyc" and not EXCLUDE_DIR_NAMES.intersection(relative_parts):
                files.append(path)
    return files


def package_project(output: Path, root: Path = REPO_ROOT) -> List[str]:
    """Writes the zip and returns the archived paths (relative, POSIX-style)."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    names = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in collect_files(root):
            name = path.relative_to(root).as_posix()
            archive.write(path, name)
            names.append(name)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Zip the project code for cloud training.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "dist" / "multitask_ast_code.zip",
        help="Output zip path (default: dist/multitask_ast_code.zip).",
    )
    args = parser.parse_args()
    names = package_project(args.output)
    size_kb = args.output.stat().st_size / 1024
    print(f"Wrote {len(names)} files ({size_kb:.0f} KB) to {args.output}")


if __name__ == "__main__":
    main()
