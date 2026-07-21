#!/usr/bin/env python3
"""Download the official Pos/Neg TSE source and baseline checkpoint."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


SOURCE_URL = "https://github.com/xu-shitong/TSE-through-Positive-Negative-Enroll/archive/refs/heads/main.zip"
CHECKPOINT_URL = "https://huggingface.co/ShitongXu/TSE-Pos-Neg-Enroll/resolve/main/improved-monaural.pt"


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        default="external_TSE_PosNeg/TSE-through-Positive-Negative-Enroll-main",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    destination = Path(args.destination)
    checkpoint = destination / "improved-monaural.pt"
    if destination.is_dir() and checkpoint.is_file() and not args.force:
        print(f"TSE source and checkpoint already exist: {destination.resolve()}")
        return 0

    with tempfile.TemporaryDirectory() as tmp_text:
        tmp = Path(tmp_text)
        archive = tmp / "source.zip"
        download(SOURCE_URL, archive)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(tmp / "source")
        extracted = next((tmp / "source").iterdir())
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(extracted, destination)

    download(CHECKPOINT_URL, checkpoint)
    print(f"Ready: {destination.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
