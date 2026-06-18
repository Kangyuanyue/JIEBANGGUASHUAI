#!/usr/bin/env python3
"""Assemble split archive parts such as cn-celeb2_v2.tar.gzaa/gzab/gzac."""

from __future__ import annotations

import argparse
from pathlib import Path


def find_parts(prefix: Path) -> list[Path]:
    parent = prefix.parent
    stem = prefix.name
    parts = sorted(parent.glob(stem + "??"))
    return [p for p in parts if p.is_file()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble split .tar.gz parts")
    parser.add_argument(
        "--prefix",
        default="cn-celeb2_v2.tar.gz",
        help="Split prefix before aa/ab/ac suffix, e.g. cn-celeb2_v2.tar.gz",
    )
    parser.add_argument("--workdir", default=".", help="Directory containing split parts")
    parser.add_argument("--output", default="", help="Output archive path")
    parser.add_argument("--force", action="store_true", help="Overwrite output if it exists")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    prefix = workdir / args.prefix
    output = Path(args.output) if args.output else prefix

    parts = find_parts(prefix)
    if not parts:
        raise SystemExit(f"No split parts found for prefix: {prefix}")

    expected_first = prefix.with_name(prefix.name + "aa")
    if parts[0].name != expected_first.name:
        names = ", ".join(p.name for p in parts[:5])
        raise SystemExit(
            f"First part should be {expected_first.name}, but found {parts[0].name}. "
            f"Current parts: {names}. Download all parts before assembling."
        )

    if output.exists() and not args.force:
        raise SystemExit(f"Output already exists: {output}. Use --force to overwrite.")

    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(output, "wb") as out:
        for part in parts:
            print(f"Appending {part.name} ({part.stat().st_size / (1024 ** 3):.2f} GB)")
            with open(part, "rb") as src:
                while True:
                    chunk = src.read(1024 * 1024 * 16)
                    if not chunk:
                        break
                    out.write(chunk)
                    total += len(chunk)

    with open(output, "rb") as f:
        magic = f.read(2)
    if magic != b"\x1f\x8b":
        raise SystemExit(
            f"Assembled file does not start with gzip magic bytes. "
            f"Check whether parts are complete and ordered. Output: {output}"
        )

    print(f"Done: {output} ({total / (1024 ** 3):.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
