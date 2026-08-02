#!/usr/bin/env python3
"""Verify that two extracted V3 benchmarks differ only in noise-related synthesis fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


IGNORED_PROVENANCE_FIELDS = {
    "snr_db",
    "noise_gain",
    "limiter_gain",
    "pre_limit_peak",
    "output_peak",
    "output_sha256",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as source:
        return [json.loads(line) for line in source if line.strip()]


def provenance_core(row: dict) -> dict:
    return {key: value for key, value in row.items() if key not in IGNORED_PROVENANCE_FIELDS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minus10", required=True, type=Path)
    parser.add_argument("--minus20", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    pos10, pos20 = read_jsonl(args.minus10 / "pos.jsonl"), read_jsonl(args.minus20 / "pos.jsonl")
    neg10, neg20 = read_jsonl(args.minus10 / "neg.jsonl"), read_jsonl(args.minus20 / "neg.jsonl")
    prov10 = read_jsonl(args.minus10 / "provenance.jsonl")
    prov20 = read_jsonl(args.minus20 / "provenance.jsonl")
    errors: list[str] = []
    if pos10 != pos20:
        errors.append("POS labels differ")
    if neg10 != neg20:
        errors.append("NEG labels differ")
    if len(prov10) != len(prov20):
        errors.append("Provenance row counts differ")
    snr_differences: list[float] = []
    core_mismatch_count = 0
    for first, second in zip(prov10, prov20):
        if provenance_core(first) != provenance_core(second):
            core_mismatch_count += 1
        snr_differences.append(float(second["snr_db"]) - float(first["snr_db"]))
    if core_mismatch_count:
        errors.append(f"Non-noise provenance differs in {core_mismatch_count} rows")
    if any(abs(value - 10.0) > 1e-6 for value in snr_differences):
        errors.append("SNR difference is not exactly 10 dB for every row")
    report = {
        "valid": not errors,
        "labels_identical": pos10 == pos20 and neg10 == neg20,
        "pos_rows": len(pos10),
        "neg_rows": len(neg10),
        "provenance_rows": len(prov10),
        "core_mismatch_count": core_mismatch_count,
        "snr_difference_db": sorted(set(round(value, 6) for value in snr_differences)),
        "errors": errors,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
