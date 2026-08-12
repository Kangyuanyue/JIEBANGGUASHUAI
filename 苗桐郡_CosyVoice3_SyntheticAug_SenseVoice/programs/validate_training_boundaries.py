#!/usr/bin/env python3
"""Reject training manifests that reference competition evaluation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


DENIED_TOKENS = (
    "dataseta",
    "three_stream_dataset",
    "v3_same_start",
    "dataset_v1_noise",
    "/data/benchmarks/v3",
    "\\data\\benchmarks\\v3",
)


def iter_strings(value: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield prefix, value
    elif isinstance(value, dict):
        for key, nested in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_strings(nested, child)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            yield from iter_strings(nested, f"{prefix}[{index}]")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            value["__line_number"] = line_number
            rows.append(value)
    return rows


def validate_manifest(path: Path, require_audio: bool) -> dict[str, Any]:
    rows = read_jsonl(path)
    violations: list[dict[str, Any]] = []
    missing_audio: list[dict[str, Any]] = []
    invalid_training_flags: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    duplicate_uids: list[str] = []

    for row in rows:
        line_number = int(row.pop("__line_number"))
        uid = str(row.get("uid", "")).strip()
        if not uid:
            violations.append({"line": line_number, "field": "uid", "reason": "missing uid"})
        elif uid in seen_uids:
            duplicate_uids.append(uid)
        seen_uids.add(uid)

        for field, value in iter_strings(row):
            normalized = value.casefold().replace("\\", "/")
            for token in DENIED_TOKENS:
                if token.replace("\\", "/") in normalized:
                    violations.append(
                        {"line": line_number, "uid": uid, "field": field, "token": token, "value": value}
                    )

        if row.get("allowed_for_training") is not True:
            invalid_training_flags.append({"line": line_number, "uid": uid})
        if require_audio:
            audio = Path(str(row.get("audio_path", "")))
            if not audio.is_file():
                missing_audio.append({"line": line_number, "uid": uid, "audio_path": str(audio)})

    passed = not violations and not missing_audio and not invalid_training_flags and not duplicate_uids
    return {
        "manifest": str(path.resolve()),
        "record_count": len(rows),
        "passed": passed,
        "denied_tokens": list(DENIED_TOKENS),
        "violations": violations,
        "missing_audio": missing_audio,
        "invalid_training_flags": invalid_training_flags,
        "duplicate_uids": sorted(set(duplicate_uids)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, action="append", type=Path)
    parser.add_argument("--require-audio", action="store_true")
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    results = []
    for value in args.manifest:
        path = (value if value.is_absolute() else project / value).resolve()
        result = validate_manifest(path, args.require_audio)
        if args.expected_count is not None and result["record_count"] != args.expected_count:
            result["passed"] = False
            result["expected_count"] = args.expected_count
            result["count_error"] = f"expected {args.expected_count}, found {result['record_count']}"
        results.append(result)

    report = {"passed": all(item["passed"] for item in results), "manifests": results}
    if args.out:
        out = (args.out if args.out.is_absolute() else project / args.out).resolve()
        if out.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {out}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out}")
    print(json.dumps({"passed": report["passed"], "manifest_count": len(results)}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
