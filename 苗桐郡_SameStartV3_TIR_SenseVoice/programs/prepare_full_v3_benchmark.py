#!/usr/bin/env python3
"""Safely extract and independently validate one complete V3 benchmark archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as source:
        return [json.loads(line) for line in source if line.strip()]


def safe_relative(name: str, root: str) -> Path:
    prefix = root + "/"
    if not name.startswith(prefix):
        raise ValueError(f"Entry is outside archive root {root}: {name}")
    relative = PurePosixPath(name[len(prefix) :])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe archive entry: {name}")
    return Path(*relative.parts)


def inspect_wav(path: Path) -> dict:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_rate = source.getframerate()
        sample_width = source.getsampwidth()
        frames = source.getnframes()
        audio = np.frombuffer(source.readframes(frames), dtype="<i2")
    peak = float(np.max(np.abs(audio.astype(np.float32))) / 32768.0) if audio.size else 0.0
    return {
        "channels": channels,
        "sample_rate": sample_rate,
        "sample_width": sample_width,
        "frames": frames,
        "duration_seconds": frames / sample_rate,
        "peak": peak,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    with zipfile.ZipFile(args.archive) as zf:
        file_entries = [entry for entry in zf.infolist() if not entry.is_dir()]
        roots = {entry.filename.split("/", 1)[0] for entry in file_entries if "/" in entry.filename}
        if len(roots) != 1:
            raise ValueError(f"Expected one archive root, found {sorted(roots)}")
        root = next(iter(roots))
        args.out.mkdir(parents=True)
        for entry in file_entries:
            relative = safe_relative(entry.filename, root)
            destination = args.out / relative
            if destination.exists():
                raise FileExistsError(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(zf.read(entry))

    pos_rows = read_jsonl(args.out / "pos.jsonl")
    neg_rows = read_jsonl(args.out / "neg.jsonl")
    provenance_rows = read_jsonl(args.out / "provenance.jsonl")
    expected_paths: set[str] = set()
    infer_rows: list[dict] = []
    for split, rows in (("pos", pos_rows), ("neg", neg_rows)):
        for index, row in enumerate(rows, start=1):
            expected_paths.update({row["唤醒音频"], row["识别音频"]})
            infer_rows.append(
                {
                    "uid": f"{args.variant}_{split}_{index:04d}",
                    "id": row.get("id"),
                    "jsonl_row": index,
                    "variant": args.variant,
                    "audio_kws": str((args.out / row["唤醒音频"]).resolve()),
                    "audio_cmd": str((args.out / row["识别音频"]).resolve()),
                    "wake_text": row.get("唤醒文本") or "",
                    "label": row.get("识别文本") or "",
                    "is_positive": split == "pos",
                    "split": split,
                }
            )
    actual_wavs = sorted(path for path in args.out.rglob("*.wav") if path.is_file())
    actual_relative = {path.relative_to(args.out).as_posix() for path in actual_wavs}
    missing = sorted(expected_paths - actual_relative)
    unexpected = sorted(actual_relative - expected_paths)
    format_counts: dict[str, int] = {}
    durations: list[float] = []
    peaks: list[float] = []
    invalid_audio: list[dict] = []
    for path in actual_wavs:
        try:
            info = inspect_wav(path)
            key = f"{info['sample_width'] * 8}bit|{info['sample_rate']}Hz|{info['channels']}ch"
            format_counts[key] = format_counts.get(key, 0) + 1
            durations.append(info["duration_seconds"])
            peaks.append(info["peak"])
            if (
                info["sample_width"] != 2
                or info["sample_rate"] != 16000
                or info["channels"] != 1
                or info["frames"] <= 0
                or info["peak"] > 0.892
            ):
                invalid_audio.append({"path": str(path), **info})
        except Exception as exc:
            invalid_audio.append({"path": str(path), "error": repr(exc)})
    infer_manifest = args.out / "infer_manifest.jsonl"
    with infer_manifest.open("w", encoding="utf-8", newline="\n") as output:
        for row in infer_rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "valid": not missing and not unexpected and not invalid_audio,
        "variant": args.variant,
        "archive": str(args.archive.resolve()),
        "output_root": str(args.out.resolve()),
        "pos_rows": len(pos_rows),
        "neg_rows": len(neg_rows),
        "expected_audio_paths": len(expected_paths),
        "actual_wav_files": len(actual_wavs),
        "provenance_rows": len(provenance_rows),
        "infer_manifest_rows": len(infer_rows),
        "format_counts": format_counts,
        "duration_seconds": {
            "min": min(durations),
            "mean": float(np.mean(durations)),
            "max": max(durations),
        },
        "peak": {"min": min(peaks), "max": max(peaks)},
        "label_hashes": {
            "pos_jsonl_sha256": sha256(args.out / "pos.jsonl"),
            "neg_jsonl_sha256": sha256(args.out / "neg.jsonl"),
        },
        "missing": missing,
        "unexpected": unexpected,
        "invalid_audio": invalid_audio,
    }
    (args.out / "independent_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
