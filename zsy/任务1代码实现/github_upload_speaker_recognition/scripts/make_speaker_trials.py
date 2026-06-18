#!/usr/bin/env python3
"""Create same/different-speaker trials from an extracted speaker dataset."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}


def infer_speaker_id(path: Path, root: Path, mode: str) -> str:
    rel = path.relative_to(root)
    parts = rel.parts
    if mode == "parent":
        return path.parent.name
    if mode == "first":
        return parts[0]
    raise ValueError(f"Unknown speaker id mode: {mode}")


def audio_duration_sec(path: Path) -> float:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return 0.0


def scan_audio(root: Path, speaker_id_mode: str, max_speakers: int = 0) -> dict[str, list[Path]]:
    by_spk: dict[str, list[Path]] = defaultdict(list)
    if speaker_id_mode == "first":
        speaker_dirs = [p for p in root.iterdir() if p.is_dir()]
        speaker_dirs = sorted(speaker_dirs)
        if max_speakers > 0:
            speaker_dirs = speaker_dirs[:max_speakers]
        for spk_dir in speaker_dirs:
            for p in spk_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                    by_spk[spk_dir.name].append(p)
        return {spk: sorted(files) for spk, files in by_spk.items() if len(files) >= 2}

    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            spk = infer_speaker_id(p, root, speaker_id_mode)
            by_spk[spk].append(p)
    out = {spk: sorted(files) for spk, files in by_spk.items() if len(files) >= 2}
    if max_speakers > 0:
        out = {spk: out[spk] for spk in sorted(out)[:max_speakers]}
    return out


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def make_trials(
    by_spk: dict[str, list[Path]],
    root: Path,
    out_csv: Path,
    positive_per_speaker: int,
    negative_per_speaker: int,
    min_duration_sec: float,
    max_duration_sec: float,
    seed: int,
) -> None:
    rng = random.Random(seed)
    speakers = sorted(by_spk)
    if len(speakers) < 2:
        raise SystemExit("Need at least two speakers with two audio files each.")

    rows = []

    duration_cache: dict[Path, float] = {}

    def is_valid(path: Path) -> bool:
        if min_duration_sec <= 0 and max_duration_sec <= 0:
            return True
        if path not in duration_cache:
            duration_cache[path] = audio_duration_sec(path)
        duration = duration_cache[path]
        if min_duration_sec > 0 and duration < min_duration_sec:
            return False
        if max_duration_sec > 0 and duration > max_duration_sec:
            return False
        return True

    def pick_valid(files: list[Path]) -> Path | None:
        for _ in range(100):
            p = rng.choice(files)
            if is_valid(p):
                return p
        valid = [p for p in files if is_valid(p)]
        return rng.choice(valid) if valid else None

    for spk in speakers:
        files = by_spk[spk]
        for _ in range(positive_per_speaker):
            a = pick_valid(files)
            b = pick_valid(files)
            if a is None or b is None or a == b:
                continue
            rows.append((rel(a, root), rel(b, root), 1))

        other_speakers = [s for s in speakers if s != spk]
        for _ in range(negative_per_speaker):
            a = pick_valid(files)
            other = rng.choice(other_speakers)
            b = pick_valid(by_spk[other])
            if a is None or b is None:
                continue
            rows.append((rel(a, root), rel(b, root), 0))

    rng.shuffle(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["enroll_audio", "test_audio", "label"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} trials for {len(speakers)} speakers -> {out_csv}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build speaker verification trials")
    parser.add_argument("--audio-root", required=True, help="Extracted dataset audio root")
    parser.add_argument("--output", default="output/speaker_trials.csv")
    parser.add_argument("--speaker-id-mode", choices=["first", "parent"], default="first")
    parser.add_argument("--positive-per-speaker", type=int, default=2)
    parser.add_argument("--negative-per-speaker", type=int, default=2)
    parser.add_argument("--min-duration-sec", type=float, default=1.0)
    parser.add_argument("--max-duration-sec", type=float, default=20.0)
    parser.add_argument("--max-speakers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    root = Path(args.audio_root).resolve()
    by_spk = scan_audio(root, args.speaker_id_mode, args.max_speakers)
    print(f"Found {sum(len(v) for v in by_spk.values())} audio files from {len(by_spk)} speakers")
    make_trials(
        by_spk=by_spk,
        root=root,
        out_csv=Path(args.output),
        positive_per_speaker=args.positive_per_speaker,
        negative_per_speaker=args.negative_per_speaker,
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
