#!/usr/bin/env python3
"""Execute a fixed mixing plan and write aligned components plus metadata."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import yaml

from audio_mix import (
    active_crop,
    aligned_track,
    apply_peak_ceiling,
    db_ratio,
    load_audio,
    match_noise,
    measured_overlap_ratio,
    overlap_offset,
    scale_for_ratio,
    scale_to_rms_dbfs,
    write_wav,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_active(path: str, sample_rate: int, ffmpeg: str) -> np.ndarray:
    return active_crop(load_audio(Path(path), sample_rate, ffmpeg=ffmpeg), sample_rate)


def build_speech_mix(
    primary: np.ndarray,
    secondary: np.ndarray | None,
    noise: np.ndarray | None,
    ratio_db: float | None,
    snr_db: float | None,
    overlap_ratio: float,
    sample_rate: int,
    audio_config: dict,
    noise_start_fraction: float,
    alignment_mode: str = "overlap_ratio",
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
    primary = scale_to_rms_dbfs(primary, float(audio_config["target_rms_dbfs"]))
    pre_roll = float(audio_config.get("pre_roll_seconds", 0.25))
    trailing = float(audio_config.get("trailing_padding_seconds", 0.25))
    min_seconds = float(audio_config.get("min_output_seconds", 3.0))
    max_seconds = float(audio_config.get("max_output_seconds", 5.0))
    primary_offset = int(round(pre_roll * sample_rate))
    secondary_offset = None
    if secondary is not None:
        secondary = scale_for_ratio(primary, secondary, float(ratio_db if ratio_db is not None else 0.0))
        secondary_offset = overlap_offset(
            primary.size,
            secondary.size,
            overlap_ratio,
            primary_offset,
            sample_rate,
            max_seconds,
            trailing,
            alignment_mode=alignment_mode,
        )
    end = primary_offset + primary.size
    if secondary is not None and secondary_offset is not None:
        end = max(end, secondary_offset + secondary.size)
    duration_policy = str(audio_config.get("duration_policy", "bounded"))
    if duration_policy == "longest_track":
        total_length = end + int(round(trailing * sample_rate))
    elif duration_policy == "bounded":
        total_length = max(int(round(min_seconds * sample_rate)), end + int(round(trailing * sample_rate)))
        total_length = min(total_length, int(round(max_seconds * sample_rate)))
    else:
        raise ValueError(f"Unsupported duration policy: {duration_policy}")
    primary_track = aligned_track(primary, primary_offset, total_length)
    secondary_track = (
        aligned_track(secondary, secondary_offset, total_length)
        if secondary is not None and secondary_offset is not None
        else None
    )
    noise_track = None
    if noise is not None and snr_db is not None:
        noise_track = match_noise(noise, total_length, noise_start_fraction)
        noise_track = scale_for_ratio(primary_track, noise_track, float(snr_db))
    tracks = [primary_track]
    if secondary_track is not None:
        tracks.append(secondary_track)
    if noise_track is not None:
        tracks.append(noise_track)
    scaled_tracks, peak_scale = apply_peak_ceiling(tracks, float(audio_config["peak_ceiling_dbfs"]))
    primary_track = scaled_tracks[0]
    cursor = 1
    if secondary_track is not None:
        secondary_track = scaled_tracks[cursor]
        cursor += 1
    if noise_track is not None:
        noise_track = scaled_tracks[cursor]
    mixture_parts = [primary_track]
    if secondary_track is not None:
        mixture_parts.append(secondary_track)
    if noise_track is not None:
        mixture_parts.append(noise_track)
    mixture = np.sum(np.stack(mixture_parts), axis=0, dtype=np.float64).astype(np.float32)
    measurements = {
        "peak_scale": peak_scale,
        "measured_speech_ratio_db": db_ratio(primary_track, secondary_track) if secondary_track is not None else None,
        "measured_snr_db": db_ratio(primary_track, noise_track) if noise_track is not None else None,
        "measured_overlap_ratio": measured_overlap_ratio(primary_track, secondary_track)
        if secondary_track is not None
        else None,
        "primary_offset_samples": primary_offset,
        "secondary_offset_samples": secondary_offset,
        "alignment_mode": alignment_mode,
        "duration_policy": duration_policy,
        "duration_seconds": total_length / sample_rate,
    }
    return mixture, primary_track, secondary_track, noise_track, measurements


def synthesize_sample(row: dict, config: dict, out_root: Path, ffmpeg: str) -> dict:
    sample_rate = int(config["audio"]["sample_rate"])
    sample_dir = out_root / row["sample_id"]
    if sample_dir.exists():
        raise FileExistsError(f"Sample output already exists: {sample_dir}")
    sample_dir.mkdir(parents=True)
    rng = random.Random(int(row["sample_seed"]))

    reference = load_active(row["reference_kws_source"], sample_rate, ffmpeg)
    reference = scale_to_rms_dbfs(reference, float(config["audio"]["target_rms_dbfs"]))
    write_wav(sample_dir / "reference_kws.wav", reference, sample_rate)

    command_sources = [load_active(path, sample_rate, ffmpeg) for path in row["command_sources"]]
    if not command_sources:
        raise ValueError(f"No command sources for {row['sample_id']}")
    primary = command_sources[0]
    secondary = command_sources[1] if len(command_sources) > 1 else None
    if row["sample_type"] == "POS" and row["interferer_command_sources"]:
        secondary = load_active(row["interferer_command_sources"][0], sample_rate, ffmpeg)
    noise = load_audio(Path(row["noise_source"]), sample_rate, ffmpeg) if row.get("noise_source") else None
    command_mix, primary_track, secondary_track, noise_track, command_metrics = build_speech_mix(
        primary=primary,
        secondary=secondary,
        noise=noise,
        ratio_db=row.get("sir_db"),
        snr_db=row.get("snr_db"),
        overlap_ratio=float(row["overlap_ratio"]),
        sample_rate=sample_rate,
        audio_config=config["audio"],
        noise_start_fraction=rng.random(),
        alignment_mode=str(row.get("alignment_mode", "overlap_ratio")),
    )
    write_wav(sample_dir / "command_mixture.wav", command_mix, sample_rate)
    target_clean = primary_track if row["sample_type"] == "POS" else np.zeros_like(primary_track)
    write_wav(sample_dir / "target_clean.wav", target_clean, sample_rate)
    if row["sample_type"] == "POS" and secondary_track is not None:
        write_wav(sample_dir / "interferer_01.wav", secondary_track, sample_rate)
    elif row["sample_type"] == "NEG":
        write_wav(sample_dir / "interferer_01.wav", primary_track, sample_rate)
        if secondary_track is None:
            raise ValueError(f"NEG sample lacks the second non-wake speaker: {row['sample_id']}")
        write_wav(sample_dir / "interferer_02.wav", secondary_track, sample_rate)
    if noise_track is not None:
        write_wav(sample_dir / "noise.wav", noise_track, sample_rate)

    wake_secondary = (
        load_active(row["wake_interferer_source"], sample_rate, ffmpeg)
        if row.get("wake_interferer_source")
        else None
    )
    wake_mix, _, _, _, wake_metrics = build_speech_mix(
        primary=reference,
        secondary=wake_secondary,
        noise=noise,
        ratio_db=row.get("sir_db"),
        snr_db=row.get("snr_db"),
        overlap_ratio=float(row["overlap_ratio"]),
        sample_rate=sample_rate,
        audio_config=config["audio"],
        noise_start_fraction=rng.random(),
        alignment_mode=str(row.get("alignment_mode", "overlap_ratio")),
    )
    write_wav(sample_dir / "wake_mixture.wav", wake_mix, sample_rate)

    metadata = {
        **row,
        "sample_rate": sample_rate,
        "channels": 1,
        "output_dir": str(sample_dir.resolve()),
        "outputs": {
            "reference_kws": str((sample_dir / "reference_kws.wav").resolve()),
            "wake_mixture": str((sample_dir / "wake_mixture.wav").resolve()),
            "command_mixture": str((sample_dir / "command_mixture.wav").resolve()),
            "target_clean": str((sample_dir / "target_clean.wav").resolve()),
        },
        "command_measurements": command_metrics,
        "wake_measurements": wake_metrics,
    }
    (sample_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    global_manifest = args.out / "metadata.jsonl"
    if args.out.exists() or global_manifest.exists():
        raise FileExistsError("Output directory already exists; choose a new directory")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    plan_rows = read_jsonl(args.plan)
    if any(bool(row.get("reverb", False)) for row in plan_rows):
        raise ValueError("Reverb is not implemented in the stage-2 baseline")
    args.out.mkdir(parents=True)
    metadata_rows = [synthesize_sample(row, config, args.out, args.ffmpeg) for row in plan_rows]
    with global_manifest.open("w", encoding="utf-8", newline="\n") as output:
        for row in metadata_rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "plan": str(args.plan.resolve()),
        "config": str(args.config.resolve()),
        "sample_count": len(metadata_rows),
        "positive_count": sum(row["sample_type"] == "POS" for row in metadata_rows),
        "negative_count": sum(row["sample_type"] == "NEG" for row in metadata_rows),
        "metadata": str(global_manifest.resolve()),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
