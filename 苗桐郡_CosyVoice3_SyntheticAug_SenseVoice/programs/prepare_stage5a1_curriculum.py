from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf


TEST_MARKERS = ("dataseta", "dataset_a", "v3_same_start", "three_stream_dataset_v3")
TAG_RE = re.compile(r"<\|[^|]+?\|>")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_rank(rows: Iterable[dict[str, Any]], seed: int, salt: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: hashlib.sha256(f"{seed}:{salt}:{row['uid']}".encode()).hexdigest())


def deterministic_index(seed: int, uid: str, modulo: int, salt: str) -> int:
    digest = hashlib.sha256(f"{seed}:{salt}:{uid}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def normalized_length(text: str) -> int:
    text = TAG_RE.sub("", text)
    return max(1, sum(not char.isspace() and not unicodedata.category(char).startswith("P") for char in text))


def to_funasr(row: dict[str, Any]) -> dict[str, Any]:
    info = sf.info(row["audio_path"])
    return {
        "key": row["uid"],
        "prompt": "<|ASR|>",
        "source": row["audio_path"],
        "source_len": max(1, int(round(info.duration * 100))),
        "target": row["text"],
        "target_len": normalized_length(row["text"]),
        "text_language": "<|zh|>",
        "with_or_wo_itn": "<|woitn|>",
    }


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64) + 1e-12))


def scale_to_rms(audio: np.ndarray, target_rms: float) -> np.ndarray:
    return audio * (target_rms / max(rms(audio), 1e-8))


def loop_to_length(audio: np.ndarray, length: int, start: int) -> np.ndarray:
    repeats = int(math.ceil((start + length) / max(1, len(audio))))
    return np.tile(audio, repeats)[start : start + length]


def mix_with_noise(
    source_path: Path,
    noise_arrays: list[np.ndarray],
    destination: Path,
    uid: str,
    snr_db: float,
    seed: int,
    sample_rate: int,
    target_rms_dbfs: float,
    peak_limit: float,
) -> None:
    target, sr = sf.read(source_path, dtype="float32")
    if target.ndim > 1:
        target = target.mean(axis=1)
    if sr != sample_rate:
        raise ValueError(f"Unexpected sample rate {sr}: {source_path}")
    target = scale_to_rms(target, 10 ** (target_rms_dbfs / 20))
    noise_index = deterministic_index(seed, uid, len(noise_arrays), "stage5a1_noise")
    noise = noise_arrays[noise_index]
    start = deterministic_index(seed, uid, max(1, len(noise)), "stage5a1_start")
    noise = loop_to_length(noise, len(target), start)
    noise = scale_to_rms(noise, rms(target) / (10 ** (snr_db / 20)))
    mixture = target + noise
    gain = min(1.0, peak_limit / max(float(np.max(np.abs(mixture))), 1e-8))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, (mixture * gain).astype(np.float32), sample_rate, subtype="PCM_16")


def assert_boundary(rows: Iterable[dict[str, Any]]) -> None:
    for row in rows:
        searchable = json.dumps(row, ensure_ascii=False).lower().replace("-", "_")
        if any(marker in searchable for marker in TEST_MARKERS):
            raise ValueError(f"Official test marker detected in {row.get('uid')}")


def curriculum_rows(
    route: str,
    phase: int,
    snr_db: float,
    real_rows: list[dict[str, Any]],
    augmentation_rows: list[dict[str, Any]],
    noise_arrays: list[np.ndarray],
    output_dir: Path,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for row in real_rows:
        rows.append({
            **row,
            "uid": f"stage5a1_{route}_p{phase}_clean_{row['uid']}",
            "curriculum_phase": phase,
            "snr_db": None,
            "augmentation": "clean_anchor",
            "route": route,
        })
    for row in augmentation_rows:
        source_path = Path(row["audio_path"]).resolve()
        uid = f"stage5a1_{route}_p{phase}_noise_{row['uid']}"
        destination = output_dir / "audio" / route / f"phase_{phase}" / f"{uid}.wav"
        mix_with_noise(
            source_path,
            noise_arrays,
            destination,
            uid,
            snr_db,
            int(config["seed"]),
            int(config["sample_rate"]),
            float(config["target_rms_dbfs"]),
            float(config["peak_limit"]),
        )
        rows.append({
            "uid": uid,
            "audio_path": str(destination.resolve()),
            "text": row["text"],
            "split": "train",
            "source": row["source"],
            "source_uid": row["uid"],
            "speaker": row["speaker"],
            "curriculum_phase": phase,
            "snr_db": snr_db,
            "augmentation": "verified_noise",
            "route": route,
            "contains_interfering_speech": False,
            "contains_official_test_data": False,
        })
    return stable_rank(rows, int(config["seed"]), f"{route}_phase_{phase}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output_dir = Path(args.out).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty Stage 5A.1 directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    real_rows = read_jsonl(Path(args.real_manifest))
    if len(real_rows) != int(config["curriculum_samples_per_source"]):
        raise ValueError(f"Expected {config['curriculum_samples_per_source']} real rows, found {len(real_rows)}")
    synthetic_pool = [
        row for row in read_jsonl(Path(args.synthetic_manifest))
        if row["split"] == "train" and row["source"] == "rule_generated"
    ]
    synthetic_rows = stable_rank(synthetic_pool, int(config["seed"]), "stage5a1_synthetic")[: len(real_rows)]
    if len(synthetic_rows) != len(real_rows):
        raise ValueError("Not enough safe synthetic training rows")

    noise_arrays = []
    for path in sorted(Path(args.noise_wav_dir).glob("*.wav")):
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != int(config["sample_rate"]):
            raise ValueError(f"Invalid noise sample rate: {path}")
        noise_arrays.append(audio)
    if len(noise_arrays) != 20:
        raise ValueError(f"Expected 20 verified noise WAVs, found {len(noise_arrays)}")

    real_control = []
    augmented = []
    phase_counts = {}
    for phase, snr_db in enumerate(config["curriculum_snr_db"], start=1):
        control_phase = curriculum_rows("real_control", phase, float(snr_db), real_rows, real_rows, noise_arrays, output_dir, config)
        augmented_phase = curriculum_rows("synthetic_augmented", phase, float(snr_db), real_rows, synthetic_rows, noise_arrays, output_dir, config)
        real_control.extend(control_phase)
        augmented.extend(augmented_phase)
        phase_counts[str(phase)] = {
            "snr_db": snr_db,
            "real_control": len(control_phase),
            "synthetic_augmented": len(augmented_phase),
        }

    assert_boundary([*real_control, *augmented])
    write_jsonl(output_dir / "real_control_provenance.jsonl", real_control)
    write_jsonl(output_dir / "synthetic_augmented_provenance.jsonl", augmented)
    write_jsonl(output_dir / "real_control_train.jsonl", map(to_funasr, real_control))
    write_jsonl(output_dir / "synthetic_augmented_train.jsonl", map(to_funasr, augmented))

    summary = {
        "seed": config["seed"],
        "curriculum_snr_db": config["curriculum_snr_db"],
        "samples_per_phase_per_route": len(real_rows) * 2,
        "real_control_total": len(real_control),
        "synthetic_augmented_total": len(augmented),
        "phase_counts": phase_counts,
        "synthetic_speaker_counts": dict(Counter(row["speaker"] for row in synthetic_rows)),
        "official_test_data_violations": 0,
        "interfering_speech_violations": 0,
        "sample_rate": config["sample_rate"],
        "channels": 1,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build equal-size Stage 5A.1 noise curricula.")
    parser.add_argument("--real-manifest", required=True)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--noise-wav-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
