from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


OFFICIAL_TEST_MARKERS = (
    "dataseta",
    "dataset_a",
    "v3_same_start",
    "three_stream_dataset_v3",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64) + 1e-12))


def scale_to_rms(audio: np.ndarray, desired_rms: float) -> np.ndarray:
    return audio * (desired_rms / max(rms(audio), 1e-8))


def pad_or_crop(audio: np.ndarray, length: int, start: int = 0) -> np.ndarray:
    if len(audio) >= start + length:
        return audio[start : start + length]
    if len(audio) == 0:
        return np.zeros(length, dtype=np.float32)
    needed = start + length
    repeats = int(math.ceil(needed / len(audio)))
    return np.tile(audio, repeats)[start : start + length]


def load_audio(path: Path, sample_rate: int, ffmpeg: str = "ffmpeg") -> np.ndarray:
    source = path
    temp_path: Path | None = None
    try:
        if path.suffix.lower() not in {".wav", ".flac", ".ogg"}:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
                temp_path = Path(temp.name)
            subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(path), "-ac", "1", "-ar", str(sample_rate), str(temp_path)],
                check=True,
            )
            source = temp_path
        audio, sr = sf.read(source, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        if sr != sample_rate:
            divisor = math.gcd(sr, sample_rate)
            mono = resample_poly(mono, sample_rate // divisor, sr // divisor).astype(np.float32)
        return mono.astype(np.float32)
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def deterministic_index(seed: int, uid: str, modulo: int, salt: str) -> int:
    digest = hashlib.sha256(f"{seed}:{salt}:{uid}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def choose_interferer(
    target: dict[str, Any], by_speaker: dict[str, list[dict[str, Any]]], seed: int, salt: str
) -> dict[str, Any]:
    speakers = sorted(speaker for speaker in by_speaker if speaker != target["speaker"])
    speaker = speakers[deterministic_index(seed, target["uid"], len(speakers), f"{salt}:speaker")]
    candidates = [row for row in by_speaker[speaker] if row["text_uid"] != target["text_uid"]]
    return candidates[deterministic_index(seed, target["uid"], len(candidates), f"{salt}:utterance")]


def mix_pos(
    target: np.ndarray,
    interferer: np.ndarray,
    noise: np.ndarray,
    tir_db: float,
    snr_db: float,
    target_rms_dbfs: float,
    peak_limit: float,
    noise_start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    length = max(len(target), len(interferer))
    target_aligned = pad_or_crop(target, length)
    interferer_aligned = pad_or_crop(interferer, length)
    desired_target_rms = 10 ** (target_rms_dbfs / 20)
    target_scaled = scale_to_rms(target_aligned, desired_target_rms)
    desired_interferer_rms = desired_target_rms / (10 ** (tir_db / 20))
    interferer_scaled = scale_to_rms(interferer_aligned, desired_interferer_rms)
    speech = target_scaled + interferer_scaled
    noise_aligned = pad_or_crop(noise, length, noise_start)
    desired_noise_rms = rms(speech) / (10 ** (snr_db / 20))
    noise_scaled = scale_to_rms(noise_aligned, desired_noise_rms)
    mixture = speech + noise_scaled
    gain = min(1.0, peak_limit / max(float(np.max(np.abs(mixture))), 1e-8))
    return (
        (mixture * gain).astype(np.float32),
        (target_scaled * gain).astype(np.float32),
        (noise_scaled * gain).astype(np.float32),
        {
            "master_gain": gain,
            "measured_tir_db": 20 * math.log10(rms(target_scaled) / rms(interferer_scaled)),
            "measured_snr_db": 20 * math.log10(rms(speech) / rms(noise_scaled)),
        },
    )


def mix_neg(
    interferer: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    speech_rms_dbfs: float,
    peak_limit: float,
    noise_start: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    length = len(interferer)
    speech = scale_to_rms(interferer, 10 ** (speech_rms_dbfs / 20))
    noise_aligned = pad_or_crop(noise, length, noise_start)
    noise_scaled = scale_to_rms(noise_aligned, rms(speech) / (10 ** (snr_db / 20)))
    mixture = speech + noise_scaled
    gain = min(1.0, peak_limit / max(float(np.max(np.abs(mixture))), 1e-8))
    return (
        (mixture * gain).astype(np.float32),
        np.zeros(length, dtype=np.float32),
        {"master_gain": gain, "measured_snr_db": 20 * math.log10(rms(speech) / rms(noise_scaled))},
    )


def save_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate, subtype="PCM_16")


def ensure_clean_output(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    contents = list(output_dir.iterdir())
    if contents:
        raise FileExistsError(f"Output directory is not empty; refusing to overwrite: {output_dir}")


def assert_training_boundary(rows: list[dict[str, Any]]) -> None:
    violations = []
    for row in rows:
        searchable = json.dumps(row, ensure_ascii=False).lower().replace("-", "_")
        if any(marker in searchable for marker in OFFICIAL_TEST_MARKERS):
            violations.append(row.get("uid", "unknown"))
    if violations:
        raise ValueError(f"Official test data found in training sources: {violations[:5]}")


def relative_to_project(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def run(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    accepted = sorted(read_jsonl(Path(args.manifest)), key=lambda row: row["uid"])
    tts_rows = {row["uid"]: row for row in read_jsonl(Path(args.tts_manifest))}
    assert_training_boundary(accepted)

    output_dir = Path(args.out).resolve()
    ensure_clean_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = project_root / "data" / "manifests"
    sample_rate = int(config["sample_rate"])
    seed = int(config["seed"])
    tir_values = list(config["tir_db"])
    snr_values = list(config["snr_db"])
    target_rms_dbfs = float(config["target_rms_dbfs"])
    peak_limit = float(config["peak_limit"])

    noise_dir = Path(args.noise_dir or config["noise_directory"])
    noise_paths = sorted(path for path in noise_dir.iterdir() if path.is_file() and path.suffix.lower() in {".m4a", ".mp3", ".wav", ".flac"})
    if len(noise_paths) != 20:
        raise ValueError(f"Expected exactly 20 verified no-human-speech noise files, found {len(noise_paths)}")
    noise_cache = {str(path): load_audio(path, sample_rate, args.ffmpeg) for path in noise_paths}
    if any(len(audio) < sample_rate for audio in noise_cache.values()):
        raise ValueError("Every noise recording must be at least one second")

    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        by_speaker[row["speaker"]].append(row)
    if len(by_speaker) < 2:
        raise ValueError("At least two speakers are required for overlap mixtures")

    audio_cache: dict[str, np.ndarray] = {}
    reference_cache: dict[str, np.ndarray] = {}

    def row_audio(row: dict[str, Any]) -> np.ndarray:
        path = str(row["audio_path"])
        if path not in audio_cache:
            audio_cache[path] = load_audio(Path(path), sample_rate, args.ffmpeg)
        return audio_cache[path]

    def reference_audio(row: dict[str, Any]) -> tuple[np.ndarray, str]:
        prompt_path = tts_rows[row["uid"]]["prompt_audio_path"]
        if prompt_path not in reference_cache:
            audio = load_audio(Path(prompt_path), sample_rate, args.ffmpeg)
            reference_cache[prompt_path] = scale_to_rms(audio, 10 ** (-22 / 20)).astype(np.float32)
        return reference_cache[prompt_path], prompt_path

    pos_rows: list[dict[str, Any]] = []
    asr_rows: list[dict[str, Any]] = []
    for index, target_row in enumerate(accepted):
        interferer_row = choose_interferer(target_row, by_speaker, seed, "pos")
        noise_path = noise_paths[deterministic_index(seed, target_row["uid"], len(noise_paths), "pos:noise")]
        noise = noise_cache[str(noise_path)]
        target = row_audio(target_row)
        interferer = row_audio(interferer_row)
        tir_db = float(tir_values[index % len(tir_values)])
        snr_db = float(snr_values[index % len(snr_values)])
        length = max(len(target), len(interferer))
        max_start = max(1, len(noise) - min(len(noise), length) + 1)
        noise_start = deterministic_index(seed, target_row["uid"], max_start, "pos:noise_start")
        mixture, target_clean, noise_component, measured = mix_pos(
            target, interferer, noise, tir_db, snr_db, target_rms_dbfs, peak_limit, noise_start
        )
        reference, reference_source = reference_audio(target_row)
        sample_dir = output_dir / "pos" / target_row["uid"]
        mixture_path = sample_dir / "mixture.wav"
        target_path = sample_dir / "target_clean.wav"
        reference_path = sample_dir / "reference.wav"
        noise_aug_path = sample_dir / "noise_augmented.wav"
        save_audio(mixture_path, mixture, sample_rate)
        save_audio(target_path, target_clean, sample_rate)
        save_audio(reference_path, reference, sample_rate)
        # ASR augmentation excludes the interfering speaker by construction.
        target_only = pad_or_crop(target, len(target))
        target_only = scale_to_rms(target_only, 10 ** (target_rms_dbfs / 20))
        noise_only = pad_or_crop(noise, len(target), noise_start)
        noise_only = scale_to_rms(noise_only, rms(target_only) / (10 ** (snr_db / 20)))
        noise_aug = target_only + noise_only
        aug_gain = min(1.0, peak_limit / max(float(np.max(np.abs(noise_aug))), 1e-8))
        save_audio(noise_aug_path, (noise_aug * aug_gain).astype(np.float32), sample_rate)

        record = {
            "uid": f"stage4_pos_{index:04d}",
            "is_positive": True,
            "label": target_row["text"],
            "speaker": target_row["speaker"],
            "interferer_speaker": interferer_row["speaker"],
            "target_source_uid": target_row["uid"],
            "interferer_source_uid": interferer_row["uid"],
            "noise_source": str(noise_path),
            "reference_source": reference_source,
            "mixture_path": relative_to_project(mixture_path, project_root),
            "target_clean_path": relative_to_project(target_path, project_root),
            "reference_path": relative_to_project(reference_path, project_root),
            "noise_augmented_path": relative_to_project(noise_aug_path, project_root),
            "split": target_row["split"],
            "tir_db": tir_db,
            "snr_db": snr_db,
            "same_start": True,
            "sample_rate": sample_rate,
            "duration_seconds": round(len(mixture) / sample_rate, 4),
            **{key: round(value, 6) for key, value in measured.items()},
            "allowed_for_asr_training": False,
            "allowed_for_tse_training": True,
            "contains_official_test_data": False,
        }
        pos_rows.append(record)
        for variant, audio_path in (("clean", target_path), ("noise_only_augmented", noise_aug_path)):
            asr_rows.append({
                "uid": f"{record['uid']}__{variant}",
                "audio_path": relative_to_project(audio_path, project_root),
                "text": target_row["text"],
                "speaker": target_row["speaker"],
                "split": target_row["split"],
                "augmentation": variant,
                "allowed_for_asr_training": True,
                "contains_interfering_speech": False,
                "contains_official_test_data": False,
            })

    rng = random.Random(seed)
    target_cycle = [accepted[i % len(accepted)] for i in range(int(config["neg_count"]))]
    rng.shuffle(target_cycle)
    neg_rows: list[dict[str, Any]] = []
    for index, nominal_target in enumerate(target_cycle):
        interferer_row = choose_interferer(nominal_target, by_speaker, seed, f"neg:{index}")
        noise_path = noise_paths[deterministic_index(seed, nominal_target["uid"], len(noise_paths), f"neg:{index}:noise")]
        noise = noise_cache[str(noise_path)]
        interferer = row_audio(interferer_row)
        snr_db = float(snr_values[index % len(snr_values)])
        max_start = max(1, len(noise) - min(len(noise), len(interferer)) + 1)
        noise_start = deterministic_index(seed, nominal_target["uid"], max_start, f"neg:{index}:noise_start")
        mixture, silent_target, measured = mix_neg(
            interferer, noise, snr_db, target_rms_dbfs, peak_limit, noise_start
        )
        reference, reference_source = reference_audio(nominal_target)
        uid = f"stage4_neg_{index:04d}"
        sample_dir = output_dir / "neg" / uid
        mixture_path = sample_dir / "mixture.wav"
        target_path = sample_dir / "target_clean.wav"
        reference_path = sample_dir / "reference.wav"
        save_audio(mixture_path, mixture, sample_rate)
        save_audio(target_path, silent_target, sample_rate)
        save_audio(reference_path, reference, sample_rate)
        neg_rows.append({
            "uid": uid,
            "is_positive": False,
            "label": "",
            "nominal_target_speaker": nominal_target["speaker"],
            "interferer_speaker": interferer_row["speaker"],
            "interferer_source_uid": interferer_row["uid"],
            "noise_source": str(noise_path),
            "reference_source": reference_source,
            "mixture_path": relative_to_project(mixture_path, project_root),
            "target_clean_path": relative_to_project(target_path, project_root),
            "reference_path": relative_to_project(reference_path, project_root),
            "split": nominal_target["split"],
            "tir_db": None,
            "snr_db": snr_db,
            "same_start": True,
            "sample_rate": sample_rate,
            "duration_seconds": round(len(mixture) / sample_rate, 4),
            **{key: round(value, 6) for key, value in measured.items()},
            "allowed_for_asr_training": False,
            "allowed_for_tse_training": True,
            "contains_target_speaker": False,
            "contains_official_test_data": False,
        })

    write_jsonl(manifest_dir / "stage4_pos.jsonl", pos_rows)
    write_jsonl(manifest_dir / "stage4_neg.jsonl", neg_rows)
    write_jsonl(manifest_dir / "stage4_tse.jsonl", [*pos_rows, *neg_rows])
    write_jsonl(manifest_dir / "stage4_asr_augmented.jsonl", asr_rows)

    summary = {
        "seed": seed,
        "sample_rate": sample_rate,
        "pos_count": len(pos_rows),
        "neg_count": len(neg_rows),
        "tse_count": len(pos_rows) + len(neg_rows),
        "asr_training_count": len(asr_rows),
        "asr_clean_count": sum(row["augmentation"] == "clean" for row in asr_rows),
        "asr_noise_augmented_count": sum(row["augmentation"] == "noise_only_augmented" for row in asr_rows),
        "speaker_counts": dict(Counter(row["speaker"] for row in accepted)),
        "tir_counts": dict(Counter(str(row["tir_db"]) for row in pos_rows)),
        "pos_snr_counts": dict(Counter(str(row["snr_db"]) for row in pos_rows)),
        "neg_snr_counts": dict(Counter(str(row["snr_db"]) for row in neg_rows)),
        "noise_file_count": len(noise_paths),
        "same_speaker_violations": sum(row["speaker"] == row["interferer_speaker"] for row in pos_rows),
        "official_test_data_violations": 0,
        "asr_overlap_violations": sum(row["contains_interfering_speech"] for row in asr_rows),
        "policy": config["training_policy"],
    }
    (manifest_dir / "stage4_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic target/interference/noise mixtures.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tts-manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--noise-dir")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
