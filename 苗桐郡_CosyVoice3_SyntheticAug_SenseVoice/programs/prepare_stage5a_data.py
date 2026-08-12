from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import unicodedata
from pathlib import Path
from typing import Any, Iterable

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


def normalized_length(text: str) -> int:
    text = TAG_RE.sub("", text)
    return sum(not char.isspace() and not unicodedata.category(char).startswith("P") for char in text)


def stable_select(rows: list[dict[str, Any]], count: int, seed: int, salt: str) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}:{salt}:{row['uid']}".encode("utf-8")).hexdigest(),
    )
    if len(ranked) < count:
        raise ValueError(f"Requested {count} rows for {salt}, only {len(ranked)} available")
    return ranked[:count]


def to_funasr(row: dict[str, Any]) -> dict[str, Any]:
    info = sf.info(row["audio_path"])
    return {
        "key": row["uid"],
        "prompt": "<|ASR|>",
        "source": row["audio_path"],
        "source_len": max(1, int(round(info.duration * 100))),
        "target": row["text"],
        "target_len": max(1, normalized_length(row["text"])),
        "text_language": "<|zh|>",
        "with_or_wo_itn": "<|woitn|>",
    }


def boundary_violations(rows: Iterable[dict[str, Any]]) -> list[str]:
    violations = []
    for row in rows:
        searchable = json.dumps(row, ensure_ascii=False).lower().replace("-", "_")
        if any(marker in searchable for marker in TEST_MARKERS):
            violations.append(row.get("uid", "unknown"))
    return violations


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.project_root).resolve()
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty Stage 5A data directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    real_audio_dir = out / "real_16k"
    real_audio_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    personal = read_jsonl(Path(args.personal_manifest))
    text_rows = read_jsonl(Path(args.text_manifest))
    split_by_source_id = {
        str(row["source_id"]): row["split"]
        for row in text_rows
        if row["source"] == "personal_verified_recording"
    }

    real_rows = []
    for row in personal:
        source_id = str(row["source_id"])
        split = split_by_source_id[source_id]
        destination = real_audio_dir / f"real_{source_id}.wav"
        subprocess.run(
            [args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", row["audio_path"], "-ac", "1", "-ar", str(cfg["sample_rate"]), str(destination)],
            check=True,
        )
        info = sf.info(destination)
        if info.samplerate != cfg["sample_rate"] or info.channels != 1 or info.frames <= 0:
            raise ValueError(f"Invalid converted real recording: {destination}")
        real_rows.append({
            "uid": f"stage5a_real_{source_id}",
            "audio_path": str(destination.resolve()),
            "text": row["text"],
            "split": split,
            "source": "personal_verified_recording",
            "speaker": "spk08_miaotongjun",
            "contains_interfering_speech": False,
            "contains_official_test_data": False,
        })

    stage4 = read_jsonl(Path(args.stage4_manifest))
    stage4_rows = []
    for row in stage4:
        audio_path = Path(row["audio_path"])
        if not audio_path.is_absolute():
            audio_path = root / audio_path
        stage4_rows.append({**row, "audio_path": str(audio_path.resolve()), "source": "stage4_safe_tts"})

    real_train_pool = [row for row in real_rows if row["split"] == "train"]
    real_train = stable_select(real_train_pool, int(cfg["real_train_count"]), int(cfg["seed"]), "real_train")
    synth_pool = [
        row for row in stage4_rows
        if row["split"] == "train" and row["augmentation"] in {"clean", "noise_only_augmented"}
    ]
    synth_train = stable_select(synth_pool, int(cfg["synthetic_train_count"]), int(cfg["seed"]), "synthetic_train")
    synth_dev_pool = [
        row for row in stage4_rows
        if row["split"] == "dev" and row["augmentation"] == "noise_only_augmented"
    ]
    synth_dev = stable_select(synth_dev_pool, int(cfg["synthetic_dev_count"]), int(cfg["seed"]), "synthetic_dev")
    real_dev = sorted((row for row in real_rows if row["split"] == "dev"), key=lambda row: row["uid"])
    real_holdout = sorted((row for row in real_rows if row["split"] == "internal-test"), key=lambda row: row["uid"])

    all_training_sources = [*real_train, *synth_train]
    violations = boundary_violations(all_training_sources)
    if violations:
        raise ValueError(f"Official test data found in Stage 5A training rows: {violations[:5]}")
    if {row["uid"] for row in real_train} & {row["uid"] for row in real_dev + real_holdout}:
        raise ValueError("Real recording train/evaluation leakage detected")

    write_jsonl(out / "real_train_eval.jsonl", real_train)
    write_jsonl(out / "synthetic_train_eval.jsonl", synth_train)
    write_jsonl(out / "internal_dev_eval.jsonl", synth_dev)
    write_jsonl(out / "real_dev_eval.jsonl", real_dev)
    write_jsonl(out / "real_holdout_eval.jsonl", real_holdout)
    write_jsonl(out / "real_only_train.jsonl", map(to_funasr, real_train))
    write_jsonl(out / "augmented_train.jsonl", map(to_funasr, [*real_train, *synth_train]))

    summary = {
        "seed": cfg["seed"],
        "real_recordings_converted": len(real_rows),
        "real_train_pool": len(real_train_pool),
        "real_train_selected": len(real_train),
        "synthetic_train_selected": len(synth_train),
        "augmented_train_total": len(real_train) + len(synth_train),
        "synthetic_internal_dev": len(synth_dev),
        "real_dev": len(real_dev),
        "real_holdout": len(real_holdout),
        "official_test_data_violations": len(violations),
        "real_train_eval_overlap": 0,
        "sample_rate": cfg["sample_rate"],
        "channels": 1,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fixed Stage 5A real and synthetic smoke subsets.")
    parser.add_argument("--personal-manifest", required=True)
    parser.add_argument("--text-manifest", required=True)
    parser.add_argument("--stage4-manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
