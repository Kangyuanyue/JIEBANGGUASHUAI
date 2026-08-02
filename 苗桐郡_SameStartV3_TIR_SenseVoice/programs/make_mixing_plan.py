#!/usr/bin/env python3
"""Create a deterministic POS/NEG mixing plan from a validated source manifest."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def choose_other(rng: random.Random, speakers: list[str], excluded: set[str], count: int) -> list[str]:
    candidates = [speaker for speaker in speakers if speaker not in excluded]
    if len(candidates) < count:
        raise ValueError(
            f"Need {count} non-target speaker(s), but only {len(candidates)} are available; "
            f"available={speakers}, excluded={sorted(excluded)}"
        )
    return rng.sample(candidates, count)


def build_plan(rows: list[dict], config: dict, positive_count: int, negative_count: int) -> list[dict]:
    seed = int(config["seed"])
    rng = random.Random(seed)
    positive_by_speaker: dict[str, list[dict]] = {}
    negative_by_speaker: dict[str, list[dict]] = {}
    noises: list[dict] = []
    for row in rows:
        if row["kind"] == "positive_pair":
            positive_by_speaker.setdefault(row["speaker_id"], []).append(row)
        elif row["kind"] == "negative_wake":
            negative_by_speaker.setdefault(row["speaker_id"], []).append(row)
        elif row["kind"] == "noise":
            noises.append(row)
    speakers = sorted(positive_by_speaker)
    if positive_count and len(speakers) < int(config["speaker_rules"]["positive"]["minimum_distinct_speakers"]):
        raise ValueError(f"POS mixing requires at least 2 ready speakers; found {len(speakers)}")
    negative_speakers = sorted(set(speakers) & set(negative_by_speaker))
    if negative_count and len(negative_speakers) < int(
        config["speaker_rules"]["negative"]["minimum_distinct_speakers"]
    ):
        raise ValueError(f"NEG mixing requires at least 3 ready speakers; found {len(negative_speakers)}")
    levels = list(config["pilot"]["levels"])
    plan: list[dict] = []

    for index in range(positive_count):
        level_name = levels[index % len(levels)]
        level = config["difficulty"][level_name]
        target_speaker = rng.choice(speakers)
        target = rng.choice(positive_by_speaker[target_speaker])
        interferer_speaker = choose_other(rng, speakers, {target_speaker}, 1)[0]
        interferer = rng.choice(positive_by_speaker[interferer_speaker])
        use_interference = level_name != "clean"
        noise = rng.choice(noises) if use_interference and noises else None
        plan.append(
            {
                "sample_id": f"pos_{index + 1:04d}_{level_name}",
                "sample_type": "POS",
                "difficulty": level_name,
                "sample_seed": rng.randrange(0, 2**31),
                "target_speaker_id": target_speaker,
                "reference_kws_source": target["wake_path"],
                "wake_text": target["wake_text"],
                "target_command_source": target["command_path"],
                "target_text": target["command_text"],
                "command_sources": [target["command_path"]],
                "interferer_speaker_ids": [interferer_speaker] if use_interference else [],
                "interferer_command_sources": [interferer["command_path"]] if use_interference else [],
                "wake_interferer_source": interferer["wake_path"] if use_interference else None,
                "noise_source": noise["noise_path"] if noise else None,
                "sir_db": level["sir_db"],
                "snr_db": level["snr_db"],
                "overlap_ratio": float(level["overlap_ratio"]),
                "alignment_mode": str(
                    level.get("alignment_mode", config["audio"].get("alignment_mode", "overlap_ratio"))
                ),
                "reverb": bool(level["reverb"]),
            }
        )

    for index in range(negative_count):
        level_name = levels[index % len(levels)]
        level = config["difficulty"][level_name]
        wake_speaker = rng.choice(negative_speakers)
        wake = rng.choice(negative_by_speaker[wake_speaker])
        speaker_b, speaker_c = choose_other(rng, negative_speakers, {wake_speaker}, 2)
        command_b = rng.choice(positive_by_speaker[speaker_b])
        command_c = rng.choice(positive_by_speaker[speaker_c])
        noise = rng.choice(noises) if level_name != "clean" and noises else None
        plan.append(
            {
                "sample_id": f"neg_{index + 1:04d}_{level_name}",
                "sample_type": "NEG",
                "difficulty": level_name,
                "sample_seed": rng.randrange(0, 2**31),
                "target_speaker_id": wake_speaker,
                "reference_kws_source": wake["wake_path"],
                "wake_text": wake["wake_text"],
                "target_command_source": None,
                "target_text": "",
                "command_sources": [command_b["command_path"], command_c["command_path"]],
                "interferer_speaker_ids": [speaker_b, speaker_c],
                "interferer_command_sources": [command_c["command_path"]],
                "wake_interferer_source": command_b["wake_path"] if level_name != "clean" else None,
                "noise_source": noise["noise_path"] if noise else None,
                "sir_db": level["sir_db"] if level["sir_db"] is not None else 0.0,
                "snr_db": level["snr_db"],
                "overlap_ratio": float(level["overlap_ratio"]),
                "alignment_mode": str(
                    level.get("alignment_mode", config["audio"].get("alignment_mode", "overlap_ratio"))
                ),
                "reverb": bool(level["reverb"]),
            }
        )
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--positive", type=int)
    parser.add_argument("--negative", type=int)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Output plan already exists; choose a new path")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    positive = int(args.positive if args.positive is not None else config["pilot"]["positive_samples"])
    negative = int(args.negative if args.negative is not None else config["pilot"]["negative_samples"])
    plan = build_plan(read_jsonl(args.sources), config, positive, negative)
    write_jsonl(args.out, plan)
    print(json.dumps({"output": str(args.out.resolve()), "count": len(plan)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
