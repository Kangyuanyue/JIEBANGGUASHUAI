#!/usr/bin/env python
"""Build the fixed 240-record stage-3 text corpus and 720-item TTS manifest."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "dev", "internal-test")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def personal_group(text: str) -> tuple[str, str]:
    if "温度" in text or re.search(r"[十二三四五六七八九]+度", text):
        return "temperature", "personal_temperature"
    if "风速" in text or "风量" in text:
        return "fan_speed", "personal_fan_speed"
    if any(token in text for token in ("风向", "风摆", "摆动", "上吹", "下吹")):
        return "fan_direction", "personal_fan_direction"
    if any(token in text for token in ("模式", "防直吹", "智控温", "ECO")):
        return "mode", "personal_mode"
    if "空调" in text or "显示屏" in text:
        return "power", "personal_power_display"
    if any(token in text for token in ("播放", "放首", "我想听", "音乐")):
        return "media", "personal_media"
    if any(token in text for token in ("吃什么", "食物", "忌口", "便秘", "腹泻", "哺乳期", "备孕")):
        return "health", "personal_health"
    if any(token in text for token in ("滤网", "故障", "步骤")):
        return "maintenance", "personal_maintenance"
    return "other", "personal_other"


def assign_personal_groups(group_sizes: dict[str, int]) -> dict[str, str]:
    groups = sorted(group_sizes)
    total = sum(group_sizes.values())
    targets = {"train": total * 0.70, "dev": total * 0.15, "internal-test": total * 0.15}
    best = None
    for choices in itertools.product(SPLITS, repeat=len(groups)):
        if len(set(choices)) < 3:
            continue
        counts = Counter()
        for group, split in zip(groups, choices):
            counts[split] += group_sizes[group]
        score = sum(abs(counts[split] - targets[split]) for split in SPLITS)
        candidate = (score, choices)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return dict(zip(groups, best[1]))


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--personal", type=Path, required=True)
    parser.add_argument("--generation-config", type=Path, required=True)
    parser.add_argument("--speaker-config", type=Path, required=True)
    parser.add_argument("--text-out", type=Path, required=True)
    parser.add_argument("--tts-out", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.text_out, args.tts_out, args.summary):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite: {output}")
    if args.audio_dir.exists():
        raise FileExistsError(f"Refusing to reuse audio directory: {args.audio_dir}")

    personal = read_jsonl(args.personal)
    if len(personal) != 80:
        raise ValueError(f"Expected 80 personal records, found {len(personal)}")
    config = json.loads(args.generation_config.read_text(encoding="utf-8"))
    speaker_config = json.loads(args.speaker_config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    rng = random.Random(seed)
    personal_texts = {row["text"] for row in personal}

    text_rows = []
    personal_groups: dict[str, list[dict]] = defaultdict(list)
    for row in personal:
        intent, group = personal_group(row["text"])
        personal_groups[group].append(row)
    personal_assignments = assign_personal_groups({key: len(value) for key, value in personal_groups.items()})
    for row in personal:
        intent, group = personal_group(row["text"])
        duplicate_key = hashlib.sha1(row["text"].encode("utf-8")).hexdigest()[:10]
        text_rows.append(
            {
                "uid": f"personal_{row['source_id']}",
                "text": row["text"],
                "source": "personal_verified_recording",
                "source_id": str(row["source_id"]),
                "intent": intent,
                "template_id": group,
                "group_id": f"{group}_{duplicate_key}",
                "split": personal_assignments[group],
                "allowed_for_training": True,
                "contains_official_test_data": False,
            }
        )

    generated_groups = [template["id"] for template in config["templates"]]
    shuffled_groups = generated_groups[:]
    rng.shuffle(shuffled_groups)
    generated_split = {
        group: "train" if index < 14 else "dev" if index < 17 else "internal-test"
        for index, group in enumerate(shuffled_groups)
    }
    generated_texts = set()
    per_template = int(config["generated_per_template"])
    for template in config["templates"]:
        slot_names = list(template["slots"])
        combinations = list(itertools.product(*(template["slots"][name] for name in slot_names)))
        rng.shuffle(combinations)
        accepted = []
        for values in combinations:
            slots = dict(zip(slot_names, values))
            text = template["format"].format(**slots)
            if text in personal_texts or text in generated_texts:
                continue
            accepted.append((text, slots))
            generated_texts.add(text)
            if len(accepted) == per_template:
                break
        if len(accepted) != per_template:
            raise ValueError(f"Template {template['id']} produced only {len(accepted)} unique texts")
        for index, (text, slots) in enumerate(accepted, start=1):
            text_rows.append(
                {
                    "uid": f"generated_{template['id']}_{index:02d}",
                    "text": text,
                    "source": "rule_generated",
                    "source_id": None,
                    "intent": template["intent"],
                    "template_id": template["id"],
                    "group_id": f"generated_{template['id']}",
                    "slots": slots,
                    "split": generated_split[template["id"]],
                    "allowed_for_training": True,
                    "contains_official_test_data": False,
                }
            )

    if len(text_rows) != 240:
        raise AssertionError(f"Expected 240 text records, found {len(text_rows)}")
    text_to_splits: dict[str, set[str]] = defaultdict(set)
    group_to_splits: dict[str, set[str]] = defaultdict(set)
    for row in text_rows:
        text_to_splits[row["text"]].add(row["split"])
        group_to_splits[row["group_id"]].add(row["split"])
    if any(len(splits) != 1 for splits in text_to_splits.values()):
        raise AssertionError("Identical text leaked across splits")
    if any(len(splits) != 1 for splits in group_to_splits.values()):
        raise AssertionError("Template group leaked across splits")

    speakers = speaker_config["speakers"]
    if len(speakers) != 3:
        raise ValueError("Stage 3 requires exactly three configured speakers")
    args.audio_dir.mkdir(parents=True)
    tts_rows = []
    speed_counts = Counter()
    for text_row in text_rows:
        for speaker_index, speaker in enumerate(speakers):
            stable = int(hashlib.sha1(f"{seed}|{text_row['uid']}|{speaker['speaker']}".encode()).hexdigest(), 16)
            speed = config["speeds"][stable % len(config["speeds"])]
            speed_counts[str(speed)] += 1
            uid = f"{text_row['uid']}__{speaker['speaker']}"
            output = (args.audio_dir / f"{uid}.wav").resolve()
            tts_rows.append(
                {
                    **text_row,
                    "uid": uid,
                    "text_uid": text_row["uid"],
                    "speaker": speaker["speaker"],
                    "speaker_display_name": speaker["display_name"],
                    "prompt_text": speaker["prompt_text"],
                    "prompt_audio_path": str(resolve_path(speaker["prompt_audio_path"])),
                    "prompt_source": speaker["source"],
                    "prompt_source_detail": speaker["source_detail"],
                    "output_audio_path": str(output),
                    "speed": speed,
                    "tts_engine": speaker_config["engine"],
                }
            )
    if len(tts_rows) != 720:
        raise AssertionError(f"Expected 720 TTS rows, found {len(tts_rows)}")

    args.text_out.parent.mkdir(parents=True, exist_ok=True)
    with args.text_out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in text_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with args.tts_out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in tts_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    split_counts = Counter(row["split"] for row in text_rows)
    source_counts = Counter(row["source"] for row in text_rows)
    summary = {
        "seed": seed,
        "text_records": len(text_rows),
        "unique_texts": len(text_to_splits),
        "duplicate_text_records": len(text_rows) - len(text_to_splits),
        "source_counts": dict(sorted(source_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "split_ratios": {key: round(value / len(text_rows), 4) for key, value in sorted(split_counts.items())},
        "generated_template_groups": len(generated_groups),
        "personal_template_groups": len(personal_groups),
        "tts_records": len(tts_rows),
        "speaker_count": len(speakers),
        "speed_counts": dict(sorted(speed_counts.items())),
        "text_leakage": False,
        "group_leakage": False,
        "official_test_data_used": False,
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
