#!/usr/bin/env python3
"""Label-free ASR consensus selection across frontends/models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from asr_consensus import select_consensus_text  # noqa: E402
from command_postprocess import normalize_command_text  # noqa: E402


def load_cache(path: Path) -> tuple[dict, dict[str, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, {str(row["id"]): row for row in payload["samples"]}


def select_text(texts: list[str], weird_weight: float, command_weight: float) -> tuple[int, list[float]]:
    _, selected, scores = select_consensus_text(texts, weird_weight, command_weight)
    return selected, scores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--caches", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", default=[])
    parser.add_argument("--weird-weight", type=float, default=0.35)
    parser.add_argument("--command-weight", type=float, default=0.10)
    parser.add_argument("--output", default="output/datasetA_asr_cache_consensus.json")
    args = parser.parse_args()

    loaded = [load_cache(Path(path)) for path in args.caches]
    primary_payload, primary_rows = loaded[0]
    row_maps = [rows for _, rows in loaded]
    names = args.names or [Path(path).stem for path in args.caches]
    if len(names) != len(row_maps):
        raise ValueError("--names must have the same length as --caches")

    output_rows = []
    selection_counts = {name: 0 for name in names}
    for sample_id, primary in primary_rows.items():
        rows = [mapping.get(sample_id, {}) for mapping in row_maps]
        texts = [row.get("asr_text", "") for row in rows]
        selected, scores = select_text(
            texts, weird_weight=args.weird_weight, command_weight=args.command_weight
        )
        selection_counts[names[selected]] += 1
        item = dict(primary)
        item["asr_text"] = normalize_command_text(texts[selected])
        item["selected_asr"] = names[selected]
        item["candidate_texts"] = {name: normalize_command_text(text) for name, text in zip(names, texts)}
        item["consensus_scores"] = {name: score for name, score in zip(names, scores)}
        output_rows.append(item)

    result = {
        "method": "label_free_character_medoid_with_grammar_tiebreak",
        "source_caches": args.caches,
        "names": names,
        "weird_weight": args.weird_weight,
        "command_weight": args.command_weight,
        "selection_counts": selection_counts,
        "n_total": len(output_rows),
        "total_duration_sec": sum(float(payload.get("total_duration_sec", 0.0)) for payload, _ in loaded),
        "samples": output_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
