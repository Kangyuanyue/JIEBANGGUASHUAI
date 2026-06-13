#!/usr/bin/env python3
"""Official-style inference entry for the miao solution.

This script provides a stable batch=1 loop, duration measurement and JSON output
format. In --mock mode it does not load real models and rejects all samples; this
is only for checking the repository structure and output schema.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from src.pipeline import CompetitionPipeline
from src.schema import make_record, write_result_json


def load_manifest(path: str | Path) -> List[Dict[str, Any]]:
    """Load flexible manifest formats.

    Supported formats:
    1. A list of sample dicts.
    2. A dict with key `samples`, `data`, or `items`.
    3. A single official-style sample dict.
    4. A dict mapping sample_id -> sample dict.
    """
    obj = json.loads(Path(path).read_text(encoding="utf-8"))

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        for key in ("samples", "data", "items"):
            if key in obj and isinstance(obj[key], list):
                return obj[key]

        # Single official-style sample.
        if any(k in obj for k in ("识别音频", "识别音频名字", "query_audio")):
            return [obj]

        # Mapping: id -> sample.
        if all(isinstance(v, dict) for v in obj.values()):
            samples = []
            for sample_id, sample in obj.items():
                sample = dict(sample)
                sample.setdefault("id", sample_id)
                samples.append(sample)
            return samples

    raise ValueError(f"Unsupported manifest format: {path}")


def get_sample_id(sample: Dict[str, Any], index: int) -> str:
    for key in ("id", "识别音频名字", "query_audio_name", "识别音频"):
        value = sample.get(key)
        if value:
            return Path(str(value)).stem
    return f"sample_{index:06d}"


def get_label(sample: Dict[str, Any]) -> str:
    # Depending on data release, Chinese keys may contain either text labels or
    # label file names. If label files are provided, adapt this function to read
    # their contents.
    for key in ("label", "识别标签", "识别文本", "识别文本名字", "text"):
        value = sample.get(key)
        if value is not None:
            return str(value)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_manifest", required=True, help="Path to test manifest JSON")
    parser.add_argument("--output_json", default="outputs/result.json", help="Output result JSON path")
    parser.add_argument("--config", default="config/config.yaml", help="Config YAML")
    parser.add_argument("--thresholds", default="config/thresholds.yaml", help="Threshold YAML")
    parser.add_argument("--mock", action="store_true", help="Run I/O test mode without real models")
    args = parser.parse_args()

    samples = load_manifest(args.input_manifest)
    pipeline = CompetitionPipeline.from_yaml(args.config, args.thresholds, mock=args.mock)

    records = []
    start = time.perf_counter()

    for idx, sample in enumerate(samples):
        sample_id = get_sample_id(sample, idx)
        label = get_label(sample)
        prediction = pipeline.infer_sample(sample)
        records.append(make_record(sample_id, prediction.content, label))

    duration = time.perf_counter() - start
    write_result_json(records, duration_sec=duration, output_path=args.output_json)
    print(f"Wrote {len(records)} records to {args.output_json}; duration={duration:.6f}s")


if __name__ == "__main__":
    main()
