"""Local scorer for result.json.

Usage:
    python -m src.scorer --result_json outputs/result.json --report_json outputs/metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .schema import load_result_json
from .text_norm import char_error_rate, mean


def compute_metrics(result_obj: Dict[str, Any]) -> Dict[str, Any]:
    records: List[Dict[str, str]] = result_obj["result"]["results"]

    positive_cers = []
    negative_count = 0
    negative_correct_reject = 0
    negative_false_accept = 0

    for r in records:
        label = str(r.get("label", ""))
        content = str(r.get("content", ""))
        if label == "":
            negative_count += 1
            if content == "":
                negative_correct_reject += 1
            else:
                negative_false_accept += 1
        else:
            positive_cers.append(char_error_rate(content, label))

    final_cer = mean(positive_cers)
    rr = negative_correct_reject / negative_count if negative_count else 0.0
    duration = float(result_obj["result"].get("duration", 0.0))

    return {
        "total_count": len(records),
        "positive_count": len(positive_cers),
        "negative_count": negative_count,
        "final_cer": final_cer,
        "rr": rr,
        "negative_correct_reject": negative_correct_reject,
        "negative_false_accept": negative_false_accept,
        "duration": duration,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_json", required=True, help="Path to result.json")
    parser.add_argument("--report_json", default="", help="Optional output path for metrics report")
    args = parser.parse_args()

    result_obj = load_result_json(args.result_json)
    metrics = compute_metrics(result_obj)
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(text)

    if args.report_json:
        out = Path(args.report_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
