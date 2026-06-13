"""Result JSON helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .text_norm import char_error_rate, mean


def make_record(sample_id: str, content: str, label: str | None = "", normalize_digits: bool = False) -> Dict[str, str]:
    label = "" if label is None else str(label)
    content = "" if content is None else str(content)
    cer = char_error_rate(content, label, normalize_digits=normalize_digits)
    return {
        "id": str(sample_id),
        "content": content,
        "label": label,
        "cer": f"{cer:.6f}",
    }


def make_result(records: List[Dict[str, str]], duration_sec: float) -> Dict[str, Any]:
    positive_cers = [float(r["cer"]) for r in records if str(r.get("label", "")) != ""]
    final_cer = mean(positive_cers)
    return {
        "result": {
            "results": records,
            "final_cer": f"{final_cer:.6f}",
            "duration": f"{duration_sec:.6f}",
        }
    }


def validate_result_obj(obj: Dict[str, Any]) -> None:
    if "result" not in obj or not isinstance(obj["result"], dict):
        raise ValueError("Missing top-level 'result' object.")
    result = obj["result"]
    if "results" not in result or not isinstance(result["results"], list):
        raise ValueError("Missing result.results list.")
    for idx, item in enumerate(result["results"]):
        for key in ("id", "content", "label", "cer"):
            if key not in item:
                raise ValueError(f"Record {idx} missing key: {key}")
    if "final_cer" not in result:
        raise ValueError("Missing result.final_cer.")
    if "duration" not in result:
        raise ValueError("Missing result.duration.")


def write_result_json(records: List[Dict[str, str]], duration_sec: float, output_path: str | Path) -> Dict[str, Any]:
    obj = make_result(records, duration_sec=duration_sec)
    validate_result_obj(obj)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return obj


def load_result_json(path: str | Path) -> Dict[str, Any]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_result_obj(obj)
    return obj
