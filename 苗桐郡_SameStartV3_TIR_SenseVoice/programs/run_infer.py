#!/usr/bin/env python3
"""Run SenseVoice-Small inference for a datasetA split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from common import normalize_text, read_jsonl, repair_mojibake, resolve_manifest, should_reject, write_jsonl


def build_model(model_name: str, vad_model: str | None, device: str, disable_update: bool):
    try:
        from funasr import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "funasr is not installed. Install requirements first, then rerun this script."
        ) from exc

    kwargs: Dict[str, Any] = {
        "model": model_name,
        "trust_remote_code": True,
        "device": device,
        "disable_update": disable_update,
    }
    if vad_model:
        kwargs["vad_model"] = vad_model
        kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}
    return AutoModel(**kwargs)


def parse_funasr_result(result: Any) -> str:
    if isinstance(result, list):
        texts = []
        for item in result:
            if isinstance(item, dict) and item.get("text") is not None:
                texts.append(str(item["text"]))
            elif isinstance(item, str):
                texts.append(item)
        return "".join(texts)
    if isinstance(result, dict):
        return str(result.get("text", ""))
    return str(result)


def generate_one(model, audio_path: str, language: str, use_itn: bool, hotword: str | None) -> str:
    kwargs: Dict[str, Any] = {
        "input": audio_path,
        "cache": {},
        "language": language,
        "use_itn": use_itn,
        "batch_size_s": 60,
        "merge_vad": True,
        "merge_length_s": 15,
    }
    if hotword:
        kwargs["hotword"] = hotword
    return parse_funasr_result(model.generate(**kwargs))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", help="Split name or manifest path")
    parser.add_argument("--manifest", type=Path, help="Explicit manifest path")
    parser.add_argument("--model", default="iic/SenseVoiceSmall")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--use-itn", action="store_true")
    parser.add_argument("--no-vad", action="store_true")
    parser.add_argument("--vad-model", default="fsmn-vad")
    parser.add_argument("--hotword-file", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--disable-update", action="store_true", default=True)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    manifest_path = args.manifest or resolve_manifest(project, args.split)
    rows = read_jsonl(manifest_path)
    if args.limit:
        rows = rows[: args.limit]
    hotword = None
    if args.hotword_file:
        hotword = args.hotword_file.read_text(encoding="utf-8").strip()

    model = build_model(
        model_name=args.model,
        vad_model=None if args.no_vad else args.vad_model,
        device=args.device,
        disable_update=args.disable_update,
    )

    preds: List[Dict[str, Any]] = []
    started = time.time()
    for index, row in enumerate(rows, start=1):
        try:
            raw_text = generate_one(model, row["audio_cmd"], args.language, args.use_itn, hotword)
            raw_text = repair_mojibake(raw_text)
            error = None
        except Exception as exc:
            raw_text = ""
            error = repr(exc)
        text = normalize_text(raw_text)
        pred = {
            "uid": row["uid"],
            "audio_cmd": row["audio_cmd"],
            "raw_text": raw_text,
            "text": text,
            "reject": should_reject(text, {"reject_empty": True, "min_chars": 1, "min_domain_score": 0.0}),
            "model": args.model,
            "use_itn": args.use_itn,
            "vad": not args.no_vad,
            "hotword_file": str(args.hotword_file) if args.hotword_file else None,
            "error": error,
        }
        preds.append(pred)
        if index % 20 == 0:
            write_jsonl(args.out, preds)
            print(f"Processed {index}/{len(rows)}")

    write_jsonl(args.out, preds)
    elapsed = time.time() - started
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(
            {
                "manifest": str(manifest_path.resolve()),
                "model": args.model,
                "device": args.device,
                "use_itn": args.use_itn,
                "vad": not args.no_vad,
                "hotword_file": str(args.hotword_file) if args.hotword_file else None,
                "count": len(preds),
                "elapsed_seconds": elapsed,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
