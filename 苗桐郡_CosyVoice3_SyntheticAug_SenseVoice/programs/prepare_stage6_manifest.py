from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(args: argparse.Namespace) -> dict:
    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Stage 6 manifest: {output}")
    rows = read_jsonl(Path(args.input))
    prepared = []
    for row in rows:
        cmd = Path(row["audio_cmd"])
        kws = Path(row["audio_kws"])
        if not cmd.is_file() or not kws.is_file():
            raise FileNotFoundError(f"Missing Stage 6 audio for {row['uid']}")
        prepared.append({
            **row,
            "audio_cmd": str(cmd.resolve()),
            "audio_kws": str(kws.resolve()),
            "dataset": args.dataset,
            "evaluation_only": True,
            "allowed_for_training": False,
            "contains_official_test_data": args.dataset == "datasetA_test",
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in prepared:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = {
        "dataset": args.dataset,
        "count": len(prepared),
        "positive_count": sum(row["is_positive"] for row in prepared),
        "negative_count": sum(not row["is_positive"] for row in prepared),
        "missing_audio": 0,
        "evaluation_only": True,
        "allowed_for_training": False,
    }
    Path(str(output) + ".summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Freeze a Stage 6 evaluation-only manifest.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
