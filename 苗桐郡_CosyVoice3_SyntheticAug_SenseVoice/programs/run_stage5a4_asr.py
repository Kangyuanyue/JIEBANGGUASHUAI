from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_stage5a import evaluate_model


def run(args: argparse.Namespace) -> dict:
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    separation_dir = Path(args.separation_dir).resolve()
    output_dir = Path(args.out).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty Stage 5A.4 ASR directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = {
        name: separation_dir / f"{name}_eval.jsonl"
        for name in ("mixture", "source0", "source1", "selected", "target_clean")
    }
    missing = [str(path) for path in manifests.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 5A.4 manifests: {missing}")
    sample_counts = {
        name: sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        for name, path in manifests.items()
    }
    if len(set(sample_counts.values())) != 1:
        raise ValueError(f"Candidate manifest counts do not match: {sample_counts}")
    metrics = evaluate_model(
        base_model_path=config["asr_base_model"],
        checkpoint_path=str(Path(args.project_root).resolve() / config["asr_checkpoint"]),
        manifests=manifests,
        device=config["device"],
        output_dir=output_dir,
    )
    result = {
        "sample_count": next(iter(sample_counts.values())),
        "base_model": config["asr_base_model"],
        "checkpoint": config["asr_checkpoint"],
        "metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all Stage 5A.4 separation candidates with one SenseVoice checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--separation-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--project-root", default=".")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({name: value["cer"] for name, value in result["metrics"].items() if isinstance(value, dict) and "cer" in value}, ensure_ascii=False, indent=2))
