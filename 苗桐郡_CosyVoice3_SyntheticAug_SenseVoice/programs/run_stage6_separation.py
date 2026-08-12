from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import torch
from speechbrain.inference.separation import SepformerSeparation
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_mono(path: Path, sample_rate: int) -> torch.Tensor:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != sample_rate:
        raise ValueError(f"Expected {sample_rate} Hz, found {sr}: {path}")
    return torch.from_numpy(audio.mean(axis=1))


def embedding(classifier: EncoderClassifier, waveform: torch.Tensor, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        value = classifier.encode_batch(waveform.unsqueeze(0).to(device)).detach().cpu().numpy().reshape(-1)
    norm = np.linalg.norm(value)
    return value / norm if norm else value


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator else 0.0


def save_audio(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    array = waveform.detach().cpu().float().numpy()
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    if peak > 0.98:
        array = array * (0.98 / peak)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, array, sample_rate, subtype="PCM_16")


def eval_row(row: dict[str, Any], audio_path: Path) -> dict[str, Any]:
    return {
        "uid": row["uid"],
        "audio_path": str(audio_path.resolve()),
        "text": row.get("label", ""),
        "is_positive": bool(row["is_positive"]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.out).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty Stage 6 separation directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(Path(args.manifest))
    if len(rows) != int(config["expected_count"]):
        raise ValueError(f"Expected {config['expected_count']} rows, found {len(rows)}")
    if any(row.get("allowed_for_training") for row in rows):
        raise ValueError("Stage 6 evaluation manifest must not allow training")

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    separator = SepformerSeparation.from_hparams(
        source=config["separator_model"],
        savedir=str(output / "model_runtime" / "separator"),
        run_opts={"device": str(device)},
        local_strategy=LocalStrategy.COPY,
    )
    speaker = EncoderClassifier.from_hparams(
        source=config["speaker_model"],
        savedir=str(output / "model_runtime" / "speaker"),
        run_opts={"device": str(device)},
        local_strategy=LocalStrategy.COPY,
    )

    sample_rate = int(config["sample_rate"])
    results = []
    eval_rows = {name: [] for name in ("mixture", "source0", "source1")}
    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        mixture_path = Path(row["audio_cmd"])
        reference_path = Path(row["audio_kws"])
        mixture = load_mono(mixture_path, sample_rate)
        reference = load_mono(reference_path, sample_rate)
        with torch.no_grad():
            estimates = separator.separate_batch(mixture.unsqueeze(0).to(device)).detach().cpu()[0]
        sources = [estimates[:, source_index] for source_index in range(estimates.shape[-1])]
        if len(sources) != 2:
            raise ValueError(f"Expected two separated sources, found {len(sources)} for {row['uid']}")
        reference_embedding = embedding(speaker, reference, device)
        source_scores = [cosine(reference_embedding, embedding(speaker, source, device)) for source in sources]
        mixture_score = cosine(reference_embedding, embedding(speaker, mixture, device))
        source_paths = []
        for source_index, source in enumerate(sources):
            path = output / "audio" / row["uid"] / f"source_{source_index}.wav"
            save_audio(path, source, sample_rate)
            source_paths.append(str(path.resolve()))
        selected_source = int(np.argmax(source_scores))
        results.append({
            "uid": row["uid"],
            "id": row.get("id", row.get("source_id", row["uid"])),
            "label": row.get("label", ""),
            "is_positive": bool(row["is_positive"]),
            "mixture_path": str(mixture_path.resolve()),
            "reference_path": str(reference_path.resolve()),
            "source_paths": source_paths,
            "speaker_scores": source_scores,
            "mixture_speaker_score": mixture_score,
            "selected_source": selected_source,
            "dataset": config["dataset_name"],
            "evaluation_only": True,
            "allowed_for_training": False,
        })
        eval_rows["mixture"].append(eval_row(row, mixture_path))
        eval_rows["source0"].append(eval_row(row, Path(source_paths[0])))
        eval_rows["source1"].append(eval_row(row, Path(source_paths[1])))
        if index % 25 == 0 or index == len(rows):
            print(f"Separated {index}/{len(rows)}")

    elapsed = time.perf_counter() - started
    write_jsonl(output / "separation_results.jsonl", results)
    for name, items in eval_rows.items():
        write_jsonl(output / f"{name}_eval.jsonl", items)
    summary = {
        "dataset": config["dataset_name"],
        "count": len(rows),
        "positive_count": sum(row["is_positive"] for row in rows),
        "negative_count": sum(not row["is_positive"] for row in rows),
        "selected_source_counts": dict(Counter(row["selected_source"] for row in results)),
        "mean_source_score_margin": float(np.mean([abs(row["speaker_scores"][0] - row["speaker_scores"][1]) for row in results])),
        "elapsed_seconds": elapsed,
        "seconds_per_sample": elapsed / len(rows),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
        "evaluation_only": True,
        "official_test_data_used_for_training": False,
    }
    (output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 6 two-source separation without target-clean labels.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
