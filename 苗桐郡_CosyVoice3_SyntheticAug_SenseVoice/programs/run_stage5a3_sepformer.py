from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
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


def stable_rank(rows: list[dict[str, Any]], seed: int, salt: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: hashlib.sha256(f"{seed}:{salt}:{row['uid']}".encode()).hexdigest())


def resolve_path(path: str, project: Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (project / candidate).resolve()


def load_mono(path: Path, sample_rate: int) -> torch.Tensor:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != sample_rate:
        raise ValueError(f"Expected {sample_rate} Hz, found {sr}: {path}")
    return torch.from_numpy(audio.mean(axis=1))


def normalized_embedding(classifier: EncoderClassifier, waveform: torch.Tensor, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        embedding = classifier.encode_batch(waveform.unsqueeze(0).to(device)).detach().cpu().numpy().reshape(-1)
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator else 0.0


def align(first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    length = min(first.numel(), second.numel())
    return first[:length].float(), second[:length].float()


def si_sdr(estimate: torch.Tensor, target: torch.Tensor) -> float:
    estimate, target = align(estimate, target)
    estimate = estimate - estimate.mean()
    target = target - target.mean()
    target_energy = torch.sum(target.square()) + 1e-8
    projection = torch.sum(estimate * target) * target / target_energy
    noise = estimate - projection
    return float(10 * torch.log10((torch.sum(projection.square()) + 1e-8) / (torch.sum(noise.square()) + 1e-8)))


def save_audio(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    array = waveform.detach().cpu().float().numpy()
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    if peak > 0.98:
        array = array * (0.98 / peak)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, array, sample_rate, subtype="PCM_16")


def select_rows(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    if config.get("all_split", False):
        allowed_tir = {float(value) for value in config["tir_db"]}
        selected = [
            row for row in rows
            if row["split"] == config["split"] and float(row["tir_db"]) in allowed_tir
        ]
        selected = stable_rank(selected, int(config["seed"]), "stage5a4_full_split")
        expected_count = config.get("expected_count")
        if expected_count is not None and len(selected) != int(expected_count):
            raise ValueError(f"Expected {expected_count} rows, found {len(selected)}")
        return selected
    selected = []
    for tir in config["tir_db"]:
        group = [row for row in rows if row["split"] == config["split"] and float(row["tir_db"]) == float(tir)]
        chosen = stable_rank(group, int(config["seed"]), f"stage5a3_tir_{tir}")[: int(config["samples_per_tir"])]
        if len(chosen) != int(config["samples_per_tir"]):
            raise ValueError(f"Not enough samples for TIR {tir}")
        selected.extend(chosen)
    return selected


def asr_row(uid: str, audio_path: Path, text: str) -> dict[str, Any]:
    return {"uid": uid, "audio_path": str(audio_path.resolve()), "text": text}


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    project = Path(args.project_root).resolve()
    output = Path(args.out).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty separation directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    audio_dir = output / "audio"

    rows = select_rows(read_jsonl(Path(args.manifest)), config)
    if any(row.get("contains_official_test_data") for row in rows):
        raise ValueError("Official test data detected in Stage 5A.3 selection")
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

    results = []
    mixture_eval = []
    source0_eval = []
    source1_eval = []
    selected_eval = []
    clean_eval = []
    started = time.perf_counter()
    for index, row in enumerate(rows):
        mixture_path = resolve_path(row["mixture_path"], project)
        target_path = resolve_path(row["target_clean_path"], project)
        reference_path = resolve_path(row["reference_path"], project)
        mixture = load_mono(mixture_path, int(config["sample_rate"]))
        target = load_mono(target_path, int(config["sample_rate"]))
        reference = load_mono(reference_path, int(config["sample_rate"]))
        with torch.no_grad():
            estimates = separator.separate_batch(mixture.unsqueeze(0).to(device)).detach().cpu()[0]
        sources = [estimates[:, source_index] for source_index in range(estimates.shape[-1])]
        if len(sources) != 2:
            raise ValueError(f"Expected two separated sources, found {len(sources)} for {row['uid']}")
        reference_embedding = normalized_embedding(speaker, reference, device)
        source_embeddings = [normalized_embedding(speaker, source, device) for source in sources]
        speaker_scores = [cosine(reference_embedding, embedding) for embedding in source_embeddings]
        mixture_speaker_score = cosine(reference_embedding, normalized_embedding(speaker, mixture, device))
        target_scores = [si_sdr(source, target) for source in sources]
        selected_index = int(np.argmax(speaker_scores))
        oracle_index = int(np.argmax(target_scores))
        selected = sources[selected_index]

        source_paths = []
        for source_index, source in enumerate(sources):
            path = audio_dir / row["uid"] / f"source_{source_index}.wav"
            save_audio(path, source, int(config["sample_rate"]))
            source_paths.append(str(path.resolve()))
        selected_path = audio_dir / row["uid"] / "selected.wav"
        save_audio(selected_path, selected, int(config["sample_rate"]))
        input_sisdr = si_sdr(mixture, target)
        selected_sisdr = target_scores[selected_index]
        results.append({
            "uid": row["uid"],
            "label": row["label"],
            "speaker": row["speaker"],
            "interferer_speaker": row["interferer_speaker"],
            "tir_db": row["tir_db"],
            "snr_db": row["snr_db"],
            "mixture_path": str(mixture_path),
            "target_clean_path": str(target_path),
            "reference_path": str(reference_path),
            "source_paths": source_paths,
            "selected_path": str(selected_path.resolve()),
            "speaker_scores": speaker_scores,
            "mixture_speaker_score": mixture_speaker_score,
            "source_target_si_sdr": target_scores,
            "selected_source": selected_index,
            "oracle_source": oracle_index,
            "selection_correct": selected_index == oracle_index,
            "input_si_sdr": input_sisdr,
            "selected_si_sdr": selected_sisdr,
            "si_sdri": selected_sisdr - input_sisdr,
            "contains_official_test_data": False,
        })
        mixture_eval.append(asr_row(row["uid"], mixture_path, row["label"]))
        source0_eval.append(asr_row(row["uid"], Path(source_paths[0]), row["label"]))
        source1_eval.append(asr_row(row["uid"], Path(source_paths[1]), row["label"]))
        selected_eval.append(asr_row(row["uid"], selected_path, row["label"]))
        clean_eval.append(asr_row(row["uid"], target_path, row["label"]))
        print(f"Separated {index + 1}/{len(rows)}: {row['uid']}")

    elapsed = time.perf_counter() - started
    write_jsonl(output / "separation_results.jsonl", results)
    write_jsonl(output / "mixture_eval.jsonl", mixture_eval)
    write_jsonl(output / "source0_eval.jsonl", source0_eval)
    write_jsonl(output / "source1_eval.jsonl", source1_eval)
    write_jsonl(output / "selected_eval.jsonl", selected_eval)
    write_jsonl(output / "target_clean_eval.jsonl", clean_eval)
    by_tir = defaultdict(list)
    for row in results:
        by_tir[str(int(row["tir_db"]))].append(row)
    summary = {
        "count": len(results),
        "device": str(device),
        "separator_model": config["separator_model"],
        "speaker_model": config["speaker_model"],
        "selection_accuracy": sum(row["selection_correct"] for row in results) / len(results),
        "mean_input_si_sdr": float(np.mean([row["input_si_sdr"] for row in results])),
        "mean_selected_si_sdr": float(np.mean([row["selected_si_sdr"] for row in results])),
        "mean_si_sdri": float(np.mean([row["si_sdri"] for row in results])),
        "mean_mixture_speaker_score": float(np.mean([row["mixture_speaker_score"] for row in results])),
        "positive_si_sdri_rate": sum(row["si_sdri"] > 0 for row in results) / len(results),
        "by_tir": {
            tir: {
                "count": len(items),
                "selection_accuracy": sum(item["selection_correct"] for item in items) / len(items),
                "mean_si_sdri": float(np.mean([item["si_sdri"] for item in items])),
            }
            for tir, items in sorted(by_tir.items(), key=lambda item: float(item[0]))
        },
        "elapsed_seconds": elapsed,
        "seconds_per_sample": elapsed / len(results),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
        "status_counts": dict(Counter("ok" for _ in results)),
        "official_test_data_violations": 0,
    }
    (output / "separation_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SepFormer plus ECAPA target-channel selection on Stage 4 mixtures.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--project-root", default=".")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
