#!/usr/bin/env python3
"""Create target-speaker command audio using wake audio as speaker cue."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from common import audio_info, load_simple_yaml, read_jsonl, write_jsonl


def import_speechbrain():
    try:
        from speechbrain.inference.separation import SepformerSeparation
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier, SepformerSeparation
    return SepformerSeparation, EncoderClassifier


def load_audio(path: Path, sample_rate: int):
    import torch
    import torchaudio
    import soundfile as sf

    array, sr = sf.read(str(path), always_2d=True, dtype="float32")
    wav = torch.from_numpy(array.T).mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav


def save_audio(path: Path, wav, sample_rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    sf.write(str(path), wav.squeeze(0).detach().cpu().numpy(), sample_rate)


def flatten_embedding(embedding) -> np.ndarray:
    array = embedding.detach().cpu().numpy().reshape(-1).astype(np.float32)
    norm = np.linalg.norm(array)
    return array / norm if norm > 0 else array


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def build_models(config: Dict[str, Any], savedir: Path):
    import torch
    from speechbrain.utils.fetching import LocalStrategy

    SepformerSeparation, EncoderClassifier = import_speechbrain()
    device = str(config.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    separator = SepformerSeparation.from_hparams(
        source=str(config.get("separator_model", "speechbrain/sepformer-whamr16k")),
        savedir=str(savedir / "separator"),
        run_opts={"device": device},
        local_strategy=LocalStrategy.COPY,
    )
    speaker = EncoderClassifier.from_hparams(
        source=str(config.get("speaker_model", "speechbrain/spkrec-ecapa-voxceleb")),
        savedir=str(savedir / "speaker"),
        run_opts={"device": device},
        local_strategy=LocalStrategy.COPY,
    )
    return separator, speaker, device


def module_device(module, default: str = "cpu"):
    import torch

    for value in getattr(module, "mods", {}).values():
        try:
            return next(value.parameters()).device
        except (AttributeError, StopIteration):
            continue
    return torch.device(default)


def split_sources(separator, cmd_path: Path, sample_rate: int) -> List[Any]:
    import torch

    wav = load_audio(cmd_path, sample_rate)
    device = module_device(separator)
    with torch.no_grad():
        estimates = separator.separate_batch(wav.to(device)).detach().cpu()
    if estimates.ndim == 3 and estimates.shape[0] == 1:
        return [estimates[0, :, idx].detach().cpu() for idx in range(estimates.shape[-1])]
    if estimates.ndim == 3:
        return [estimates[idx, :, 0].detach().cpu() for idx in range(estimates.shape[0])]
    if estimates.ndim == 2:
        if estimates.shape[0] <= 4:
            return [estimates[idx, :].detach().cpu() for idx in range(estimates.shape[0])]
        return [estimates[:, idx].detach().cpu() for idx in range(estimates.shape[1])]
    return [estimates.reshape(-1).detach().cpu()]


def speaker_embedding(classifier, wav) -> np.ndarray:
    import torch

    with torch.no_grad():
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        return flatten_embedding(classifier.encode_batch(wav.to(module_device(classifier))))


def choose_target_source(classifier, wake_wav, sources: List[Any]) -> Tuple[int, List[float]]:
    wake_emb = speaker_embedding(classifier, wake_wav)
    scores = [cosine(wake_emb, speaker_embedding(classifier, source)) for source in sources]
    best = int(np.argmax(scores)) if scores else 0
    return best, scores


def process_row(
    row: Dict[str, Any],
    out_audio_dir: Path,
    separator,
    speaker,
    sample_rate: int,
    fallback: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    uid = row["uid"]
    cmd_path = Path(row["audio_cmd"])
    kws_path = Path(row["audio_kws"])
    out_audio = out_audio_dir / f"{uid}.wav"
    meta: Dict[str, Any] = {
        "uid": uid,
        "input_cmd": str(cmd_path),
        "input_kws": str(kws_path),
        "output": str(out_audio),
        "status": "ok",
        "selected_source": None,
        "scores": [],
        "error": None,
    }
    try:
        wake_wav = load_audio(kws_path, sample_rate)
        sources = split_sources(separator, cmd_path, sample_rate)
        best, scores = choose_target_source(speaker, wake_wav, sources)
        save_audio(out_audio, sources[best], sample_rate)
        meta["selected_source"] = best
        meta["scores"] = scores
    except Exception as exc:
        meta["status"] = "fallback_copy" if fallback == "copy" else "failed"
        meta["error"] = repr(exc)
        if fallback == "copy":
            out_audio.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cmd_path, out_audio)
        else:
            raise
    new_row = dict(row)
    new_row["audio_cmd_original"] = row["audio_cmd"]
    new_row["audio_cmd"] = str(out_audio)
    info = audio_info(out_audio)
    new_row["cmd_duration"] = round(float(info["duration"]), 6)
    new_row["sample_rate"] = info["sample_rate"]
    new_row["channels"] = info["channels"]
    new_row["sample_width"] = info["sample_width"]
    return new_row, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    parser.add_argument("--out-audio-dir", required=True, type=Path)
    parser.add_argument("--meta", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/tse_speechbrain.yaml"), type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fallback", choices=["copy", "fail"])
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else project / args.config
    config = load_simple_yaml(config_path)
    if args.fallback:
        config["fallback"] = args.fallback
    sample_rate = int(config.get("sample_rate", 16000))
    fallback = str(config.get("fallback", "copy"))
    rows = read_jsonl(args.manifest)
    if args.limit:
        rows = rows[: args.limit]

    started = time.time()
    separator = speaker = None
    model_error: Optional[str] = None
    try:
        separator, speaker, device = build_models(config, project / "exp" / "speechbrain")
    except Exception as exc:
        if fallback != "copy":
            raise
        device = "unavailable"
        model_error = repr(exc)

    out_rows: List[Dict[str, Any]] = []
    meta_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if separator is None or speaker is None:
            out_audio = args.out_audio_dir / f"{row['uid']}.wav"
            out_audio.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row["audio_cmd"], out_audio)
            new_row = dict(row)
            new_row["audio_cmd_original"] = row["audio_cmd"]
            new_row["audio_cmd"] = str(out_audio)
            meta_row = {
                "uid": row["uid"],
                "input_cmd": row["audio_cmd"],
                "input_kws": row["audio_kws"],
                "output": str(out_audio),
                "status": "fallback_copy",
                "selected_source": None,
                "scores": [],
                "error": model_error,
            }
        else:
            new_row, meta_row = process_row(
                row=row,
                out_audio_dir=args.out_audio_dir,
                separator=separator,
                speaker=speaker,
                sample_rate=sample_rate,
                fallback=fallback,
            )
        out_rows.append(new_row)
        meta_rows.append(meta_row)
        if index % 10 == 0:
            print(f"TSE processed {index}/{len(rows)}")

    write_jsonl(args.out_manifest, out_rows)
    args.meta.parent.mkdir(parents=True, exist_ok=True)
    args.meta.write_text(
        json.dumps(
            {
                "manifest": str(args.manifest.resolve()),
                "out_manifest": str(args.out_manifest.resolve()),
                "out_audio_dir": str(args.out_audio_dir.resolve()),
                "config": config,
                "device": device,
                "count": len(out_rows),
                "status_counts": {
                    status: sum(1 for item in meta_rows if item["status"] == status)
                    for status in sorted({item["status"] for item in meta_rows})
                },
                "elapsed_seconds": time.time() - started,
                "items": meta_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out_manifest}")
    print(f"Wrote {args.meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
