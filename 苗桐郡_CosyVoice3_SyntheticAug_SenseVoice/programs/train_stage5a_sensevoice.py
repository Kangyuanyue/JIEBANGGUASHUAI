from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from funasr import AutoModel
from funasr.download.download_model_from_hub import download_model
from funasr.register import tables


SUPPORT_FILES = (
    "config.yaml",
    "configuration.json",
    "chn_jpn_yue_eng_ko_spectok.bpe.model",
    "am.mvn",
    "tokens.json",
)


def resolve_model_reference(value: str) -> str:
    path = Path(value).expanduser()
    return str(path.resolve()) if path.exists() else value


def freeze_parameters(model: torch.nn.Module, prefixes: list[str]) -> None:
    for name, parameter in model.named_parameters():
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            parameter.requires_grad = False


def unfreeze_parameters(model: torch.nn.Module, prefixes: list[str]) -> None:
    for name, parameter in model.named_parameters():
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            parameter.requires_grad = True


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def scalar_stats(stats: dict[str, Any]) -> dict[str, float]:
    result = {}
    for key, value in stats.items():
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            result[key] = float(value.detach().cpu())
        elif isinstance(value, (int, float)):
            result[key] = float(value)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(args.out).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty training directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Stage 5A requires the isolated CUDA environment")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    resolved_config = download_model(
        is_training=True,
        model=resolve_model_reference(config["base_model"]),
        device=str(device),
        train_data_set_list=str(Path(args.train_manifest).resolve()),
    )
    wrapper = AutoModel(**resolved_config)
    model = wrapper.model
    if args.init_checkpoint:
        state_dict = torch.load(Path(args.init_checkpoint) / "model.pt", map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        del state_dict
        torch.cuda.empty_cache()
    freeze_parameters(model, list(config["freeze_prefixes"]))
    unfreeze_parameters(model, list(config.get("unfreeze_prefixes", [])))
    model.train()
    model.encoder.eval()

    kwargs = wrapper.kwargs
    dataset_class = tables.dataset_classes.get(kwargs.get("dataset", "SenseVoiceCTCDataset"))
    dataset_config = dict(kwargs.get("dataset_conf", {}))
    dataset_config.update({
        "data_split_num": 1,
        "batch_size": 400,
        "batch_type": "token",
        "max_token_length": 2000,
        "min_token_length": 1,
        "max_source_length": 2000,
        "min_source_length": 1,
        "max_target_length": 200,
        "min_target_length": 0,
        "shuffle": False,
        "num_workers": 0,
        "retry": 3,
    })
    dataset = dataset_class(
        str(Path(args.train_manifest).resolve()),
        frontend=kwargs.get("frontend"),
        tokenizer=kwargs.get("tokenizer"),
        is_training=True,
        **dataset_config,
    )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=float(config["learning_rate"]))
    precision = str(config.get("precision", "fp16" if config.get("fp16", False) else "fp32")).lower()
    if precision not in {"fp16", "bf16", "fp32"}:
        raise ValueError(f"Unsupported training precision: {precision}")
    use_autocast = precision in {"fp16", "bf16"}
    autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    accumulation = int(config["gradient_accumulation"])
    losses: list[dict[str, Any]] = []
    optimizer_steps = 0
    skipped = 0
    nonfinite_samples: list[dict[str, Any]] = []
    valid_since_step = 0
    optimizer.zero_grad(set_to_none=True)

    indices = list(range(len(dataset)))
    if bool(config.get("shuffle_training", True)):
        random.Random(seed).shuffle(indices)
    for epoch in range(int(config["epochs"])):
        for position, sample_index in enumerate(indices, start=1):
            sample = dataset[sample_index]
            if sample is None:
                skipped += 1
                continue
            batch = move_batch(dataset.collator([sample]), device)
            with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=use_autocast):
                loss, stats, _ = model(**batch)
                scaled_loss = loss / accumulation
            if not torch.isfinite(loss):
                nonfinite_samples.append({
                    "sample_position": position,
                    "sample_index": sample_index,
                    "reason": "nonfinite_ctc_loss",
                })
                del scaled_loss, loss, stats, batch, sample
                torch.cuda.empty_cache()
                continue
            scaler.scale(scaled_loss).backward()
            valid_since_step += 1
            should_step = valid_since_step == accumulation
            if should_step:
                scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 5.0).detach().cpu())
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                valid_since_step = 0
            else:
                grad_norm = None
            losses.append({
                "epoch": epoch + 1,
                "sample_position": position,
                "sample_index": sample_index,
                "loss": float(loss.detach().cpu()),
                "stats": scalar_stats(stats),
                "optimizer_step": should_step,
                "gradient_norm": grad_norm,
            })
            del scaled_loss, loss, stats, batch, sample
            if should_step:
                torch.cuda.empty_cache()

    if valid_since_step:
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 5.0).detach().cpu())
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        losses[-1]["optimizer_step"] = True
        losses[-1]["gradient_norm"] = grad_norm

    source_model = Path(config["base_model"])
    for name in SUPPORT_FILES:
        source = source_model / name
        if source.exists():
            shutil.copy2(source, checkpoint_dir / name)
    torch.save(model.state_dict(), checkpoint_dir / "model.pt")
    elapsed = time.perf_counter() - started
    result = {
        "route": args.route,
        "init_checkpoint": str(Path(args.init_checkpoint).resolve()) if args.init_checkpoint else None,
        "train_manifest": str(Path(args.train_manifest).resolve()),
        "dataset_length": len(dataset),
        "epochs": int(config["epochs"]),
        "batch_size": int(config["batch_size"]),
        "gradient_accumulation": accumulation,
        "optimizer_steps": optimizer_steps,
        "learning_rate": float(config["learning_rate"]),
        "fp16": precision == "fp16",
        "precision": precision,
        "frozen_prefixes": config["freeze_prefixes"],
        "unfrozen_prefixes": config.get("unfreeze_prefixes", []),
        "shuffle_training": bool(config.get("shuffle_training", True)),
        "total_parameters": total_count,
        "trainable_parameters": trainable_count,
        "trainable_ratio": trainable_count / total_count,
        "skipped_samples": skipped,
        "nonfinite_sample_count": len(nonfinite_samples),
        "nonfinite_samples": nonfinite_samples,
        "initial_loss": losses[0]["loss"],
        "final_loss": losses[-1]["loss"],
        "mean_loss": sum(item["loss"] for item in losses) / len(losses),
        "elapsed_seconds": elapsed,
        "samples_per_second": len(losses) / elapsed,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_saved": (checkpoint_dir / "model.pt").exists(),
        "loss_curve": losses,
    }
    (out_dir / "train_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one frozen-encoder SenseVoice Stage 5A smoke epoch.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--route", required=True, choices=("real_only", "real_plus_synthetic"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--init-checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({key: value for key, value in result.items() if key != "loss_curve"}, ensure_ascii=False, indent=2))
