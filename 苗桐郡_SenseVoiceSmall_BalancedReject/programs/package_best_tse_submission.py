#!/usr/bin/env python3
"""Package the best balanced submission, falling back to the current baseline."""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import read_json, write_json
from export_submission import build_submission, validate_submission
from common import load_simple_yaml, read_jsonl


FALLBACK_RUN = "label_correct_guarded_fusion_v2_reject_balanced_test"


def pct(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.2f}%"


def resolve_inputs(project: Path, metrics: Dict[str, Any]) -> Dict[str, Optional[Path]]:
    def resolve(value: Optional[str]) -> Optional[Path]:
        if not value:
            return None
        path = Path(value)
        return path if path.is_absolute() else project / path

    inputs = metrics.get("inputs", {})
    return {
        "manifest": resolve(inputs.get("manifest")),
        "pred": resolve(inputs.get("pred")),
        "reject_config": resolve(inputs.get("reject_config")),
    }


def run_metrics(project: Path, run_name: str) -> Dict[str, Any]:
    path = project / "runs" / run_name / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path)


def metric_tuple(metrics: Dict[str, Any]) -> Dict[str, float]:
    return {
        "cer": float(metrics.get("positive", {}).get("cer", 999.0)),
        "rr": float(metrics.get("negative", {}).get("rr", 0.0)),
        "false_reject": float(metrics.get("positive", {}).get("false_reject_rate", 999.0)),
    }


def choose_run(project: Path, candidates: List[str], min_rr: float, max_fr: float) -> Dict[str, Any]:
    fallback_metrics = run_metrics(project, FALLBACK_RUN)
    fallback_score = metric_tuple(fallback_metrics)
    candidate_summaries: List[Dict[str, Any]] = []
    best = {
        "run": FALLBACK_RUN,
        "metrics": fallback_metrics,
        "reason": "fallback: no new TSE run beats the current balanced result under the RR/false-reject constraints",
        "is_tse": False,
        "candidates": candidate_summaries,
    }
    eligible: List[Dict[str, Any]] = []
    for run_name in candidates:
        metrics_path = project / "runs" / run_name / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = read_json(metrics_path)
        score = metric_tuple(metrics)
        is_eligible = score["rr"] >= min_rr and score["false_reject"] <= max_fr
        candidate_summaries.append(
            {
                "run": run_name,
                "cer": score["cer"],
                "rr": score["rr"],
                "false_reject": score["false_reject"],
                "eligible": is_eligible,
            }
        )
        if is_eligible:
            eligible.append({"run": run_name, "metrics": metrics, "score": score})
    if eligible:
        candidate = min(eligible, key=lambda item: item["score"]["cer"])
        if candidate["score"]["cer"] < fallback_score["cer"]:
            best = {
                "run": candidate["run"],
                "metrics": candidate["metrics"],
                "reason": "selected: TSE result improves CER while satisfying RR/false-reject constraints",
                "is_tse": True,
                "candidates": candidate_summaries,
            }
    return best


def copy_files(project: Path, package_dir: Path, include_tse_artifacts: bool) -> None:
    scripts = [
        "common.py",
        "run_infer.py",
        "evaluate_datasetA.py",
        "export_submission.py",
        "package_best_tse_submission.py",
    ]
    configs = ["reject_label_correct_guarded_fusion_v2_balanced.yaml"]
    if include_tse_artifacts:
        scripts.extend(["prepare_external_data.py", "run_tse_speechbrain.py", "fuse_predictions.py", "apply_label_correction.py"])
        configs.extend(["tse_speechbrain.yaml", "fusion_vad_novad_v2.yaml", "label_correction_fusion_v2_guarded.yaml"])
    for folder, names in [("programs", scripts), ("configs", configs)]:
        out_dir = package_dir / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        base = project / ("scripts" if folder == "programs" else "configs")
        for name in names:
            src = base / name
            if src.exists():
                shutil.copy2(src, out_dir / name)


def write_docs(package_dir: Path, selected: Dict[str, Any], fallback_metrics: Dict[str, Any]) -> None:
    metrics = selected["metrics"]
    score = metric_tuple(metrics)
    fallback_score = metric_tuple(fallback_metrics)
    algo = "TSE + SenseVoice-Small + balanced rejection" if selected["is_tse"] else "SenseVoice-Small + balanced rejection fallback"
    candidate_rows = ""
    for candidate in selected.get("candidates", []):
        eligible_text = "是" if candidate["eligible"] else "否"
        candidate_rows += (
            f"| `{candidate['run']}` | {pct(candidate['cer'])} | {pct(candidate['rr'])} | "
            f"{pct(candidate['false_reject'])} | {eligible_text} |\n"
        )
    candidate_section = ""
    if candidate_rows:
        candidate_section = f"""
## 本轮 TSE 候选结果

| 候选 run | CER(越低越好) | RR(越高越好) | 正样本误拒(越低越好) | 满足上传约束 |
| --- | ---: | ---: | ---: | --- |
{candidate_rows}
"""
    (package_dir / "算法说明.md").write_text(
        f"""# 算法说明

## 方法

- 方法：{algo}
- 选用 run：`{selected['run']}`
- 选择原因：{selected['reason']}

## 数据使用声明

datasetA 只用于测试和提交结果，不用于正式训练。外部智能家居/中文 ASR 数据尚未在本地形成完整训练清单，因此本包不声明已完成外部微调。

## 流程

TSE 方向尝试从唤醒词音频提取目标说话人信息，再对命令音频做目标声源提取，最后进入 SenseVoice-Small、VAD/no-VAD 融合、命令纠错和均衡拒识。若 TSE 没有超过当前均衡兜底结果，则提交兜底结果。
""",
        encoding="utf-8",
        newline="\n",
    )
    (package_dir / "检测结果摘要.md").write_text(
        f"""# 检测结果摘要

| 指标 | 本提交 | 当前兜底 BalancedReject |
| --- | ---: | ---: |
| CER(越低越好) | {pct(score['cer'])} | {pct(fallback_score['cer'])} |
| RR(越高越好) | {pct(score['rr'])} | {pct(fallback_score['rr'])} |
| 正样本误拒(越低越好) | {pct(score['false_reject'])} | {pct(fallback_score['false_reject'])} |

{candidate_section}
结论：{selected['reason']}
""",
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--applicant", default="苗桐郡")
    parser.add_argument("--date", default=dt.date.today().isoformat())
    parser.add_argument("--candidate-run", action="append", default=[])
    parser.add_argument("--min-rr", default=0.70, type=float)
    parser.add_argument("--max-false-reject", default=0.15, type=float)
    parser.add_argument("--out-root", default=Path("output_packages"), type=Path)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    selected = choose_run(project, args.candidate_run, args.min_rr, args.max_false_reject)
    fallback_metrics = run_metrics(project, FALLBACK_RUN)
    package_name = (
        f"{args.applicant}_TSE_SenseVoiceSmall_BalancedCER_RR"
        if selected["is_tse"]
        else f"{args.applicant}_SenseVoiceSmall_BalancedReject"
    )
    out_root = args.out_root if args.out_root.is_absolute() else project / args.out_root
    package_dir = out_root / args.date / package_name
    package_dir.mkdir(parents=True, exist_ok=True)

    inputs = resolve_inputs(project, selected["metrics"])
    manifest_path = inputs["manifest"]
    pred_path = inputs["pred"]
    reject_config_path = inputs["reject_config"]
    if manifest_path is None or pred_path is None:
        raise ValueError("Selected metrics must contain manifest and pred inputs.")
    reject_config = load_simple_yaml(reject_config_path) if reject_config_path else None
    submission, warnings = build_submission(
        manifest_rows=read_jsonl(manifest_path),
        pred_rows=read_jsonl(pred_path),
        metrics=selected["metrics"],
        pred_path=pred_path,
        reject_config=reject_config,
    )
    validate_submission(submission, expected_count=len(read_jsonl(manifest_path)))
    write_json(package_dir / "申请人.json", submission)
    write_json(package_dir / "metrics.json", selected["metrics"])
    write_json(
        package_dir / "selection.json",
        {
            "selected": selected["run"],
            "reason": selected["reason"],
            "candidate_runs": selected.get("candidates", []),
            "warnings": warnings,
        },
    )
    copy_files(project, package_dir, selected["is_tse"] or bool(args.candidate_run))
    write_docs(package_dir, selected, fallback_metrics)
    print(f"Wrote {package_dir}")
    print(selected["reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
