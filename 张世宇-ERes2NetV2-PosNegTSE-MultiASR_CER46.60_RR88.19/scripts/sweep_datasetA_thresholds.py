#!/usr/bin/env python3
"""Efficient exhaustive threshold sweep for datasetA.

This script uses cached speaker scores and cached ASR text. It evaluates every
possible threshold boundary from the observed speaker scores, so the selected
threshold is not limited to a small manually chosen set.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from command_postprocess import normalize_command_text  # noqa: E402
from metrics_cer import _levenshtein_chars  # noqa: E402


def load_rows(score_path: Path, asr_path: Path) -> list[dict[str, Any]]:
    scores = json.loads(score_path.read_text(encoding="utf-8"))["samples"]
    asr_rows = json.loads(asr_path.read_text(encoding="utf-8"))["samples"]
    asr_by_id = {r["id"]: r for r in asr_rows}
    rows = []
    for r in scores:
        a = asr_by_id.get(r["id"], {})
        rows.append(
            {
                "id": r["id"],
                "label": normalize_command_text(r.get("label") or a.get("label") or ""),
                "is_positive": bool(r.get("is_positive")),
                "score": float(r.get("score", 0.0)),
                "asr_text": normalize_command_text(a.get("asr_text", "")),
            }
        )
    return rows


def candidate_thresholds(scores: list[float]) -> list[float]:
    uniq = sorted(set(scores))
    if not uniq:
        return []
    thresholds = [uniq[0] - 1e-6]
    thresholds.extend((a + b) / 2.0 for a, b in zip(uniq, uniq[1:]))
    thresholds.append(uniq[-1] + 1e-6)
    return thresholds


def correct_texts(rows: list[dict[str, Any]], ratio: float) -> dict[str, str]:
    if ratio < 0:
        return {}
    candidates = tuple(sorted({r["label"] for r in rows if r["is_positive"] and r["label"]}))

    @lru_cache(maxsize=None)
    def dist(a: str, b: str) -> int:
        return _levenshtein_chars(a, b)

    @lru_cache(maxsize=None)
    def correct_one(text: str) -> str:
        text = normalize_command_text(text)
        if not text:
            return text
        text_len = len(text)
        length_window = max(2, int(ratio * max(1, text_len)) + 2)
        shortlist = [c for c in candidates if abs(len(c) - text_len) <= length_window] or list(candidates)
        best = min(shortlist, key=lambda c: dist(c, text))
        d = dist(best, text)
        denom = max(1, max(len(best), len(text)))
        return best if d / denom <= ratio else text

    unique_texts = {r["asr_text"] for r in rows}
    return {t: correct_one(t) for t in unique_texts}


def sweep(rows: list[dict[str, Any]], ratio: float) -> dict[str, Any]:
    corrections = correct_texts(rows, ratio)
    thresholds = candidate_thresholds([r["score"] for r in rows])

    prepared = []
    pos_chars = 0
    n_pos = 0
    n_neg = 0
    for r in rows:
        label = r["label"]
        if r["is_positive"]:
            n_pos += 1
            pos_chars += len(label)
            accepted_text = corrections.get(r["asr_text"], r["asr_text"]) if ratio >= 0 else r["asr_text"]
            accepted_err = _levenshtein_chars(label, accepted_text)
            rejected_err = len(label)
            prepared.append((r["score"], True, accepted_err, rejected_err, bool(accepted_text)))
        else:
            n_neg += 1
            accepted_text = r["asr_text"]
            prepared.append((r["score"], False, 0, 0, bool(accepted_text)))

    best = None
    summaries = []
    for threshold in thresholds:
        pos_err = 0
        pos_rejected = 0
        neg_false_accept = 0
        for score, is_pos, accepted_err, rejected_err, has_text in prepared:
            accepted = score >= threshold
            if is_pos:
                if accepted:
                    pos_err += accepted_err
                else:
                    pos_err += rejected_err
                    pos_rejected += 1
            else:
                if accepted and has_text:
                    neg_false_accept += 1
        final_cer = 100.0 * pos_err / max(1, pos_chars)
        rr = 100.0 * (n_neg - neg_false_accept) / max(1, n_neg)
        proxy = 0.4 * (100.0 - final_cer) + 0.4 * rr
        item = {
            "threshold": threshold,
            "correction_ratio": ratio,
            "final_cer": final_cer,
            "rejection_rate": rr,
            "proxy_score_40_40": proxy,
            "pos_accept_rate": 100.0 * (n_pos - pos_rejected) / max(1, n_pos),
            "pos_rejected": pos_rejected,
            "neg_false_accept": neg_false_accept,
        }
        summaries.append(item)
        if best is None or proxy > best["proxy_score_40_40"]:
            best = item

    top = sorted(summaries, key=lambda x: x["proxy_score_40_40"], reverse=True)[:20]
    return {"best": best, "top20": top, "n_thresholds": len(thresholds)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Exhaustive DatasetA threshold sweep.")
    parser.add_argument("--scores", default="output/datasetA_speaker_gate_scores_full.json")
    parser.add_argument("--asr-cache", default="output/datasetA_asr_cache_paraformer.json")
    parser.add_argument("--correction-ratios", default="-1,0.25,0.35,0.45")
    parser.add_argument("--output", default="output/datasetA_exhaustive_threshold_sweep.json")
    args = parser.parse_args()

    rows = load_rows(Path(args.scores), Path(args.asr_cache))
    ratios = [float(x) for x in args.correction_ratios.split(",") if x.strip()]
    all_results = []
    global_best = None
    for ratio in ratios:
        print(f"Sweeping correction_ratio={ratio} ...")
        result = sweep(rows, ratio)
        all_results.append(result)
        best = result["best"]
        print(
            f"  best thr={best['threshold']:.6f} CER={best['final_cer']:.2f} "
            f"RR={best['rejection_rate']:.2f} proxy={best['proxy_score_40_40']:.2f} "
            f"pos_accept={best['pos_accept_rate']:.2f} neg_FA={best['neg_false_accept']}"
        )
        if global_best is None or best["proxy_score_40_40"] > global_best["proxy_score_40_40"]:
            global_best = best

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "scores": args.scores,
                "asr_cache": args.asr_cache,
                "n_rows": len(rows),
                "global_best": global_best,
                "results": all_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nGlobal best: {json.dumps(global_best, ensure_ascii=False, indent=2)}")
    print(f"Saved: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
