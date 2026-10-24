"""Evaluate the zero-parameter local/global Native3D contrast on DND21."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .core import NativeStateBank, StateConfig, bidirectional_features
from .data import iter_labeled_text
from .evaluate_dnd21 import PAPER_AUC, SCENE_FILES
from .analytic import analytic_contrast_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--edmamba-results", type=Path, default=None)
    parser.add_argument("--rates", nargs="+", default=list(PAPER_AUC))
    parser.add_argument("--scenes", nargs="+", default=list(SCENE_FILES))
    parser.add_argument("--block-size", type=int, default=10_240)
    parser.add_argument("--max-blocks", type=int, default=None)
    return parser.parse_args()


def evaluate_stream(path, config, block_size, max_blocks):
    block_aucs = {"float": [], "fixed_q12_8": [], "fixed_q8_4": []}
    events = 0
    started = time.perf_counter()
    for values in iter_labeled_text(path, block_size):
        if max_blocks is not None and len(block_aucs["float"]) >= max_blocks:
            break
        if len(values) != block_size:
            break
        raw = bidirectional_features(values[:, :4], config)
        labels = values[:, 4]
        scores = {
            "float": analytic_contrast_score(raw, config),
            "fixed_q12_8": analytic_contrast_score(raw, config, "q12.8"),
            "fixed_q8_4": analytic_contrast_score(raw, config, "q8.4"),
        }
        for identifier, score in scores.items():
            block_aucs[identifier].append(float(roc_auc_score(labels, score)))
        events += len(labels)
    if not block_aucs["float"]:
        raise RuntimeError(f"No complete blocks found in {path}")
    return {
        "source": str(path.resolve()),
        "blocks": len(block_aucs["float"]),
        "events": events,
        "wall_seconds": time.perf_counter() - started,
        "candidates": {
            identifier: {
                "mean_block_auc": float(np.mean(values)),
                "std_block_auc": (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                ),
                "block_aucs": [float(value) for value in values],
            }
            for identifier, values in block_aucs.items()
        },
    }


def paired_cluster_bootstrap(
    paired_differences, seed=1024, replicates=10_000, cluster_blocks=10
):
    """Condition-stratified contiguous-block cluster bootstrap of paired AUCs."""

    observed = float(np.mean([np.mean(values) for values in paired_differences]))
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    clusters_by_condition = []
    for values in paired_differences:
        values = np.asarray(values, dtype=np.float64)
        clusters = []
        for start in range(0, len(values), cluster_blocks):
            chunk = values[start : start + cluster_blocks]
            clusters.append((float(chunk.sum()), len(chunk)))
        clusters_by_condition.append(clusters)
    for replicate in range(replicates):
        condition_means = []
        for clusters in clusters_by_condition:
            indices = rng.integers(0, len(clusters), size=len(clusters))
            total = sum(clusters[index][0] for index in indices)
            count = sum(clusters[index][1] for index in indices)
            condition_means.append(total / count)
        bootstrap[replicate] = np.mean(condition_means)
    return {
        "observed_paired_mean_delta": observed,
        "95ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "bootstrap_replicates": replicates,
        "cluster_blocks": cluster_blocks,
        "cluster_events": cluster_blocks * 10_240,
        "stratification": "unweighted mean of ten condition-level paired deltas",
    }


def summarize(results, edmamba):
    summary = {}
    for candidate in ("float", "fixed_q12_8", "fixed_q8_4"):
        values = []
        variances = []
        paper_targets = []
        reproduced_targets = []
        paired_differences = []
        paper_wins = 0
        reproduced_wins = 0
        for rate, scenes in results.items():
            for scene, stream in scenes.items():
                metrics = stream["candidates"][candidate]
                value = metrics["mean_block_auc"]
                values.append(value)
                variances.append(
                    metrics["std_block_auc"] ** 2 / stream["blocks"]
                )
                paper = PAPER_AUC[rate][scene]
                paper_targets.append(paper)
                paper_wins += value > paper
                if edmamba is not None:
                    reproduced_metrics = edmamba[rate][scene]
                    reproduced = reproduced_metrics["mean_block_auc"]
                    reproduced_targets.append(reproduced)
                    reproduced_wins += value > reproduced
                    if "block_aucs" in metrics and "block_aucs" in reproduced_metrics:
                        candidate_blocks = np.asarray(metrics["block_aucs"])
                        reproduced_blocks = np.asarray(reproduced_metrics["block_aucs"])
                        if candidate_blocks.shape != reproduced_blocks.shape:
                            raise ValueError("Paired DND21 block arrays have different shapes")
                        paired_differences.append(candidate_blocks - reproduced_blocks)
        mean = float(np.mean(values))
        standard_error = math.sqrt(sum(variances)) / len(values)
        record = {
            "mean_auc": mean,
            "normal_approximation_standard_error": standard_error,
            "normal_approximation_95ci": [
                mean - 1.96 * standard_error,
                mean + 1.96 * standard_error,
            ],
            "paper_mean_auc": float(np.mean(paper_targets)),
            "delta_vs_paper_mean": mean - float(np.mean(paper_targets)),
            "conditions_beating_paper": paper_wins,
            "conditions": len(values),
        }
        if reproduced_targets:
            reproduced_mean = float(np.mean(reproduced_targets))
            record.update(
                {
                    "reproduced_edmamba_mean_auc": reproduced_mean,
                    "delta_vs_reproduced_edmamba_mean": mean - reproduced_mean,
                    "delta_vs_reproduced_edmamba_95ci_fixed_baseline": [
                        mean - reproduced_mean - 1.96 * standard_error,
                        mean - reproduced_mean + 1.96 * standard_error,
                    ],
                    "conditions_beating_reproduced_edmamba": reproduced_wins,
                }
            )
            if len(paired_differences) == len(values):
                record["paired_cluster_bootstrap"] = paired_cluster_bootstrap(
                    paired_differences
                )
        summary[candidate] = record
    return summary


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    config = StateConfig()
    edmamba = None
    if args.edmamba_results is not None:
        edmamba = json.loads(args.edmamba_results.read_text())["results"]
    payload = {
        "method": "bidirectional 250-us same-polarity local/global log contrast",
        "parameters": 0,
        "state_config": config.to_dict(),
        "state_bytes_float32": NativeStateBank(config).state_bytes,
        "protocol": {
            "block_size": args.block_size,
            "tail_policy": "drop incomplete final block",
            "fixed_point": {
                "fixed_q12_8": "unsigned 12-bit, 8 fractional bits, saturating",
                "fixed_q8_4": "unsigned 8-bit, 4 fractional bits, saturating",
            },
        },
        "results": {},
    }
    for rate in args.rates:
        payload["results"][rate] = {}
        for scene in args.scenes:
            stream = evaluate_stream(
                args.dataset_root / rate / SCENE_FILES[scene],
                config,
                args.block_size,
                args.max_blocks,
            )
            target = PAPER_AUC[rate][scene]
            stream["edmamba_paper_auc"] = target
            for metrics in stream["candidates"].values():
                metrics["delta_vs_edmamba_paper"] = (
                    metrics["mean_block_auc"] - target
                )
            payload["results"][rate][scene] = stream
            payload["summary"] = summarize(payload["results"], edmamba)
            _write(args.output, payload)
            print(
                f"RESULT {rate} {scene}: "
                f"float={stream['candidates']['float']['mean_block_auc']:.6f} "
                f"q12.8={stream['candidates']['fixed_q12_8']['mean_block_auc']:.6f} "
                f"q8.4={stream['candidates']['fixed_q8_4']['mean_block_auc']:.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
