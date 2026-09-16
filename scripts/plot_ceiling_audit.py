"""Export descriptive audit figures and their exact source table (no training)."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mechanism-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events = EventAccumulator(str(args.run_dir / "summaries"), size_guidance={"scalars": 0})
    events.Reload()
    blocks = [{int(e.step): float(e.value) for e in events.Scalars(
        "successes_per_block/block_{}/frame".format(i))} for i in range(64)]
    steps = np.array(sorted(set.intersection(*(set(b) for b in blocks))))
    values = np.array([[b[int(step)] for b in blocks] for step in steps])
    means = values.mean(axis=1)
    with (args.output_dir / "training_curve.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "population_cached_mean", "master_cached_mean"])
        writer.writerows(zip(steps.tolist(), means.tolist(), values[:, -1].tolist()))
    comparisons = json.loads(args.mechanism_results.read_text())["comparisons"]
    with (args.output_dir / "optimizer_comparisons.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    ax.plot(steps / 1e9, values[:, -1], color="#999999", alpha=0.7, label="Master (cached training metric)")
    ax.plot(steps / 1e9, means, color="#1d3557", label="64-block mean (cached)")
    ax.axhline(50, color="black", linestyle=":", label="Episode counter cap, not proven optimum")
    late = steps >= 4e9
    slope, intercept = np.polyfit(steps[late] / 1e9, means[late], 1)
    ax.plot(steps[late] / 1e9, slope * steps[late] / 1e9 + intercept,
            color="#ba4a32", linewidth=2, label="Late descriptive slope: {:.2f} / B".format(slope))
    ax.set(xlabel="Environment frames (billions)", ylabel="Success count", ylim=(-1, 54),
           title="E51: finite training budget, one seed")
    ax.legend(fontsize=7.5, loc="upper left")
    ax = axes[1]
    for history, color, label in (("fresh_adam", "#1d3557", "Fresh Adam update"),
                                  ("matched_moments", "#ba4a32", "Adam with matched moment history")):
        rows = sorted((r for r in comparisons if r["history"] == history and r["step_in_arm"] == 0),
                      key=lambda r: r["amp_scale"])
        ax.semilogx([r["amp_scale"] for r in rows],
                    [r["legacy_over_candidate_parameter_delta"] for r in rows], "o-", color=color, label=label)
    rows = sorted((r for r in comparisons if r["history"] == "fresh_adam" and r["step_in_arm"] == 0),
                  key=lambda r: r["amp_scale"])
    ax.semilogx([r["amp_scale"] for r in rows], [r["legacy_over_candidate_shared_gradient"] for r in rows],
                "s--", color="#888888", label="Shared gradient after clipping")
    ax.set(xlabel="AMP scale (log scale)", ylabel="Legacy / optimizer-only clipping", ylim=(-0.04, 1.1),
           title="CUDA mechanism: gradient ratio differs from Adam update")
    ax.legend(fontsize=8, loc="lower left")
    fig.text(0.5, 0.01, "Descriptive diagnostics only. No multi-seed performance claim; toy optimizer experiment is not RL training.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(str(args.output_dir / "ceiling_audit.png"), dpi=180)
    fig.savefig(str(args.output_dir / "ceiling_audit.pdf"))
    print(args.output_dir / "ceiling_audit.png")


if __name__ == "__main__":
    main()
