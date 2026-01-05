#!/usr/bin/env python3
"""Plot privacy/divergence/utility metrics with anonymization-aware coloring."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

CACHE_ROOT = Path(".cache")
MPL_ROOT = Path(".matplotlib")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
MPL_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT.resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_ROOT.resolve()))

import matplotlib

matplotlib.use("macosx")

from matplotlib import pyplot as plt
import numpy as np
import yaml


def read_sample_risk(filepath: Path, idx: int) -> np.ndarray:
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            if i == idx:
                data = json.loads(line)
                scores = data["scores"]
                if isinstance(scores, dict):
                    return np.array(list(scores.values()))
                return np.array(scores)
    raise IndexError("Sample index out of range")

def samples_to_probs(samples: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.exp(samples / temperature)
    scaled /= np.sum(scaled)
    return scaled

def plot_risk_distribution(samples: np.ndarray, temperature: float, thresholds: List[float], ax: plt.Axes) -> None:
    samples_prop = (samples - np.min(samples)) / (np.max(samples) - np.min(samples) + 1e-10)
    samples_disprop = (np.max(samples) - samples) / (np.max(samples) - np.min(samples) + 1e-10)
    probs_prop = samples_to_probs(samples_prop, temperature)
    probs_disprop = samples_to_probs(samples_disprop, temperature)
    sorted_idx_prop = np.argsort(probs_prop)[::-1]
    sorted_idx_disprop = np.argsort(probs_disprop)[::-1]
    sorted_probs_prop = probs_prop[sorted_idx_prop]
    sorted_probs_disprop = probs_disprop[sorted_idx_disprop]
    cumsum_prop = np.cumsum(sorted_probs_prop)
    ax.plot(sorted_probs_disprop, label="Privacy Budget", color="green")
    ax.plot(sorted_probs_prop, label="Risk probability", color="blue")
    ax2 = ax.twinx()
    ax2.plot(cumsum_prop, label="Cumulative Risk Probability", color="orange")
    colors = plt.cm.viridis(np.linspace(0, 1, len(thresholds)))
    for threshold, color in zip(thresholds, colors):
        idx = np.searchsorted(cumsum_prop, 1 - threshold)
        ax.axvline(idx, linestyle="--", label=f"Threshold {threshold:.2f}", color=color)
    ax.set_title(f"Temperature: {temperature}")
    ax.set_xlabel("Sorted Sample Index")
    ax.set_ylabel("Probability")
    ax2.set_ylabel("Cumulative Risk Probability")
    ax.legend(loc="upper right")
    ax2.legend(loc="lower right")

def load_config(config_path: Path) -> Dict[str, object]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config.exists() else {}
    temps = cfg.get("temperatures", [0.8, 0.9])
    thresholds = cfg.get("thresholds", [0.8])

    samples = read_sample_risk(args.file, args.idx)
    print(samples)
    fig, axes = plt.subplots(1, len(temps), figsize=(5 * len(temps), 4), squeeze=False)
    for i, temp in enumerate(temps):
        plot_risk_distribution(samples, float(temp), list(thresholds), axes[0, i])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
