from matplotlib import pyplot as plt
import numpy as np

def risk_probs(risk_scores: np.ndarray, tau_r: float = 0.1) -> np.ndarray:
    norm_negated_scores = max(risk_scores) - risk_scores / (max(risk_scores) - min(risk_scores) + 1e-9)
    a = np.exp(norm_negated_scores / tau_r)
    return a / np.sum(a)

def sample_shap_values() -> np.ndarray:
    np.random.seed(42)
    return np.random.uniform(-5, 5, size=(100, 10))

def skewed_shap_values(n: int = 100, seed: int = 7) -> np.ndarray:
    """Simulate a more realistic risk-score spread: most tokens cluster at
    low/moderate risk, with a moderate high-risk tail, instead of a uniform ramp."""
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=0.0, scale=1.1, size=n)
    tail = rng.exponential(scale=0.9, size=n) * (rng.random(n) < 0.12)
    scores = base + tail
    lo, hi = np.percentile(scores, [2, 98])
    return np.sort(np.clip(scores, lo, hi))

if __name__ == "__main__":
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.edgecolor": "0.3",
        "axes.linewidth": 0.8,
    })

    shap_values = skewed_shap_values()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    fig.patch.set_facecolor("white")

    epsilon = 10
    selected_tau = 0.1
    taus = [0.025, 0.05, 0.1, 0.2, 0.4, 1.6]
    all_probs = [risk_probs(shap_values, tau_r=tau_r) for tau_r in taus]

    cmap = plt.get_cmap("plasma")
    tau_colors = {tau: cmap(0.05 + 0.85 * i / (len(taus) - 1)) for i, tau in enumerate(taus)}

    ax0 = axes[1]
    for tau_r, probs in zip(taus, all_probs):
        budget = epsilon * len(shap_values) * probs
        color = tau_colors[tau_r]
        if tau_r == selected_tau:
            ax0.plot(shap_values, budget, color="black", linewidth=3.5, zorder=5)
            ax0.plot(shap_values, budget, color=color, linewidth=2.2,
                      label=f"τ={tau_r} (*)", zorder=6)
            ax0.fill_between(shap_values, budget, alpha=0.15, color=color, zorder=1)
        else:
            ax0.plot(shap_values, budget, color=color, linewidth=1.8, alpha=0.85,
                      label=f"τ={tau_r}")

    ax0.set_title(f"Privacy Budget vs. Risk Score Under Different Temperatures (ε={epsilon})")
    ax0.set_xlabel("Token-level re-id. risk →")
    ax0.set_ylabel("Token-level privacy budget")
    ax0.grid(alpha=0.3, linestyle="--")
    ax0.set_facecolor("#fbfbfb")
    ax0.legend(frameon=False, loc="upper right")

    taus_smaller = [taus[0], taus[2], taus[4], taus[5]]
    all_probs_smaller = [all_probs[0], all_probs[2], all_probs[4], all_probs[5]]
    cumprobs = [np.cumsum(probs) for probs in all_probs_smaller]
    rho_thresholds = np.arange(0.0, 1.1, 0.1)

    ax1 = axes[0]
    for tau_r, cumprob in zip(taus_smaller, cumprobs):
        color = tau_colors[tau_r]
        x = np.arange(1, len(cumprob) + 1)
        is_selected = tau_r == selected_tau
        ax1.plot(x, cumprob, color=color, linewidth=3.0 if is_selected else 1.8,
                  alpha=1.0 if is_selected else 0.85,
                  label=f"τ={tau_r}" + (" (*)" if is_selected else ""))
        ax1.fill_between(x, cumprob, alpha=0.12 if is_selected else 0.05, color=color)
        for rho in rho_thresholds:
            idx = np.searchsorted(cumprob, rho) + 1
            if idx > len(cumprob):
                continue
            ax1.plot(idx, rho, marker="o", markersize=3, color=color, alpha=0.35)

    ax1.set_xlabel("Top percentile of tokens")
    ax1.set_ylabel("Cumulative risk probability")
    ax1.set_title("Cumulative Risk-Saturation Curves")
    ax1.set_xlim(1, len(cumprobs[0]))
    ax1.set_xticklabels(f"{int(x) / 100:.1f}" for x in ax1.get_xticks())
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.set_facecolor("#fbfbfb")
    ax1.legend(frameon=False, loc="lower right")

    plt.tight_layout()
    plt.savefig("docs/paper/latex/plots/risk_temperature.png", dpi=300, bbox_inches="tight")
    plt.show()
