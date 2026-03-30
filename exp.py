from matplotlib import pyplot as plt
import numpy as np

def risk_probs(risk_scores: np.ndarray, tau_r: float = 0.1) -> np.ndarray:
    norm_negated_scores = max(risk_scores) - risk_scores / (max(risk_scores) - min(risk_scores) + 1e-9)
    a = np.exp(norm_negated_scores / tau_r)
    return a / np.sum(a)

def sample_shap_values() -> np.ndarray:
    np.random.seed(42)
    return np.random.uniform(-5, 5, size=(100, 10))

def uniform_shap_values() -> np.ndarray:
    return (np.arange(100) - 50.0) / 20.0 
    
if __name__ == "__main__":
    # shap_values = np.mean(sample_shap_values(), axis=1)
    shap_values = uniform_shap_values()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epsilon = 10
    selected_tau = 0.1
    taus = [0.025, 0.05, 0.1, 0.2, 0.4]
    all_probs = []
    for tau_r in taus:
        probs = risk_probs(shap_values, tau_r=tau_r)
        all_probs.append(probs)

    ax0 = axes[0]
    for tau_r, probs in zip(taus, all_probs):
        if tau_r == selected_tau:
            ax0.plot(shap_values, epsilon * len(shap_values) * probs, alpha=0.9, linewidth=3, label="τ=0.1 (selected)")
        else:
            ax0.plot(shap_values, epsilon * len(shap_values) * probs, alpha=0.7)
    ax0.set_facecolor((1.0, 1.0, 1.0))
    xlim = ax0.get_xlim()
    ylim = ax0.get_ylim()
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = (X - xlim[0]) / (xlim[1] - xlim[0])
    ax0.imshow(Z, extent=(xlim[0], xlim[1], ylim[0], ylim[1]), aspect='auto', origin='lower', cmap="RdYlGn_r", alpha=0.05)
    ax0.set_title("Privacy Budget vs. Risk Scores under Different Temperatures (ε=10)")
    ax0.set_xlabel("Token-level re-id. risk")
    ax0.set_ylabel("Token-level privacy budget")
    ax0.grid()
    ax0.legend([f"τ={tau} (selected)" if tau == selected_tau else f"τ={tau}" for tau in taus])

    taus_smaller = taus[::2]
    all_probs_smaller = all_probs[::2]
    cumprobs = [np.cumsum(probs) for probs in all_probs_smaller]
    rho_thresholds = np.arange(0.0, 1.1, 0.1)

    ax1 = axes[1]
    for tau_r, probs, cumprob in zip(taus_smaller, all_probs_smaller, cumprobs):
        ax1.plot(np.arange(1, len(cumprob) + 1), cumprob, label=f"τ={tau_r}")
        for rho in rho_thresholds:
            idx = np.searchsorted(cumprob, rho) + 1
            if idx > len(cumprob):
                continue
            ax1.axvline(idx, ymin=0, ymax=rho, color=ax1.lines[-1].get_color(), linestyle="--", alpha=0.2)
    ax1.set_xlabel("Top percentile of tokens")
    ax1.set_ylabel("Cumulative risk probability")
    ax1.set_title("Cumulative risk saturation curves")
    ax1.set_xlim(1, len(cumprobs[0]))
    ax1.set_xticklabels(f"{int(x) / 100:.1f}" for x in ax1.get_xticks())
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    plt.tight_layout()
    plt.show()
