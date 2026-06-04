"""
Regime-Shift · Step 2: HMM Regime Classifier (fast version)
Uses covariance_type='diag' and saves plot as PNG without plt.show()
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no popup window, just saves PNG
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ─────────────────────────────────────────────────────────────────────
N_STATES    = 3
N_ITER      = 200
RANDOM_SEED = 42
HMM_FEATURES = ["SPY", "TLT", "GLD", "VIX", "SPY_vol21", "TLT_vol21", "GLD_vol21"]

# ── LOAD ───────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("regime_features.csv", index_col=0, parse_dates=True)
print(f"Loaded {df.shape[0]} rows")

# ── SCALE ──────────────────────────────────────────────────────────────────────
print("Scaling features...")
scaler = StandardScaler()
X = scaler.fit_transform(df[HMM_FEATURES])

# ── FIT HMM ───────────────────────────────────────────────────────────────────
print("Fitting HMM... (may take 1-2 mins)")
model = GaussianHMM(
    n_components=N_STATES,
    covariance_type="diag",
    n_iter=N_ITER,
    random_state=RANDOM_SEED,
    verbose=False,
)
model.fit(X)
print(f"Converged: {model.monitor_.converged}")
print(f"Log-likelihood: {model.monitor_.history[-1]:.2f}")

# ── DECODE ─────────────────────────────────────────────────────────────────────
print("Decoding regimes...")
states = model.predict(X)

# ── LABEL STATES ──────────────────────────────────────────────────────────────
means_scaled = model.means_
means_orig   = scaler.inverse_transform(means_scaled)
means_df     = pd.DataFrame(means_orig, columns=HMM_FEATURES)

print("\n── State means (original scale) ───────────────")
print(means_df[["SPY", "VIX"]].round(4))

spy_means = means_df["SPY"].values
vix_means = means_df["VIX"].values
ranked    = np.argsort(spy_means)

labels = {}
labels[ranked[2]] = "Bull"
if vix_means[ranked[0]] > vix_means[ranked[1]]:
    labels[ranked[0]] = "Crisis"
    labels[ranked[1]] = "Bear"
else:
    labels[ranked[1]] = "Crisis"
    labels[ranked[0]] = "Bear"

print("\n── State labels ────────────────────────────────")
for state, name in labels.items():
    print(f"  State {state}: {name}  (SPY mean={spy_means[state]:.4f}, VIX mean={vix_means[state]:.2f})")

# ── TRANSITION MATRIX ─────────────────────────────────────────────────────────
state_names = [labels[i] for i in range(N_STATES)]
tmat = pd.DataFrame(model.transmat_, index=state_names, columns=state_names)
print("\n── Transition probabilities ────────────────────")
print(tmat.round(3))

# ── ATTACH TO DATAFRAME ───────────────────────────────────────────────────────
df["state"]  = states
df["regime"] = df["state"].map(labels)

print("\n── Regime distribution ─────────────────────────")
print(df["regime"].value_counts())

# ── PLOT (saved to PNG, no popup) ─────────────────────────────────────────────
print("\nSaving regime chart...")
REGIME_COLORS = {"Bull": "#2ecc71", "Bear": "#e67e22", "Crisis": "#e74c3c"}

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

spy_cum = np.exp(df["SPY"].cumsum())
axes[0].plot(df.index, spy_cum, color="black", lw=1.2, zorder=3)
axes[0].set_ylabel("SPY cumulative return")
axes[0].set_title("HMM Regime Detection overlaid on SPY")
axes[0].grid(alpha=0.3)

axes[1].plot(df.index, df["VIX"], color="crimson", lw=0.9, zorder=3)
axes[1].axhline(30, color="gray", ls="--", lw=0.7)
axes[1].set_ylabel("VIX level")
axes[1].grid(alpha=0.3)

prev_regime = df["regime"].iloc[0]
start_date  = df.index[0]
for date, regime in df["regime"].items():
    if regime != prev_regime:
        color = REGIME_COLORS[prev_regime]
        for ax in axes:
            ax.axvspan(start_date, date, alpha=0.15, color=color, zorder=1)
        start_date  = date
        prev_regime = regime
color = REGIME_COLORS[prev_regime]
for ax in axes:
    ax.axvspan(start_date, df.index[-1], alpha=0.15, color=color, zorder=1)

patches = [mpatches.Patch(color=c, alpha=0.4, label=r) for r, c in REGIME_COLORS.items()]
axes[0].legend(handles=patches, loc="upper left")

plt.tight_layout()
plt.savefig("regime_chart.png", dpi=150)
plt.close()
print("Chart saved to regime_chart.png")

# ── SAVE ──────────────────────────────────────────────────────────────────────
df.to_csv("regime_labels.csv")
print("Saved regime_labels.csv")
print("\nDone! Ready for Step 3 — portfolio optimizer.")