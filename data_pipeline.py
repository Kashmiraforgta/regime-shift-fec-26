"""
Regime-Shift · Step 1: Data Pipeline
Pulls SPY, TLT, GLD daily returns + VIX from Yahoo Finance.
Output: a clean DataFrame ready for HMM feature engineering.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ── 1. CONFIG ──────────────────────────────────────────────────────────────────

ASSETS   = ["SPY", "TLT", "GLD"]   # equities, bonds, gold
VIX      = "^VIX"                   # CBOE Volatility Index
START    = "2005-01-01"             # covers multiple regimes (GFC, COVID, etc.)
END      = "2024-12-31"


# ── 2. DOWNLOAD PRICES ────────────────────────────────────────────────────────

def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted closing prices for a list of tickers.
    yfinance returns a multi-level column DataFrame; we flatten it.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance returns ("Close", "SPY"), ("Close", "TLT"), ... when multi-ticker
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # single ticker edge-case
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    print(f"Downloaded {prices.shape[1]} tickers, {prices.shape[0]} trading days.")
    return prices


def download_vix(start: str, end: str) -> pd.Series:
    """
    VIX level (not returns) — used as a raw fear gauge feature for the HMM.
    We keep the level, not daily returns, because VIX magnitude matters.
    """
    raw = yf.download(VIX, start=start, end=end, auto_adjust=True, progress=False)
    vix = raw["Close"].squeeze()          # squeeze to Series
    vix.name = "VIX"
    print(f"Downloaded VIX: {vix.shape[0]} trading days.")
    return vix


# ── 3. COMPUTE RETURNS ────────────────────────────────────────────────────────

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Log returns: r_t = ln(P_t / P_{t-1})
    Why log? Additive across time, better distributional properties for HMM.
    """
    returns = np.log(prices / prices.shift(1))
    returns.dropna(inplace=True)          # drop the first NaN row
    return returns


# ── 4. CLEAN & ALIGN ──────────────────────────────────────────────────────────

def build_feature_matrix(returns: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    """
    Align asset returns with VIX on a common date index.
    Also adds a rolling 21-day volatility feature per asset (useful for HMM).
    """
    # inner join: keep only dates where ALL series have data
    df = returns.join(vix, how="inner")

    # rolling 21-day realised vol (annualised) for each asset
    for col in ASSETS:
        df[f"{col}_vol21"] = df[col].rolling(21).std() * np.sqrt(252)

    df.dropna(inplace=True)   # rolling window creates NaNs at the start

    print(f"\nFeature matrix shape: {df.shape}")
    print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"\nColumns: {list(df.columns)}")
    return df


# ── 5. SANITY CHECKS ──────────────────────────────────────────────────────────

def sanity_check(df: pd.DataFrame):
    """Quick checks before passing data to the HMM."""
    print("\n── Sanity checks ──────────────────────────────")

    # Missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"WARNING: missing values found:\n{missing[missing > 0]}")
    else:
        print("✓ No missing values")

    # Returns in a sane range (daily log returns shouldn't exceed ±20%)
    for col in ASSETS:
        max_ret = df[col].abs().max()
        if max_ret > 0.20:
            print(f"WARNING: {col} has a daily return of {max_ret:.1%} — check for splits/errors")
        else:
            print(f"✓ {col} max daily |return|: {max_ret:.2%}")

    # Basic stats
    print("\n── Return statistics ───────────────────────────")
    print(df[ASSETS].describe().round(4))


# ── 6. QUICK VISUALISATION ────────────────────────────────────────────────────

def plot_overview(df: pd.DataFrame):
    """
    Plot 1: cumulative returns of each asset.
    Plot 2: VIX level over time.
    Useful to visually confirm the data looks right before any modelling.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Cumulative returns
    cum_ret = df[ASSETS].cumsum().apply(np.exp)   # exp(sum of log returns)
    cum_ret.plot(ax=axes[0], lw=1.5)
    axes[0].set_title("Cumulative returns — SPY, TLT, GLD")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # VIX
    df["VIX"].plot(ax=axes[1], color="crimson", lw=1)
    axes[1].axhline(30, color="gray", ls="--", lw=0.8, label="VIX = 30 (stress threshold)")
    axes[1].set_title("CBOE VIX — fear gauge")
    axes[1].set_ylabel("VIX level")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("data_overview.png", dpi=150)
    plt.show()
    print("\nPlot saved to data_overview.png")


# ── 7. MAIN ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step A: download
    prices = download_prices(ASSETS, START, END)
    vix    = download_vix(START, END)

    # Step B: returns
    returns = compute_returns(prices)

    # Step C: feature matrix
    features = build_feature_matrix(returns, vix)

    # Step D: checks
    sanity_check(features)

    # Step E: visualise
    plot_overview(features)

    # Step F: save — this CSV is your input to the HMM in the next step
    features.to_csv("regime_features.csv")
    print("\nSaved feature matrix to regime_features.csv")
    print("Ready for HMM training.")
