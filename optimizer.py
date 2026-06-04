
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.stats import skew
import warnings
warnings.filterwarnings("ignore")

ASSETS          = ["SPY", "TLT", "GLD"]
TURNOVER_COST   = 0.0010   
TRAIN_YEARS     = 2       
REBAL_FREQ      = 21      
RISK_FREE       = 0.0001   
print("Loading regime labels...")
df = pd.read_csv("regime_labels.csv", index_col=0, parse_dates=True)
returns = df[ASSETS]
regimes = df["regime"]
print(f"Loaded {len(df)} rows | Regimes: {regimes.value_counts().to_dict()}")
def optimize(mu: np.ndarray, sigma: np.ndarray,
             prev_weights: np.ndarray, regime: str) -> np.ndarray:
    
    n = len(ASSETS)
    w = cp.Variable(n)

    portfolio_return   = mu @ w
    portfolio_variance = cp.quad_form(w, sigma)
    turnover_penalty   = TURNOVER_COST * cp.norm1(w - prev_weights)
    constraints = [
        cp.sum(w) == 1,      
        w >= 0.05,            
        w <= 0.80,            
    ]

    if regime == "Bull":
        objective = cp.Maximize(portfolio_return - 0.5 * portfolio_variance - turnover_penalty)

    elif regime == "Bear":
        constraints.append(portfolio_variance <= (0.12 / np.sqrt(252)) ** 2)
        objective = cp.Maximize(portfolio_return - 1.0 * portfolio_variance - turnover_penalty)

    else: 
        objective = cp.Minimize(portfolio_variance + turnover_penalty)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if prob.status not in ["optimal", "optimal_inaccurate"] or w.value is None:
        return prev_weights   

    weights = np.array(w.value)
    weights = np.clip(weights, 0, 1)
    weights /= weights.sum()  
    return weights


def run_backtest(returns: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
    train_days  = TRAIN_YEARS * 252
    dates       = returns.index
    n_assets    = len(ASSETS)

    prev_weights = np.ones(n_assets) / n_assets
    current_weights = prev_weights.copy()

    results = []

    print(f"\nRunning walk-forward backtest...")
    print(f"Training window: {TRAIN_YEARS} years | Rebalance: every {REBAL_FREQ} days")

    for i in range(train_days, len(dates)):


        if (i - train_days) % REBAL_FREQ == 0:
            
            train_returns = returns.iloc[i - train_days:i]
            current_regime = regimes.iloc[i]

            mu    = train_returns.mean().values * 252       
            sigma = train_returns.cov().values   * 252       

            current_weights = optimize(mu, sigma, prev_weights, current_regime)
            prev_weights    = current_weights.copy()

        daily_ret = returns.iloc[i].values @ current_weights
        results.append({
            "date":    dates[i],
            "return":  daily_ret,
            "regime":  regimes.iloc[i],
            "w_SPY":   current_weights[0],
            "w_TLT":   current_weights[1],
            "w_GLD":   current_weights[2],
        })

        if i % 500 == 0:
            print(f"  Progress: {i}/{len(dates)} days done...")

    print("Backtest complete.")
    return pd.DataFrame(results).set_index("date")

def compute_benchmarks(returns: pd.DataFrame, start_idx: int) -> pd.DataFrame:
    r = returns.iloc[start_idx:].copy()
    bench = pd.DataFrame(index=r.index)
    bench["60_40"]       = r["SPY"] * 0.60 + r["TLT"] * 0.40
    bench["equal_weight"] = r[ASSETS].mean(axis=1)
    return bench

def performance_metrics(daily_returns: pd.Series, label: str) -> dict:
    ann_ret   = daily_returns.mean() * 252
    ann_vol   = daily_returns.std()  * np.sqrt(252)
    sharpe    = (daily_returns.mean() - RISK_FREE) / daily_returns.std() * np.sqrt(252)

    downside  = daily_returns[daily_returns < 0].std() * np.sqrt(252)
    sortino   = (ann_ret - RISK_FREE * 252) / downside if downside > 0 else np.nan

    cum       = (1 + daily_returns).cumprod()
    peak      = cum.cummax()
    drawdown  = (cum - peak) / peak
    max_dd    = drawdown.min()

    calmar    = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    print(f"\n {label} ")
    print(f"  Annual Return : {ann_ret:.2%}")
    print(f"  Annual Vol    : {ann_vol:.2%}")
    print(f"  Sharpe Ratio  : {sharpe:.2f}")
    print(f"  Sortino Ratio : {sortino:.2f}")
    print(f"  Max Drawdown  : {max_dd:.2%}")
    print(f"  Calmar Ratio  : {calmar:.2f}")

    return {"label": label, "ann_ret": ann_ret, "ann_vol": ann_vol,
            "sharpe": sharpe, "sortino": sortino, "max_dd": max_dd, "calmar": calmar}



REGIME_COLORS = {"Bull": "#2ecc71", "Bear": "#e67e22", "Crisis": "#e74c3c"}

def plot_tearsheet(results: pd.DataFrame, bench: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)

    strat_cum = (1 + results["return"]).cumprod()
    b6040_cum = (1 + bench["60_40"]).cumprod()
    ew_cum    = (1 + bench["equal_weight"]).cumprod()

    axes[0].plot(strat_cum.index, strat_cum,   lw=1.5, label="Regime-Shift", color="#3498db")
    axes[0].plot(b6040_cum.index, b6040_cum,   lw=1.2, label="60/40",        color="#e67e22", ls="--")
    axes[0].plot(ew_cum.index,    ew_cum,       lw=1.2, label="Equal Weight", color="#9b59b6", ls="--")
    axes[0].set_title("Equity Curves — Regime-Shift vs Benchmarks")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    prev_r = results["regime"].iloc[0]
    start  = results.index[0]
    for date, reg in results["regime"].items():
        if reg != prev_r:
            axes[0].axvspan(start, date, alpha=0.08, color=REGIME_COLORS[prev_r])
            start  = date
            prev_r = reg
    axes[0].axvspan(start, results.index[-1], alpha=0.08, color=REGIME_COLORS[prev_r])

    cum   = (1 + results["return"]).cumprod()
    peak  = cum.cummax()
    dd    = (cum - peak) / peak
    axes[1].fill_between(dd.index, dd, 0, color="#e74c3c", alpha=0.4, label="Drawdown")
    axes[1].set_title("Strategy Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].stackplot(results.index,
                      results["w_SPY"], results["w_TLT"], results["w_GLD"],
                      labels=["SPY", "TLT", "GLD"],
                      colors=["#3498db", "#2ecc71", "#f1c40f"], alpha=0.7)
    axes[2].set_title("Portfolio Weights Over Time")
    axes[2].set_ylabel("Weight")
    axes[2].legend(loc="upper left")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("tearsheet.png", dpi=150)
    plt.close()
    print("\nTearsheet saved to tearsheet.png")


if __name__ == "__main__":
    train_days = TRAIN_YEARS * 252

    results = run_backtest(returns, regimes)
    results.to_csv("backtest_results.csv")
    print("Saved backtest_results.csv")

    bench = compute_benchmarks(returns, train_days)

    metrics = []
    metrics.append(performance_metrics(results["return"],))
    metrics.append(performance_metrics(bench["60_40"],))
    metrics.append(performance_metrics(bench["equal_weight"],))
    summary = pd.DataFrame(metrics).set_index("label")
    print("\nSummary Table")
    print(summary.round(3))
    summary.to_csv("performance_summary.csv")
    print("\nSaved performance_summary.csv")
    plot_tearsheet(results, bench)
    print("\nAll done! Files saved:")
    print("  backtest_results.csv")
    print("  performance_summary.csv")
    print("  tearsheet.png")
