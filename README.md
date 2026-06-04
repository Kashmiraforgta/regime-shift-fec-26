# Regime-Shift

A macro-aware tactical asset allocation engine built for FEC IIT Guwahati's DIY '26 project series.

The core idea is simple — markets aren't static. A portfolio that works perfectly during a bull run can get destroyed during a crisis, and a static 60/40 allocation has no way of knowing the difference. This project tries to fix that by detecting which "regime" the market is currently in and adjusting the portfolio accordingly.

---

## What it actually does

The system has three main pieces that run in sequence:

**1. Data Pipeline** — pulls daily price data for SPY (equities), TLT (bonds), and GLD (gold) from Yahoo Finance along with the CBOE VIX index. Converts prices to log returns and computes rolling 21-day volatility for each asset. Everything gets saved to a single clean CSV.

**2. HMM Regime Classifier** — feeds that feature matrix into a Gaussian Hidden Markov Model with 3 states. The model figures out on its own that markets tend to cluster into distinct behaviours — it doesn't need any manual labels. After training, each trading day gets assigned one of three regimes: Bull, Bear, or Crisis. The Viterbi algorithm finds the most likely sequence of these hidden states through the entire history.

**3. Portfolio Optimizer + Backtester** — this is where the regime information actually gets used. A CVXPY convex optimizer runs every 21 trading days, but the objective function changes depending on the current regime. In a Bull market it tries to maximize the Sharpe ratio. In a Crisis it switches to pure volatility minimization — just preserve capital. Every rebalance also gets hit with a 10 basis point transaction cost so the results reflect something closer to reality. The whole thing runs in a strict walk-forward loop, meaning the model never touches future data.

---

## Results

Tested from 2007 to 2024, covering the 2008 financial crisis, the 2020 COVID crash, and the 2022 rate hike cycle.

| | Annual Return | Sharpe | Max Drawdown | Calmar |
|---|---|---|---|---|
| Regime-Shift | 6.86% | 0.34 | -24.79% | 0.28 |
| 60/40 | 7.07% | 0.40 | -34.61% | 0.20 |
| Equal Weight | 6.73% | 0.43 | -23.99% | 0.28 |

The strategy doesn't beat 60/40 on raw returns — that wasn't really the goal. The interesting number is the max drawdown. During the worst periods the strategy lost about 25% peak to trough, while 60/40 lost nearly 35%. The Calmar ratio (return divided by max drawdown) comes out better for Regime-Shift as a result.

The HMM identified three fairly clean regimes from the data:

- **Bull** — VIX around 14, slightly positive daily SPY returns. Covers most of 2013–2019 and 2021–2023.
- **Bear** — VIX around 22, flat to slightly positive returns. Choppy sideways markets.
- **Crisis** — VIX spiking to 44, negative SPY returns. Only 313 days out of 5000+, but they're the ones that matter.

---

## How to run it

```bash
git clone https://github.com/Kashmiraforgta/regime-shift-fec-26.git
cd regime-shift-fec-26

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

pip install yfinance pandas numpy matplotlib hmmlearn scikit-learn cvxpy scipy

python data_pipeline.py
python hmm_classifier.py
python optimizer.py
```

Running all three scripts takes about 5–7 minutes total, mostly the HMM fitting and the walk-forward loop. Outputs are saved as CSVs and PNGs in the same folder.

---

## Files

```
data_pipeline.py        pulls and cleans market data
hmm_classifier.py       fits the HMM and labels regimes
optimizer.py            runs the walk-forward backtest
regime_features.csv     cleaned input features
regime_labels.csv       features + regime label per day
backtest_results.csv    daily returns and weights from the strategy
performance_summary.csv metrics vs benchmarks
regime_chart.png        regimes overlaid on SPY price history
tearsheet.png           equity curves, drawdown, weight allocation over time
```

---

## Some decisions worth explaining

**Why log returns instead of simple returns** — log returns are additive across time which makes the math cleaner, and their distribution is closer to normal which helps the Gaussian HMM.

**Why diagonal covariance for the HMM** — full covariance would let each state model correlations between all features, which sounds better but actually overfits badly on financial data. Diagonal is faster and generalizes better here.

**Why walk-forward instead of a normal backtest** — a standard backtest fits the model on all available data and then tests on the same data. That's cheating. Walk-forward refits only on past data at each step, which is the only honest way to evaluate a strategy that uses a trained model.

**Why 10bps transaction cost** — ignoring trading costs makes every strategy look better than it is. 10bps per rebalance is conservative for liquid ETFs but it's real enough to show whether the strategy survives friction.

---

## Dependencies

- `hmmlearn` — Gaussian HMM implementation
- `cvxpy` — convex optimization for portfolio weights
- `yfinance` — market data
- `scikit-learn` — feature scaling
- `pandas / numpy / matplotlib / scipy` — everything else

---

