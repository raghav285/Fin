# Volatility Skew Trading Strategy Backtester

A multi-timeframe volatility skew analysis backtester that generates trading signals based on Z-score thresholds.

## Strategy Overview

The strategy uses three timeframes to generate trading signals:

| Timeframe | Description | Threshold |
|-----------|------------|-----------|
| ST (Short-term) | 5-day Z-score | > 1.0 |
| MT (Medium-term) | 21-day Z-score | > 0.8 |
| LT (Long-term) | 63-day Z-score | > 1.0 |

Signals are combined into a composite signal (equal-weighted average) and used to generate long/short positions.

---

## Performance Metrics

Based on historical backtest:

| Metric | Value |
|--------|-------|
| Total Return | 43.89% |
| Sharpe Ratio | 1.63 |
| Win Rate | 60.28% |
| Max Drawdown | -20.29% |
| Number of Trades | 282 |
| Avg Win | +0.77% |
| Avg Loss | -0.78% |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/raghav285/skew-backtester.git
cd skew-backtester

# Install dependencies
pip install -r requirements.txt
