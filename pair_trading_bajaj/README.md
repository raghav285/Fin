# Bajaj Finance vs Bajaj Finserv Pairs Trading Strategy

A sophisticated statistical arbitrage strategy implementing mean reversion trading between Bajaj Finance and Bajaj Finserv stocks using cointegration and z-score based signals.

## Strategy Overview

This implementation employs a quantitative pairs trading approach based on the statistical relationship between Bajaj Finance (BF) and Bajaj Finserv (BFS). The strategy identifies mispricings in the relative valuation of these two fundamentally linked securities and executes mean-reverting trades when deviations exceed statistical thresholds.

### Core Methodology

1. **Cointegration Testing**: Uses Engle-Granger two-step method to establish long-term equilibrium relationship
2. **Spread Calculation**: Constructs hedged portfolio: `Spread = BFS - β·BF - α`
3. **Signal Generation**: Implements z-score normalization with dynamic rolling statistics
4. **Position Management**: Enters trades at extreme z-scores, exits at mean reversion

### Key Features

- **Dynamic Hedge Ratio**: Continuously updated β coefficient from OLS regression
- **Risk-Free Allocation**: Earns risk-free returns during non-trading periods
- **Comprehensive Analytics**: Trade-level performance attribution and risk metrics
- **Parameter Optimization**: Configurable entry/exit thresholds and lookback periods

## Installation & Requirements

```bash
pip install pandas numpy matplotlib seaborn statsmodels pathlib argparse