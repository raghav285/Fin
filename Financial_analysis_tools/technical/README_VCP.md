Volatility Contraction Pattern (VCP) Scanner
Purpose
Systematically detects VCP breakout setups—high-probability price contraction patterns—across liquid Indian equities for actionable technical trading.

Trading Logic
Data: Fetches and stores daily price/volume data for all NSE-listed stocks.

Filter: Screens for strong liquidity (ADV, price filter).

Pattern Detection:

Identifies consecutive price contractions (tightenings).

Each contraction must show lower volatility and reduced volume.

Looks for risk compression before breakout.

Output: Flags tickers meeting VCP criteria and ranks by contraction quality; results saved for downstream trading or analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/raghav285/Fin.git
cd Fin/Financial_analysis_tools/technical
