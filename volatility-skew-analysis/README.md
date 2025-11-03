# Volatility Skew Analysis

A professional quantitative trading strategy analyzing SPX volatility skew across multiple timeframes to generate alpha.

##  Strategy Performance

- **Strategy Type**: Trend-following on volatility skew  
- **Asset**: SPX Index options data  
- **Timeframe**: Multi-timeframe (5D, 21D, 63D)  
- **Performance**: Sharpe 0.78, Profit Factor 1.4, Win Rate 38.5%

## Quick Start

```bash
# Clone repository
git clone https://github.com/raghav285/volatility-skew-analysis.git
cd volatility-skew-analysis

# Install dependencies
pip install -r requirements.txt

# Run analysis
jupyter notebook notebooks/02_strategy_development.ipynb
