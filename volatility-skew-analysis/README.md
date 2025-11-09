# Volatility Skew Trading Strategy Backtester

A multi-timeframe volatility skew analysis backtester that generates trading signals based on Z-score thresholds.

## Strategy Overview

The strategy uses three timeframes to generate trading signals:
- **ST (Short-term)**: 5-day Z-score with threshold > 1.0
- **MT (Medium-term)**: 21-day Z-score with threshold > 0.8
- **LT (Long-term)**: 63-day Z-score with threshold > 1.0

Signals are combined into a composite signal (equal-weighted average) and used to generate long/short positions.

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

## Installation

```bash
# Clone the repository
git clone https://github.com/raghav285/skew-backtester.git
cd skew-backtester

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Update the `DATA_PATH` in `main()` function to point to your CSV file
2. Run the backtester:

```bash
python skew_backtester.py
```

### Example

```python
from skew_backtester import SkewStrategyBacktester

# Initialize backtester
backtester = SkewStrategyBacktester("/path/to/your/data.csv")

# Run the full pipeline
backtester.load_and_clean_data()
backtester.generate_signals()
backtester.run_backtest()

# Get performance metrics
metrics = backtester.calculate_metrics()
print(metrics)

# Save results
backtester.save_results("output.csv")
backtester.plot_results("plot.png")
```

## Required Data Format

Your CSV file must contain the following columns:

| Column | Description |
|--------|-------------|
| `Date` | Trading date (will be parsed to datetime) |
| `1M_Skew_Value` | 1-month skew value |
| `Skew_5D_ZScore` | 5-day rolling Z-score |
| `Skew_21D_ZScore` | 21-day rolling Z-score |
| `Skew_63D_ZScore` | 63-day rolling Z-score |
| `Daily_RET` | Daily returns |

## Signal Generation Logic

### Short-term Signal (ST)
```
IF |Skew_5D_ZScore| > 1.0:
    ST_Signal = sign(1M_Skew_Value)
ELSE:
    ST_Signal = 0
```

### Medium-term Signal (MT)
```
IF |Skew_21D_ZScore| > 0.8:
    MT_Signal = sign(1M_Skew_Value)
ELSE:
    MT_Signal = 0
```

### Long-term Signal (LT)
```
IF Skew_63D_ZScore > 1:
    LT_Signal = 1
ELIF Skew_63D_ZScore < -1:
    LT_Signal = -1
ELSE:
    LT_Signal = 0
```

### Composite Signal
```
Composite_Signal = (ST_Signal + MT_Signal + LT_Signal) / 3
```

### Position Sizing
```
IF Composite_Signal > 0:
    Position = +1 (Long)
ELIF Composite_Signal < 0:
    Position = -1 (Short)
ELSE:
    Position = 0 (Flat)
```

## Output

The backtester generates two files:

1. **CSV File**: Detailed results including:
   - Date
   - All Z-scores
   - Individual signals (ST, MT, LT)
   - Composite signal
   - Position
   - Daily returns
   - Strategy returns
   - Cumulative P&L

2. **PNG File**: Visualization with two subplots:
   - Cumulative P&L over time
   - All trading signals (ST, MT, LT, Composite)



```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results. Use at your own risk.

## Author

Raghav Marwaha

## Acknowledgments

- Strategy based on multi-timeframe volatility skew analysis
- Inspired by options market microstructure research
