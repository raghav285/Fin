"""
Volatility Skew Trading Strategy Backtester

Multi-timeframe volatility skew analysis backtester.
Generates ST (5-day), MT (21-day), and LT (63-day) signals based on Z-score thresholds.

Author: Your Name
Date: 2025-11-09
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SkewStrategyBacktester:
    """Backtests a multi-timeframe volatility skew trading strategy."""

    def __init__(self, data_path: str):
        """
        Initialize the backtester with data path.

        Args:
            data_path: Path to the CSV file containing market data
        """
        self.data_path = Path(data_path)
        self.data: Optional[pd.DataFrame] = None

    def load_and_clean_data(self) -> pd.DataFrame:
        """
        Load and clean the data from CSV file.

        Returns:
            Cleaned DataFrame

        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If required columns are missing
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        logger.info(f"Loading data from {self.data_path}")
        data = pd.read_csv(self.data_path)

        # Remove error rows
        data = data[data['Date'] != '#ERROR!']

        # Parse dates
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data = data.dropna(subset=['Date'])
        data = data.sort_values('Date').reset_index(drop=True)

        # Fill NaN values in numeric columns with 0
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].fillna(0)

        logger.info(f"Data loaded: {len(data)} rows, {len(data.columns)} columns")

        # Validate required columns
        required_cols = ['Date', '1M_Skew_Value', 'Skew_5D_ZScore',
                        'Skew_21D_ZScore', 'Skew_63D_ZScore', 'Daily_RET']
        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.data = data
        return data

    def generate_signals(self) -> pd.DataFrame:
        """
        Generate trading signals based on Z-score thresholds.

        Signal Logic:
        - ST (Short-term): |Skew_5D_ZScore| > 1.0 → sign(1M_Skew_Value)
        - MT (Medium-term): |Skew_21D_ZScore| > 0.8 → sign(1M_Skew_Value)
        - LT (Long-term): Skew_63D_ZScore > 1 → +1, < -1 → -1

        Returns:
            DataFrame with signal columns added
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_and_clean_data() first.")

        logger.info("Generating trading signals")

        data = self.data.copy()

        # Initialize signal columns
        data['ST_Signal'] = 0
        data['MT_Signal'] = 0
        data['LT_Signal'] = 0

        # Vectorized signal generation (much faster than loops)
        # ST Signal: |Z-score| > 1 → use sign of 1M skew
        st_condition = np.abs(data['Skew_5D_ZScore']) > 1
        data.loc[st_condition, 'ST_Signal'] = np.sign(data.loc[st_condition, '1M_Skew_Value'])

        # MT Signal: |Z-score| > 0.8 → use sign of 1M skew
        mt_condition = np.abs(data['Skew_21D_ZScore']) > 0.8
        data.loc[mt_condition, 'MT_Signal'] = np.sign(data.loc[mt_condition, '1M_Skew_Value'])

        # LT Signal: Direct thresholding on Z-score
        data.loc[data['Skew_63D_ZScore'] > 1, 'LT_Signal'] = 1
        data.loc[data['Skew_63D_ZScore'] < -1, 'LT_Signal'] = -1

        # Composite signal (equal-weighted average)
        data['Composite_Signal'] = (
            data['ST_Signal'] + data['MT_Signal'] + data['LT_Signal']
        ) / 3

        self.data = data
        return data

    def run_backtest(self) -> pd.DataFrame:
        """
        Execute backtest based on composite signals.

        Position Logic:
        - Composite_Signal > 0 → Long (+1)
        - Composite_Signal < 0 → Short (-1)
        - Composite_Signal = 0 → Flat (0)

        Returns:
            DataFrame with backtest results
        """
        if self.data is None or 'Composite_Signal' not in self.data.columns:
            raise ValueError("Signals not generated. Call generate_signals() first.")

        logger.info("Running backtest")

        data = self.data.copy()

        # Generate positions based on signals
        data['Position'] = np.select(
            [data['Composite_Signal'] > 0, data['Composite_Signal'] < 0],
            [1, -1],
            default=0
        )

        # Calculate strategy returns
        data['Strategy_Return'] = data['Position'].shift(1) * data['Daily_RET']
        data['Strategy_Return'] = data['Strategy_Return'].fillna(0)

        # Calculate cumulative P&L
        data['Cumulative_PnL'] = data['Strategy_Return'].cumsum()

        self.data = data
        return data

    def calculate_metrics(self) -> dict:
        """
        Calculate performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        if self.data is None or 'Strategy_Return' not in self.data.columns:
            raise ValueError("Backtest not run. Call run_backtest() first.")

        returns = self.data['Strategy_Return']

        # Calculate metrics
        num_trades = len(returns[returns != 0])
        winning_trades = len(returns[returns > 0])
        losing_trades = len(returns[returns < 0])

        metrics = {
            'total_return': self.data['Cumulative_PnL'].iloc[-1],
            'num_trades': num_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / num_trades if num_trades > 0 else 0,
            'avg_win': returns[returns > 0].mean() if winning_trades > 0 else 0,
            'avg_loss': returns[returns < 0].mean() if losing_trades > 0 else 0,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            'max_drawdown': (self.data['Cumulative_PnL'] - self.data['Cumulative_PnL'].cummax()).min(),
        }

        return metrics

    def save_results(self, output_path: str):
        """
        Save backtest results to CSV.

        Args:
            output_path: Path to save results
        """
        if self.data is None:
            raise ValueError("No results to save. Run backtest first.")

        output_cols = [
            'Date', '1M_Skew_Value', 'Skew_5D_ZScore', 'Skew_21D_ZScore', 'Skew_63D_ZScore',
            'ST_Signal', 'MT_Signal', 'LT_Signal', 'Composite_Signal',
            'Position', 'Daily_RET', 'Strategy_Return', 'Cumulative_PnL'
        ]

        # Only include columns that exist
        available_cols = [col for col in output_cols if col in self.data.columns]

        self.data[available_cols].to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot cumulative P&L and signals.

        Args:
            save_path: Optional path to save the plot
        """
        if self.data is None or 'Cumulative_PnL' not in self.data.columns:
            raise ValueError("Backtest not run. Call run_backtest() first.")

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Plot cumulative P&L
        axes[0].plot(self.data['Date'], self.data['Cumulative_PnL'],
                    linewidth=2, label='Strategy P&L', color='#2E86AB')
        axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[0].set_title('Cumulative P&L', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Cumulative Return')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot signals
        axes[1].plot(self.data['Date'], self.data['ST_Signal'],
                    label='ST Signal (5D)', alpha=0.6, linewidth=1.5)
        axes[1].plot(self.data['Date'], self.data['MT_Signal'],
                    label='MT Signal (21D)', alpha=0.6, linewidth=1.5)
        axes[1].plot(self.data['Date'], self.data['LT_Signal'],
                    label='LT Signal (63D)', alpha=0.6, linewidth=1.5)
        axes[1].plot(self.data['Date'], self.data['Composite_Signal'],
                    label='Composite Signal', linewidth=2.5, color='black')
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        axes[1].set_title('Trading Signals', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Signal Value')
        axes[1].set_xlabel('Date')
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()


def main():
    """Main execution function."""
    # Hardcoded configuration
    DATA_PATH = "/Users/raghav/Downloads/vol - Multitimeframe.csv"
    OUTPUT_PATH = "/Users/raghav/Downloads/backtest_results.csv"
    PLOT_PATH = "/Users/raghav/Downloads/backtest_plot.png"

    try:
        # Initialize backtester
        backtester = SkewStrategyBacktester(DATA_PATH)

        # Run analysis pipeline
        backtester.load_and_clean_data()
        backtester.generate_signals()
        backtester.run_backtest()

        # Display results
        metrics = backtester.calculate_metrics()
        print("\n" + "="*80)
        print("BACKTEST PERFORMANCE METRICS")
        print("="*80)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key.replace('_', ' ').title()}: {value:.4f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")

        # Save results
        backtester.save_results(OUTPUT_PATH)
        backtester.plot_results(PLOT_PATH)

        logger.info("Backtest completed successfully!")

    except Exception as e:
        logger.error(f"Error during backtest: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
