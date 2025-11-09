

```python
"""
Bajaj Finance vs Bajaj Finserv Pairs Trading Strategy

A statistical arbitrage strategy using z-score based mean reversion
on Bajaj Finance and Bajaj Finserv stock prices.

Author: Raghav
License: MIT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import warnings
from pathlib import Path
from typing import Tuple, Dict, Optional
import argparse

warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
pd.set_option('display.precision', 4)


class BajajPairsTrading:
    """
    Pairs trading strategy for Bajaj Finance and Bajaj Finserv.
    
    Implements statistical arbitrage using cointegration and mean reversion principles.
    """
    
    def __init__(self, entry_z: float = 2.0, exit_z: float = 0.5, 
                 lookback: int = 63, risk_free_rate: float = 0.07):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.lookback = lookback
        self.risk_free_rate = risk_free_rate
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load and preprocess price data from CSV file."""
        df = pd.read_csv(filepath, header=None)
        
        clean_data = pd.DataFrame()
        clean_data['Date'] = pd.to_datetime(df.iloc[2:, 1], errors='coerce')
        clean_data['Bajaj_Finance'] = pd.to_numeric(df.iloc[2:, 2], errors='coerce')
        clean_data['Bajaj_Finserv'] = pd.to_numeric(df.iloc[2:, 4], errors='coerce')
        
        return clean_data.dropna().reset_index(drop=True)
    
    def engle_granger_test(self, X: pd.Series, Y: pd.Series) -> Tuple[bool, float, float, pd.Series]:
        """Perform Engle-Granger cointegration test between two series."""
        X_const = sm.add_constant(X)
        model = sm.OLS(Y, X_const).fit()
        residuals = model.resid
        adf_pvalue = adfuller(residuals)[1]
        return adf_pvalue < 0.05, model.params[1], model.params[0], residuals
    
    def calculate_zscore(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling z-score for spread series."""
        return (series - series.rolling(window).mean()) / series.rolling(window).std()
    
    def generate_signals(self, z_scores: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Generate trading signals based on z-score thresholds."""
        signals = pd.Series(0, index=z_scores.index)
        positions = pd.Series(0, index=z_scores.index)
        current_pos = 0
        
        for i, (date, z) in enumerate(z_scores.items()):
            if pd.isna(z):
                continue
                
            # Exit conditions
            if current_pos == 1 and z >= -self.exit_z:
                signals[date], current_pos = -1, 0
            elif current_pos == -1 and z <= self.exit_z:
                signals[date], current_pos = 1, 0
                
            # Entry conditions  
            elif current_pos == 0:
                if z < -self.entry_z:
                    signals[date], current_pos = 1, 1
                elif z > self.entry_z:
                    signals[date], current_pos = -1, -1
            
            positions[date] = current_pos
            
        return signals, positions
    
    def run_strategy(self, filepath: str) -> Tuple[pd.DataFrame, bool, float]:
        """Execute complete pairs trading strategy."""
        # Load and validate data
        data = self.load_data(filepath)
        print(f"Data loaded: {len(data)} trading days")
        print(f"Date range: {data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}")
        
        # Cointegration test
        is_coint, hedge_ratio, intercept, residuals = self.engle_granger_test(
            data['Bajaj_Finance'], data['Bajaj_Finserv']
        )
        print(f"\nCointegration: {is_coint}")
        print(f"Hedge Ratio: {hedge_ratio:.6f}")
        print(f"Intercept: {intercept:.6f}")
        
        # Calculate spread and z-score
        data['Spread'] = data['Bajaj_Finserv'] - hedge_ratio * data['Bajaj_Finance'] - intercept
        data['Z_Score'] = self.calculate_zscore(data['Spread'], self.lookback)
        
        # Generate signals
        data['Signal'], data['Position'] = self.generate_signals(data['Z_Score'])
        
        # Calculate returns
        data['BF_Returns'] = data['Bajaj_Finance'].pct_change()
        data['BFS_Returns'] = data['Bajaj_Finserv'].pct_change()
        data['Strategy_Returns'] = 0.0
        
        # Daily risk-free rate
        daily_rf = (1 + self.risk_free_rate) ** (1/252) - 1
        
        long_mask = data['Position'] == 1
        short_mask = data['Position'] == -1
        no_position_mask = data['Position'] == 0
        
        # Strategy returns calculation
        data.loc[long_mask, 'Strategy_Returns'] = (
            data.loc[long_mask, 'BFS_Returns'] - hedge_ratio * data.loc[long_mask, 'BF_Returns']
        )
        data.loc[short_mask, 'Strategy_Returns'] = (
            -data.loc[short_mask, 'BFS_Returns'] + hedge_ratio * data.loc[short_mask, 'BF_Returns']
        )
        data.loc[no_position_mask, 'Strategy_Returns'] = daily_rf
        
        # Cumulative returns
        data['Cumulative_Strategy'] = (1 + data['Strategy_Returns']).cumprod()
        data['Cumulative_BF'] = (1 + data['BF_Returns']).cumprod()
        data['Cumulative_BFS'] = (1 + data['BFS_Returns']).cumprod()
        
        return data, is_coint, hedge_ratio
    
    def analyze_trades(self, data: pd.DataFrame) -> pd.DataFrame:
        """Analyze individual trades with holding period returns."""
        trades = []
        entry_idx = None
        entry_value = None
        entry_type = None
        
        for i in range(len(data)):
            if data['Signal'].iloc[i] != 0 and entry_idx is None:
                entry_idx = i
                entry_value = data['Cumulative_Strategy'].iloc[i]
                entry_type = data['Signal'].iloc[i]
            
            elif data['Signal'].iloc[i] != 0 and entry_idx is not None:
                exit_idx = i
                exit_value = data['Cumulative_Strategy'].iloc[i]
                
                trade_return = (exit_value / entry_value) - 1
                days_held = (data['Date'].iloc[exit_idx] - data['Date'].iloc[entry_idx]).days
                
                if days_held > 0:
                    annualized_return = ((1 + trade_return) ** (365.25 / days_held)) - 1
                else:
                    annualized_return = 0
                
                trades.append({
                    'Entry Date': data['Date'].iloc[entry_idx],
                    'Exit Date': data['Date'].iloc[exit_idx],
                    'Type': 'Long' if entry_type == 1 else 'Short',
                    'Entry Z-Score': data['Z_Score'].iloc[entry_idx],
                    'Exit Z-Score': data['Z_Score'].iloc[exit_idx],
                    'Days Held': days_held,
                    'Trade Return': trade_return,
                    'Annualized Return': annualized_return
                })
                
                entry_idx = None
        
        return pd.DataFrame(trades)
    
    def performance_metrics(self, data: pd.DataFrame, trades_df: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics."""
        returns = data['Strategy_Returns'].dropna()
        cum_returns = data['Cumulative_Strategy'].dropna()
        
        # Portfolio metrics
        total_return = cum_returns.iloc[-1] - 1
        years = (data['Date'].iloc[-1] - data['Date'].iloc[0]).days / 365.25
        cagr = ((cum_returns.iloc[-1]) ** (1 / years)) - 1
        
        volatility = returns.std() * np.sqrt(252)
        sharpe = (returns.mean() * 252 - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade metrics
        winning_trades = trades_df[trades_df['Trade Return'] > 0]
        losing_trades = trades_df[trades_df['Trade Return'] <= 0]
        
        trade_win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
        profit_factor = (winning_trades['Trade Return'].sum() / 
                        abs(losing_trades['Trade Return'].sum())) if len(losing_trades) > 0 else np.inf
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'sharpe': sharpe,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'trade_win_rate': trade_win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades)
        }


def main():
    """Command-line interface for strategy execution."""
    parser = argparse.ArgumentParser(description='Bajaj Pairs Trading Strategy')
    parser.add_argument('--data', type=str, required=True, help='Path to data CSV file')
    parser.add_argument('--entry-z', type=float, default=2.0, help='Entry z-score threshold')
    parser.add_argument('--exit-z', type=float, default=0.5, help='Exit z-score threshold')
    parser.add_argument('--lookback', type=int, default=63, help='Lookback window for z-score')
    parser.add_argument('--risk-free-rate', type=float, default=0.07, help='Annual risk-free rate')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    strategy = BajajPairsTrading(
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        lookback=args.lookback,
        risk_free_rate=args.risk_free_rate
    )
    
    results, is_coint, hedge_ratio = strategy.run_strategy(args.data)
    trades = strategy.analyze_trades(results)
    metrics = strategy.performance_metrics(results, trades)
    
    print(f"\nStrategy completed: {len(trades)} trades executed")
    print(f"Cointegration: {is_coint}, Hedge Ratio: {hedge_ratio:.4f}")


if __name__ == "__main__":
    main()
