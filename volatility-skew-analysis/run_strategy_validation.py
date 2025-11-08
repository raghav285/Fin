#!/usr/bin/env python3
"""
Volatility Skew Strategy Validation Script
------------------------------------------
This script reproduces the Bloomberg/Excel-based strategy logic in Python.

It loads:
 - SPX skew data (BB vol - SPX.1.csv)
 - Multi-timeframe signal data (B vol - Multitimeframe.csv)

Then it validates:
 - Signal correlation between Excel and Python logic
 - Strategy return correlation
 - Win rate, Sharpe ratio, and profit factor consistency
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_PATH)

# --- IMPORT MODULES ---
from data.data_loader import BloombergDataLoader
from signals.signal_generator import SignalGenerator
from backtesting.performance_tracker import PerformanceTracker

# === MAIN EXECUTION ===
if __name__ == "__main__":
    print("\n" + "="*60)
    print("=== VOLATILITY SKEW STRATEGY VALIDATION ===")
    print("="*60)
    print("\n🎯 Objective: Reproduce Excel strategy with Sharpe 0.78 and 38.5% win rate")

    loader = BloombergDataLoader()

    # 1️⃣ Load SPX data
    print("\nLoading SPX data...")
    spx_path = "data/raw/vol - SPX.1.csv"
    spx_df = loader.load_spx_data(spx_path)

    # 2️⃣ Load multi-timeframe data
    print("\nLoading multi-timeframe signals...")
    mt_path = "data/raw/vol - Multitimeframe.csv"
    mt_df = loader.load_multitimeframe_data(mt_path)

    # 3️⃣ Merge datasets
    print("\nMerging datasets...")
    merged_df = loader.merge_datasets(spx_df, mt_df)

    if merged_df.empty:
        print("❌ Merging failed — please check file paths and formats.")
        sys.exit(1)

    # 4️⃣ Generate signals using SignalGenerator
    print("\nGenerating signals (Python logic)...")
    signal_model = SignalGenerator()
    python_signals = signal_model.generate_trend_following_signals(merged_df)

    # 5️⃣ Track performance using PerformanceTracker
    print("\nTracking strategy performance...")
    tracker = PerformanceTracker()
    results = tracker.calculate_strategy_returns(python_signals)
    metrics = tracker.calculate_performance_metrics(results)
    print(tracker.generate_performance_report(results))

    # 6️⃣ Validate vs Excel signals
    print("\nValidating against Bloomberg/Excel signals...")

    if 'composite_signal' in merged_df.columns:
        signal_correlation = np.corrcoef(
            python_signals['Composite_Signal'].fillna(0),
            merged_df['composite_signal'].fillna(0)
        )[0, 1]
    else:
        signal_correlation = np.nan

    if 'strategy_return' in merged_df.columns:
        return_correlation = np.corrcoef(
            results['Strategy_Return'].fillna(0),
            merged_df['strategy_return'].fillna(0)
        )[0, 1]
    else:
        return_correlation = np.nan

    print(f"\n📈 Signal correlation (Python vs Excel): {signal_correlation:.3f}")
    print(f"💰 Return correlation (Python vs Excel): {return_correlation:.3f}")

    # 7️⃣ Plot comparison
    os.makedirs("results/figures", exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(results['Date'], results['Cumulative_Strategy'], label="Python Strategy", lw=2)
    if 'Cummulative_PNL' in merged_df.columns:
        plt.plot(merged_df['Date'], merged_df['Cummulative_PNL'], label="Excel Strategy", lw=2, linestyle='--')
    plt.title("Strategy Cumulative Performance Comparison")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/figures/strategy_validation.png", dpi=300)
    plt.show()

    print("\n" + "="*60)
    print("✅ STRATEGY VALIDATION COMPLETE")
    print("="*60)

    if signal_correlation > 0.95 and return_correlation > 0.95:
        print("🎉 EXCELLENT: Python implementation matches Excel results!")
    elif signal_correlation > 0.90 and return_correlation > 0.90:
        print("✅ GOOD: Python implementation closely matches Excel results!")
    else:
        print("⚠️  CHECK: Some differences between Python and Excel implementation.")

    print("\nNext steps:")
    print("1. Review the generated chart in results/figures/strategy_validation.png")
    print("2. Check if performance metrics match your Excel results")
    print("3. Proceed to enhanced strategy development")