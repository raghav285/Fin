#!/usr/bin/env python3
"""
Test script for the updated BloombergDataLoader
"""

import sys
import os
import glob

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_loader import BloombergDataLoader


def find_file(pattern: str):
    """
    Helper function to automatically find files in data/raw/
    that match a pattern like '*SPX*' or '*Multitimeframe*'
    """
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    matches = glob.glob(os.path.join(data_dir, pattern))
    if matches:
        print(f"✅ Found file for pattern '{pattern}': {matches[0]}")
        return matches[0]
    else:
        print(f"❌ No file found for pattern '{pattern}' in {data_dir}")
        return None


def main():
    print("🚀 Testing Updated BloombergDataLoader...\n")

    # Initialize loader
    loader = BloombergDataLoader()

    # Auto-detect your CSV files
    spx_file = find_file("*SPX*.csv")
    mt_file = find_file("*Multitimeframe*.csv")

    # 1️⃣ Load SPX data
    print("\n1️⃣ Loading SPX data...")
    spx_data = loader.load_spx_data(spx_file) if spx_file else None

    # 2️⃣ Load multi-timeframe data
    print("\n2️⃣ Loading multi-timeframe signals...")
    mt_data = loader.load_multitimeframe_data(mt_file) if mt_file else None

    # 3️⃣ Merge datasets
    print("\n3️⃣ Merging datasets...")
    if spx_data is not None and not spx_data.empty and mt_data is not None and not mt_data.empty:
        merged_data = loader.merge_datasets(spx_data, mt_data)
        print(f"\n✅ SUCCESS! Loaded and merged {len(merged_data)} rows")

        print("\n📊 Available columns in merged data:")
        for col in merged_data.columns:
            print(f"  - {col}")
    else:
        print("\n❌ One or both datasets failed to load. Check filenames or CSV content.")

    print("\n🎉 Data loading pipeline test complete!")


if __name__ == "__main__":
    main()