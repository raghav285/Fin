"""
Bloomberg data loader for volatility skew analysis
Handles actual CSV files with real column names from Bloomberg data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

class BloombergDataLoader:
    """
    Load and process Bloomberg volatility data for SPX options
    """
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
    
    def load_spx_data(self, file_path: str) -> pd.DataFrame:
        """
        Load SPX volatility data from CSV file with actual column names
        
        Args:
            file_path: Path to CSV file with Bloomberg data (BB vol - SPX.1.csv)
            
        Returns:
            DataFrame with SPX volatility data
        """
        try:
            df = pd.read_csv(file_path)
            
            # Clean column names (remove extra spaces, special characters)
            df.columns = df.columns.str.strip()
            
            print(f"Raw columns found: {list(df.columns)}")
            
            # Convert date column - handle different date formats
            date_columns = ['Date', 'DATE', 'date']
            date_col = next((col for col in date_columns if col in df.columns), None)
            
            if date_col:
                df['Date'] = pd.to_datetime(df[date_col])
                print(f"Using '{date_col}' as date column")
            else:
                # If no date column found, check first column
                if pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
                    df['Date'] = pd.to_datetime(df.iloc[:, 0])
                    print(f"Using first column '{df.columns[0]}' as date column")
                else:
                    # Create date range as fallback
                    print("Warning: No date column found. Creating date range.")
                    df['Date'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
            
            # Sort by date and reset index
            df = df.sort_values('Date').reset_index(drop=True)
            
            # Map to standardized column names for easier processing
            df = self._standardize_column_names(df)
            
            print(f"✅ Loaded SPX data with {len(df)} rows")
            print(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
            print(f"📊 Available columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading SPX data: {e}")
            return pd.DataFrame()
    
    def load_multitimeframe_data(self, file_path: str) -> pd.DataFrame:
        """
        Load multi-timeframe signals data from CSV file
        
        Args:
            file_path: Path to CSV file with multi-timeframe signals (B vol - Multitimeframe.csv)
            
        Returns:
            DataFrame with multi-timeframe signals
        """
        try:
            df = pd.read_csv(file_path)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            print(f"Multi-timeframe raw columns: {list(df.columns)}")
            
            # Convert date column
            date_columns = ['Date', 'DATE', 'date']
            date_col = next((col for col in date_columns if col in df.columns), None)
            
            if date_col:
                df['Date'] = pd.to_datetime(df[date_col])
            else:
                if pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
                    df['Date'] = pd.to_datetime(df.iloc[:, 0])
                else:
                    df['Date'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
            
            df = df.sort_values('Date').reset_index(drop=True)
            
            # Standardize column names
            df = self._standardize_multitimeframe_columns(df)
            
            print(f"✅ Loaded multi-timeframe data with {len(df)} rows")
            print(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading multi-timeframe data: {e}")
            return pd.DataFrame()
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names for SPX raw data based on actual CSV structure
        
        Your actual columns from BB vol - SPX.1.csv:
        Date, PRICE, VOLUME, 25DELTA_PUT_1M, 25DELTA_CALL_1M, 25DELTA_PUT_2M, 
        25DELTA_CALL_2M, LEAP_25DELTA_P, LEAP_25DELTA_C, 30D hist_vol, 
        1M_25D_skew, 2M_25D_SKEW, Leap_25D_SKEW, Term_slope_1,2m, 
        Term_slope_2m,leap, IV,HV_PREM_1M, IV,HV_PREM_2M
        """
        df_standardized = df.copy()
        
        # Column mapping from actual to standardized names
        column_mapping = {
            # Price and volume
            'PRICE': 'price',
            'VOLUME': 'volume',
            
            # Raw implied volatilities
            '25DELTA_PUT_1M': 'put_iv_1m_25d',
            '25DELTA_CALL_1M': 'call_iv_1m_25d', 
            '25DELTA_PUT_2M': 'put_iv_2m_25d',
            '25DELTA_CALL_2M': 'call_iv_2m_25d',
            'LEAP_25DELTA_P': 'put_iv_leap_25d',
            'LEAP_25DELTA_C': 'call_iv_leap_25d',
            
            # Historical volatility
            '30D hist_vol': 'historical_vol_30d',
            
            # Pre-calculated skews (from your Excel)
            '1M_25D_skew': 'skew_1m_25d',
            '2M_25D_SKEW': 'skew_2m_25d', 
            'Leap_25D_SKEW': 'skew_leap_25d',
            
            # Term structure
            'Term_slope_1,2m': 'term_slope_1m_2m',
            'Term_slope_2m,leap': 'term_slope_2m_leap',
            
            # Implied vs Historical premium
            'IV,HV_PREM_1M': 'iv_hv_premium_1m',
            'IV,HV_PREM_2M': 'iv_hv_premium_2m'
        }
        
        # Apply column mapping (only for columns that exist)
        for old_name, new_name in column_mapping.items():
            if old_name in df_standardized.columns:
                df_standardized[new_name] = df_standardized[old_name]
                print(f"   Mapped '{old_name}' -> '{new_name}'")
        
        # Calculate daily returns if price data exists
        if 'price' in df_standardized.columns:
            df_standardized['daily_return'] = df_standardized['price'].pct_change()
            print("   Calculated daily returns from price data")
        
        # Verify we have the essential skew column
        if 'skew_1m_25d' not in df_standardized.columns:
            print("⚠️  Warning: 1M_25D_skew column not found. This is essential for strategy.")
        
        return df_standardized
    
    def _standardize_multitimeframe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names for multi-timeframe signals data
        
        Your actual columns from B vol - Multitimeframe.csv:
        Date, 1M_Skew_Value, Skew_5D_MA, Skew_5D_StdDev, Skew_5D_ZScore,
        Skew_21D_MA, Skew_21D_StdDev, Skew_21D_ZScore, Skew_63D_MA, 
        Skew_63D_StdDev, Skew_63D_ZScore, ST_Signal (5-day), MT_Signal (21-day), 
        LT_Signal (63-day), Composite_Signal, Daily_RET, Strategy_Ret, 
        Cummulative_PNL, Active trades, Winning trades, True win rate, 
        sharpe ratio, average win, average loss, average win/loss, Profit factor - multi
        """
        df_standardized = df.copy()
        
        # Column mapping for multi-timeframe data
        column_mapping = {
            # Skew values and statistics
            '1M_Skew_Value': 'skew_1m_value',
            'Skew_5D_MA': 'skew_5d_ma',
            'Skew_5D_StdDev': 'skew_5d_std',
            'Skew_5D_ZScore': 'skew_5d_zscore',
            'Skew_21D_MA': 'skew_21d_ma', 
            'Skew_21D_StdDev': 'skew_21d_std',
            'Skew_21D_ZScore': 'skew_21d_zscore',
            'Skew_63D_MA': 'skew_63d_ma',
            'Skew_63D_StdDev': 'skew_63d_std',
            'Skew_63D_ZScore': 'skew_63d_zscore',
            
            # Signals
            'ST_Signal (5-day)': 'signal_short_term',
            'MT_Signal (21-day)': 'signal_medium_term', 
            'LT_Signal (63-day)': 'signal_long_term',
            'Composite_Signal': 'composite_signal',
            
            # Returns and performance
            'Daily_RET': 'daily_return',
            'Strategy_Ret': 'strategy_return',
            'Cummulative_PNL': 'cumulative_pnl',
            
            # Performance metrics
            'Active trades': 'active_trades',
            'Winning trades': 'winning_trades', 
            'True win rate': 'win_rate',
            'sharpe ratio': 'sharpe_ratio',
            'average win': 'avg_win',
            'average loss': 'avg_loss',
            'average win/loss': 'win_loss_ratio',
            'Profit factor - multi': 'profit_factor'
        }
        
        # Apply column mapping
        for old_name, new_name in column_mapping.items():
            if old_name in df_standardized.columns:
                df_standardized[new_name] = df_standardized[old_name]
                print(f"   Mapped '{old_name}' -> '{new_name}'")
        
        return df_standardized
    
    def merge_datasets(self, spx_df: pd.DataFrame, mt_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge SPX raw data with multi-timeframe signals
        
        Args:
            spx_df: SPX raw data DataFrame
            mt_df: Multi-timeframe signals DataFrame
            
        Returns:
            Merged DataFrame with all data
        """
        try:
            # Merge on Date column
            merged_df = pd.merge(
                spx_df, 
                mt_df, 
                on='Date', 
                how='inner',
                suffixes=('_spx', '_mt')
            )
            
            print(f"✅ Merged datasets: {len(merged_df)} common dates")
            print(f"📊 Final columns: {len(merged_df.columns)}")
            
            return merged_df
            
        except Exception as e:
            print(f"❌ Error merging datasets: {e}")
            return pd.DataFrame()
    
    def validate_data_quality(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """
        Validate data quality and completeness
        
        Args:
            df: DataFrame to validate
            dataset_name: Name of dataset for reporting
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'dataset': dataset_name,
            'total_rows': len(df),
            'date_range_start': df['Date'].min(),
            'date_range_end': df['Date'].max(), 
            'missing_dates': self._check_missing_dates(df),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_dates': df['Date'].duplicated().sum()
        }
        
        print(f"\n📋 DATA VALIDATION - {dataset_name}:")
        print(f"   Rows: {validation['total_rows']}")
        print(f"   Date Range: {validation['date_range_start']} to {validation['date_range_end']}")
        print(f"   Missing Dates: {validation['missing_dates']}")
        print(f"   Duplicate Dates: {validation['duplicate_dates']}")
        
        # Show columns with most missing values
        missing_series = df.isnull().sum()
        if missing_series.sum() > 0:
            print("   Columns with missing values:")
            for col, missing_count in missing_series[missing_series > 0].items():
                print(f"     - {col}: {missing_count} missing ({missing_count/len(df)*100:.1f}%)")
        
        return validation
    
    def _check_missing_dates(self, df: pd.DataFrame) -> int:
        """Check for missing dates in the time series"""
        if 'Date' not in df.columns:
            return 0
            
        full_date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
        missing_dates = full_date_range.difference(df['Date'])
        return len(missing_dates)

# Example usage and testing
if __name__ == "__main__":
    print("Testing BloombergDataLoader...")
    
    # Initialize loader
    loader = BloombergDataLoader()
    
    # Test loading SPX data
    print("\n" + "="*50)
    print("TESTING SPX DATA LOADING")
    print("="*50)
    
    spx_data = loader.load_spx_data('data/raw/BB vol - SPX.1.csv')
    
    if not spx_data.empty:
        # Validate data quality
        spx_validation = loader.validate_data_quality(spx_data, "SPX Raw Data")
        
        # Show first few rows
        print("\nFirst 3 rows of SPX data:")
        print(spx_data[['Date', 'price', 'skew_1m_25d', 'daily_return']].head(3))
    
    # Test loading multi-timeframe data  
    print("\n" + "="*50)
    print("TESTING MULTI-TIMEFRAME DATA LOADING")
    print("="*50)
    
    mt_data = loader.load_multitimeframe_data('data/raw/B vol - Multitimeframe.csv')
    
    if not mt_data.empty:
        # Validate data quality
        mt_validation = loader.validate_data_quality(mt_data, "Multi-timeframe Data")
        
        # Show first few rows
        print("\nFirst 3 rows of multi-timeframe data:")
        """
Bloomberg data loader for volatility skew analysis
Handles actual CSV files with real column names from Bloomberg data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

class BloombergDataLoader:
    """
    Load and process Bloomberg volatility data for SPX options
    """

    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None

    def load_spx_data(self, file_path: str) -> pd.DataFrame:
        """
        Load SPX volatility data from CSV file with actual column names
        """
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()
            print(f"Raw columns found: {list(df.columns)}")

            # Detect and safely parse date column
            date_columns = ['Date', 'DATE', 'date']
            date_col = next((col for col in date_columns if col in df.columns), None)

            if date_col:
                df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
                df = df.dropna(subset=['Date'])
                print(f"Using '{date_col}' as date column (invalid entries removed)")
            else:
                print("⚠️ No date column found — creating synthetic date range.")
                df['Date'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')

            df = df.sort_values('Date').reset_index(drop=True)
            df = self._standardize_column_names(df)

            print(f"✅ Loaded SPX data with {len(df)} valid rows")
            print(f"📅 Date range: {df['Date'].min()} → {df['Date'].max()}")
            return df

        except Exception as e:
            print(f"❌ Error loading SPX data: {e}")
            return pd.DataFrame()

    def load_multitimeframe_data(self, file_path: str) -> pd.DataFrame:
        """
        Load multi-timeframe signals data from CSV file
        """
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()
            print(f"Multi-timeframe raw columns: {list(df.columns)}")

            date_columns = ['Date', 'DATE', 'date']
            date_col = next((col for col in date_columns if col in df.columns), None)

            if date_col:
                df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
                df = df.dropna(subset=['Date'])
                print(f"Using '{date_col}' as date column (invalid entries removed)")
            else:
                print("⚠️ No date column found — creating synthetic date range.")
                df['Date'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')

            df = df.sort_values('Date').reset_index(drop=True)
            df = self._standardize_multitimeframe_columns(df)

            print(f"✅ Loaded multi-timeframe data with {len(df)} valid rows")
            print(f"📅 Date range: {df['Date'].min()} → {df['Date'].max()}")
            return df

        except Exception as e:
            print(f"❌ Error loading multi-timeframe data: {e}")
            return pd.DataFrame()

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names for SPX raw data
        """
        df_standardized = df.copy()
        column_mapping = {
            'PRICE': 'price',
            'VOLUME': 'volume',
            '25DELTA_PUT_1M': 'put_iv_1m_25d',
            '25DELTA_CALL_1M': 'call_iv_1m_25d',
            '25DELTA_PUT_2M': 'put_iv_2m_25d',
            '25DELTA_CALL_2M': 'call_iv_2m_25d',
            'LEAP_25DELTA_P': 'put_iv_leap_25d',
            'LEAP_25DELTA_C': 'call_iv_leap_25d',
            '30D hist_vol': 'historical_vol_30d',
            '1M_25D_skew': 'skew_1m_25d',
            '2M_25D_SKEW': 'skew_2m_25d',
            'Leap_25D_SKEW': 'skew_leap_25d',
            'Term_slope_1,2m': 'term_slope_1m_2m',
            'Term_slope_2m,leap': 'term_slope_2m_leap',
            'IV,HV_PREM_1M': 'iv_hv_premium_1m',
            'IV,HV_PREM_2M': 'iv_hv_premium_2m'
        }

        for old_name, new_name in column_mapping.items():
            if old_name in df_standardized.columns:
                df_standardized[new_name] = df_standardized[old_name]
                print(f"   Mapped '{old_name}' -> '{new_name}'")

        if 'price' in df_standardized.columns:
            df_standardized['daily_return'] = df_standardized['price'].pct_change()
            print("   Calculated daily returns from price data")

        return df_standardized

    def _standardize_multitimeframe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names for multi-timeframe signals data
        """
        df_standardized = df.copy()
        column_mapping = {
            '1M_Skew_Value': 'skew_1m_value',
            'Skew_5D_MA': 'skew_5d_ma',
            'Skew_5D_StdDev': 'skew_5d_std',
            'Skew_5D_ZScore': 'skew_5d_zscore',
            'Skew_21D_MA': 'skew_21d_ma',
            'Skew_21D_StdDev': 'skew_21d_std',
            'Skew_21D_ZScore': 'skew_21d_zscore',
            'Skew_63D_MA': 'skew_63d_ma',
            'Skew_63D_StdDev': 'skew_63d_std',
            'Skew_63D_ZScore': 'skew_63d_zscore',
            'ST_Signal (5-day)': 'signal_short_term',
            'MT_Signal (21-day)': 'signal_medium_term',
            'LT_Signal (63-day)': 'signal_long_term',
            'Composite_Signal': 'composite_signal',
            'Daily_RET': 'daily_return',
            'Strategy_Ret': 'strategy_return',
            'Cummulative_PNL': 'cumulative_pnl',
            'Active trades': 'active_trades',
            'Winning trades': 'winning_trades',
            'True win rate': 'win_rate',
            'sharpe ratio': 'sharpe_ratio',
            'average win': 'avg_win',
            'average loss': 'avg_loss',
            'average win/loss': 'win_loss_ratio',
            'Profit factor': 'profit_factor'
        }

        for old_name, new_name in column_mapping.items():
            if old_name in df_standardized.columns:
                df_standardized[new_name] = df_standardized[old_name]
                print(f"   Mapped '{old_name}' -> '{new_name}'")

        return df_standardized

    def merge_datasets(self, spx_df: pd.DataFrame, mt_df: pd.DataFrame) -> pd.DataFrame:
        """Merge SPX raw data with multi-timeframe signals"""
        try:
            merged_df = pd.merge(spx_df, mt_df, on='Date', how='inner', suffixes=('_spx', '_mt'))
            print(f"✅ Merged datasets: {len(merged_df)} common dates")
            print(f"📊 Final columns: {len(merged_df.columns)}")
            return merged_df
        except Exception as e:
            print(f"❌ Error merging datasets: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    print("Testing BloombergDataLoader...\n")

    loader = BloombergDataLoader()
    spx_data = loader.load_spx_data('data/raw/vol - SPX.1.csv')
    mt_data = loader.load_multitimeframe_data('data/raw/vol - Multitimeframe.csv')

    if not spx_data.empty and not mt_data.empty:
        merged = loader.merge_datasets(spx_data, mt_data)
        print(f"\n✅ Successfully merged {len(merged)} rows.")
