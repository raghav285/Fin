"""
Bloomberg data loader for volatility skew analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

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
        Load SPX volatility data from Excel file
        
        Args:
            file_path: Path to Excel file with Bloomberg data
            
        Returns:
            DataFrame with SPX volatility data
        """
        try:
            df = pd.read_excel(file_path, sheet_name='SPX.1')
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            print(f"Loaded SPX data with {len(df)} rows")
            return df
        except Exception as e:
            print(f"Error loading SPX data: {e}")
            return pd.DataFrame()
    
    def calculate_skew_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility skew metrics from raw data
        
        Args:
            df: Raw DataFrame with Bloomberg data
            
        Returns:
            DataFrame with calculated skew metrics
        """
        df_processed = df.copy()
        
        # Calculate volatility skews (Put IV - Call IV)
        df_processed['1M_25D_Skew'] = (
            df_processed['25DELTA_PUT_1M'] - df_processed['25DELTA_CALL_1M']
        )
        df_processed['2M_25D_Skew'] = (
            df_processed['25DELTA_PUT_2M'] - df_processed['25DELTA_CALL_2M']
        )
        df_processed['LEAP_25D_Skew'] = (
            df_processed['LEAP_25DELTA_P'] - df_processed['LEAP_25DELTA_C']
        )
        
        # Calculate daily returns
        df_processed['Daily_Return'] = df_processed['PRICE'].pct_change()
        
        # Calculate term structure slopes
        df_processed['Term_Slope_1M_2M'] = (
            df_processed['2M_ATM_IV'] - df_processed['1M_ATM_IV']
        )
        
        print("Calculated skew metrics and returns")
        return df_processed

if __name__ == "__main__":
    loader = BloombergDataLoader()
    print("BloombergDataLoader class ready for use")
