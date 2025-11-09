"""
VCP (Volatility Contraction Pattern) Stock Scanner
A comprehensive tool for detecting VCP patterns in stock data.

Author: Raghav
Repository: https://github.com/raghav285/Fin
"""

import pandas as pd
import numpy as np
import datetime
import requests
from io import StringIO
import logging
from pathlib import Path
import sqlite3
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
import json


# Configuration
@dataclass
class Config:
    """Configuration settings for the VCP scanner."""
    
    # Data source
    DATA_URL_TEMPLATE: str = 'https://raw.githubusercontent.com/girishg4t/nse-bse-bhavcopy/master/nse/{date}.csv'
    
    # Database
    DB_PATH: str = 'vcp_scanner.db'
    CACHE_DIR: str = 'cache'
    
    # Analysis parameters
    LOOKBACK_DAYS: int = 90
    MIN_PRICE: float = 10.0
    MIN_ADV: float = 10000
    TOP_N_STOCKS: int = 500

    # VCP detection parameters
    MIN_CONTRACTIONS: int = 2
    MAX_CONTRACTION_WEEKS: int = 20
    FINAL_CONTRACTION_MAX: float = 0.30
    VOLUME_DECREASE_THRESHOLD: float = 0.95

    # Moving averages
    MA_SHORT: int = 20
    MA_LONG: int = 50

    # Relative strength
    RS_LOOKBACK: int = 63

    # Logging
    LOG_FILE: str = 'vcp_scanner.log'
    LOG_LEVEL: int = logging.INFO


config = Config()

# Set up logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Database:
    """Handles all database operations for stock data and VCP results."""

    def __init__(self, db_path: str = config.DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            # Stock data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stock_data (
                    symbol TEXT NOT NULL,
                    timestamp DATE NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    PRIMARY KEY (symbol, timestamp)
                )
            """)

            # VCP results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vcp_results (
                    symbol TEXT NOT NULL,
                    detection_date DATE NOT NULL,
                    close_price REAL,
                    num_contractions INTEGER,
                    final_contraction_range REAL,
                    volume_decrease REAL,
                    rs_rating REAL,
                    above_ma50 INTEGER,
                    above_ma200 INTEGER,
                    details TEXT,
                    PRIMARY KEY (symbol, detection_date)
                )
            """)

            # Create indexes for performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_stock_data_date
                ON stock_data(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_vcp_results_date
                ON vcp_results(detection_date)
            """)

    def save_stock_data(self, df: pd.DataFrame):
        """Save stock data to database."""
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql('stock_data', conn, if_exists='append', index=False)
            logger.info(f"Saved {len(df)} records to database")

    def get_stock_data(self, symbol: str, start_date: datetime.date,
                       end_date: datetime.date) -> pd.DataFrame:
        """Retrieve stock data for a symbol."""
        query = """
            SELECT * FROM stock_data
            WHERE symbol = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df

    def get_all_symbols(self) -> List[str]:
        """Get list of all symbols in database."""
        query = "SELECT DISTINCT symbol FROM stock_data"
        with sqlite3.connect(self.db_path) as conn:
            return [row[0] for row in conn.execute(query)]

    def get_date_range(self) -> Tuple[Optional[datetime.date], Optional[datetime.date]]:
        """Get min and max dates in database."""
        query = "SELECT MIN(timestamp), MAX(timestamp) FROM stock_data"
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(query).fetchone()
            if result[0] and result[1]:
                return (datetime.datetime.strptime(result[0], '%Y-%m-%d').date(),
                        datetime.datetime.strptime(result[1], '%Y-%m-%d').date())
            return None, None

    def save_vcp_result(self, result: dict):
        """Save VCP detection result."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO vcp_results VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result['symbol'],
                result['detection_date'],
                result['close_price'],
                result['num_contractions'],
                result['final_contraction_range'],
                result['volume_decrease'],
                result['rs_rating'],
                result['above_ma50'],
                result['above_ma200'],
                json.dumps(result.get('details', {}))
            ))

    def get_vcp_results(self, start_date: datetime.date = None) -> pd.DataFrame:
        """Get VCP detection results."""
        query = "SELECT * FROM vcp_results"
        params = []
        if start_date:
            query += " WHERE detection_date >= ?"
            params.append(start_date)
        query += " ORDER BY detection_date DESC, rs_rating DESC"

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)


class DataFetcher:
    """Handles fetching data from external sources."""

    def __init__(self, db: Database):
        self.db = db
        Path(config.CACHE_DIR).mkdir(exist_ok=True)

    @staticmethod
    def transform_date(date: datetime.date) -> str:
        """Transform date to API format."""
        return date.strftime("%d%b%Y").upper().zfill(9)

    def fetch_daily_data(self, date: datetime.date) -> Optional[pd.DataFrame]:
        """Fetch data for a single date."""
        if date.weekday() >= 5:  # Skip weekends
            return None

        try:
            transformed_date = self.transform_date(date)
            url = config.DATA_URL_TEMPLATE.format(date=transformed_date)

            response = requests.get(url, timeout=10)
            response.raise_for_status()

            df = pd.read_csv(StringIO(response.text))

            # Keep data in long format
            df = df[['SYMBOL', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TOTTRDQTY', 'TIMESTAMP']]
            df.columns = ['symbol', 'open', 'high', 'low', 'close', 'volume', 'timestamp']
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%b-%Y')
            df = df.drop_duplicates(subset=['timestamp', 'symbol'])

            logger.info(f"Fetched {len(df)} records for {date}")
            return df

        except requests.RequestException as e:
            logger.warning(f"Failed to fetch data for {date}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing data for {date}: {e}")
            return None

    def fetch_date_range(self, start_date: datetime.date,
                        end_date: datetime.date) -> pd.DataFrame:
        """Fetch data for a date range and save to database."""
        all_data = []
        current_date = start_date

        while current_date <= end_date:
            df = self.fetch_daily_data(current_date)
            if df is not None and not df.empty:
                all_data.append(df)
            current_date += datetime.timedelta(days=1)

        if not all_data:
            logger.warning("No data fetched for date range")
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)

        # Save to database
        try:
            self.db.save_stock_data(result)
        except sqlite3.IntegrityError:
            logger.info("Some records already exist in database, skipping duplicates")

        return result

    def update_data(self, start_date: datetime.date = None, end_date: datetime.date = None):
        """Update database with latest data."""
        if start_date is None or end_date is None:
            start_date = datetime.date(2024, 1, 1)
            end_date = datetime.date(2024, 6, 30)

        logger.info(f"Fetching data from {start_date} to {end_date}")

        if start_date <= end_date:
            self.fetch_date_range(start_date, end_date)
        else:
            logger.info("Invalid date range")


class TechnicalAnalyzer:
    """Performs technical analysis calculations."""

    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        df = df.copy()

        # Moving averages
        df['ma50'] = df['close'].rolling(window=config.MA_SHORT).mean()
        df['ma200'] = df['close'].rolling(window=config.MA_LONG).mean()

        # Volume moving average
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()
        df['volume_ma50'] = df['volume'].rolling(window=50).mean()

        # Price ranges
        df['high_low_range'] = ((df['high'] - df['low']) / df['low'] * 100)

        # Relative highs
        df['high_52week'] = df['high'].rolling(window=252).max()
        df['distance_from_high'] = (df['high_52week'] - df['close']) / df['high_52week'] * 100

        return df

    @staticmethod
    def calculate_relative_strength(df: pd.DataFrame, index_df: pd.DataFrame) -> float:
        """
        Calculate relative strength vs market index.
        Returns RS rating (0-100 scale).
        """
        if len(df) < config.RS_LOOKBACK or len(index_df) < config.RS_LOOKBACK:
            return 0.0

        # Calculate percentage returns
        stock_return = (df['close'].iloc[-1] / df['close'].iloc[-config.RS_LOOKBACK] - 1) * 100
        index_return = (index_df['close'].iloc[-1] / index_df['close'].iloc[-config.RS_LOOKBACK] - 1) * 100

        # RS is relative performance
        rs = stock_return - index_return

        # Normalize to 0-100 scale
        rs_rating = max(0, min(100, 50 + rs))

        return round(rs_rating, 2)


@dataclass
class Contraction:
    """Represents a single contraction in the VCP pattern."""
    start_idx: int
    end_idx: int
    high_price: float
    low_price: float
    range_pct: float
    avg_volume: float
    weeks: int


class VCPDetector:
    """Detects VCP patterns in stock data."""

    @staticmethod
    def find_pivot_highs(df: pd.DataFrame, window: int = 10) -> List[int]:
        """Find pivot high points in price action."""
        highs = []
        for i in range(window, len(df) - window):
            if df['high'].iloc[i] == df['high'].iloc[i-window:i+window+1].max():
                highs.append(i)
        return highs

    @staticmethod
    def identify_contractions(df: pd.DataFrame) -> List[Contraction]:
        """Identify contraction phases in the pattern."""
        if len(df) < 50:
            return []

        pivot_highs = VCPDetector.find_pivot_highs(df)

        if len(pivot_highs) < config.MIN_CONTRACTIONS + 1:
            return []

        contractions = []

        # Analyze each contraction between pivot highs
        for i in range(len(pivot_highs) - 1):
            start_idx = pivot_highs[i]
            end_idx = pivot_highs[i + 1]

            # Get data for this contraction
            contraction_data = df.iloc[start_idx:end_idx + 1]

            high_price = contraction_data['high'].max()
            low_price = contraction_data['low'].min()
            range_pct = ((high_price - low_price) / low_price) * 100
            avg_volume = contraction_data['volume'].mean()
            weeks = len(contraction_data) / 5  # Approximate weeks

            # Skip if contraction is too long
            if weeks > config.MAX_CONTRACTION_WEEKS:
                continue

            contractions.append(Contraction(
                start_idx=start_idx,
                end_idx=end_idx,
                high_price=high_price,
                low_price=low_price,
                range_pct=range_pct,
                avg_volume=avg_volume,
                weeks=weeks
            ))

        return contractions

    @staticmethod
    def validate_vcp_pattern(contractions: List[Contraction]) -> Tuple[bool, dict]:
        """
        Validate if contractions form a valid VCP pattern.
        Returns (is_valid, details).
        """
        if len(contractions) < config.MIN_CONTRACTIONS:
            return False, {'reason': 'Not enough contractions'}

        # Check if contractions are tightening
        tightening_count = 0
        for i in range(1, len(contractions)):
            if contractions[i].range_pct < contractions[i-1].range_pct:
                tightening_count += 1

        if tightening_count < len(contractions) - 2:
            return False, {'reason': 'Not enough tightening contractions'}

        # Check volume trend
        volumes = [c.avg_volume for c in contractions]
        volume_decrease = (volumes[-1] / volumes[0]) if volumes[0] > 0 else 1.0

        if volume_decrease > config.VOLUME_DECREASE_THRESHOLD:
            return False, {'reason': f'Volume not decreasing: {volume_decrease:.2f}'}

        # Check final contraction
        final_range = contractions[-1].range_pct
        if final_range > config.FINAL_CONTRACTION_MAX * 100:
            return False, {'reason': f'Final contraction too wide: {final_range:.2f}%'}

        details = {
            'num_contractions': len(contractions),
            'ranges': [round(c.range_pct, 2) for c in contractions],
            'final_range': round(final_range, 2),
            'volume_decrease': round(volume_decrease, 2),
            'total_weeks': sum(c.weeks for c in contractions)
        }

        return True, details

    @staticmethod
    def detect_vcp(df: pd.DataFrame, symbol: str) -> Optional[dict]:
        """
        Main VCP detection function.
        Returns result dict if VCP found, None otherwise.
        """
        if len(df) < config.MA_LONG:
            logger.debug(f"{symbol}: Not enough data ({len(df)} rows)")
            return None

        # Calculate indicators
        df = TechnicalAnalyzer.calculate_indicators(df)

        # Check basic requirements
        latest = df.iloc[-1]

        # Check for missing MA data
        if pd.isna(latest['ma50']) or pd.isna(latest['ma200']):
            logger.debug(f"{symbol}: MA data not available")
            # Continue anyway for relaxed detection

        # Check if above moving averages
        above_ma50 = latest['close'] > latest['ma50'] if not pd.isna(latest['ma50']) else True
        above_ma200 = latest['close'] > latest['ma200'] if not pd.isna(latest['ma200']) else True

        # Price requirements
        if latest['close'] < config.MIN_PRICE:
            logger.debug(f"{symbol}: Price too low (${latest['close']:.2f})")
            return None

        # Volume requirement
        avg_volume = df['volume'].tail(50).mean()
        if avg_volume < config.MIN_ADV:
            logger.debug(f"{symbol}: Volume too low ({avg_volume:.0f})")
            return None

        # Find contractions
        contractions = VCPDetector.identify_contractions(df)

        if not contractions:
            logger.debug(f"{symbol}: No contractions found")
            return None

        logger.debug(f"{symbol}: Found {len(contractions)} contractions")

        # Validate VCP pattern
        is_valid, details = VCPDetector.validate_vcp_pattern(contractions)

        if not is_valid:
            logger.debug(f"{symbol}: Pattern invalid - {details.get('reason', 'unknown')}")
            return None

        # Calculate relative strength (placeholder)
        rs_rating = 0.0

        result = {
            'symbol': symbol,
            'detection_date': latest['timestamp'].date(),
            'close_price': round(latest['close'], 2),
            'num_contractions': details['num_contractions'],
            'final_contraction_range': details['final_range'],
            'volume_decrease': details['volume_decrease'],
            'rs_rating': rs_rating,
            'above_ma50': 1 if above_ma50 else 0,
            'above_ma200': 1 if above_ma200 else 0,
            'details': details
        }

        return result


class VCPScanner:
    """Main scanner orchestrator."""

    def __init__(self):
        self.db = Database()
        self.fetcher = DataFetcher(self.db)
        self.detector = VCPDetector()

    def update_and_scan(self):
        """Update data and scan for VCP patterns."""
        logger.info("Starting VCP scan...")

        # Update data
        logger.info("Fetching stock data...")
        start_date = datetime.date(2024, 1, 1)
        end_date = datetime.date(2024, 6, 30)
        self.fetcher.update_data(start_date, end_date)

        # Get top stocks by volume
        logger.info("Selecting top stocks by volume...")
        stocks = self._get_top_stocks_by_volume(start_date, end_date)
        
        logger.info(f"Scanning {len(stocks)} stocks for VCP patterns...")
        
        # Scan each stock
        results = []
        for i, symbol in enumerate(stocks, 1):
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{len(stocks)} stocks scanned")
            
            df = self.db.get_stock_data(symbol, start_date, end_date)
            
            if df.empty or len(df) < config.MA_LONG:
                continue
            
            result = self.detector.detect_vcp(df, symbol)
            
            if result:
                self.db.save_vcp_result(result)
                results.append(result)
                logger.info(f"VCP detected: {symbol} - {result['num_contractions']} contractions")
        
        logger.info(f"Scan complete. Found {len(results)} VCP patterns.")
        
        return results
    
    def _get_top_stocks_by_volume(self, start_date: datetime.date, 
                                   end_date: datetime.date) -> List[str]:
        """Get top N stocks by average daily volume."""
        query = """
            SELECT symbol, AVG(volume) as avg_volume
            FROM stock_data
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY symbol
            HAVING avg_volume >= ?
            ORDER BY avg_volume DESC
            LIMIT ?
        """
        
        with sqlite3.connect(self.db.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=(
                start_date, end_date, config.MIN_ADV, config.TOP_N_STOCKS
            ))
        
        return df['symbol'].tolist()
    
    def get_latest_results(self, days: int = 7) -> pd.DataFrame:
        """Get VCP results from the last N days."""
        start_date = datetime.date.today() - datetime.timedelta(days=days)
        return self.db.get_vcp_results(start_date)
    
    def export_results(self, filename: str = 'vcp_results.csv'):
        """Export latest results to CSV."""
        df = self.get_latest_results()
        if not df.empty:
            df.to_csv(filename, index=False)
            logger.info(f"Results exported to {filename}")
        else:
            logger.info("No results to export")


def main():
    """Main execution function."""
    scanner = VCPScanner()
    
    # Run scan
    results = scanner.update_and_scan()
    
    # Export results
    if results:
        scanner.export_results()
        
        # Print summary
        print("\n" + "="*60)
        print("VCP SCAN RESULTS")
        print("="*60)
        for result in results:
            print(f"\n{result['symbol']}")
            print(f"  Date: {result['detection_date']}")
            print(f"  Price: ${result['close_price']}")
            print(f"  Contractions: {result['num_contractions']}")
            print(f"  Final Range: {result['final_contraction_range']:.2f}%")
            print(f"  Volume Decrease: {result['volume_decrease']:.2f}")
    else:
        print("\nNo VCP patterns detected in this scan.")


if __name__ == "__main__":
    main()
