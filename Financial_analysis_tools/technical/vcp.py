import pandas as pd
import numpy as np
import datetime
import requests
from io import StringIO
import logging
from multiprocessing import Pool, cpu_count
import csv
import time

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='vcp_search_debug.log',
                    filemode='w')

def transform_date(date):
    return date.strftime("%d%b%Y").upper().zfill(9)

def get_data(start_date, end_date, chunk_size=30):
    all_data = []
    current_date = start_date
    
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Skip weekends
            try:
                transformed_date = transform_date(current_date)
                csv_url = f'https://raw.githubusercontent.com/girishg4t/nse-bse-bhavcopy/master/nse/{transformed_date}.csv'

                response = requests.get(csv_url, timeout=10)
                response.raise_for_status()

                df = pd.read_csv(StringIO(response.text))
                df = df[['SYMBOL', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TOTTRDQTY', 'TIMESTAMP']]
                df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%d-%b-%Y')
                df = df.drop_duplicates(subset=['TIMESTAMP', 'SYMBOL'])

                all_data.append(df)
                logging.info(f"Fetched data for {current_date}")
            except requests.RequestException as e:
                logging.warning(f"Failed to fetch data for {current_date}: {str(e)}")
            except Exception as e:
                logging.error(f"Error processing data for {current_date}: {str(e)}")
        
        current_date += datetime.timedelta(days=1)

    if not all_data:
        logging.warning("No data was successfully fetched for the given date range")
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    logging.info(f"Final dataframe shape: {result.shape}")
    return result

def check_data(df):
    logging.debug(f"DataFrame shape: {df.shape}")
    logging.debug(f"DataFrame columns: {df.columns}")
    logging.debug(f"First few rows:\n{df.head()}")
    logging.debug(f"Data types:\n{df.dtypes}")
    logging.debug(f"Missing values:\n{df.isnull().sum()}")

def get_top_stocks_by_adv(df, n=500):  # Reduced for testing
    stock_adv = {}
    for column in df.columns:
        if '_TOTTRDQTY' in column:
            symbol = column.split('_')[0]
            adv = df[column].mean()
            stock_adv[symbol] = adv

    sorted_stocks = sorted(stock_adv.items(), key=lambda x: x[1], reverse=True)
    return [symbol for symbol, _ in sorted_stocks[:n]]

def vcp_search(stock_data):
    # Simplified VCP detection
    if len(stock_data) < 50:
        return pd.DataFrame()

    try:
        # Calculate basic metrics
        stock_data = stock_data.copy()
        stock_data['high_20'] = stock_data['high'].rolling(20).max()
        stock_data['volume_ma'] = stock_data['volume'].rolling(20).mean()
        
        # Simple VCP conditions
        recent_high = stock_data['high'].iloc[-1]
        recent_low = stock_data['low'].iloc[-1]
        recent_volume = stock_data['volume'].iloc[-1]
        avg_volume = stock_data['volume_ma'].iloc[-1]
        
        # Check if near highs and volume is decreasing
        near_high = recent_high > stock_data['high'].iloc[-20:-1].max() * 0.95
        volume_decreasing = recent_volume < avg_volume * 0.8
        
        if near_high and volume_decreasing:
            return stock_data.iloc[[-1]]
        else:
            return pd.DataFrame()
            
    except Exception as e:
        logging.error(f"Error in VCP search: {e}")
        return pd.DataFrame()

def evaluate_stock(df, symbol):
    try:
        required_columns = [f'{symbol}_OPEN', f'{symbol}_HIGH', f'{symbol}_LOW', f'{symbol}_CLOSE', f'{symbol}_TOTTRDQTY']
        if not all(col in df.columns for col in required_columns):
            logging.warning(f"Missing required columns for {symbol}")
            return None

        stock_data = df[['TIMESTAMP'] + required_columns].copy()
        stock_data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        stock_data = stock_data.dropna()

        if stock_data.empty:
            logging.warning(f"No valid data for {symbol} after dropping NaN values")
            return None

        stock_data['timestamp'] = pd.to_datetime(stock_data['timestamp'])
        stock_data = stock_data.set_index('timestamp')
        stock_data = stock_data.resample('B').ffill()

        stock_data = stock_data.apply(pd.to_numeric, errors='coerce')
        stock_data = stock_data.dropna()

        if len(stock_data) < 50:
            logging.warning(f"Not enough data points for {symbol}")
            return None

        if stock_data['close'].isnull().any() or (stock_data['close'] <= 0).any():
            logging.warning(f"Invalid price data found for {symbol}")
            return None

        logging.debug(f"Evaluating stock: {symbol}")
        logging.debug(f"Stock data shape: {stock_data.shape}")

        vcp_results = vcp_search(stock_data)

        if not vcp_results.empty:
            result = {
                'symbol': symbol,
                'date': vcp_results.index[-1],
                'close': vcp_results['close'].iloc[-1],
                'adv': stock_data['volume'].rolling(window=50).mean().iloc[-1]
            }
            logging.info(f"VCP pattern detected for {symbol}")
            return result
        else:
            logging.debug(f"No VCP pattern detected for {symbol}")
            return None

    except Exception as e:
        logging.error(f"Error processing stock {symbol}: {str(e)}", exc_info=True)
        return None

def process_stock_wrapper(args):
    return evaluate_stock(*args)

def main():
    # Use recent dates
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=90)

    while start_date.weekday() >= 5:
        start_date += datetime.timedelta(days=1)

    while end_date.weekday() >= 5:
        end_date -= datetime.timedelta(days=1)

    logging.info(f"Fetching data from {start_date} to {end_date}")
    df = get_data(start_date, end_date)

    if df.empty:
        logging.error("No data available for the specified date range.")
        return

    check_data(df)

    top_stocks = get_top_stocks_by_adv(df, n=200)  # Reduced for testing
    logging.info(f"Processing top {len(top_stocks)} stocks by ADV")

    tasks = [(df, symbol) for symbol in top_stocks]
    results = []

    num_processes = min(cpu_count(), 8)  # Reduced for stability
    logging.info(f"Using {num_processes} processes for parallel processing")

    start_time = time.time()
    total_stocks = len(tasks)
    with Pool(processes=num_processes) as pool:
        for i, result in enumerate(pool.imap_unordered(process_stock_wrapper, tasks, chunksize=5), 1):
            if result is not None:
                results.append(result)

            if i % 10 == 0 or i == total_stocks:
                progress = (i / total_stocks) * 100
                elapsed_time = time.time() - start_time
                estimated_total_time = elapsed_time * (total_stocks / i)
                remaining_time = estimated_total_time - elapsed_time
                logging.info(f"Processed {i}/{total_stocks} stocks ({progress:.2f}%) - "
                             f"Elapsed: {elapsed_time/60:.2f} min, "
                             f"Remaining: {remaining_time/60:.2f} min")

    logging.info(f"Total results found: {len(results)}")

    # Save results to CSV
    output_file = 'vcp_stocks_debug.csv'
    if results:
        pd.DataFrame(results).to_csv(output_file, index=False)
        logging.info(f"Results saved to {output_file}")
    else:
        logging.warning("No VCP patterns found")

    total_time = time.time() - start_time
    logging.info(f"Total execution time: {total_time/60:.2f} minutes")

if __name__ == "__main__":
    main()
