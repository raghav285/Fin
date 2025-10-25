import pandas as pd
import numpy as np
import datetime
import logging
import requests
from io import StringIO
import csv
from scipy.stats import linregress

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='stock_data_fetcher.log',
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

def calculate_adv(df, window=30):
    """Calculate Average Daily Volume for each symbol"""
    adv_data = []
    for symbol in df['SYMBOL'].unique():
        symbol_data = df[df['SYMBOL'] == symbol].copy()
        symbol_data = symbol_data.sort_values('TIMESTAMP')
        symbol_data['ADV'] = symbol_data['TOTTRDQTY'].rolling(window=window, min_periods=1).mean()
        adv_data.append(symbol_data)
    
    return pd.concat(adv_data, ignore_index=True)

def calculate_dma(data, period):
    return data.rolling(window=period).mean()

def calculate_slope(y):
    if len(y) < 2:
        return 0
    x = np.arange(len(y))
    mask = ~np.isnan(y)
    if np.sum(mask) < 2:
        return 0
    slope, _, _, _, _ = linregress(x[mask], y[mask])
    return np.degrees(np.arctan(slope))

def dma_signal(data, period, angle=40):
    data = data.copy()
    data['DMA'] = calculate_dma(data['CLOSE'], period)
    data['DMA_prev'] = data['DMA'].shift(1)
    
    # Calculate slope safely
    data['DMA_slope'] = data['DMA'].rolling(window=3).apply(
        calculate_slope, raw=False
    )
    data['DMA_slope_prev'] = data['DMA_slope'].shift(1)

    # Check for concave to convex transition and slope > specified angle
    data['signal'] = (
        (data['DMA_slope_prev'] < 0) & 
        (data['DMA_slope'] > 0) & 
        (data['DMA_slope'].abs() > angle)
    ).astype(int)

    return data

def main():
    # Use recent dates
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=90)  # Last 3 months
    min_adv = 100000

    logging.info(f"Fetching data from {start_date} to {end_date}")
    df = get_data(start_date, end_date)

    if df.empty:
        logging.error("No data available for the specified date range.")
        return

    # Calculate ADV properly
    df = calculate_adv(df)

    # Filter stocks based on ADV
    latest_adv = df.groupby('SYMBOL')['ADV'].last().reset_index()
    top_stocks = latest_adv[latest_adv['ADV'] >= min_adv]['SYMBOL'].tolist()

    # Prepare data for CSV
    output_data = []
    for symbol in top_stocks[:50]:  # Limit to top 50 for testing
        stock_data = df[df['SYMBOL'] == symbol].sort_values('TIMESTAMP')
        
        if len(stock_data) < 30:  # Need enough data for DMA
            continue
            
        # Calculate DMA signal for daily timeframe
        stock_data = dma_signal(stock_data, period=30, angle=32.5)
        
        latest_data = stock_data.iloc[-1]  # Get the most recent data point
        output_data.append({
            'symbol': symbol,
            'date': latest_data['TIMESTAMP'],
            'close': latest_data['CLOSE'],
            'adv': latest_data['ADV'],
            'dma_signal': latest_data['signal']
        })

    # Save results to CSV
    output_file = 'stock_data_with_dma_signal.csv'
    if output_data:
        pd.DataFrame(output_data).to_csv(output_file, index=False)
        logging.info(f"Results saved to {output_file} with {len(output_data)} stocks")
    else:
        logging.warning("No data to save")

if __name__ == "__main__":
    main()
