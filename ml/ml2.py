import os
import pandas as pd
import numpy as np
import datetime
import pandas_ta as ta
from multiprocessing import Pool, cpu_count
import logging
import requests
from io import StringIO
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import csv
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='stock_processing_ml.log',
                    filemode='w')


def transform_date(date):
    return date.strftime("%d%b%Y").upper().zfill(9)


def get_data(start_date, end_date, chunk_size=30):
    all_data = []
    current_date = start_date
    chunk_end = min(
        current_date + datetime.timedelta(days=chunk_size), end_date)

    while current_date <= chunk_end:
        chunk_data = []  # Initialize chunk_data at the beginning of each chunk
        if current_date.weekday() < 5:  # Skip weekends
            try:
                transformed_date = transform_date(current_date)
                csv_url = f'https://raw.githubusercontent.com/girishg4t/nse-bse-bhavcopy/master/nse/{transformed_date}.csv'

                response = requests.get(csv_url)
                response.raise_for_status()

                # If we reach this point, the request was successful
                df = pd.read_csv(StringIO(response.text))
                df = df[['SYMBOL', 'OPEN', 'HIGH', 'LOW',
                         'CLOSE', 'TOTTRDQTY', 'TIMESTAMP']]
                df['TIMESTAMP'] = pd.to_datetime(
                    df['TIMESTAMP'], format='%d-%b-%Y')
                df = df.drop_duplicates(subset=['TIMESTAMP', 'SYMBOL'])

                chunk_data.append(df)
                logging.info(f"Fetched data for {current_date}")
            except requests.RequestException as e:
                logging.warning(
                    f"Failed to fetch data for {current_date}: {str(e)}")
            except Exception as e:
                logging.error(
                    f"Error processing data for {current_date}: {str(e)}")
        else:
            logging.info(f"Skipping weekend date: {current_date}")

        current_date += datetime.timedelta(days=1)

        if chunk_data:
            chunk_df = pd.concat(chunk_data, ignore_index=True)
            chunk_pivot = chunk_df.pivot(index='TIMESTAMP', columns='SYMBOL', values=[
                                         'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TOTTRDQTY'])
            chunk_pivot.columns = [
                f'{col[1]}_{col[0]}' for col in chunk_pivot.columns]
            chunk_pivot = chunk_pivot.reset_index()
            all_data.append(chunk_pivot)

        chunk_end = min(
            current_date + datetime.timedelta(days=chunk_size), end_date)

    if not all_data:
        logging.warning(
            "No data was successfully fetched for the given date range")
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    logging.info(f"Final dataframe shape: {result.shape}")
    return result


def wave_trend(data, n1=10, n2=21):
    ap = (data['high'] + data['low'] + data['close']) / 3
    esa = ap.ewm(span=n1, adjust=False).mean()
    d = (ap - esa).abs().ewm(span=n1, adjust=False).mean()
    ci = (ap - esa) / (0.015 * d)
    tci = ci.ewm(span=n2, adjust=False).mean()
    wt1 = tci
    wt2 = wt1.rolling(4).mean()
    return pd.DataFrame({'WT1': wt1, 'WT2': wt2})


def cci(data, length=20):
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    sma = typical_price.rolling(window=length).mean()
    mad = typical_price.rolling(window=length).apply(
        lambda x: np.abs(x - x.mean()).mean())
    cci = (typical_price - sma) / (0.015 * mad)
    return pd.Series(cci, name=f'CCI_{length}')


def adx(data, length=14):
    high, low, close = data['high'], data['low'], data['close']
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1, join='inner').max(axis=1)
    atr = tr.rolling(window=length).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr)
    minus_di = abs(
        100 * (minus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1 / length, adjust=False).mean()
    return pd.Series(adx, name=f'ADX_{length}')


def handle_nan_inf(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df


def calculate_adv(volume_data, window=30):
    return volume_data.rolling(window=window).mean()


def generate_enhanced_features(stock_data):
    features = pd.DataFrame(index=stock_data.index)

    features['RSI'] = stock_data['RSI_14']
    features['CCI'] = stock_data['CCI_20']
    features['ADX'] = stock_data['ADX_14']
    features['MFI'] = stock_data['MFI_14']
    features['WT1'] = stock_data['WT1']
    features['WT2'] = stock_data['WT2']
    features['Price_Change'] = stock_data['close'].pct_change()
    features['Price_Volatility'] = stock_data['close'].rolling(window=20).std()
    features['Volume_Change'] = stock_data['volume'].pct_change()
    features['Volume_MA_Ratio'] = stock_data['volume'] / \
        stock_data['volume'].rolling(window=20).mean()
    features['MA_10'] = stock_data['close'].rolling(window=10).mean()
    features['MA_50'] = stock_data['close'].rolling(window=50).mean()
    features['MA_Ratio'] = features['MA_10'] / features['MA_50']

    macd = stock_data.ta.macd()
    features['MACD'] = macd['MACD_12_26_9']
    features['MACD_Signal'] = macd['MACDs_12_26_9']
    features['MACD_Hist'] = macd['MACDh_12_26_9']

    features['ATR'] = stock_data.ta.atr(length=14)
    features['ROC'] = stock_data.ta.roc(length=10)
    features['CMF'] = stock_data.ta.cmf(length=20)

    features['OBV'] = stock_data.ta.obv()

    for period in [5, 10, 20, 50]:
        features[f'Price_Momentum_{period}'] = stock_data['close'].pct_change(
            period)

    features['Relative_Volume'] = stock_data['volume'] / \
        stock_data['volume'].rolling(window=20).mean()

    features['Day_of_Week'] = stock_data.index.dayofweek
    features['Month'] = stock_data.index.month

    return features.dropna()


def generate_labels(stock_data, lookahead=5, threshold=0.02):
    future_returns = stock_data['close'].pct_change(
        periods=lookahead).shift(-lookahead)
    labels = pd.Series(1, index=stock_data.index)  # Default to hold (1)
    labels[future_returns > threshold] = 2  # Buy signal
    labels[future_returns < -threshold] = 0  # Sell signal
    return labels


def train_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42)

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Label distribution in training set: {np.bincount(y_train)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    selector = SelectFromModel(lgb.LGBMClassifier(
        random_state=42), max_features=20)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'max_depth': -1,
        'n_jobs': -1,
        'verbose': -1
    }

    train_data = lgb.Dataset(X_train_selected, label=y_train)
    valid_data = lgb.Dataset(
        X_test_selected, label=y_test, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    y_pred = model.predict(X_test_selected)
    y_pred_class = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_test, y_pred_class)
    print(f"Model Accuracy: {accuracy}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_class,
                                target_names=['Sell', 'Hold', 'Buy']))

    feature_importance = analyze_feature_importance(
        model, features.columns[selector.get_support()])
    print("Feature Importance:")
    for feature, importance in feature_importance:
        print(f"{feature}: {importance}")

    return model, accuracy, scaler, selector


def analyze_feature_importance(model, feature_names):
    importances = model.feature_importance()
    return sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)


def score_stock(stock_data, features, model, scaler, selector, weights=None):
    score = 0
    signals = {}

    scaled_features = scaler.transform(features.to_frame().T)
    selected_features = selector.transform(scaled_features)

    prediction = model.predict(selected_features)
    model_signal = np.argmax(prediction, axis=1)[
        0] - 1  # Convert 0, 1, 2 to -1, 0, 1

    signals['RSI'] = 1 if features['RSI'] < 30 else (
        -1 if features['RSI'] > 70 else 0)
    signals['MACD'] = 1 if features['MACD'] > features['MACD_Signal'] else -1
    signals['ROC'] = 1 if features['ROC'] > 0 else -1
    signals['CMF'] = 1 if features['CMF'] > 0 else -1
    signals['MA_Cross'] = 1 if features['MA_10'] > features['MA_50'] else -1
    signals['Volume_Spike'] = 1 if features['Relative_Volume'] > 2 else 0
    signals['Price_Momentum'] = 1 if features['Price_Momentum_20'] > 0 else -1
    signals['ML_Model'] = model_signal

    if weights is None:
        weights = {
            'RSI': 1, 'MACD': 1, 'ROC': 1.5, 'CMF': 1.5, 'MA_Cross': 1,
            'Volume_Spike': 1, 'Price_Momentum': 1.5, 'ML_Model': 2
        }

    for signal, value in signals.items():
        score += value * weights.get(signal, 1)

    return score, signals


def evaluate_stock(df, symbol, min_adv, scoring_threshold=2):
    MIN_DATA_POINTS = 100

    if len(df) < MIN_DATA_POINTS:
        logging.warning(f"Not enough data to process {symbol}")
        return None

    try:
        required_columns = [f'{symbol}_OPEN', f'{symbol}_HIGH',
                            f'{symbol}_LOW', f'{symbol}_CLOSE', f'{symbol}_TOTTRDQTY']
        if not all(col in df.columns for col in required_columns):
            logging.warning(f"Missing required columns for {symbol}")
            return None

        stock_data = df[['TIMESTAMP'] + required_columns].copy()
        stock_data.columns = ['TIMESTAMP', 'open',
                              'high', 'low', 'close', 'volume']
        stock_data = stock_data.dropna()

        if stock_data.empty:
            logging.warning(
                f"No valid data for {symbol} after dropping NaN values")
            return None

        stock_data['TIMESTAMP'] = pd.to_datetime(stock_data['TIMESTAMP'])
        stock_data = stock_data.set_index('TIMESTAMP')
        stock_data = stock_data.resample('B').ffill()

        stock_data = stock_data.apply(pd.to_numeric, errors='coerce')
        stock_data = stock_data.dropna()

        if len(stock_data) < MIN_DATA_POINTS:
            logging.warning(
                f"Not enough valid data points for {symbol} after conversion")
            return None

        stock_data = stock_data.sort_index()

        if stock_data['close'].isnull().any() or (stock_data['close'] <= 0).any():
            logging.warning(f"Invalid price data found for {symbol}")
            return None

        # Calculate ADV
        stock_data['ADV'] = calculate_adv(stock_data['volume'])

        # Apply volume filter
        if stock_data['ADV'].iloc[-1] < min_adv:
            logging.info(
                f"Skipping {symbol} due to low ADV: {stock_data['ADV'].iloc[-1]:.2f}")
            return None

        # Calculate technical indicators
        stock_data.ta.rsi(length=14, append=True)
        stock_data['CCI_20'] = cci(stock_data, length=20)
        stock_data['ADX_14'] = adx(stock_data, length=14)
        stock_data.ta.mfi(length=14, append=True)
        wt = wave_trend(stock_data)
        stock_data['WT1'] = wt['WT1']
        stock_data['WT2'] = wt['WT2']

        stock_data = handle_nan_inf(stock_data)

        features = generate_enhanced_features(stock_data)
        labels = generate_labels(stock_data)

        # Align features and labels
        valid_data = features.dropna().index.intersection(labels.dropna().index)
        features = features.loc[valid_data]
        labels = labels.loc[valid_data]

        if len(features) < MIN_DATA_POINTS:
            logging.warning(
                f"Not enough valid data points for {symbol} after feature generation")
            return None

        model, accuracy, scaler, selector = train_model(features, labels)

        # Generate predictions for the most recent data point
        latest_features = features.iloc[-1]

        # Calculate final score
        final_score, final_signals = score_stock(
            stock_data, latest_features, model, scaler, selector)

        if abs(final_score) >= scoring_threshold:
            result = {
                'symbol': symbol,
                'date': stock_data.index[-1],
                'close': stock_data['close'].iloc[-1],
                'adv': stock_data['ADV'].iloc[-1],
                'score': final_score,
                'signals': final_signals,
                'model_accuracy': accuracy
            }
            logging.info(
                f"Successfully processed {symbol} with score {final_score}")
            return result
        else:
            logging.info(
                f"No strong signal for {symbol} (score: {final_score})")
            return None

    except Exception as e:
        logging.error(
            f"Error processing stock {symbol}: {str(e)}", exc_info=True)
        return None


def process_stock_wrapper(args):
    return evaluate_stock(*args)


def filter_stocks(input_file, output_file_filtered, output_file_remaining):
    filter_stocks = set()
    with open('filter_stocks.txt', 'r') as f:
        for line in f:
            filter_stocks.add(line.strip())

    with open(input_file, 'r') as infile, \
            open(output_file_filtered, 'w', newline='') as outfile_filtered, \
            open(output_file_remaining, 'w', newline='') as outfile_remaining:

        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        writer_filtered = csv.DictWriter(
            outfile_filtered, fieldnames=fieldnames)
        writer_remaining = csv.DictWriter(
            outfile_remaining, fieldnames=fieldnames)

        writer_filtered.writeheader()
        writer_remaining.writeheader()

        for row in reader:
            if row['symbol'] in filter_stocks:
                writer_filtered.writerow(row)
            else:
                writer_remaining.writerow(row)


def get_top_stocks_by_adv(df, n=1000):
    stock_adv = {}
    for column in df.columns:
        if '_TOTTRDQTY' in column:
            symbol = column.split('_')[0]
            adv = df[column].mean()
            stock_adv[symbol] = adv

    sorted_stocks = sorted(stock_adv.items(), key=lambda x: x[1], reverse=True)
    return [symbol for symbol, _ in sorted_stocks[:n]]


def main():
    start_date = datetime.date(2020, 4, 1)
    end_date = datetime.date(2024, 7, 4)
    min_adv = 100000  # Minimum Average Daily Volume

    while start_date.weekday() >= 5:
        start_date += datetime.timedelta(days=1)

    while end_date.weekday() >= 5:
        end_date -= datetime.timedelta(days=1)

    logging.info(f"Fetching data from {start_date} to {end_date}")
    df = get_data(start_date, end_date)

    if df.empty:
        logging.error("No data available for the specified date range.")
        return

    top_stocks = get_top_stocks_by_adv(df, n=1000)
    logging.info(f"Processing top {len(top_stocks)} stocks by ADV")

    tasks = [(df, symbol, min_adv) for symbol in top_stocks]
    buy_signals = {}
    sell_signals = {}
    successful_stocks = 0
    failed_stocks = 0

    num_processes = min(cpu_count(), 16)
    logging.info(f"Using {num_processes} processes for parallel processing")

    start_time = time.time()
    total_stocks = len(tasks)
    with Pool(processes=num_processes) as pool:
        for i, result in enumerate(pool.imap_unordered(process_stock_wrapper, tasks, chunksize=10), 1):
            if result is not None:
                if result['score'] > 0:
                    buy_signals[result['symbol']] = result
                elif result['score'] < 0:
                    sell_signals[result['symbol']] = result
                successful_stocks += 1
            else:
                failed_stocks += 1

            if i % 10 == 0 or i == total_stocks:
                progress = (i / total_stocks) * 100
                elapsed_time = time.time() - start_time
                estimated_total_time = elapsed_time * (total_stocks / i)
                remaining_time = estimated_total_time - elapsed_time
                logging.info(f"Processed {i}/{total_stocks} stocks ({progress:.2f}%) - "
                             f"Elapsed: {elapsed_time/60:.2f} min, "
                             f"Remaining: {remaining_time/60:.2f} min")

    # Save results to CSV
    output_file = 'stock_signals.csv'
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['symbol', 'date', 'close', 'adv',
                      'score', 'signals', 'model_accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for signal in buy_signals.values():
            writer.writerow(signal)
        for signal in sell_signals.values():
            writer.writerow(signal)

    logging.info(f"Results saved to {output_file}")

    # Filter stocks
    input_file = 'stock_signals.csv'
    output_file_filtered = 'mlfiltered_stock_signals.csv'
    output_file_remaining = 'mlremaining_stock_signals.csv'

    filter_stocks(input_file, output_file_filtered, output_file_remaining)

    logging.info(f"Filtered results saved to {output_file_filtered}")
    logging.info(f"Remaining results saved to {output_file_remaining}")

    total_time = time.time() - start_time
    logging.info(f"Total execution time: {total_time/60:.2f} minutes")


if __name__ == "__main__":
    main()
