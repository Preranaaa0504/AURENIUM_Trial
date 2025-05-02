import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
from alpaca_trade_api import REST, TimeFrame

# Configuration file and directory setup
main_path = os.path.dirname(os.path.realpath(__file__))
config_file = os.path.join(main_path, 'app_config.json')
data_path = os.path.join(main_path, 'data_storage')
os.makedirs(data_path, exist_ok=True)

stock_file = os.path.join(data_path, 'stocks.csv')
price_file = os.path.join(data_path, 'prices.csv')

# StockManager handles stock and price data using CSV
class StockManager:
    def __init__(self):
        self.stock_file = stock_file
        self.price_file = price_file

        # Load or create stock dataframe
        if os.path.exists(self.stock_file):
            self.df_stocks = pd.read_csv(self.stock_file)
        else:
            self.df_stocks = pd.DataFrame(columns=['symbol', 'name', 'sector', 'sp_500'])

        # Load or create price dataframe
        if os.path.exists(self.price_file):
            self.df_prices = pd.read_csv(self.price_file, parse_dates=['timestamp'])
        else:
            self.df_prices = pd.DataFrame(columns=[
                'timestamp', 'symbol', 'open', 'close', 'high',
                'low', 'volume', 'trade_count', 'vwap'
            ])

    def save_stocks(self):
        self.df_stocks.to_csv(self.stock_file, index=False)

    def save_prices(self):
        self.df_prices.to_csv(self.price_file, index=False)

    def add_update_stock(self, symbol, name, sector, sp_500):
        self.df_stocks = self.df_stocks[self.df_stocks.symbol != symbol]  # Remove old entry
        self.df_stocks = pd.concat([self.df_stocks, pd.DataFrame([{
            'symbol': symbol,
            'name': name,
            'sector': sector,
            'sp_500': sp_500
        }])], ignore_index=True)
        self.save_stocks()

    def add_update_stock_prices(self, df_new_prices):
        # Drop existing records for the same symbol and timestamp
        for _, row in df_new_prices.iterrows():
            self.df_prices = self.df_prices[~(
                (self.df_prices['symbol'] == row['symbol']) & 
                (self.df_prices['timestamp'] == row['timestamp'])
            )]
        self.df_prices = pd.concat([self.df_prices, df_new_prices], ignore_index=True)
        self.save_prices()

if __name__ == '__main__':
    # Load configuration
    with open(config_file, "r") as f:
        conf = json.load(f)

    # Set API keys and Alpaca base URL
    api_key = conf['API_KEY']
    secret_key = conf['SECRET_KEY']
    base_url = conf['BASE_URL']

    # Date range: past 2 years until yesterday
    end_date = datetime.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=365 * 2)

    # Initialize stock manager (CSV-based)
    db_stock = StockManager()

    # Insert 3M stock info
    db_stock.add_update_stock('TSLA', 'Tesla Inc.', 'Consumer Discretionary', True)


    # Fetch prices using Alpaca API
    api = REST(key_id=api_key, secret_key=secret_key, base_url=base_url, api_version='v2')
    print(f'Fetching bars from {start_date.date()} to {end_date.date()}...')

    symbols = ['NVDA']
    df_barset = api.get_bars(
        symbols, TimeFrame.Day,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        adjustment='all'
    ).df

    df_barset = df_barset.reset_index()
    df_barset = df_barset.rename(columns={'time': 'timestamp'})
    df_barset = df_barset[[
        'timestamp', 'symbol', 'open', 'close', 'high',
        'low', 'volume', 'trade_count', 'vwap'
    ]]

    db_stock.add_update_stock_prices(df_barset)
    print('Done fetching and saving stock data.')
