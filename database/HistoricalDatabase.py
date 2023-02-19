from datetime import datetime
import pandas as pd
import os
from database.request import download_yf_data


class HistoricalDatabase:
    def __init__(self, tickers: list, name: str = None):
        self.exchange = "NASDAQ"
        self.path_to_data = os.path.abspath(__file__).replace('\database\HistoricalDatabase.py', "\data")
        self.init(tickers, name)

    @staticmethod
    def transform_data(df):
        df.dropna(inplace=True)
        df.index = [index.replace(tzinfo=None) for index in df.index]
        return df

    def init(self, tickers: list, name: str = None):
        try:
            data = pd.read_hdf(f'{self.path_to_data}/{name}.h5', 'df')
        except FileNotFoundError:
            data = download_yf_data(tickers, name, self.path_to_data)
        data = self.transform_data(data)
        self.start_date = data.index[0]
        self.end_date = data.index[-1]
        print(f'Number of instances per asset: {len(data)}')
        self.data = dict(map(lambda tick: (tick, data[tick]), data.columns.levels[0]))
        self.calendar = list(data.index)

    def get_next_timestep(self, timestamp: datetime, ticker: str):
        return self.data[ticker].loc[self.data[ticker].index >= timestamp].index[0]

    def get_last_snapshot(self, timestamp: datetime, ticker: str):
        return self.data[ticker].loc[self.data[ticker].index <= timestamp].iloc[-1]
