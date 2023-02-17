import yfinance as yf

def download_yf_data(tickers: list, name: str = None, path_do_data: str = None):
    #Requesting from yahoo finance Open, High, Low, Close, Volume data for a list of tickers over the last year
    tickers = " ".join(tickers)
    field = list("High,Low,Close,Volume".split(','))
    print(f'Downloading intraday data {tickers} from Yfinance with 1-hour bar over the 2 last year')
    df = yf.download(tickers, period='2y', interval='1h')[field]
    df = df.T.swaplevel(1, 0).T
    file_path = f'{path_do_data}/{name}.h5'
    df.to_hdf(file_path, 'df')
    return df