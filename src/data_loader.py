import yfinance as yf
import pandas as pd
import numpy as np

def load_symbol(symbol: str, start: str, end: str, interval: str = '1d') -> pd.DataFrame:
    df = yf.download(
        symbol, 
        start, 
        end,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError(
            f"No data found for {symbol} at interval {interval}"
        )
    
    if isinstance(df.columns, pd.MultiIndex):
         df.columns = df.columns.droplevel(1)

    df.columns = [str(c).lower() for c in df.columns]

    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    return df