import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    log_returns = np.log(data['close']).diff()

    data['log_return'] = log_returns
    data['momentum_5']  = np.log(
        data['close'] / data['close'].shift(5)
    )
    data['momentum_20']  = np.log(
        data['close'] / data['close'].shift(20)
    )

    data["volatility_5"] = log_returns.rolling(5).std()
    data["volatility_20"] = log_returns.rolling(20).std()

    data['volatility_ratio'] = (
        data['volatility_5'] /
        data['volatility_20']
    )

    data['autocorr_20'] = (
        log_returns.rolling(20).corr(log_returns.shift(1))
    )

    data['vol_of_vol'] = (
        data['volatility_5'].rolling(20).std()
    )

    downside = log_returns.clip(upper=0)

    data['downside_volatility'] = np.sqrt(downside.pow(2).rolling(20).mean())

    rolling_max = data['close'].rolling(20).max()
    data['drawdown_20'] = (
        data['close'] / rolling_max - 1
    )

    if "volume" in data.columns:
        volume_mean = data["volume"].rolling(20).mean()
        volume_std = data["volume"].rolling(20).std()

        data["volume_zscore"] = (
            (data["volume"] - volume_mean) / volume_std
        )

    feature_cols = [
        'log_return', 'momentum_5', 'momentum_20',
        'volatility_5', 'volatility_20', 'volatility_ratio',
        'autocorr_20', 'vol_of_vol', 'downside_volatility', 'drawdown_20',
    ]

    if 'volume_zscore' in data.columns:
        feature_cols.append('volume_zscore')

    data = data[feature_cols].replace([np.inf, -np.inf], np.nan)
    return data.dropna()

    
