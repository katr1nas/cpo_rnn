import numpy as np
import pandas as pd


class Backtest:
    def __init__(
        self,
        predictions: pd.Series,
        actual_log_returns: pd.Series,
        zscore_threshold: float = 1.0,
        zscore_window: int = 60,
    ):
        if zscore_threshold < 0:
            raise ValueError("zscore_threshold must be non-negative")

        if zscore_window < 2:
            raise ValueError("zscore_window must be at least 2")

        if not predictions.index.is_unique:
            raise ValueError("Prediction index must be unique")

        if not actual_log_returns.index.is_unique:
            raise ValueError("Return index must be unique")

        self.predictions, self.actual_log_returns = (
            predictions.align(actual_log_returns, join="inner")
        )

        if self.predictions.empty:
            raise ValueError("No overlapping dates between inputs")

        if not np.isfinite(self.predictions).all():
            raise ValueError("Predictions contain missing or infinite values")

        if not np.isfinite(self.actual_log_returns).all():
            raise ValueError("Actual returns contain missing or infinite values")

        self.zscore_threshold = zscore_threshold
        self.zscore_window = zscore_window

    def prediction_zscore(self):
        mean = self.predictions.rolling(
            self.zscore_window
        ).mean()

        std = self.predictions.rolling(
            self.zscore_window
        ).std()

        self.zscore = (
            (self.predictions - mean) /
            std.replace(0, np.nan)
        )

        return self.zscore

    def generate_signals(self) -> pd.Series:
        zscore = self.prediction_zscore()

        signals = pd.Series(0.0, index=self.predictions.index)

        signals[zscore > self.zscore_threshold] = 1.0
        signals[zscore < -self.zscore_threshold] = -1.0

        self.signals = signals

        return signals

    def order(self) -> pd.DataFrame:
        positions = self.generate_signals()

        strategy_log_returns = (
            positions * self.actual_log_returns
        )

        strategy_returns = np.expm1(strategy_log_returns)

        return pd.DataFrame({
            "prediction": self.predictions,
            "zscore": self.zscore,
            "actual_log_returns": self.actual_log_returns,
            "position": positions,
            "strategy_return": strategy_returns,
        })
