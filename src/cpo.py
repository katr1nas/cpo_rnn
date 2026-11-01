
import pandas as pd
from src.backtest import Backtest


class CPO:
    def __init__(
        self,
        combinations,
        predictions,
        actual_log_return
    ):
        self.combinations = combinations
        self.predictions = predictions
        self.actual_log_return = actual_log_return

    def evaluate(self):
        results = []

        for zscore_threshold, zscore_window in self.combinations:

            back = Backtest(
                self.predictions,
                self.actual_log_return,
                zscore_threshold,
                zscore_window
            )

            strategy = back.order()
            returns = strategy["strategy_return"]

            volatility = returns.std()

            sharpe = (
                returns.mean() / volatility
                if volatility > 0 else 0.0
            )

            results.append({
                "zscore_threshold": zscore_threshold,
                "zscore_window": zscore_window,
                "total_return": (1 + returns).prod() - 1,
                "annualized_volatility": volatility * (252 ** 0.5),
                "sharpe": sharpe * (252 ** 0.5),
                "observations": len(returns),
            })

        return pd.DataFrame(results).sort_values(
            "sharpe",
            ascending=False
        ).reset_index(drop=True)
