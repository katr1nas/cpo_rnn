# cpo_rnn

Conditional Parameter Optimization (CPO) on ES=F futures with hand-written recurrent networks (RNN, GRU, LSTM) in PyTorch.

Each model predicts the next-day log return. A rolling z-score of those predictions drives a long/short signal, and the signal parameters (z-score threshold and window) are chosen on a validation block. Everything is evaluated walk-forward and compared against a baseline that uses random predictions.

The goal of the project is a technically correct pipeline (no look-ahead, same procedure for every model), not a profitable strategy.

## Results

Walk-forward out-of-sample, ES=F, roughly 2014-06 to 2025-05 (about 2,600 trading days). Returns are simple daily strategy returns, no costs.

Seed 0 (the series saved in `outputs/`):

| Model  | Total return | CAGR  | Volatility | Sharpe | Sortino | Max DD  | Calmar |
|--------|-------------:|------:|-----------:|-------:|--------:|--------:|-------:|
| RNN    | 72.03%       | 5.51% | 9.82%      | 0.59   | 0.93    | -12.10% | 0.46   |
| GRU    | 93.56%       | 6.61% | 10.69%     | 0.65   | 1.09    | -13.77% | 0.48   |
| LSTM   | 6.93%        | 0.65% | 10.98%     | 0.11   | 0.16    | -29.10% | 0.02   |
| random | 71.68%       | 5.41% | 7.95%      | 0.70   | 1.13    | -11.10% | 0.49   |

Three seeds (Sharpe of the concatenated OOS series):

| Model  | Sharpe mean | Sharpe std | Total return mean | Seeds above random median |
|--------|------------:|-----------:|------------------:|--------------------------:|
| GRU    | 0.63        | 0.02       | 97.9%             | 2 of 3                    |
| RNN    | 0.31        | 0.29       | 34.4%             | 0 of 3                    |
| LSTM   | 0.23        | 0.14       | 20.9%             | 0 of 3                    |
| random | 0.53        | 0.22       | 55.0%             | n/a                       |

Takeaways:

- No model shows a reliable edge over the random baseline. The GRU result sits inside the spread of random.
- The random baseline has a positive Sharpe, so the z-score scheme itself (and the upward drift of ES) contributes. Models should be judged against random, not against zero.
- LSTM is the weakest: low return and a 29% drawdown.
- GRU and RNN equity curves have long flat stretches, probably because their predictions are close to constant and the z-score rarely crosses the threshold. This has not been verified.
- Three seeds is too few to separate the models statistically.

## How it works

1. **Data.** `src/data_loader.py` downloads daily ES=F bars with `yfinance` (2005-01-01 to 2026-01-01 in the main notebook).
2. **Features.** `src/features.py` builds log return, 5/20-day momentum, 5/20-day volatility and their ratio, 20-day autocorrelation, volatility of volatility, downside volatility, 20-day drawdown and a volume z-score. Raw prices are not fed to the models. The target is the next-day log return.
3. **Walk-forward folds.** The first 40% of the data is the minimum training set. Each fold then uses an expanding training window, a 250-day validation block and a 250-day test block. Folds do not overlap and the test blocks are concatenated into one out-of-sample series.
4. **Models.** `src/models/rnn.py`, `gru.py` and `lstm.py` implement the cells by hand. Each takes input of shape `(batch, seq_len, features)` and returns `(batch, 1)`. The main notebook uses a 30-day window, hidden size 32, Adam (lr 1e-3, weight decay 1e-4), gradient clipping at 1.0 and early stopping on validation loss. Scalers are fitted on the training part of each fold only.
5. **CPO.** `src/backtest.py` turns predictions into signals: long when the rolling z-score of predictions is above the threshold, short when it is below the negative threshold, flat otherwise. `src/cpo.py` evaluates every `(threshold, window)` pair from a grid (thresholds 0.5, 1.0, 1.5, 2.0 and windows 5, 10, 20, 30). The best pair by Sharpe on validation is applied to the test block.
6. **Baseline.** Random Gaussian predictions go through exactly the same procedure.

## Project structure

```
src/
  data_loader.py     ES=F download via yfinance
  features.py        feature engineering
  dataset.py         sequence helpers and DataLoader
  preprocessing.py   StandardScaler wrapper
  backtest.py        z-score signals and strategy returns
  cpo.py             parameter grid evaluation
  models/
    rnn.py gru.py lstm.py
outputs/
  strategy_returns_{RNN,GRU,LSTM,random}.csv
test_cpo_compare.ipynb   main experiment (walk-forward CPO, all models)
test_compare.ipynb       single split comparison without CPO
test_fixed.ipynb         single split GRU pipeline
test.ipynb               early prototype
main.py, configs/default.yaml, src/trainer.py, src/export.py   placeholders
```

## Setup

```bash
git clone https://github.com/katr1nas/cpo_rnn.git
cd cpo_rnn
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Open `test_cpo_compare.ipynb` and run all cells from the repository root, or from the terminal:

```bash
pip install nbconvert
jupyter nbconvert --to notebook --execute --inplace test_cpo_compare.ipynb
```

For a quick check set `SEEDS = [0]` in the first cell. The full run trains each model on every fold for every seed, so it takes a while on CPU.

Results are written to `outputs/strategy_returns_<model>.csv` (seed 0 only) with columns `date,return`.

## Limitations

- No commissions or slippage.
- One instrument. The `ES=F` series is a stitched continuous contract, so roll dates may contain artificial gaps.
- Three seeds and about 2,600 out-of-sample days are not enough to rank the models.
- Equity curves and the first table come from seed 0 only.
- Planned: more seeds, a check of prediction variance per fold, transaction costs, and a comparison with a random-forest CPO using the same features.
