# Bitcoin Forecasting with a Hybrid LSTM-SGARCH Model

## Overview

This repository contains a reproducible research project on short-term Bitcoin return forecasting using a hybrid LSTM-SGARCH model.

The target variable is the next-day daily Bitcoin log return. The hybrid model combines:

- an LSTM neural network to forecast the conditional mean of the next daily log return,
- an SGARCH model to describe conditional volatility in the LSTM residuals.

The project evaluates whether the hybrid framework improves forecast accuracy relative to classical benchmarks and a standalone LSTM model.

## Research Question

Can a hybrid LSTM-SGARCH model improve daily Bitcoin log-return forecasting compared with standalone models and simple benchmarks?

## Target Variable

The daily log return is defined as:

```text
r_t = log(P_t / P_{t-1})
```

where `P_t` is the Bitcoin daily close price.

The one-step-ahead forecasting target is:

```text
target_t = r_{t+1}
```

All main models are aligned to this target.

## Data

The project uses BTC-USD spot candles from Coinbase at 5-minute frequency:

- `data/BTC/BTC_USD_coinbase_spot_5min.csv`

Daily features are derived from the raw 5-minute candles:

- `data/BTC/BTC_USD_coinbase_spot_daily_features.csv`
- `data/BTC/BTC_USD_coinbase_spot_daily_features.metadata.json`
- `data/BTC/BTC_USD_coinbase_spot_daily_features.scaler.pkl`

The daily feature file contains close-to-close log returns, rolling return features, lagged features, a next-day return target, and chronological split labels.

## Models

The project compares:

- simple walk-forward benchmarks: naive, rolling means, SES,
- ARIMA benchmark on the daily log-return target,
- standalone LSTM,
- hybrid LSTM-SGARCH.

The hybrid setup uses the LSTM for the next-day return mean forecast and SGARCH(1,1) for residual volatility.

## Notebook Workflow

Run the notebooks in this order:

1. `EDA.ipynb`
2. `ARIMA_modeling.ipynb`
3. `LSTM_5min_walk_forward.ipynb`
4. `LSTM_SGARCH_5min_walk_forward.ipynb`
5. `Model_Comparison_ARIMA_LSTM_LSTM_SGARCH.ipynb`

The LSTM notebooks are GPU-first and are intended to be run in WSL2. See `WSL_SETUP.md` for the full setup.

Generated training artifacts are written under `artifacts/`. Most intermediate artifacts are ignored by git; rerun the notebooks to regenerate fold predictions, metadata, plots, and comparison tables.

## Current Results

The saved model-comparison notebook reports the following test-set metrics:

| Model | RMSE | MAE | Direction Accuracy |
| --- | ---: | ---: | ---: |
| LSTM | 0.021907 | 0.017668 | 0.566667 |
| ARIMA | 0.026249 | 0.020850 | 0.433333 |
| LSTM-SGARCH | 0.070476 | 0.052921 | 0.466667 |

In the current experiment, the standalone LSTM has the best forecast accuracy. The LSTM-SGARCH hybrid does not improve daily return forecast accuracy relative to the standalone LSTM.

This does not reject the usefulness of volatility modelling in general. It means that, for this dataset, target definition, feature set, and walk-forward design, the SGARCH residual-volatility component did not translate into better conditional mean forecasts.

## Reproducibility

The repository is designed for a WSL2-first TensorFlow GPU workflow.

Quick setup:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\wsl_install_admin.ps1
powershell -ExecutionPolicy Bypass -File scripts\wsl_bootstrap_from_windows.ps1
```

Then, inside WSL:

```bash
bash scripts/wsl_jupyter.sh
```

In Jupyter, select the kernel:

```text
Python (LSTM WSL GPU)
```

For details, see `WSL_SETUP.md`.

## Project Structure

```text
.
|-- EDA.ipynb
|-- ARIMA_modeling.ipynb
|-- LSTM_5min_walk_forward.ipynb
|-- LSTM_SGARCH_5min_walk_forward.ipynb
|-- Model_Comparison_ARIMA_LSTM_LSTM_SGARCH.ipynb
|-- preprocess_btc.py
|-- data/
|-- artifacts/
|-- scripts/
|-- pyproject.toml
`-- WSL_SETUP.md
```

## Contributors

- Filip Bronisz
- Michał Sucharzewski
- Andrzej Żernaczuk
- Piotr Radziszewski

## Conclusion

The main research pipeline is implemented: data preparation, EDA, classical benchmarks, LSTM, hybrid LSTM-SGARCH, and unified model comparison.

The current empirical answer to the research question is negative: the hybrid LSTM-SGARCH model does not outperform the standalone LSTM on daily Bitcoin log-return forecast accuracy in the saved results.
