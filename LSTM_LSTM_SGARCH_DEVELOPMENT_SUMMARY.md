# LSTM and LSTM-SGARCH Development Summary

Date: 2026-06-03

This document briefly summarizes the main modelling, parameter, validation, environment, and refactoring changes made while preparing the standalone LSTM and hybrid LSTM-SGARCH models for Bitcoin daily log-return forecasting.

## Modelling Motivation

Bitcoin returns show nonlinear dynamics, high volatility, and volatility clustering. The modelling work was structured around two related models:

- Standalone LSTM: forecasts the conditional mean of the next daily Bitcoin log return.
- Hybrid LSTM-SGARCH: reuses the same LSTM mean forecast and fits SGARCH(1,1) on recent LSTM residuals to model remaining conditional volatility.



## Target Variable

The target was aligned across LSTM, LSTM-SGARCH, ARIMA comparison, and model evaluation:

```text
r_t = log(P_t / P_{t-1})
```

In the final LSTM notebooks this is implemented as:

```python
TARGET_COL = "target_daily_log_return"
daily[TARGET_COL] = daily["log_return"]
```

The model uses prior sequence observations to predict the next row return in the walk-forward setup.


## Data Preparation

The raw input remains the Coinbase BTC/USD 5-minute candle file:

```text
data/BTC/BTC_USD_coinbase_spot_5min.csv
```

The notebooks aggregate the data to daily observations and build:

- daily close-to-close log return,
- absolute and squared return transforms,
- rolling return statistics,
- realized volatility from intraday 5-minute returns,
- realized-volatility lag and rolling averages.

Raw price levels were excluded from LSTM features because the EDA stationarity checks did not support using raw prices directly as predictors.

## Feature-Set Changes

Several feature-set versions were tested during development.

### Initial Simple Feature Set

The earliest LSTM setup used a small feature set:

```text
daily_log_return
daily_volatility
volatility_7d
```

This was useful as a simple baseline but did not fully reflect the EDA results.

### Full EDA-Selected Feature Set

The model was then expanded to the EDA-selected feature set. This set is currently restored and used in both LSTM notebooks.

Current feature count: `32`

Current feature groups:

- signed daily return dynamics,
- rolling return means,
- rolling return volatility,
- absolute return features,
- squared return features,
- lagged return, absolute-return, and squared-return terms,
- realized volatility features.

Current feature list:

```text
log_return
abs_log_return
squared_log_return
return_mean_7
return_volatility_7
abs_return_mean_7
squared_return_mean_7
return_mean_14
return_volatility_14
abs_return_mean_14
squared_return_mean_14
return_mean_30
return_volatility_30
abs_return_mean_30
squared_return_mean_30
log_return_lag_1
abs_log_return_lag_1
squared_log_return_lag_1
log_return_lag_2
abs_log_return_lag_2
squared_log_return_lag_2
log_return_lag_3
abs_log_return_lag_3
squared_log_return_lag_3
log_return_lag_7
abs_log_return_lag_7
squared_log_return_lag_7
realized_volatility
log_realized_volatility
realized_volatility_lag_1
realized_volatility_30d_mean
realized_volatility_90d_mean
```

### Volatility-Only Feature Test

A reduced volatility-only set was also tested to decrease model complexity. It removed signed return-level and return-mean features. This increased RMSE in the saved grid-search outputs, so the full 32-feature EDA set was restored.

Observed result from saved artifacts:

```text
32 features, 180-day train window: best RMSE around 0.055351
24 volatility-only features, 30-day train window: best RMSE around 0.107677
```


## Walk-Forward Validation Design

The project moved from simple train/validation/test splits toward walk-forward validation to better respect time order and reduce leakage risk.

Current LSTM setup:

```python
TRAIN_WINDOW_MODE = "rolling"
TRAIN_WINDOW_DAYS = 180
VALIDATION_PERIOD_DAYS = 90
TEST_PERIOD_DAYS = 30
VALIDATION_FOLD_STEP_DAYS = 7
VALIDATION_FOLD_HORIZON_DAYS = 3
TEST_FOLD_STEP_DAYS = 3
TEST_FOLD_HORIZON_DAYS = 3
GRID_VALIDATION_FOLD_LIMIT = 5
```

Interpretation:

- Each fold trains only on observations before the evaluation interval.
- The training window is a rolling 180-day historical window.
- Hyperparameter search uses validation folds sampled from the 90-day validation period.
- Final testing uses walk-forward folds over the last 30 days.


## LSTM Architecture

The LSTM model was kept intentionally small to reduce overfitting and training cost.

Current architecture:

```text
Input(sequence_length, n_features)
LSTM(units)
Dropout(dropout)
Dense(16, relu)
Dense(1, float32)
```

Key design choices:

- Many-to-one sequence design.
- One LSTM layer only.
- `recurrent_dropout = 0.0` to keep the fast cuDNN GPU path available.
- Final output layer uses `float32`, which is safer with mixed precision.
- Loss: mean squared error.
- Optimizer: Adam.

## Hyperparameter Search Changes

The grid search was adjusted multiple times during development.

Earlier tested settings included:

- wider dropout options: `[0.0, 0.1, 0.2]`,
- batch sizes: `[32, 64]`,
- learning rates: `[0.001, 0.0005, 0.0002]`,
- sequence lengths: `[3, 5, 7, 10, 14, 21]`,
- rolling windows of both 30 and 180 days.

Saved artifact review showed the better RMSE came from the 180-day training-window setup and the full EDA feature set.

Current grid:

```python
HYPERPARAMETER_GRID = {
    "seq_len": [3, 5, 7, 10, 14, 21],
    "lstm_units": [16, 32],
    "dropout": [0.2],
    "learning_rate": [0.001, 0.0005, 0.0002],
    "batch_size": [64],
}
```

Current grid size:

```text
36 combinations
```

Current training controls:

```python
MAX_EPOCHS = 30
PATIENCE = 5
```

Reason for this latest change:

- The grid was narrowed by fixing dropout and batch size.
- Extra epochs became affordable.
- Early stopping still prevents the model from always using all 30 epochs.


## Refactoring for Reliable Long Training

Several training-loop changes were made to reduce loss of progress and runtime issues.

Changes:

- `ModelCheckpoint` saves the best model for each fold.
- `BackupAndRestore` is used so interrupted training can resume.
- Best fold model is copied to `best_model_overall.keras`.
- Live grid-search CSV is written after each combination.
- Final grid-search CSV is written after the full run.
- Keras `.predict()` calls inside loops were replaced with direct model inference to reduce retracing warnings.
- `tf.keras.backend.clear_session()` added after folds to reduce graph/session buildup.
- Notebook outputs cleared after structural changes to avoid stale results.

## LSTM-SGARCH Hybrid Refactor

The hybrid model was changed so it no longer trains a separate LSTM.

Current hybrid design:

1. Run the standalone LSTM notebook first.
2. Save fold-specific LSTM models during final evaluation.
3. In the LSTM-SGARCH notebook, load the exact same fold-specific LSTM models.
4. Predict LSTM conditional mean.
5. Compute LSTM residuals.
6. Fit SGARCH(1,1) only on recent LSTM residuals.
7. Produce hybrid outputs with:
   - LSTM mean forecast,
   - SGARCH volatility forecast,
   - residual diagnostics.

Current SGARCH residual window:

```python
SGARCH_RESIDUAL_WINDOW_DAYS = 180
```

This makes the comparison between LSTM and LSTM-SGARCH cleaner because both models share the same LSTM mean component.

## Artifact and Signature Handling

Run-specific artifact directories were introduced so different configurations do not overwrite each other.

Current expected signatures:

```text
LSTM run signature: 60536b3ea525
LSTM-SGARCH run signature: 0faa02bca6ee
```

Artifact roots:

```text
artifacts/lstm_walk_forward/runs/<signature>/
artifacts/lstm_sgarch_walk_forward/runs/<signature>/
```

Latest-run pointers are written after successful runs:

```text
artifacts/lstm_walk_forward/latest_run.json
artifacts/lstm_sgarch_walk_forward/latest_run.json
```

The hybrid notebook checks the LSTM metadata to ensure it is using compatible LSTM artifacts.

## Model Comparison Changes

The model comparison notebook was extended to compare:

- ARIMA,
- standalone LSTM,
- LSTM-SGARCH.

Forecast metrics:

- RMSE,
- MAE,
- MAPE,
- SMAPE,
- R2,
- correlation,
- direction accuracy.

Rolling-window metrics:

- rolling RMSE,
- rolling MAE,
- rolling direction accuracy.

Strategy metrics:

- total return,
- annualized return,
- annualized volatility,
- Sharpe,
- Sortino,
- max drawdown,
- hit rate,
- turnover.

Visualization changes:

- Forecast dashboard added.
- Strategy return metrics dashboard added.
- Graphs are saved to:

```text
artifacts/model_comparison_daily/forecast_model_comparison_dashboard.png
artifacts/model_comparison_daily/strategy_return_metrics_dashboard.png
```

## Current Run Order

Use this order after changing LSTM configuration:

```text
1. LSTM_5min_walk_forward.ipynb
2. LSTM_SGARCH_5min_walk_forward.ipynb
3. Model_Comparison_ARIMA_LSTM_LSTM_SGARCH.ipynb
```

The LSTM-SGARCH notebook depends on the fold-specific LSTM artifacts, so it must be run after the standalone LSTM notebook.

## Current Configuration Snapshot

```text
Target: target_daily_log_return
Feature count: 32
Training window: 180 days
Validation period: 90 days
Test period: 30 days
Validation fold step: 7 days
Validation fold horizon: 3 days
Selected validation folds for grid: 5
Test fold step: 3 days
Test fold horizon: 3 days
Grid combinations: 36
Max epochs: 30
Early-stopping patience: 5
CUDA required: True
Mixed precision: True
XLA: False
Grid selection: weighted-rank composite validation score
```

## Current Grid-Search Selection Rule

The grid search no longer selects the best hyperparameter combination by RMSE alone. Each completed combination is scored using a weighted rank across validation forecast and strategy metrics.

Forecast metrics included:

- RMSE,
- MAE,
- MAPE,
- SMAPE,
- R2,
- correlation,
- direction accuracy.

Strategy metrics included:

- total return,
- annualized return,
- annualized volatility,
- Sharpe,
- Sortino,
- max drawdown,
- hit rate,
- turnover.

Lower-is-better metrics are ranked ascending, and higher-is-better metrics are ranked descending. The lowest composite score is selected as the best grid-search model.


## Remaining Notes and Risks

- The best saved RMSE so far came from the 180-day/full-feature setup, but the latest forced `batch_size = 64` still needs to be validated.
- If RMSE increases, the first hyperparameter to relax should be `batch_size`, returning to `[32, 64]`.
- Volatility-only features are better aligned with risk/variance modelling than return-mean prediction, so they increased RMSE for the return target in the observed runs.
- SGARCH improves volatility modelling, but it does not automatically guarantee better return-direction forecasts.
- RMSE alone may reward models that forecast near zero; direction accuracy and trading metrics should also be considered.

