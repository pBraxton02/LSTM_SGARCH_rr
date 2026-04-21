# Bitcoin Forecasting with LSTM and SGARCH

## Overview

This repository contains a reproducible research project focused on forecasting Bitcoin market behaviour using a combination of deep learning and classical volatility modelling methods. The project investigates whether a framework based on **Long Short-Term Memory (LSTM)** networks and **SGARCH** models can provide useful short-term forecasts for Bitcoin data and whether such an approach offers advantages over more traditional time series methods.

The project was developed as part of a **reproducible research** course. For that reason, the repository emphasizes not only empirical results, but also transparency, version control, clear project structure, and the ability to fully reproduce the workflow from data collection to final evaluation.

## Motivation

Forecasting Bitcoin is a difficult problem because the market is highly volatile, noisy, and sensitive to external shocks. Standard linear time series models often struggle to capture complex nonlinear patterns, while deep learning models may detect temporal dependencies that are harder to model using classical approaches alone.

At the same time, volatility modelling remains highly relevant in financial time series. This motivates combining or comparing **LSTM-based forecasting** with **SGARCH-based volatility modelling** in order to better understand whether such methods are useful for short-horizon Bitcoin prediction.

## Research Background

This project is based on the research direction presented in the work of **Michańków, Sakowski, and Ślepaczuk**, which studies algorithmic investment strategies using **Long Short-Term Memory** and **time series models** in financial markets. Their work provides an important methodological motivation for comparing modern deep learning methods with more classical financial time series approaches, especially in settings involving volatile assets and return dynamics.

Inspired by that line of research, this repository adapts the idea to a Bitcoin forecasting setting and focuses more directly on the relationship between:

- **deep learning sequence models (LSTM)**,
- **conditional volatility models (SGARCH)**,
- and their usefulness in short-term financial forecasting.

The project does not assume in advance that the more complex model will perform better. Instead, it tests whether the additional modelling complexity is justified empirically.

## Research Question

**Can an LSTM-SGARCH forecasting framework improve short-term Bitcoin prediction compared with traditional time series approaches?**

## Project Objective

The main objective of this project is to evaluate forecasting performance on Bitcoin data using methods drawn from both deep learning and classical econometrics.

More specifically, the project aims to:

- collect and preprocess historical Bitcoin price data,
- transform the data into a form suitable for time series and neural models,
- model return dynamics and volatility,
- train and evaluate LSTM-based forecasting models,
- estimate SGARCH-based volatility models,
- compare the predictive performance of the models under a common evaluation framework,
- document the entire process in a reproducible way.

## Methodology

The project follows several stages:

### 1. Data collection
Historical Bitcoin data is downloaded from a public source or API and stored in a reproducible format.

### 2. Data preprocessing
The raw data is cleaned and transformed. This may include:

- handling missing values,
- computing log returns,
- resampling data if needed,
- scaling features for neural network models,
- creating lagged features or rolling statistics.

### 3. Exploratory data analysis
The data is analysed to understand:

- trends and volatility patterns,
- return distribution,
- clustering of volatility,
- possible nonstationarity,
- autocorrelation structure.

### 4. Model estimation
The empirical part of the project includes models such as:

- benchmark time series models,
- **SGARCH** models for volatility estimation,
- **LSTM** models for sequence-based prediction,
- optionally, a combined or comparative **LSTM-SGARCH** framework.

### 5. Evaluation
Models are compared using selected forecasting metrics, for example:

- MAE,
- RMSE,
- MAPE,
- directional accuracy,
- and, if included, simple trading or strategy-based evaluation.

### 6. Reporting
The final stage summarizes the results, discusses model limitations, and evaluates whether the LSTM-SGARCH approach is empirically justified.

## Why this project fits reproducible research

This repository is designed to meet the requirements of reproducible empirical research. In particular, it includes:

- version-controlled code and analysis,
- a clear folder structure,
- documented dependencies,
- explicit environment setup instructions,
- scripts or notebooks that reproduce each stage of the workflow,
- transparent reporting of methods and results.

The goal is that another person should be able to clone the repository, install the required environment, run the code, and reproduce the main findings.

## Repository Structure

```text
.
├── data/                  # raw and processed Bitcoin data
├── notebooks/             # exploratory analysis and experiments
├── src/                   # source code
│   ├── data/              # data download and preprocessing
│   ├── features/          # feature engineering
│   ├── models/            # LSTM, SGARCH, and benchmark models
│   ├── evaluation/        # metrics and model comparison
│   └── utils/             # helper functions
├── results/               # saved outputs, forecasts, figures, tables
├── reports/               # final report or summary documents
├── requirements.txt       # Python dependencies
├── environment.yml        # Conda environment definition
└── README.md
