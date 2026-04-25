# Bitcoin Forecasting with a Hybrid LSTM-SGARCH Model

## Overview

This repository contains a reproducible research project on short-term Bitcoin return forecasting using a **hybrid LSTM-SGARCH model**.

The target variable is the **daily Bitcoin log return**. The main model combines:

- an **LSTM neural network** to predict the next daily log return,
- an **SGARCH model** to model the conditional volatility of the LSTM residuals.

The project evaluates whether this hybrid framework improves both:

1. **forecast accuracy**, and  
2. **investment performance**

relative to several benchmark models.

The project was developed for a **reproducible research** course, with emphasis on transparent methodology, version control, clear documentation, and reproducible results.

---

## Contributions

This research is conducted by:

- Filip Bronisz
- Michał Sucharzewski
- Andrzej Żernaczuk
- Piotr Radziszewski

---

## Motivation

Bitcoin is a highly volatile asset with nonlinear return dynamics and volatility clustering. These properties make it difficult to model using simple forecasting methods.

This project is based on the idea that:

- **LSTM** can capture nonlinear temporal dependencies in Bitcoin log returns,
- **SGARCH** can describe time-varying conditional volatility,
- combining both models may improve forecasting and investment decisions.

The hybrid model is designed so that the LSTM forecasts the conditional mean of the next log return, while SGARCH models the volatility structure remaining in the LSTM residuals.

---

## Research Background

The project is inspired by the work of **Michańków, Sakowski, and Ślepaczuk**, who studied the use of LSTM-based and time-series models in financial forecasting and algorithmic investment strategies.

Building on this research direction, the present project applies a hybrid neural-network and volatility-modelling approach to the Bitcoin market.

---

## Research Question

**Can a hybrid LSTM-SGARCH model improve daily Bitcoin log-return forecasting and investment performance compared with standalone models and simple benchmarks?**

---

## Target Variable

The target variable is the daily Bitcoin log return:

```text
r_t = log(P_t / P_{t-1})
