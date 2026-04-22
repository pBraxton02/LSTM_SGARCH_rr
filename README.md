# Bitcoin Forecasting with a Hybrid LSTM-SGARCH Model

## Overview

This repository contains a reproducible research project focused on short-term Bitcoin forecasting using a **hybrid LSTM-SGARCH model**. The aim of the project is to examine whether combining **Long Short-Term Memory (LSTM)** networks with **SGARCH** volatility modelling can improve predictive performance in the Bitcoin market.

The project was developed for a **reproducible research** course, with emphasis on transparency, clear documentation, version control, and full reproducibility of the results.

## Contributions

This research is conducted by:
Filip Bronisz
Michał Sucharzewski 
Andrzej Żernaczuk 
Piotr Radziszewski 

## Motivation

Bitcoin is a highly volatile asset with nonlinear dynamics and volatility clustering. These properties make it difficult to model using simple linear methods alone.

This project is based on the idea that:

- **LSTM** can capture nonlinear temporal patterns in sequential data,
- **SGARCH** can model time-varying conditional volatility,
- combining both may provide a better forecasting framework for Bitcoin returns.

## Research Background

The project is inspired by the work of **Michańków, Sakowski, and Ślepaczuk**, who studied the use of **LSTM** and **time series models** in financial forecasting and algorithmic investment strategies.

Building on this research direction, the present project applies the idea to Bitcoin and focuses on a **hybrid LSTM-SGARCH model** that combines sequence learning with volatility modelling in one framework.

## Research Question

**Can a hybrid LSTM-SGARCH model improve short-term Bitcoin forecasting compared with standard time series approaches?**

## Project Objective

The main objective of this project is to design and evaluate a **hybrid LSTM-SGARCH forecasting framework** for Bitcoin data.

The project includes:

- collecting and preprocessing Bitcoin data,
- modelling return dynamics and volatility,
- building the hybrid LSTM-SGARCH model,
- comparing it with benchmark approaches,
- documenting the workflow in a reproducible way.

## Methodology

The project workflow includes:

1. data collection,  
2. preprocessing and feature preparation,  
3. exploratory data analysis,  
4. estimation of LSTM, SGARCH, and hybrid models,  
5. forecasting evaluation using standard metrics,  
6. reporting and interpretation of results.

## Repository Structure

```text
.
├── data/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── utils/
├── results/
├── reports/
├── requirements.txt
├── environment.yml
└── README.md
