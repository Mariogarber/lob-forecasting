# Wunder Predictorium - LOB Price Movement Forecasting

Repository for our solution to the **Wunder Predictorium** challenge by Wunder Fund.

The goal of the competition is to build a machine learning model that predicts future market indicators from sequences of previous Limit Order Book (LOB) states and trades. This repository contains the training pipeline, feature engineering utilities, validation experiments, and the final submission code.

> Competition website: https://wundernn.io/predictorium

---

## Project Overview

This project tackles a high-frequency financial sequence modeling problem. Given a sequence of anonymized market states, the model predicts two future target indicators:

- `t0`
- `t1`

The solution is designed around three principles:

1. **Temporal modeling**  
   The input is sequential, so the model must preserve order-dependent information.

2. **Robust validation**  
   Market data is noisy and non-stationary, so validation must avoid leakage and should respect temporal splits.
