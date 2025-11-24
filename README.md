# Advanced Time Series Forecasting with Deep Learning and XAI

This project implements a multivariate energy-consumption forecasting pipeline using recurrent deep learning
(LSTM/GRU) plus Explainable AI (XAI) for feature attribution.

The code is designed to satisfy typical assignment requirements:

- Programmatically generate a **multivariate** dataset with:
  - at least 5 input features (e.g., temperature, humidity, day-of-week, holiday flag, lag features),
  - 1 target variable representing energy load,
  - clear trend and seasonal components over multiple years.
- Design, train, and hyperparameter-tune an **LSTM/GRU** model for multi-step forecasting
  using a modern optimization library (**Optuna**).
- Evaluate the best model against a classical baseline (ARIMA) with metrics:
  **MAE, RMSE, MAPE**.
- Apply **Explainable AI** (XAI) using SHAP and Integrated Gradients to explain
  the model’s predictions and highlight important features.

## Project Structure

- `src/data_generator.py` – synthetic multivariate time series generator.
- `src/dataset.py` – PyTorch dataset & dataloader utilities with sliding windows.
- `src/models.py` – LSTM/GRU forecasting model definitions.
- `src/train_optuna.py` – end-to-end training & Optuna hyperparameter search.
- `src/baseline_arima.py` – ARIMA baseline on the target series.
- `src/evaluate.py` – metrics (MAE, RMSE, MAPE) and plotting helpers.
- `src/xai_analysis.py` – SHAP & Integrated Gradients analysis scripts.
- `data/energy_synthetic.csv` – example generated dataset.
- `report/report.md` – written report with methodology and analysis notes.

## Quick Start

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Generate / regenerate dataset**

```bash
python src/data_generator.py --years 3 --freq H --output data/energy_synthetic.csv
```

3. **Run Optuna hyperparameter search and train best LSTM**

```bash
python src/train_optuna.py \
  --data-path data/energy_synthetic.csv \
  --input-length 48 \
  --forecast-horizon 24 \
  --n-trials 20 \
  --device cpu
```

4. **Run ARIMA baseline**

```bash
python src/baseline_arima.py \
  --data-path data/energy_synthetic.csv \
  --forecast-horizon 24
```

5. **Run XAI analysis (SHAP + Integrated Gradients)**

```bash
python src/xai_analysis.py \
  --data-path data/energy_synthetic.csv \
  --model-path outputs/best_lstm.pt
```

Results (metrics, plots, feature importance values) are saved under `outputs/`.

You can use the report template in `report/report.md` to write your assignment submission
explaining the dataset, model, hyperparameter search strategy, baseline comparison, and XAI findings.
