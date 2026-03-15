<div align="center">

# 📈 Stock Market Prediction Using Machine Learning

> Predicting TCS stock price movements using classical ML algorithms — regression, classification, and sentiment analysis combined.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Models Implemented](#-models-implemented)
  - [Regression Models](#regression-models-predict-price)
  - [Classification Models](#classification-models-predict-direction)
- [Sentiment Analysis Pipeline](#-sentiment-analysis-pipeline)
- [Feature Engineering](#-feature-engineering)
- [Getting Started](#-getting-started)
- [Model Comparison](#-model-comparison)
- [Key Findings](#-key-findings)

---

## 🌐 Overview

The stock market is highly volatile and nonlinear — classical ML models struggle, but with the right feature engineering they can still extract meaningful signals.

This project applies **6 different ML algorithms** (3 regression + 3 classification) to TCS (Tata Consultancy Services) historical stock data, enriched with **news sentiment scores** from a fine-tuned FinBERT model. Each script is self-contained: load data → preprocess → train → evaluate.

### Goals
- Predict the **next day's opening price** (regression)
- Predict whether price will go **UP or DOWN** the next day (classification)
- Measure the impact of **lag features** and **sentiment** on prediction quality
- Compare all models side-by-side with a visualizer script

---

## 🏆 Key Results

### Regression (Predicting Next Day's Price)

| Model | R² Score | RMSE |
|---|---|---|
| **Linear Regression** | **0.9848** | **49.81** ✅ Best |
| Random Forest Regressor | 0.6836 | 227.71 |
| KNN Regressor | 0.3824 | 318.15 |

### Classification (Predicting UP/DOWN Direction)

| Model | Accuracy | Notes |
|---|---|---|
| **Random Forest** | **75.68%** ✅ Best | Highest overall |
| Logistic Regression | 75.09% | Best precision/recall balance |
| KNN Classifier | 70.99% | Improved significantly with lag features |

> **Baseline:** ~50% (coin flip). All models beat the baseline, with Random Forest and Logistic Regression reaching ~75%.

---

## 📁 Project Structure

```
Stock-Market-ML/
│
├── 📊 Data
│   └── Stocks_TCS2.csv              # Main dataset: OHLCV + Sentiment scores
│
├── 🤖 Models
│   ├── StockMarket_LinearRegression.py      # LR for price prediction
│   ├── StockMarket_LogisticRegression.py    # LogReg for direction (+ ROC curve, confusion matrix)
│   ├── StockMarket_KNN.py                   # KNN classifier
│   ├── StockMarket_KNNReggresor.py          # KNN regressor
│   ├── StockMarket_RandomForestClassifier.py # RFC with decision tree visualization
│   └── StockMarket_RandomForestRegressor.py  # RFR
│
├── 🔧 Data Pipeline
│   ├── textb.py         # Downloads TCS stock data via yfinance
│   ├── merge.py         # Merges stock data with FinBERT sentiment scores
│   └── remover.py       # Cleans rows with zero sentiment values
│
├── 📈 Analysis
│   └── comparizer.py    # Side-by-side bar charts for all model results
│
├── requirements.txt
└── README.md
```

---

## 📦 Dataset

**File:** `Stocks_TCS2.csv`

| Column | Description |
|---|---|
| `Close` | Closing price |
| `High` | Daily high price |
| `Low` | Daily low price |
| `Open` | Opening price |
| `Volume` | Trading volume |
| `Sentiment` | FinBERT sentiment score from TCS news headlines (`-1` to `+1`) |

- **Source:** TCS.NS historical data via `yfinance` (2015–2025)
- **News Source:** CF-AN equities TCS headlines dataset
- **Sentiment Model:** [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert) — a BERT model fine-tuned on financial text

### How Sentiment Was Calculated

```
Positive headline → +score
Negative headline → -score
Neutral headline  →  0
Daily sentiment   = mean of all headline scores for that trading day
```

---

## 🤖 Models Implemented

### Regression Models (Predict Price)

All regression models predict the **next day's opening price** using `Open.shift(-1)` as the target.

#### Linear Regression
```
Target: Next day's Open price
Features: OHLCV columns
Result: R² = 0.9848, RMSE = 49.81
Note: Lag features slightly hurt LR performance — dropped for this model
```

#### KNN Regressor
```
Target: Next day's Open price
Features: OHLCV + lag features
n_neighbors: 10
Scaler: StandardScaler (required for KNN)
Result: R² = 0.3824, RMSE = 318.15
```

#### Random Forest Regressor
```
Target: Next day's Open price
Features: OHLCV + lag features
n_estimators: 100
Result: R² = 0.6836, RMSE = 227.71
```

---

### Classification Models (Predict Direction)

All classification models predict `1` (price UP) or `0` (price DOWN) for the next day.

```python
Target = (Open.shift(-1) > Open).astype(int)
```

#### KNN Classifier
```
n_neighbors: 10
Scaler: StandardScaler
Accuracy: ~70.99%
Impact of lag features: +20% accuracy improvement (from ~50% to ~70%)
```

#### Logistic Regression
```
max_iter: 1200
class_weight: balanced
Outputs: Confusion Matrix + ROC Curve
Accuracy: ~75.09%
Note: Lag features had minimal impact on LogReg
```

#### Random Forest Classifier
```
n_estimators: 200
class_weight: balanced
Outputs: Decision tree visualization
Accuracy: ~75.68%
Impact of lag features: Huge — from ~55% to ~70%+
```

---

## 💬 Sentiment Analysis Pipeline

The sentiment pipeline uses **FinBERT** to score TCS news headlines and append daily sentiment to the stock dataset.

```
textb.py     →  Downloads TCS.NS price data from yfinance → saves TCS_STOCKS!2.csv
     ↓
merge.py     →  Loads headlines CSV + stock CSV
             →  Filters stocks to only dates with news coverage
             →  Runs FinBERT on each headline
             →  Averages daily sentiment scores
             →  Saves Stocks_TCS2_date.csv
     ↓
remover.py   →  Removes rows where sentiment = 0 (no news day)
             →  Saves Stocks_TCS4.csv
```

> ⚠️ **GPU recommended** — `merge.py` runs FinBERT with `device=0` (CUDA). Switch to `device=-1` for CPU.

---

## 🔧 Feature Engineering

Lag features are the most impactful preprocessing step in this project:

```python
stocks['lag_1']       = stocks['Open'].shift(1)       # Yesterday's open
stocks['lag_2']       = stocks['Open'].shift(2)       # Two days ago open
stocks['rolling_3']   = stocks['Open'].rolling(3).mean()  # 3-day moving average
stocks['per%_change'] = stocks['Close'].pct_change()  # Daily % change
```

### Impact of Lag Features by Model

| Model | Without Lag Features | With Lag Features | Δ |
|---|---|---|---|
| KNN Classifier | ~50% | ~70% | **+20%** |
| Random Forest Classifier | ~55% | ~75% | **+20%** |
| Logistic Regression | ~75% | ~75% | ~0% |
| Linear Regression | 0.985 R² | 0.985 R² | Slightly worse |
| KNN Regressor | baseline | +0.01 R², -800 RMSE | Slight improvement |
| Random Forest Regressor | baseline | +0.003 R², -1000 RMSE | Marginal improvement |

---

## 🚀 Getting Started

### Prerequisites

```bash
Python >= 3.11
CUDA GPU (optional, for faster FinBERT inference)
```

### Installation

```bash
git clone https://github.com/your-username/Stock-Market-ML.git
cd Stock-Market-ML
pip install -r requirements.txt
```

### requirements.txt

```
pandas
numpy
scikit-learn
matplotlib
transformers
yfinance
statistics
torch  # for FinBERT GPU inference
```

> Note: The `requirements.txt` lists `Transformer` — install as `transformers` (lowercase) from Hugging Face.

### Running the Models

```bash
# Run any individual model
python StockMarket_LinearRegression.py
python StockMarket_RandomForestClassifier.py
python StockMarket_LogisticRegression.py

# Compare all results visually
python comparizer.py

# Re-build the dataset from scratch (requires news headlines CSV)
python textb.py          # Download stock data
python merge.py          # Merge with sentiment
python remover.py        # Clean zeros
```

> All model scripts expect `Stocks_TCS2.csv` to be present. The pre-built dataset is already included in the repo.

---

## 📊 Model Comparison

Run `comparizer.py` to generate side-by-side bar charts for both regression (RMSE) and classification (accuracy):

```
Regression:
┌─────────────────────────────────────┐
│  KNN   │ RF    │ Lin Reg            │
│ 318.15 │ 227.7 │ 49.81 ← Best      │
└─────────────────────────────────────┘

Classification:
┌─────────────────────────────────────┐
│  KNN   │ Log Reg │ Random Forest    │
│ 70.98% │ 75.08%  │ 75.68% ← Best   │
└─────────────────────────────────────┘
```

---

## 🔍 Key Findings

**1. Linear Regression dominates regression tasks**
With an R² of 0.985, linear regression is surprisingly the best price predictor. Stock prices on adjacent days are highly correlated — which is exactly what LR captures well. Lag features actually hurt it slightly, suggesting multicollinearity.

**2. Lag features are critical for tree-based and KNN classifiers**
Adding just two lag values and a rolling mean pushed KNN and Random Forest from ~50–55% (barely better than random) to ~70–75%. This is the single biggest lever in the project.

**3. FinBERT sentiment adds signal**
Incorporating news sentiment into the feature set provides additional predictive information beyond price history alone, particularly on high-news days.

**4. SVM underperformed**
The SVM (with RBF kernel, C=100, gamma=0.1) performed worse than random on this dataset — noted in the script itself as *"worse than deciding on coin toss."* Kernel and hyperparameter tuning would be needed for improvement.

**5. The 75% ceiling**
Both Logistic Regression and Random Forest plateau around 75% accuracy for direction prediction. Breaching this ceiling likely requires deeper features — options data, order book depth, or LSTM-based sequence modeling.

---

## 🔮 Potential Improvements

- Add more technical indicators (RSI, MACD, Bollinger Bands)
- Try LSTM or Transformer-based sequence models
- Expand sentiment to include social media (Twitter/Reddit)
- Hyperparameter tuning via GridSearchCV
- Walk-forward validation instead of a single train/test split

---

<div align="center">
  <sub>Built with 🐍 Python · scikit-learn · FinBERT · yfinance</sub>
</div>