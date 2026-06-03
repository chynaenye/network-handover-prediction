## 📡 Cellular Network Handover Prediction

A machine learning project to predict whether a cellular network handover event will occur based on signal quality metrics, device motion, and location data from real network logs.

---

## 📌 Project Overview

In cellular networks, a **handover** (or handoff) occurs when a mobile device transfers its connection from one cell tower to another — identified by a change in the **Physical Cell Identifier (PCI)**. Predicting these events in advance can help networks pre-allocate resources and reduce dropped connections.

This project builds and evaluates several classification models to predict handover events, working through the full ML pipeline: data cleaning, feature engineering, class imbalance handling, model training, and threshold tuning.

**Dataset:** [Cellular Network Handover Prediction Dataset](https://www.kaggle.com/datasets/meruvakodandasuraj/cellular-network-handover-prediction-dataset) via Kaggle

---

## 📁 Project Structure

```
├── notebook.ipynb         # Main analysis notebook
├── README.md
 ```

---

## 🔄 Pipeline Walkthrough

### 1. Data Loading & Initial Exploration

The dataset (`network_logs_1.csv`) is downloaded via `kagglehub` and loaded into a Pandas DataFrame. Initial exploration covers:

- `.info()`, `.dtypes`, `.nunique()` — to understand column types and cardinality
- `.apply(type).value_counts()` — to detect **mixed types** in signal columns (`RSRP`, `RSRQ`, `SINR`), which turned out to contain string values with units (e.g., `"-85 dBm"`)

---

### 2. Data Preprocessing

#### Unit Stripping & Type Conversion
Several columns stored numeric values as strings with measurement units. These were stripped and cast to `float`:

| Column | Unit Removed |
|---|---|
| `RSRP` | ` dBm` |
| `RSRQ` | ` dB` |
| `SINR` | ` dB` |
| `Velocity(km/h)` | ` km/h` |
| `Downlink(Mbps)` | ` Mbps` |
| `Uplink(Mbps)` | ` Mbps` |

#### Handling Missing Values
- Rows missing `Latitude`, `Longitude`, or `PCI` were **dropped** — these are essential for handover logic and cannot be imputed meaningfully.
- `Downlink(Mbps)` and `Uplink(Mbps)` missing values were filled with their respective **medians**.

#### Dropping Irrelevant Columns
`DeviceID`, `deviceModel`, `deviceMake`, `Network provi.`, and `NetworkType` were removed. These have high cardinality or are not relevant signal-level features for handover prediction.

#### Signal Range Validation
Physically impossible signal values were set to `NaN` and later handled:

- `RSRP`: valid range `[-140, -44]` dBm
- `RSRQ`: valid range `[-20, -3]` dB
- `SINR`: the sentinel value `2147483647` (integer overflow artifact) was replaced with `NaN`, then filled with the median

---

### 3. Feature Engineering (Round 1)

The `Timestamp` column was parsed into datetime and the data sorted chronologically. Three time-of-day features were extracted:

- `hour`, `minute`, `second`

#### Target Variable: `Handover`
The target is derived from changes in `PCI` between consecutive rows:

```python
df['Handover'] = (df['PCI'] != df['PCI'].shift(1)).astype(int)
df.loc[0, 'Handover'] = 0  # First row has no predecessor
```

A `1` indicates a handover occurred; `0` means the device stayed connected to the same cell tower.

#### Class Imbalance
The dataset is heavily imbalanced — only **3.09%** of records are handover events (317 out of 10,251). This is addressed in modeling via SMOTE and class weighting.

| Class | Count | Share |
|---|---|---|
| No Handover (0) | 9,934 | 96.91% |
| Handover (1) | 317 | 3.09% |

---

### 4. Modeling — Baseline

All models were trained on **scaled features** using `StandardScaler` to ensure uniformity across experiments and fair comparisons between algorithms, even for tree-based models that do not strictly require it.

#### Train/Test Split
An **80/20 time-series split** was used (no shuffling) to prevent data leakage — the test set contains only future observations relative to training.

```python
split_idx = int(len(df) * 0.8)
```

#### Scaling
`StandardScaler` was fit **only on the training set**, then applied to both train and test sets — avoiding any information bleed from the test distribution.

#### Model 1 — Logistic Regression with `class_weight='balanced'`
Class weights inversely proportional to class frequencies are applied, penalising misclassification of the minority class more heavily.

#### Model 2 — Logistic Regression with SMOTE
SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic minority-class samples to create a balanced training set before fitting a standard logistic regression.

Both models were evaluated using `classification_report` and `confusion_matrix`, focusing on **recall for class 1 (handover)**.

#### Model 3 — Random Forest with `class_weight='balanced'`
A 300-estimator Random Forest was trained to capture non-linear decision boundaries that logistic regression cannot.

#### Threshold Tuning
For logistic regression, the default 0.5 decision threshold was examined. Adjusting the threshold allows trading precision for recall on the minority class depending on operational priorities.

---

### 5. Feature Engineering (Round 2)

Additional signal-dynamics features were engineered to capture the **rate of change** of signal conditions leading up to a potential handover:

| Feature | Description |
|---|---|
| `RSRP_diff` | Change in RSRP from previous record |
| `RSRQ_diff` | Change in RSRQ from previous record |
| `SINR_diff` | Change in SINR from previous record |
| `velocity_diff` | Change in device velocity |
| `RSRP_rolling_mean` | 5-sample rolling mean of RSRP |

`minute` and `second` columns were dropped as they introduced noise without meaningful signal. NaN values in the new diff/rolling features (first rows) were filled with `0`.

---

### 6. Modeling — With Engineered Features

The expanded feature set was used to retrain models under the same scaled + time-series split regime:

#### Logistic Regression (class weight & SMOTE variants)
Performance was compared against baseline to quantify the contribution of the new features.

#### Random Forest with `RandomizedSearchCV`
Hyperparameter tuning over:
- `n_estimators`: [200, 300, 400, 500]
- `max_depth`: [10, 15, 20, None]
- `min_samples_leaf`: [1, 2, 5, 10]
- `min_samples_split`: [2, 5, 10]
- `max_features`: ["sqrt", "log2"]

Scoring was optimised on **F1** with 3-fold cross-validation and 20 random iterations. Threshold sweeps (0.2–0.6) were then applied to the best model's probability outputs.

#### Feature Importance
The trained Random Forest was used to rank feature importances, surfacing which signal and derived features most strongly predict handovers.

#### Decision Tree Classifier
A shallower, interpretable model (`max_depth=15`) was tested as a simpler alternative, offering transparency at the cost of some performance.

#### XGBoost Classifier
Gradient boosting was applied with `scale_pos_weight` set to the ratio of negative to positive samples — XGBoost's native mechanism for handling class imbalance.

```python
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
```

Threshold sweeps (0.2, 0.3, 0.4, 0.5) were run on XGBoost probability outputs to find the optimal operating point.

---

## 📊 Results

### Class Imbalance Context
The test set contained **2,051 samples** — 2,021 non-handover and only **30 handover events** — making recall for class 1 the critical metric.

---

### Baseline Models (Original Features)

| Model | Imbalance Handling | Precision (H=1) | Recall (H=1) | F1 (H=1) | Accuracy |
|---|---|---|---|---|---|
| Logistic Regression | Class Weight | 0.02 | 0.37 | 0.04 | 77% |
| Logistic Regression | SMOTE | 0.02 | 0.43 | 0.05 | 73% |
| Random Forest (300 trees) | Class Weight | 0.00 | 0.00 | 0.00 | 99% |

> **Key observation:** The baseline Random Forest completely failed on the minority class — it predicted no handovers at all, achieving 99% accuracy purely by predicting the majority class. The logistic regression models caught some handovers but with very poor precision (2%). This confirmed the need for richer features.

---

### Models with Engineered Features

| Model | Imbalance Handling | Precision (H=1) | Recall (H=1) | F1 (H=1) | Accuracy |
|---|---|---|---|---|---|
| Logistic Regression | Class Weight | 1.00 | 0.07 | 0.12 | 99% |
| Logistic Regression | SMOTE | 1.00 | 0.07 | 0.12 | 99% |
| Random Forest (tuned, RandomizedSearchCV) | Class Weight | 0.53 | **0.97** | **0.68** | 99% |
| Random Forest (300 trees, max_depth=15) | Class Weight | 0.92 | 0.37 | 0.52 | 99% |
| Decision Tree (max_depth=15) | Class Weight | 0.53 | 0.57 | 0.55 | 99% |
| XGBoost (300 trees, max_depth=15) | scale_pos_weight | 0.61 | 0.77 | **0.68** | 99% |

> **Key observation:** Engineered diff features (`RSRP_diff`, `RSRQ_diff`, `SINR_diff`) transformed model performance dramatically. The tuned Random Forest achieved **97% recall** — missing only 1 out of 30 handovers — making it the strongest model for minimising missed events.

---

### Threshold Tuning — Tuned Random Forest

Sweeping thresholds from 0.2 to 0.55 on the tuned Random Forest's probability outputs:

| Threshold | Recall | Precision | F1 |
|---|---|---|---|
| 0.20 | **1.00** | 0.46 | 0.63 |
| 0.35 | **1.00** | 0.47 | 0.64 |
| 0.45 | **1.00** | 0.51 | 0.67 |
| 0.50 | 0.97 | 0.53 | **0.68** |
| 0.55 | 0.87 | 0.53 | 0.66 |

> At threshold ≤ 0.45, the model catches **every single handover** with ~50% precision. The default 0.5 threshold gives the best F1 (0.68) while still catching 97% of handovers.

---

### Threshold Tuning — XGBoost

| Threshold | Recall | Precision | F1 | Confusion Matrix |
|---|---|---|---|---|
| 0.20 | 0.87 | 0.59 | **0.70** | TN=2003, FP=18, FN=4, TP=26 |
| 0.30 | 0.83 | 0.60 | 0.69 | TN=2004, FP=17, FN=5, TP=25 |
| 0.40 | 0.83 | 0.61 | **0.70** | TN=2005, FP=16, FN=5, TP=25 |
| 0.50 | 0.77 | 0.61 | 0.68 | TN=2006, FP=15, FN=7, TP=23 |

> XGBoost at threshold 0.2 delivers the best balance of precision and recall, with only 4 missed handovers.

---

### Feature Importance (Random Forest, max_depth=15)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `RSRP_diff` | 0.2496 |
| 2 | `SINR_diff` | 0.2143 |
| 3 | `RSRQ_diff` | 0.1653 |
| 4 | `Latitude` | 0.0721 |
| 5 | `Longitude` | 0.0686 |
| 6 | `Velocity(km/h)` | 0.0380 |
| 7 | `SINR` | 0.0323 |
| 8 | `RSRP_rolling_mean` | 0.0302 |
| 9 | `RSRP` | 0.0279 |
| 10 | `hour` | 0.0276 |

> The top 3 features are all **engineered diff features**, confirming that the *rate of change* in signal quality is far more predictive of handovers than absolute signal values alone. Location (`Latitude`, `Longitude`) also ranks highly, reflecting that handovers are spatially driven by tower coverage boundaries.

---

### Best Model Summary

| Criterion | Best Model |
|---|---|
| Highest Recall (catch the most handovers) | Tuned RF @ threshold ≤ 0.45 — **Recall: 1.00** |
| Best F1 Balance | Tuned RF @ threshold 0.50 or XGBoost @ threshold 0.2–0.4 — **F1: 0.68–0.70** |
| Most Interpretable | Decision Tree — F1: 0.55, Recall: 0.57 |

---

## 🧰 Tech Stack

- **Python 3**
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — preprocessing, models, evaluation
- `imbalanced-learn` — SMOTE
- `xgboost` — gradient boosting
- `matplotlib` — visualisation
- `kagglehub` — dataset download

---

## ⚠️ Notes

- All models were trained on **StandardScaler-transformed features** for experimental consistency, including tree-based models (Random Forest, Decision Tree, XGBoost) which are scale-invariant by nature. This was a deliberate choice to keep the preprocessing pipeline uniform across all experiments.
- The time-series split is critical — random splitting would leak future signal patterns into training and produce unrealistically optimistic metrics.
- SMOTE is applied **after** scaling and **only to training data** to avoid synthetic samples contaminating the test evaluation.

---

## 🚀 Getting Started

```bash
pip install kagglehub pandas numpy scikit-learn imbalanced-learn xgboost matplotlib
```

Then run the notebook top-to-bottom. Ensure Kaggle API credentials are configured for `kagglehub` to download the dataset automatically.





