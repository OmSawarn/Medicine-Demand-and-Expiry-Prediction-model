# Medicine-Demand-and-Expiry-Prediction-model
A machine learning decision-support system designed to forecast product-level medicine demand and evaluate inventory shelf-life risk across public health facilities.

---

## Project Overview

In resource-constrained healthcare systems, medical supply chains face a persistent dual challenge:
1. **Critical Stockouts:** Sudden depletion of life-saving medicines (such as oxytocin, oral rehydration salts, and antibiotics) endangers patient care.
2. **Expired Inventory Wastage:** Overstocked medicines sit stagnant in clinic storerooms past their expiration date, leading to avoidable financial waste and disposal costs.

This project builds an end-to-end, leak-free machine learning pipeline and an interactive Streamlit decision-support dashboard. By evaluating routine health information records and central warehouse stock, the system provides **1-month-ahead demand forecasts** and identifies **batches at risk of stockout or expiration** under the **First Expiry, First Out (FEFO)** principle.

---

## Data Sourcing & Attribution
## Data set link: https://drive.google.com/drive/folders/1fOXZm5IDdE9Wz39SPlMnTfSKDmgYqdMp?usp=drive_link

### Dataset Descriptions

| File | Records | Key Columns | Description |
| :--- | :--- | :--- | :--- |
| `data/S2_Dhis2Data.csv` | 457,225 rows, 16 features | `hf_pk`, `name1`, `date`, `consumption`, `received`, `closeBalance`, `openBalance`, `stockout`, `facility_type`, `district`, `productID` | Routine monthly DHIS2 logistics reports from **1,091 public health facilities** across **16 districts** over **50 consecutive months** (Oct 2019 – Nov 2023) covering 36 essential medicines. |
| `data/S3.csv` | 341 rows, 3 features | `Item`, `Stock`, `Quarter` | Quarterly central medical warehouse stock inventory across 150 items spanning 4 quarters (2022Q4 to 2023Q3). |

### Data Integrity & Verification Note
Programmatic inspection confirmed that routine DHIS2 reports record aggregate facility-level balances and stockouts, but do not record manufacturer batch lot numbers or exact expiry dates. Rather than fabricating fake batch lot dates into real records:
1. On real data, shelf-life risk is assessed via **Months of Stock (MOS)** ($\text{Closing Inventory} / \text{Monthly Consumption}$), identifying overstocked products ($>6$ months MOS) at severe risk of shelf stagnation.
2. A separate, documented **FEFO (First Expiry, First Out) Batch Decision Prototype** is implemented to demonstrate batch-level precision, recall, and classification when warehouse management system (WMS) batch attributes are present.

---

## Machine Learning Methodology

###  Problem Formulation
- **Prediction Unit:** Product-Month per reporting cycle across 1,091 healthcare clinics.
- **Target Variable:** Total consumption quantity in period $t+1$ (next month's demand).
- **Forecasting Horizon:** 1 month forward (matching routine procurement reorder cadences).
- **Stock Rotation Policy:** First Expiry, First Out (FEFO).

###  Leak-Free Temporal Feature Engineering
To strictly prevent future lookahead bias, features are generated using backward historical shifts (`shift(1)`):
- **Calendar Signals:** `month` (1–12), `year`, `quarter_num` (1–4).
- **Lag Features:** `lag_1`, `lag_2`, `lag_3` (consumption at $t-1, t-2, t-3$).
- **Rolling Statistics:** `rolling_mean_3` and `rolling_std_3` over the preceding 3-month window.
- **Supply Signals:** `received`, `open_balance`, `close_balance`, `stockout_rate`, `active_facilities`.

###  Chronological Splitting 
Random shuffling is strictly avoided in time series. The 1,720 product-month observations were partitioned chronologically:
- **Training Set (70%):** Jan 2020 – Sep 2022 (1,142 records)
- **Validation Set (15%):** Oct 2022 – Apr 2023 (240 records)
- **Held-Out Test Set (15%):** May 2023 – Oct 2023 (194 records)

---

##  Model Evaluation & Benchmark Results

###  Demand Forecasting (Regression)
Evaluated on the held-out test split using **Weighted Absolute Percentage Error (WAPE)**, **MAE**, and **RMSE**:

$$\text{WAPE} = \frac{\sum |y_i - \hat{y}_i|}{\sum y_i} \times 100\%$$

| Model | MAE (Units) | RMSE (Units) | WAPE (%) | Accuracy ($100 - \text{WAPE}$) | Error Reduction vs. Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **3-Month Moving Average Baseline** | 14,860.18 | 36,576.20 | 34.60% | 65.40% | Baseline |
| **Gradient Boosting Regressor** | 11,447.07 | 26,115.04 | 26.65% | 73.35% | 23.0% improvement |
| **Random Forest Regressor (Selected)** | **10,507.48** | **25,669.49** | **24.46%** | **75.54%** | **29.3% improvement** |

*The Random Forest Regressor captures non-linear interactions between closing balances, facility counts, and seasonal trends, reducing forecast error by nearly 30% relative to moving averages.*

###  Expiry-Risk Classification (FEFO Rule)
Using the FEFO condition ($\text{Remaining Batch Stock} > \text{Days to Expiry} \times \text{Daily Demand Rate}$) evaluated on 240 held-out test batches:
- **Accuracy:** **99.17%** (238 / 240 correctly classified)
- **Precision:** **99.04%**
- **Recall:** **99.04%**
- **F1-Score:** **99.04%**
- **Confusion Matrix:** 135 True Negatives, 103 True Positives, 1 False Positive, 1 False Negative.

---

## Limitations & Responsible AI

- **Temporal Granularity:** DHIS2 records are reported on a monthly frequency; daily clinic surges or emergency weekend spikes are aggregated.
- **Lot Tracking Gap:** Public health systems track aggregate inventory counts; integration with hospital warehouse management systems (WMS) is required to capture manufacturer lot numbers.
- **Operational Advisory Scope:** All model forecasts, stockout flags, and redistribution guidance are decision-support recommendations for authorized supply officers, never autonomous clinical or medical treatment advice.
