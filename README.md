# Rainfall Prediction Classifier 🌧️

A machine learning project that predicts whether it will rain **today** in the Melbourne, Australia area, using historical weather observations. Built as the final project for IBM's *Machine Learning with Python* course.

## Overview

This project simulates a real-world data science task at a fictional company, **WeatherTech Inc.**, where the goal is to build a classifier that predicts rainfall based on weather features like temperature, humidity, wind speed, and atmospheric pressure.

The notebook walks through the full ML workflow:

1. **Data exploration & feature engineering** — cleaning missing values, reframing the prediction target to avoid data leakage, engineering a `Season` feature from the date, and narrowing the dataset to a single climate region (Melbourne, Melbourne Airport, Watsonia).
2. **Pipeline building** — a `scikit-learn` `Pipeline` combining preprocessing (scaling + one-hot encoding) with a classifier, tuned via `GridSearchCV` and stratified k-fold cross-validation.
3. **Model evaluation** — classification reports, confusion matrices, and feature importance analysis, comparing a **Random Forest Classifier** against a **Logistic Regression** model.

## Dataset

- **Source:** [Kaggle — Weather Dataset Rattle Package](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) (originally from the [Australian Bureau of Meteorology](http://www.bom.gov.au/climate/dwo/))
- **File:** `weatherAUS.csv`
- **Period:** Daily weather observations from 2008–2017 across multiple Australian locations
- **Target:** `RainToday` (Yes/No) — reframed from the original `RainTomorrow` column to avoid using same-day features that wouldn't be available in a real forecasting scenario

## Key Steps

| Step | Description |
|------|-------------|
| Data leakage check | Identified and excluded same-day features (e.g. `Sunshine`, `Cloud9am/3pm`) that wouldn't be known in advance |
| Location filtering | Restricted to Melbourne, Melbourne Airport, and Watsonia for a localized, more predictable model |
| Feature engineering | Mapped `Date` → `Season` (Summer/Autumn/Winter/Spring) |
| Preprocessing | `StandardScaler` for numeric features, `OneHotEncoder` for categorical features via `ColumnTransformer` |
| Modeling | `RandomForestClassifier` tuned with `GridSearchCV` (n_estimators, max_depth, min_samples_split) |
| Model comparison | `LogisticRegression` (liblinear solver, L1/L2 penalty, class weighting) trained on the same pipeline for comparison |
| Evaluation | Classification report, confusion matrix, and Random Forest feature importances |

## Results

- **Class balance:** ~76% No rain / ~24% Yes rain — an imbalanced target, so accuracy alone is not a reliable metric
- **Random Forest:** ~84% test accuracy, 51% recall (true positive rate) for rainy days, 75% precision
- **Logistic Regression:** ~83% test accuracy, 51% recall for rainy days, more false positives than Random Forest
- **Most important feature:** `Humidity3pm`, followed by `Pressure3pm`, `Pressure9am`, and `Sunshine`

Both models perform similarly overall, with Random Forest holding a slight edge in accuracy and precision. Recall on the minority "rain" class remains the main limitation of both models — a reminder that high accuracy can mask weak performance on the class that matters most.

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn (`Pipeline`, `ColumnTransformer`, `GridSearchCV`, `RandomForestClassifier`, `LogisticRegression`)
- matplotlib, seaborn

## Project Structure

```
.
├── FinalProject_AUSWeather_completed.ipynb   # Main notebook with full analysis
├── weatherAUS.csv                            # Dataset (or fetched via URL in-notebook)
└── README.md
```

## How to Run

1. Clone this repository
2. Open `FinalProject_AUSWeather_completed.ipynb` in Jupyter or [Google Colab](https://colab.research.google.com/)
3. If running in Colab, either:
   - Upload `weatherAUS.csv` using the file picker cell, **or**
   - Use the course-hosted dataset URL included in the notebook
4. Run all cells in order

## Acknowledgements

- Course: *Machine Learning with Python* — IBM / Skills Network
- Original lab authors: Jeff Grossman, Abhishek Gagneja
- Data: Australian Government Bureau of Meteorology, via Kaggle
