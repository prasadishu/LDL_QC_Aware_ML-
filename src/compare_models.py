"""
compare_models.py

This script compares multiple LDL-C estimation approaches:
1. CatBoost Regressor
2. XGBoost Regressor
3. Random Forest Regressor
4. Support Vector Regression (SVR)
5. Friedewald Formula (clinical baseline)

All models are trained and evaluated using the SAME dataset
and SAME train-test split to ensure fair comparison.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
def load_data(df):
    """
    Assumes dataframe contains:
    TC, HDL, TG, Age, Gender, LDL_direct
    """
    X = df[["Total_Cholesterol", "HDL", "Triglycerides", "Age", "Gender"]]
    y = df["LDL_direct"]
    return X, y


# ------------------------------------------------------------------
# 2. Friedewald calculation (clinical comparator)
# ------------------------------------------------------------------
def friedewald_ldl(tc, hdl, tg):
    """
    Friedewald formula (valid only when TG < 400 mg/dL)
    LDL = TC - HDL - (TG / 5)
    """
    return tc - hdl - (tg / 5.0)


# ------------------------------------------------------------------
# 3. Evaluation function
# ------------------------------------------------------------------
def evaluate_model(y_true, y_pred):
    """
    Returns MAE, RMSE, R²
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


# ------------------------------------------------------------------
# 4. Main comparison pipeline
# ------------------------------------------------------------------
def compare_all_models(df):

    X, y = load_data(df)

    # SAME split for all models (critical for fairness)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    # -----------------------------
    # CatBoost
    # -----------------------------
    cat = CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        verbose=False
    )
    cat.fit(X_train, y_train)
    pred_cat = cat.predict(X_test)
    results["CatBoost"] = evaluate_model(y_test, pred_cat)

    # -----------------------------
    # XGBoost
    # -----------------------------
    xgb = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        objective="reg:squarederror",
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    pred_xgb = xgb.predict(X_test)
    results["XGBoost"] = evaluate_model(y_test, pred_xgb)

    # -----------------------------
    # Random Forest
    # -----------------------------
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    results["RandomForest"] = evaluate_model(y_test, pred_rf)

    # -----------------------------
    # Support Vector Regression
    # -----------------------------
    svr = SVR(kernel="rbf", C=100, gamma=0.1)
    svr.fit(X_train, y_train)
    pred_svr = svr.predict(X_test)
    results["SVR"] = evaluate_model(y_test, pred_svr)

    # -----------------------------
    # Friedewald (baseline)
    # -----------------------------
    friedewald_pred = friedewald_ldl(
        X_test["Total_Cholesterol"],
        X_test["HDL"],
        X_test["Triglycerides"]
    )
    results["Friedewald"] = evaluate_model(y_test, friedewald_pred)

    # Convert results to DataFrame
    results_df = pd.DataFrame(
        results,
        index=["MAE", "RMSE", "R2"]
    ).T

    return results_df
