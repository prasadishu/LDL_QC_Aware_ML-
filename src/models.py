from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

def catboost_model():
    return CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        verbose=0
    )

def xgboost_model():
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        objective='reg:squarederror'
    )

def rf_model():
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )

def svr_model():
    return SVR(kernel='rbf', C=100, epsilon=5)
