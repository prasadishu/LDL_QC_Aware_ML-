import sys
import pandas as pd
from catboost import CatBoostRegressor

def predict(tc, hdl, tg, age, gender, qc):
    model = CatBoostRegressor()
    model.load_model("catboost_ldl_model.cbm")

    X = pd.DataFrame({
        "Total_Cholesterol":[tc],
        "HDL":[hdl],
        "Triglycerides":[tg],
        "Age":[age],
        "Gender":[gender]
    })

    return round(model.predict(X)[0] * qc, 1)

if __name__ == "__main__":
    tc, hdl, tg, age, gender, qc = map(float, sys.argv[1:])
    print(predict(tc, hdl, tg, age, int(gender), qc))
