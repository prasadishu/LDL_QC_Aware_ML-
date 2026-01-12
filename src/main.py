from data_loader import load_data
from preprocessing import preprocess
from qc_module import generate_qc_scores, apply_qc
from models import catboost_model
from train import split_data
from evaluate import evaluate

df = preprocess(load_data("data/LDL-C.xlsx"))
X_train, X_test, y_train, y_test = split_data(df)

model = catboost_model()
model.fit(X_train, y_train)

qc_scores = generate_qc_scores(len(y_test))
preds = model.predict(X_test)
preds_qc = apply_qc(preds, qc_scores)

print(evaluate(y_test, preds_qc))
