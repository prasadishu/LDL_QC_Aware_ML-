from src.data_loader import load_data
from src.preprocessing import preprocess
from src.qc_module import generate_qc_scores, apply_qc
from src.models import catboost_model
from src.train import split_data
from src.evaluate import evaluate

df = preprocess(load_data("data/LDL-C.xlsx"))
X_train, X_test, y_train, y_test = split_data(df)

model = catboost_model()
model.fit(X_train, y_train)

qc_scores = generate_qc_scores(len(y_test))
preds = model.predict(X_test)
preds_qc = apply_qc(preds, qc_scores)

print(evaluate(y_test, preds_qc))
