from src.data_loader import load_data
from src.preprocessing import preprocess
from src.qc_module import generate_qc_scores, apply_qc
from src.models import catboost_model
from src.train import split_data
from src.evaluate import evaluate

# Load data
df = load_data("../examples/example_input.csv")

X = preprocess(df)
y = df["LDL_direct"]

# Train model
model = get_model()
model = train_model(model, X, y)

# Predict
predicted = model.predict(X)

# QC setup
from qc_module import WestgardQC, apply_qc, conservative_adjustment

qc_engine = WestgardQC(mean=y.mean(), sd=y.std())

history = []
final_predictions = []

for value in predicted:
    qc_result = apply_qc(value, history, qc_engine)

    if qc_result["qc_status"] == "FAIL":
        value = conservative_adjustment(value, y.mean())

    final_predictions.append(value)
    history.append(value)

# Evaluation
metrics = evaluate(y, final_results)
print(metrics)
