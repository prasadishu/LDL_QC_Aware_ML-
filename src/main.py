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
qc = WestgardQC(mean=y.mean(), sd=y.std())

final_results = []

for val in predicted:
    status = apply_qc(val, qc)
    if status == "FAIL":
        val = conservative_adjustment(val, y.mean())
    final_results.append(val)

# Evaluation
metrics = evaluate(y, final_results)
print(metrics)
