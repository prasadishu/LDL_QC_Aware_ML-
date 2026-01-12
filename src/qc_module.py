import numpy as np

def generate_qc_scores(n):
    return np.random.choice(
        [1.0, 0.95, 0.9, 0.85],
        size=n,
        p=[0.4, 0.3, 0.2, 0.1]
    )

def apply_qc(predictions, qc_scores):
    return predictions * qc_scores
