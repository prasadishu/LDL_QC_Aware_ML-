import numpy as np

def bland_altman(y_true, y_pred):
    diff = y_pred - y_true
    bias = diff.mean()
    sd = diff.std()
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd
    return bias, loa_upper, loa_lower
