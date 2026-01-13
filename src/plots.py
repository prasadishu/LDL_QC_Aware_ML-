import matplotlib.pyplot as plt

def plot_mae(results_df):
    plt.figure()
    plt.bar(results_df.index, results_df["MAE"])
    plt.ylabel("MAE (mg/dL)")
    plt.title("Comparison of MAE Across LDL-C Estimation Methods")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
  
  def plot_rmse(results_df):
    plt.figure()
    plt.bar(results_df.index, results_df["RMSE"])
    plt.ylabel("RMSE (mg/dL)")
    plt.title("Comparison of RMSE Across LDL-C Estimation Methods")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_r2(results_df):
    plt.figure()
    plt.bar(results_df.index, results_df["R2"])
    plt.ylabel("R²")
    plt.title("Explained Variance (R²) Across Models")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def bland_altman_plot(y_true, y_pred):
    mean = (y_true + y_pred) / 2
    diff = y_pred - y_true

    plt.figure()
    plt.scatter(mean, diff)
    plt.axhline(diff.mean(), linestyle="--")
    plt.axhline(diff.mean() + 1.96 * diff.std(), linestyle="--")
    plt.axhline(diff.mean() - 1.96 * diff.std(), linestyle="--")
    plt.xlabel("Mean of Methods")
    plt.ylabel("Difference (Predicted - Direct)")
    plt.title("Bland–Altman Plot for LDL-C Prediction")
    plt.show()
