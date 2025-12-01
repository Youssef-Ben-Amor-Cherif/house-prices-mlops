# src/eval.py

import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def evaluate_model(data_dir="data/processed", model_path="models/model.pkl"):
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze()

    model = joblib.load(model_path)

    y_pred_test = model.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)

    print(f"[EVAL] Test RMSE: {rmse_test:.2f}")
    print(f"[EVAL] Test R2: {r2_test:.2f}")

if __name__ == "__main__":
    evaluate_model()
