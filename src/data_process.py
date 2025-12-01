# src/data_process.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_data(input_path="data/train.csv", output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path)

    numeric_features = df.select_dtypes(include=np.number)
    numeric_features = numeric_features.dropna()

    X = numeric_features.drop("SalePrice", axis=1)
    y = numeric_features["SalePrice"]

    y_binned = pd.qcut(y, q=10, duplicates="drop", labels=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_binned
    )

    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

if __name__ == "__main__":
    prepare_data()
