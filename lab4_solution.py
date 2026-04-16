import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

DATASETS = [
    {
        "name": "adult",
        "file": "dataset/processed_adult.csv",
        "model": "DNN/model_processed_adult.h5",
        "label": "Class-label",
        "sensitive": ["race", "gender", "age"]
    },
    {
        "name": "compas",
        "file": "dataset/processed_compas.csv",
        "model": "DNN/model_processed_compas.h5",
        "label": "Recidivism",
        "sensitive": ["Sex", "Age", "Race"]
    },
    {
        "name": "dutch",
        "file": "dataset/processed_dutch.csv",
        "model": "DNN/model_processed_dutch.h5",
        "label": "occupation",
        "sensitive": ["sex", "age"]
    },
    {
        "name": "german",
        "file": "dataset/processed_german.csv",
        "model": "DNN/model_processed_greman_cleaned.h5",
        "label": "CREDITRATING",
        "sensitive": ["PersonStatusSex", "AgeInYears"]
    },
    {
        "name": "law",
        "file": "dataset/processed_law_school.csv",
        "model": "DNN/model_processed_law_school_cleaned.h5",
        "label": "pass_bar",
        "sensitive": ["male", "race"]
    },
    {
        "name": "credit",
        "file": "dataset/processed_credit_with_numerical.csv",
        "model": "DNN/model_processed_credit.h5",
        "label": "class",
        "sensitive": ["SEX", "AGE"]
    },
    {
        "name": "crime",
        "file": "dataset/processed_communities_crime.csv",
        "model": "DNN/model_processed_communities_crime.h5",
        "label": "class",
        "sensitive": ["Black"]
    },
    {
        "name": "kdd",
        "file": "dataset/processed_kdd.csv",
        "model": "DNN/model_processed_kdd_cleaned.h5",
        "label": "income",
        "sensitive": ["age", "race", "sex"]
    },
]


def load_and_preprocess_data(file_path, target_column):
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_column])
    X = X.astype(float)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def generate_sample_pair(X_test, sensitive_columns, non_sensitive_columns):
    sample_a = X_test.iloc[np.random.choice(len(X_test))].copy()
    sample_b = sample_a.copy()

    for col in sensitive_columns:
        if col in X_test.columns:
            unique_values = X_test[col].unique()
            sample_b[col] = np.random.choice(unique_values)

    for col in non_sensitive_columns:
        if col in X_test.columns:
            min_val = float(X_test[col].min())
            max_val = float(X_test[col].max())
            perturbation = np.random.uniform(
                -0.1 * (max_val - min_val),
                 0.1 * (max_val - min_val)
            )
            sample_a[col] = float(np.clip(float(sample_a[col]) + perturbation, min_val, max_val))
            sample_b[col] = float(np.clip(float(sample_b[col]) + perturbation, min_val, max_val))

    return sample_a, sample_b


def evaluate_discrimination(model, sample_a, sample_b, threshold=0.05):
    sample_a = np.array(sample_a, dtype=float)
    sample_b = np.array(sample_b, dtype=float)

    prediction_a = model.predict(sample_a.reshape(1, -1), verbose=0)
    prediction_b = model.predict(sample_b.reshape(1, -1), verbose=0)

    pred_a = prediction_a[0][0]
    pred_b = prediction_b[0][0]

    if abs(pred_a - pred_b) > threshold:
        return 1
    else:
        return 0


def calculate_idi_ratio(model, X_test, sensitive_columns, non_sensitive_columns, num_samples=1000):
    discrimination_count = 0
    for _ in range(num_samples):
        sample_a, sample_b = generate_sample_pair(X_test, sensitive_columns, non_sensitive_columns)
        discrimination_count += evaluate_discrimination(model, sample_a, sample_b)
    return discrimination_count / num_samples


def main():
    os.makedirs("results", exist_ok=True)
    all_results = []

    for ds in DATASETS:
        print(f"\nRunning baseline on: {ds['name']}")
        try:
            X_train, X_test, y_train, y_test = load_and_preprocess_data(ds["file"], ds["label"])
            model = keras.models.load_model(ds["model"])
            non_sensitive = [c for c in X_test.columns if c not in ds["sensitive"]]
            ratio = calculate_idi_ratio(model, X_test, ds["sensitive"], non_sensitive)
            print(f"  IDI Ratio: {ratio:.4f}")
            all_results.append({"dataset": ds["name"], "IDI_ratio": ratio})
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({"dataset": ds["name"], "IDI_ratio": None})

    df = pd.DataFrame(all_results)
    print("\n===== BASELINE RESULTS =====")
    print(df.to_string(index=False))
    df.to_csv("results/baseline_results.csv", index=False)
    print("\nSaved to results/baseline_results.csv")


if __name__ == "__main__":
    main()