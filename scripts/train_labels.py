# Train text-based models for labels

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.utils import resample

from src.models.models import make_tfidf_logreg, save_model

DATA_PATH = Path("data/project_data_150.csv")
MODELS_DIR = Path("models")


def balance_df(df, col):
    """
    Oversample the minority class to reduce class imbalance.
    """
    df_majority = df[df[col] == 0]
    df_minority = df[df[col] == 1]

    # Skip balancing if there are no positive samples
    if len(df_minority) == 0:
        print(f"WARNING: No positive samples for {col} — cannot balance.")
        return df

    # Oversample minority class
    target_samples = max(1, len(df_majority) // 2)

    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=target_samples,
        random_state=42
    )

    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Balanced dataset for {col}: {len(df_balanced)} rows")

    return df_balanced

def train_for_column(df, col):
    print(f"\n=== Training semantic model for {col} ===")

    # Balance classes before training
    df_balanced = balance_df(df, col)

    X = df_balanced["description"]
    y = df_balanced[col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = make_tfidf_logreg()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    save_model(model, col, MODELS_DIR)


def main():
    df = pd.read_csv(DATA_PATH)

    train_for_column(df, "table_2_2_activity")
    train_for_column(df, "Vv")  # new connection lexical model


if __name__ == "__main__":
    main()