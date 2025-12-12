# Evaluate the full stormwater classification pipeline
import argparse
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from src.classify import StormwaterClassifier
from src.lookups.lookup_client import PlutoLookupClient
from src.parsing.description_parser import parse_description


FINAL_LABELS = ["ESC", "WQ", "RR", "Vv", "NNI"]

INTERMEDIATE_LABELS = [
    "disturb_20000_sf",
    "new_imp",
    "new_imp_5000_sf",
    "table_2_2_activity",
    "in_ms4",
]


# Convert common boolean-like values to 0 or 1
def to_bool(val):
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (int, float)):
        return int(val)
    val = str(val).strip().lower()
    if val in ("true", "1", "yes", "y"):
        return 1
    if val in ("false", "0", "no", "n", "", "none"):
        return 0
    raise ValueError(f"Cannot interpret boolean value: {val}")


def evaluate_classifier(data_path: Path, models_dir: Path):
    print(f"\nEvaluating classifier on {data_path.name}\n")

    df = pd.read_csv(data_path)
    clf = StormwaterClassifier(
        lookup_client=PlutoLookupClient(Path("data")),
        models_dir=models_dir,
    )

    y_true = {lbl: [] for lbl in FINAL_LABELS}
    y_pred = {lbl: [] for lbl in FINAL_LABELS}
    y_true_inter = {lbl: [] for lbl in INTERMEDIATE_LABELS}
    y_pred_inter = {lbl: [] for lbl in INTERMEDIATE_LABELS}

    for _, row in df.iterrows():
        desc = row["description"]
        parsed = parse_description(desc)

        final, inter = clf.classify_with_explanation(desc)

        # Read ground-truth labels from CSV
        for lbl in FINAL_LABELS:
            if lbl == "NNI":
                # Ground-truth NNI from CSV
                val = str(row[lbl]).strip().lower()
                true_nni = 0 if val in ("false", "", "none", "na") else 1
                y_true[lbl].append(true_nni)

                # Predicted NNI from classifier output
                pred_val = getattr(final, lbl)
                pred_nni = 0 if (pred_val is False or pred_val == [] or pred_val is None) else 1
                y_pred[lbl].append(pred_nni)
            else:
                true_val = to_bool(row[lbl])
                pred_val = int(getattr(final, lbl))
                y_true[lbl].append(true_val)
                y_pred[lbl].append(pred_val)

        # Evaluate intermediate labels
        for ilbl in INTERMEDIATE_LABELS:
            # Ground-truth value
            true_inter = to_bool(row[ilbl])
            y_true_inter[ilbl].append(true_inter)

            # Predicted value from classifier
            pred_inter = inter.__dict__.get(ilbl, False)
            pred_inter_val = int(bool(pred_inter))
            y_pred_inter[ilbl].append(pred_inter_val)

    print("Label    Prec    Rec     F1     Support")

    all_true = []
    all_pred = []

    for lbl in FINAL_LABELS:
        p = precision_score(y_true[lbl], y_pred[lbl], zero_division=0)
        r = recall_score(y_true[lbl], y_pred[lbl], zero_division=0)
        f = f1_score(y_true[lbl], y_pred[lbl], zero_division=0)
        s = sum(y_true[lbl])

        print(f"{lbl:5s}   {p:0.3f}   {r:0.3f}   {f:0.3f}   {s:5d}")

        # For micro/macro F1
        all_true.extend(y_true[lbl])
        all_pred.extend(y_pred[lbl])

    print("\nIntermediate Label Performance")
    print("Label                Prec    Rec     F1     Support")
    print("-----------------------------------------------------")
    for lbl in INTERMEDIATE_LABELS:
        p = precision_score(y_true_inter[lbl], y_pred_inter[lbl], zero_division=0)
        r = recall_score(y_true_inter[lbl], y_pred_inter[lbl], zero_division=0)
        f = f1_score(y_true_inter[lbl], y_pred_inter[lbl], zero_division=0)
        s = sum(y_true_inter[lbl])
        print(f"{lbl:20s} {p:0.3f}   {r:0.3f}   {f:0.3f}   {s:5d}")

    # Aggregate metrics
    macro_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(all_true, all_pred, average="micro", zero_division=0)
    weighted_f1 = f1_score(all_true, all_pred, average="weighted", zero_division=0)
    acc = accuracy_score(all_true, all_pred)

    print("\nAggregate Metrics")
    print(f"Macro F1:     {macro_f1:0.3f}")
    print(f"Micro F1:     {micro_f1:0.3f}")
    print(f"Weighted F1:  {weighted_f1:0.3f}")
    print(f"Accuracy:     {acc:0.3f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--models_dir", default="models", help="Directory containing semantic models")
    args = parser.parse_args()

    evaluate_classifier(
        data_path=Path(args.data),
        models_dir=Path(args.models_dir),
    )


if __name__ == "__main__":
    main()