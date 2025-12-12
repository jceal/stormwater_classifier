# Text classification models and save/load helpers

import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Build text classification pipeline
def make_tfidf_logreg():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.90,
            max_features=3000,
            stop_words="english"
        )),
        ("clf", LogisticRegression(max_iter=300, class_weight="balanced", C=2.0))
    ])

# Model save and load utilities
def save_model(model, label_name, models_dir):
    # Use a different filename for the Vv label
    if label_name == "Vv":
        filename = "new_connection.joblib"
    else:
        filename = f"{label_name}.joblib"

    path = models_dir / filename
    joblib.dump(model, path)


def load_models(models_dir: Path):
    """
    Load saved models if they exist.
    """
    models = {}
    for key in ["table_2_2_activity", "new_connection"]:
        path = models_dir / f"{key}.joblib"
        if path.exists():
            models[key] = joblib.load(path)
    return models