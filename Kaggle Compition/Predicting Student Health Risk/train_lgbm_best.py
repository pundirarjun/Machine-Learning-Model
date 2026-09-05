"""Train the selected native-categorical LightGBM model and create a Kaggle submission."""

from pathlib import Path

import lightgbm as lgb
import pandas as pd


ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "train.csv"
TEST = ROOT / "test.csv"
SAMPLE = ROOT / "sample_submission.csv"
OUTPUT = ROOT / "submissions" / "submission_lgbm_balanced_calibrated_cv94931.csv"
TARGET = "health_condition"


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove the identifier and preserve categorical columns for LightGBM."""
    result = frame.drop(columns=["id"]).copy()
    for column in result.select_dtypes(include=["object", "str"]).columns:
        result[column] = result[column].astype("category")
    return result


def main() -> None:
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)
    sample = pd.read_csv(SAMPLE)

    y = train.pop(TARGET).astype("category")
    X = prepare(train)
    X_test = prepare(test)
    categorical_columns = X.select_dtypes(include=["category"]).columns.tolist()

    model = lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=350,
        learning_rate=0.06,
        num_leaves=31,
        min_child_samples=35,
        reg_alpha=0.3,
        reg_lambda=2.0,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=2026,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(X, y, categorical_feature=categorical_columns)

    # Selected on a stratified validation split for the competition's balanced
    # accuracy metric (not ordinary accuracy).
    probabilities = model.predict_proba(X_test)
    probabilities *= [1.0, 1.45, 1.45]  # at-risk, fit, unhealthy
    predictions = model.classes_[probabilities.argmax(axis=1)]
    submission = sample.copy()
    submission[TARGET] = predictions
    OUTPUT.parent.mkdir(exist_ok=True)
    submission.to_csv(OUTPUT, index=False)

    assert submission.columns.tolist() == sample.columns.tolist()
    assert submission["id"].equals(sample["id"])
    assert len(submission) == len(test)
    print(f"Wrote {OUTPUT}")
    print(submission[TARGET].value_counts().to_dict())


if __name__ == "__main__":
    main()
