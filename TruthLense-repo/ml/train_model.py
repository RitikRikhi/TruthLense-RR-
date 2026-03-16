import argparse
import json
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Dataset at {csv_path} is empty.")

    return df


def split_features_target(
    df: pd.DataFrame, target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        available = ", ".join(df.columns.astype(str))
        raise KeyError(
            f"Target column '{target_col}' not found in dataset. "
            f"Available columns: {available}"
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    if X.empty:
        raise ValueError("No feature columns remaining after dropping target.")

    return X, y


def infer_problem_type(y: pd.Series) -> str:
    """
    Heuristically decide whether this is a classification or regression task.
    """
    unique_values = y.dropna().unique()
    n_unique = len(unique_values)

    # Non-numeric target: definitely classification
    if not np.issubdtype(y.dtype, np.number):
        return "classification"

    # Numeric but few unique values -> classification
    if n_unique <= 15:
        return "classification"

    # Otherwise assume regression
    return "regression"


def build_pipeline(X: pd.DataFrame, problem_type: str) -> Pipeline:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    if not numeric_features and not categorical_features:
        raise ValueError("No usable feature columns found (numeric or categorical).")

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    if problem_type == "classification":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
        )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return clf


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    problem_type: str,
) -> dict:
    y_pred = pipeline.predict(X_test)

    if problem_type == "classification":
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        return {
            "problem_type": problem_type,
            "accuracy": acc,
            "classification_report": report,
        }

    # Regression
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, y_pred)

    return {
        "problem_type": problem_type,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }


def save_model_and_metadata(
    pipeline: Pipeline,
    metrics: dict,
    model_dir: Path,
    model_name: str,
    data_path: Path,
    target_col: str,
    feature_names: List[str],
) -> None:
    from joblib import dump

    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{model_name}.joblib"
    metadata_path = model_dir / f"{model_name}_metadata.json"

    dump(pipeline, model_path)

    metadata = {
        "model_path": str(model_path),
        "data_path": str(data_path),
        "target_col": target_col,
        "feature_names": feature_names,
        "metrics": metrics,
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def train(
    data_path: Path,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    model_dir: Path = Path("ml") / "models",
    model_name: str = "truthlense_model",
) -> None:
    df = load_dataset(data_path)
    X, y = split_features_target(df, target_col)

    feature_names = X.columns.tolist()

    problem_type = infer_problem_type(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if problem_type == "classification" else None,
    )

    pipeline = build_pipeline(X_train, problem_type)
    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(pipeline, X_test, y_test, problem_type)

    save_model_and_metadata(
        pipeline=pipeline,
        metrics=metrics,
        model_dir=model_dir,
        model_name=model_name,
        data_path=data_path,
        target_col=target_col,
        feature_names=feature_names,
    )

    print("Training complete.")
    print(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TruthLense tabular model on a CSV dataset."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the CSV dataset.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        required=True,
        help="Name of the target column in the dataset.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use as test set.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(Path("ml") / "models"),
        help="Directory where the trained model and metadata will be saved.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="truthlense_model",
        help="Base name for the saved model files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data_path).resolve()
    model_dir = Path(args.model_dir).resolve()

    train(
        data_path=data_path,
        target_col=args.target_col,
        test_size=args.test_size,
        model_dir=model_dir,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()

