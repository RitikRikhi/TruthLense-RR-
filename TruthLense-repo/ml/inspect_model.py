import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from joblib import load


def load_metadata(metadata_path: Path) -> Dict[str, Any]:
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found at: {metadata_path}. "
            f"Make sure you have trained the model first."
        )

    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sample_and_query_model(
    metadata: Dict[str, Any],
    n_samples: int = 3,
    random_state: int = 42,
) -> None:
    model_path = Path(metadata["model_path"])
    data_path = Path(metadata["data_path"])
    target_col = metadata["target_col"]
    metrics = metadata.get("metrics", {})
    problem_type = metrics.get("problem_type", "classification")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model file not found at: {model_path}. "
            f"Train the model before running this script."
        )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at: {data_path}. "
            f"The path is stored in the metadata; did you move the dataset?"
        )

    print(f"Loading model from: {model_path}")
    pipeline = load(model_path)

    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    if target_col not in df.columns:
        available = ", ".join(df.columns.astype(str))
        raise KeyError(
            f"Target column '{target_col}' not found in dataset. "
            f"Available columns: {available}"
        )

    df = df.dropna(subset=[target_col])
    if df.empty:
        raise ValueError(
            "No rows remaining after dropping missing target values. "
            "Cannot sample examples for inspection."
        )

    n_samples = min(n_samples, len(df))
    samples = df.sample(n=n_samples, random_state=random_state)

    print(f"\n=== Sampling {n_samples} random example(s) from the dataset ===\n")

    for idx, (_, row) in enumerate(samples.iterrows(), start=1):
        x = row.drop(labels=[target_col])
        y_true = row[target_col]

        x_df = x.to_frame().T
        y_pred = pipeline.predict(x_df)[0]

        print(f"Example {idx}:")
        print("Input features:")
        for col, val in x.items():
            print(f"  - {col}: {val}")

        print(f"\nTrue target ({target_col}): {y_true}")
        print(f"Model prediction: {y_pred}")

        if problem_type == "classification" and hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(x_df)[0]
            model = getattr(pipeline, "named_steps", {}).get("model")
            classes = getattr(model, "classes_", None) if model is not None else None

            print("Prediction probabilities:")
            if classes is not None:
                for cls, p in zip(classes, proba):
                    print(f"  - P({cls}) = {p:.4f}")
            else:
                for i, p in enumerate(proba):
                    print(f"  - class_{i}: {p:.4f}")

        print("\n" + "-" * 60 + "\n")

    print("Finished querying random examples.")
    print("\nStored evaluation metrics (from training):")
    print(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    default_metadata = Path("ml") / "models" / "truthlense_model_metadata.json"

    parser = argparse.ArgumentParser(
        description=(
            "Inspect the trained TruthLense model by querying it on "
            "random examples from the original dataset."
        )
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=str(default_metadata),
        help="Path to the model metadata JSON file.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=3,
        help="Number of random examples to query.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata_path).resolve()
    metadata = load_metadata(metadata_path)
    sample_and_query_model(
        metadata=metadata,
        n_samples=args.n_samples,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()

