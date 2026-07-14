from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/crop_data.csv")
MODEL_DIRECTORY = Path("models")
MODEL_PATH = MODEL_DIRECTORY / "crop_model.pkl"


FEATURE_COLUMNS = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]


TARGET_COLUMN = "label"


def load_crop_data():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Crop dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    print(f"Dataset loaded successfully.")
    print(f"Total rows: {len(data)}")

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    return data


def train_crop_model():

    print("=" * 60)
    print("CROP RECOMMENDATION MODEL TRAINING")
    print("=" * 60)

    data = load_crop_data()

    X = data[FEATURE_COLUMNS]

    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print()
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    )

    print()
    print("Training Random Forest model...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print()
    print(f"Model Accuracy: {accuracy:.4f}")

    print()
    print("Classification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print()
    print(
        f"Model saved successfully: {MODEL_PATH}"
    )

    return model


if __name__ == "__main__":

    train_crop_model()