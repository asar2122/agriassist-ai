from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path("data/fertilizer_data.csv")

MODEL_DIRECTORY = Path("models")

MODEL_PATH = MODEL_DIRECTORY / "fertilizer_model.pkl"


NUMERIC_FEATURES = [
    "temperature",
    "humidity",
    "moisture",
    "nitrogen",
    "potassium",
    "phosphorus",
]


CATEGORICAL_FEATURES = [
    "soil_type",
    "crop_type",
]


TARGET_COLUMN = "fertilizer"


FEATURE_COLUMNS = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


def load_fertilizer_data():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Fertilizer dataset not found: "
            f"{DATA_PATH}"
        )

    data = pd.read_csv(
        DATA_PATH
    )

    # Clean column names
    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
    )

    # Rename dataset columns
    data = data.rename(
        columns={
            "temparature": "temperature",
            "humidity": "humidity",
            "moisture": "moisture",
            "soil type": "soil_type",
            "crop type": "crop_type",
            "nitrogen": "nitrogen",
            "potassium": "potassium",
            "phosphorous": "phosphorus",
            "fertilizer name": "fertilizer",
        }
    )

    print(
        "Dataset loaded successfully."
    )

    print(
        f"Total rows: {len(data)}"
    )

    return data


def train_fertilizer_model():

    print("=" * 60)

    print(
        "FERTILIZER RECOMMENDATION MODEL TRAINING"
    )

    print("=" * 60)


    data = load_fertilizer_data()


    X = data[FEATURE_COLUMNS]

    y = data[TARGET_COLUMN]


    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=None,
        )
    )


    print()

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )


    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    )


    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                model,
            ),
        ]
    )


    print()

    print(
        "Training Random Forest model..."
    )


    pipeline.fit(
        X_train,
        y_train,
    )


    predictions = pipeline.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        predictions,
    )


    print()

    print(
        f"Model Accuracy: {accuracy:.4f}"
    )


    print()

    print(
        "Classification Report:"
    )


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
        pipeline,
        MODEL_PATH,
    )


    print()

    print(
        f"Model saved successfully: "
        f"{MODEL_PATH}"
    )


    return pipeline


if __name__ == "__main__":

    train_fertilizer_model()