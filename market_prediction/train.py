from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------
# PATHS
# --------------------------------------------------

DATA_PATH = Path(
    "data/market_data.csv"
)

MODEL_DIRECTORY = Path(
    "models"
)

MODEL_PATH = (
    MODEL_DIRECTORY
    / "market_model.pkl"
)


# --------------------------------------------------
# COLUMN CONFIGURATION
# --------------------------------------------------

TARGET_COLUMN = "avg_modal_price"


NUMERIC_FEATURES = [
    "avg_min_price",
    "avg_max_price",
]


CATEGORICAL_FEATURES = [
    "month",
    "commodity_name",
    "state_name",
    "district_name",
]


FEATURE_COLUMNS = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


# --------------------------------------------------
# LOAD MARKET DATA
# --------------------------------------------------

def load_market_data():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Market dataset not found: "
            f"{DATA_PATH}"
        )


    data = pd.read_csv(
        DATA_PATH
    )


    print(
        "Dataset loaded successfully."
    )


    print(
        f"Total rows before cleaning: "
        f"{len(data)}"
    )


    print()


    print(
        "Dataset Columns:"
    )


    for column in data.columns:

        print(
            f"- {column}"
        )


    # --------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # --------------------------------------------------

    required_columns = (
        FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )


    missing_columns = [

        column

        for column in required_columns

        if column not in data.columns
    ]


    if missing_columns:

        raise ValueError(
            f"Missing required columns: "
            f"{missing_columns}"
        )


    # --------------------------------------------------
    # KEEP ONLY REQUIRED COLUMNS
    # --------------------------------------------------

    data = data[
        required_columns
    ].copy()


    # --------------------------------------------------
    # CONVERT NUMERIC COLUMNS
    # --------------------------------------------------

    numeric_columns = [
        "avg_min_price",
        "avg_max_price",
        "avg_modal_price",
    ]


    for column in numeric_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )


    # --------------------------------------------------
    # REMOVE MISSING VALUES
    # --------------------------------------------------

    data = data.dropna(
        subset=required_columns
    )


    # --------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------

    data = data.drop_duplicates()


    print()

    print(
        f"Total rows after cleaning: "
        f"{len(data)}"
    )


    if len(data) == 0:

        raise ValueError(
            "No valid rows remain after "
            "cleaning the market dataset."
        )


    return data


# --------------------------------------------------
# CREATE PREPROCESSOR
# --------------------------------------------------

def create_preprocessor():

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


    return preprocessor


# --------------------------------------------------
# CREATE MODEL
# --------------------------------------------------

def create_model():

    model = RandomForestRegressor(

        n_estimators=200,

        random_state=42,

        n_jobs=-1,
    )


    return model


# --------------------------------------------------
# TRAIN MARKET MODEL
# --------------------------------------------------

def train_market_model():

    print("=" * 60)

    print(
        "MARKET PRICE PREDICTION MODEL TRAINING"
    )

    print("=" * 60)


    # --------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------

    data = load_market_data()


    # --------------------------------------------------
    # CREATE INPUT AND TARGET
    # --------------------------------------------------

    X = data[
        FEATURE_COLUMNS
    ]


    y = data[
        TARGET_COLUMN
    ]


    # --------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,
    )


    print()

    print(
        f"Training samples: "
        f"{len(X_train)}"
    )


    print(
        f"Testing samples: "
        f"{len(X_test)}"
    )


    # --------------------------------------------------
    # CREATE PREPROCESSOR
    # --------------------------------------------------

    preprocessor = (
        create_preprocessor()
    )


    # --------------------------------------------------
    # CREATE RANDOM FOREST MODEL
    # --------------------------------------------------

    model = create_model()


    # --------------------------------------------------
    # CREATE MACHINE LEARNING PIPELINE
    # --------------------------------------------------

    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor,
            ),

            (
                "regressor",
                model,
            ),

        ]
    )


    # --------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------

    print()

    print(
        "Training Random Forest "
        "Regression model..."
    )


    pipeline.fit(
        X_train,
        y_train,
    )


    # --------------------------------------------------
    # MAKE TEST PREDICTIONS
    # --------------------------------------------------

    predictions = pipeline.predict(
        X_test
    )


    # --------------------------------------------------
    # MODEL EVALUATION
    # --------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions,
    )


    mse = mean_squared_error(
        y_test,
        predictions,
    )


    rmse = mse ** 0.5


    r2 = r2_score(
        y_test,
        predictions,
    )


    print()

    print("=" * 60)

    print(
        "MODEL EVALUATION"
    )

    print("=" * 60)


    print(
        f"Mean Absolute Error (MAE): "
        f"{mae:.4f}"
    )


    print(
        f"Mean Squared Error (MSE): "
        f"{mse:.4f}"
    )


    print(
        f"Root Mean Squared Error (RMSE): "
        f"{rmse:.4f}"
    )


    print(
        f"R² Score: "
        f"{r2:.4f}"
    )


    # --------------------------------------------------
    # CREATE MODELS DIRECTORY
    # --------------------------------------------------

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------
    # SAVE COMPLETE MODEL PIPELINE
    # --------------------------------------------------

    joblib.dump(
        pipeline,
        MODEL_PATH,
    )


    print()

    print("=" * 60)

    print(
        "TRAINING COMPLETE"
    )

    print("=" * 60)


    print(
        f"Model saved successfully: "
        f"{MODEL_PATH}"
    )


    return pipeline


# --------------------------------------------------
# RUN TRAINING
# --------------------------------------------------

if __name__ == "__main__":

    train_market_model()