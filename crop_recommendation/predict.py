from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/crop_model.pkl")

FEATURE_COLUMNS = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]


class CropPredictor:

    def __init__(self):

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Crop model not found: {MODEL_PATH}. "
                "Train the model first."
            )

        self.model = joblib.load(MODEL_PATH)


    def predict(
        self,
        nitrogen,
        phosphorus,
        potassium,
        temperature,
        humidity,
        ph,
        rainfall,
    ):

        input_data = pd.DataFrame(
            [
                {
                    "N": nitrogen,
                    "P": phosphorus,
                    "K": potassium,
                    "temperature": temperature,
                    "humidity": humidity,
                    "ph": ph,
                    "rainfall": rainfall,
                }
            ],
            columns=FEATURE_COLUMNS,
        )

        prediction = self.model.predict(input_data)

        return prediction[0]


if __name__ == "__main__":

    predictor = CropPredictor()

    result = predictor.predict(
        nitrogen=90,
        phosphorus=42,
        potassium=43,
        temperature=20.8,
        humidity=82.0,
        ph=6.5,
        rainfall=202.9,
    )

    print(f"Recommended Crop: {result}")