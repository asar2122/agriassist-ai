from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(
    "models/fertilizer_model.pkl"
)


FEATURE_COLUMNS = [
    "temperature",
    "humidity",
    "moisture",
    "nitrogen",
    "potassium",
    "phosphorus",
    "soil_type",
    "crop_type",
]


class FertilizerPredictor:

    def __init__(self):

        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                f"Fertilizer model not found: "
                f"{MODEL_PATH}. "
                "Train the model first."
            )

        self.model = joblib.load(
            MODEL_PATH
        )


    def predict(
        self,
        temperature,
        humidity,
        moisture,
        nitrogen,
        potassium,
        phosphorus,
        soil_type,
        crop_type,
    ):

        input_data = pd.DataFrame(
            [
                {
                    "temperature": temperature,
                    "humidity": humidity,
                    "moisture": moisture,
                    "nitrogen": nitrogen,
                    "potassium": potassium,
                    "phosphorus": phosphorus,
                    "soil_type": soil_type,
                    "crop_type": crop_type,
                }
            ],
            columns=FEATURE_COLUMNS,
        )


        prediction = self.model.predict(
            input_data
        )


        return prediction[0]


if __name__ == "__main__":

    predictor = FertilizerPredictor()


    result = predictor.predict(
        temperature=26,
        humidity=52,
        moisture=38,
        nitrogen=37,
        potassium=0,
        phosphorus=0,
        soil_type="Sandy",
        crop_type="Maize",
    )


    print(
        f"Recommended Fertilizer: "
        f"{result}"
    )