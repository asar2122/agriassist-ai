from pathlib import Path

import joblib
import pandas as pd


# --------------------------------------------------
# MODEL PATH
# --------------------------------------------------

MODEL_PATH = Path(
    "models/market_model.pkl"
)


# --------------------------------------------------
# FEATURE COLUMNS
# --------------------------------------------------

FEATURE_COLUMNS = [
    "avg_min_price",
    "avg_max_price",
    "month",
    "commodity_name",
    "state_name",
    "district_name",
]


# --------------------------------------------------
# MARKET PRICE PREDICTOR
# --------------------------------------------------

class MarketPricePredictor:

    def __init__(self):

        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                f"Market model not found: "
                f"{MODEL_PATH}. "
                "Train the model first."
            )

        self.model = joblib.load(
            MODEL_PATH
        )


    def predict(
        self,
        month,
        commodity_name,
        avg_min_price,
        avg_max_price,
        state_name,
        district_name,
    ):

        input_data = pd.DataFrame(
            [
                {
                    "avg_min_price": avg_min_price,
                    "avg_max_price": avg_max_price,
                    "month": month,
                    "commodity_name": commodity_name,
                    "state_name": state_name,
                    "district_name": district_name,
                }
            ],
            columns=FEATURE_COLUMNS,
        )


        prediction = self.model.predict(
            input_data
        )


        predicted_price = float(
            prediction[0]
        )


        return predicted_price


# --------------------------------------------------
# TEST PREDICTION
# --------------------------------------------------

if __name__ == "__main__":

    predictor = MarketPricePredictor()


    result = predictor.predict(
        month="January",
        commodity_name="Rice",
        avg_min_price=2000,
        avg_max_price=2500,
        state_name="Tamil Nadu",
        district_name="Chennai",
    )


    print("=" * 60)

    print(
        "MARKET PRICE PREDICTION"
    )

    print("=" * 60)


    print(
        f"Predicted Modal Price: "
        f"{result:.2f}"
    )