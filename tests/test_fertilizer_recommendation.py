from fertilizer_recommendation.predict import FertilizerPredictor


def test_fertilizer_prediction():

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

    assert result is not None
    assert isinstance(result, str)