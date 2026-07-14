from crop_recommendation.predict import CropPredictor


def test_crop_prediction():

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

    assert result is not None
    assert isinstance(result, str)