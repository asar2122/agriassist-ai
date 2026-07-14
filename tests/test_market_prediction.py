from market_prediction.predict import MarketPricePredictor


def test_market_prediction():

    predictor = MarketPricePredictor()

    result = predictor.predict(
        month="2010-01-01",
        commodity_name="Barley (Jau)",
        avg_min_price=2000,
        avg_max_price=2500,
        state_name="India",
        district_name="All",
    )

    assert result is not None
    assert isinstance(result, float)
    assert result >= 0