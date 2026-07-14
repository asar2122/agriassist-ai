from pathlib import Path

from disease_detection.predict import DiseasePredictor


TEST_IMAGE_PATH = Path(
    "data/disease_dataset/Leaf Blast/"
    "aug_0_3.jpg"
)


def test_disease_prediction():

    assert TEST_IMAGE_PATH.exists(), (
        f"Test image not found: {TEST_IMAGE_PATH}"
    )

    predictor = DiseasePredictor()

    result = predictor.predict(
        TEST_IMAGE_PATH
    )

    assert result is not None

    assert isinstance(
        result,
        dict,
    )

    assert "disease" in result

    assert "confidence" in result

    assert isinstance(
        result["disease"],
        str,
    )

    assert isinstance(
        result["confidence"],
        float,
    )

    assert 0.0 <= result["confidence"] <= 1.0