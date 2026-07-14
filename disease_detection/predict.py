from pathlib import Path

import torch
from PIL import Image

from disease_detection.image_processing import (
    get_prediction_transforms,
)

from disease_detection.model import RiceDiseaseModel


MODEL_PATH = Path("models/disease_model.pth")


class DiseasePredictor:

    def __init__(self):

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Disease model not found: {MODEL_PATH}. "
                "Train the model first."
            )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=self.device,
        )

        self.class_names = checkpoint["class_names"]

        number_of_classes = checkpoint[
            "number_of_classes"
        ]

        self.model = RiceDiseaseModel(
            number_of_classes=number_of_classes,
            use_pretrained=False,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model = self.model.to(self.device)

        self.model.eval()

        self.transform = get_prediction_transforms()


    def predict(self, image):

        # Accept either a file path or PIL image

        if isinstance(image, (str, Path)):

            image = Image.open(image)

        # Ensure RGB format

        image = image.convert("RGB")

        # Apply image preprocessing

        image_tensor = self.transform(image)

        # Add batch dimension:
        # [3, 224, 224] -> [1, 3, 224, 224]

        image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)


        with torch.no_grad():

            outputs = self.model(image_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            confidence, predicted_index = torch.max(
                probabilities,
                dim=1,
            )


        predicted_class = self.class_names[
            predicted_index.item()
        ]

        confidence_score = confidence.item()


        return {
            "disease": predicted_class,
            "confidence": confidence_score,
        }


if __name__ == "__main__":

    predictor = DiseasePredictor()

    print("Disease model loaded successfully.")

    print("Classes:")

    for class_name in predictor.class_names:
        print(f"- {class_name}")