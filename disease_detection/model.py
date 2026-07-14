import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)


class RiceDiseaseModel(nn.Module):

    def __init__(
        self,
        number_of_classes,
        use_pretrained=True,
    ):

        super().__init__()


        if use_pretrained:

            weights = (
                ResNet18_Weights.DEFAULT
            )

        else:

            weights = None


        self.model = resnet18(
            weights=weights
        )


        input_features = (
            self.model.fc.in_features
        )


        self.model.fc = nn.Linear(
            input_features,
            number_of_classes,
        )


    def forward(self, images):

        return self.model(images)