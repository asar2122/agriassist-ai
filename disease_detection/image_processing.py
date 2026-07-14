from torchvision import transforms


IMAGE_SIZE = 224


def get_training_transforms():

    return transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),

            transforms.RandomHorizontalFlip(),

            transforms.RandomRotation(
                degrees=15
            ),

            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )


def get_validation_transforms():

    return transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )


def get_prediction_transforms():

    return get_validation_transforms()