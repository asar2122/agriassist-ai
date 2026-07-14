from pathlib import Path
import copy

import torch
import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

from disease_detection.image_processing import (
    get_training_transforms,
    get_validation_transforms,
)

from disease_detection.model import RiceDiseaseModel


# --------------------------------------------------
# PATHS
# --------------------------------------------------

DATASET_PATH = Path(
    "data/disease_dataset"
)

MODEL_DIRECTORY = Path(
    "models"
)

MODEL_PATH = MODEL_DIRECTORY / "disease_model.pth"


# --------------------------------------------------
# TRAINING CONFIGURATION
# --------------------------------------------------

BATCH_SIZE = 16

NUMBER_OF_EPOCHS = 5

LEARNING_RATE = 0.001

VALIDATION_RATIO = 0.20

RANDOM_SEED = 42


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# --------------------------------------------------
# CREATE DATASETS
# --------------------------------------------------

def create_datasets():

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Disease dataset not found: "
            f"{DATASET_PATH}"
        )


    # Dataset used to discover images and classes

    complete_dataset = ImageFolder(
        root=DATASET_PATH
    )


    total_images = len(
        complete_dataset
    )


    number_of_classes = len(
        complete_dataset.classes
    )


    validation_size = int(
        total_images
        * VALIDATION_RATIO
    )


    training_size = (
        total_images
        - validation_size
    )


    generator = torch.Generator().manual_seed(
        RANDOM_SEED
    )


    training_indices, validation_indices = random_split(
        range(total_images),
        [
            training_size,
            validation_size,
        ],
        generator=generator,
    )


    # Training dataset with augmentation

    training_dataset = ImageFolder(
        root=DATASET_PATH,
        transform=get_training_transforms(),
    )


    # Validation dataset without augmentation

    validation_dataset = ImageFolder(
        root=DATASET_PATH,
        transform=get_validation_transforms(),
    )


    training_dataset = torch.utils.data.Subset(
        training_dataset,
        training_indices.indices,
    )


    validation_dataset = torch.utils.data.Subset(
        validation_dataset,
        validation_indices.indices,
    )


    return (
        training_dataset,
        validation_dataset,
        complete_dataset.classes,
    )


# --------------------------------------------------
# CREATE DATA LOADERS
# --------------------------------------------------

def create_data_loaders(
    training_dataset,
    validation_dataset,
):

    training_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )


    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )


    return (
        training_loader,
        validation_loader,
    )


# --------------------------------------------------
# TRAIN ONE EPOCH
# --------------------------------------------------

def train_one_epoch(
    model,
    data_loader,
    loss_function,
    optimizer,
):

    model.train()


    running_loss = 0.0

    correct_predictions = 0

    total_predictions = 0


    for images, labels in data_loader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        optimizer.zero_grad()


        outputs = model(
            images
        )


        loss = loss_function(
            outputs,
            labels,
        )


        loss.backward()


        optimizer.step()


        running_loss += (
            loss.item()
            * images.size(0)
        )


        _, predictions = torch.max(
            outputs,
            1,
        )


        correct_predictions += (
            predictions == labels
        ).sum().item()


        total_predictions += (
            labels.size(0)
        )


    epoch_loss = (
        running_loss
        / total_predictions
    )


    epoch_accuracy = (
        correct_predictions
        / total_predictions
    )


    return (
        epoch_loss,
        epoch_accuracy,
    )


# --------------------------------------------------
# VALIDATE MODEL
# --------------------------------------------------

def validate_model(
    model,
    data_loader,
    loss_function,
):

    model.eval()


    running_loss = 0.0

    correct_predictions = 0

    total_predictions = 0


    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )


            outputs = model(
                images
            )


            loss = loss_function(
                outputs,
                labels,
            )


            running_loss += (
                loss.item()
                * images.size(0)
            )


            _, predictions = torch.max(
                outputs,
                1,
            )


            correct_predictions += (
                predictions == labels
            ).sum().item()


            total_predictions += (
                labels.size(0)
            )


    validation_loss = (
        running_loss
        / total_predictions
    )


    validation_accuracy = (
        correct_predictions
        / total_predictions
    )


    return (
        validation_loss,
        validation_accuracy,
    )


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

def train_disease_model():

    print("=" * 60)

    print(
        "RICE DISEASE DETECTION MODEL TRAINING"
    )

    print("=" * 60)


    print(
        f"Device: {DEVICE}"
    )


    print()

    print(
        "Loading disease dataset..."
    )


    (
        training_dataset,
        validation_dataset,
        class_names,
    ) = create_datasets()


    print(
        f"Training images: "
        f"{len(training_dataset)}"
    )


    print(
        f"Validation images: "
        f"{len(validation_dataset)}"
    )


    print(
        f"Number of classes: "
        f"{len(class_names)}"
    )


    print()

    print(
        "Classes:"
    )


    for number, class_name in enumerate(
        class_names,
        start=1,
    ):

        print(
            f"{number}. {class_name}"
        )


    (
        training_loader,
        validation_loader,
    ) = create_data_loaders(
        training_dataset,
        validation_dataset,
    )


    print()

    print(
        "Creating ResNet18 model..."
    )


    model = RiceDiseaseModel(
        number_of_classes=len(
            class_names
        ),
        use_pretrained=True,
    )


    model = model.to(
        DEVICE
    )


    loss_function = (
        nn.CrossEntropyLoss()
    )


    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )


    best_validation_accuracy = 0.0

    best_model_weights = None


    print()

    print(
        "Starting training..."
    )

    print()


    for epoch in range(
        NUMBER_OF_EPOCHS
    ):

        print(
            f"Epoch "
            f"{epoch + 1}/"
            f"{NUMBER_OF_EPOCHS}"
        )


        training_loss, training_accuracy = (
            train_one_epoch(
                model=model,
                data_loader=training_loader,
                loss_function=loss_function,
                optimizer=optimizer,
            )
        )


        (
            validation_loss,
            validation_accuracy,
        ) = validate_model(
            model=model,
            data_loader=validation_loader,
            loss_function=loss_function,
        )


        print(
            f"Training Loss: "
            f"{training_loss:.4f}"
        )


        print(
            f"Training Accuracy: "
            f"{training_accuracy:.4f}"
        )


        print(
            f"Validation Loss: "
            f"{validation_loss:.4f}"
        )


        print(
            f"Validation Accuracy: "
            f"{validation_accuracy:.4f}"
        )


        if (
            validation_accuracy
            > best_validation_accuracy
        ):

            best_validation_accuracy = (
                validation_accuracy
            )


            best_model_weights = copy.deepcopy(
                model.state_dict()
            )


            print(
                "Best model updated."
            )


        print("-" * 60)


    # --------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------

    if best_model_weights is None:

        raise RuntimeError(
            "Training completed without "
            "creating model weights."
        )


    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


    checkpoint = {
        "model_state_dict": best_model_weights,
        "class_names": class_names,
        "number_of_classes": len(
            class_names
        ),
        "best_validation_accuracy": (
            best_validation_accuracy
        ),
    }


    torch.save(
        checkpoint,
        MODEL_PATH,
    )


    print()

    print("=" * 60)

    print(
        "TRAINING COMPLETE"
    )

    print("=" * 60)


    print(
        f"Best Validation Accuracy: "
        f"{best_validation_accuracy:.4f}"
    )


    print(
        f"Model saved successfully: "
        f"{MODEL_PATH}"
    )


# --------------------------------------------------
# RUN TRAINING
# --------------------------------------------------

if __name__ == "__main__":

    train_disease_model()