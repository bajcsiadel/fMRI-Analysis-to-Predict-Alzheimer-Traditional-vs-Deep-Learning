import dataclasses as dc
from pathlib import Path

import hydra
import numpy as np
import omegaconf
import torch

from icecream import ic
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms, datasets

from utils.config.output import OutputConfig
from utils.config.resolvers import resolve_results_location
from utils.environment import get_env
from utils.helpers import get_device
from utils.logger import TrainLogger


@dc.dataclass
class Config:
    model: str
    n_epochs: int
    batch_size: int
    learning_rate: float
    disable_gpu: bool
    pretrained: bool
    seed: int
    out_dirs: OutputConfig
    cifar_location: Path


def get_train_valid_loader(
        data_dir: str | Path,
        batch_size: int | None,
        augment: bool,
        random_seed: int,
        valid_size: float = 0.1,
        shuffle: bool = True,
        image_shape: tuple[int, int] = (32, 32),
) -> (DataLoader, DataLoader):
    """
    Get train and validation dataloaders

    :param data_dir: location where the data is stored
    :param batch_size: if `None` the entire dataset is loaded
    :param augment: masks if train data is augmented
    :param random_seed:
    :param valid_size: size of the validation set. Defaults to `0.1`
    :param shuffle: Defaults to `True`
    :param image_shape: shape of the image. Defaults to `(32, 32)`
    :return:
    """
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    valid_transform = transforms.Compose([
        transforms.Resize(image_shape),
        transforms.ToTensor(),
        normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(image_shape),
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    batch_size = batch_size or len(indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)

    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, valid_loader


def get_test_loader(
        data_dir: str | Path,
        batch_size: int | None,
        shuffle: bool = True,
        image_shape: tuple[int, int] = (32, 32),
) -> DataLoader:
    """
    Get test dataloader

    :param data_dir: location where the data is stored
    :param batch_size:
    :param shuffle: Defaults to `True`
    :param image_shape: shape of the image. Defaults to `(32, 32)`
    :return:
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transform
    transform = transforms.Compose([
        transforms.Resize(image_shape),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=transform,
    )

    batch_size = batch_size or len(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    return data_loader


def compute_accuracy(model: torch.nn.Module, device: torch.device, data_loader: DataLoader) -> float:
    """
    Compute the accuracy of the model on the data loader.

    :param model:
    :param device:
    :param data_loader:
    :return:
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            if type(outputs) is not torch.Tensor:
                outputs = outputs[0]

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
    return correct / total


def train_model_CIFAR10(
    model: str,
    cifar_location: str,
    logger: TrainLogger,
    n_epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 0.005,
    disable_gpu: bool = False,
    pretrained: bool = False,
    seed: int = 2024,
):
    n_classes = 10
    image_shape = (227, 227)

    torch.manual_seed(seed)
    np.random.seed(seed)

    match model:
        case str():
            match model.lower():
                case "alexnet":
                    from model.neural_net.alexnet import AlexNet
                    model = AlexNet(n_classes=n_classes, image_shape=image_shape)
                case "inceptionv2":
                    from model.neural_net.inception_v2 import InceptionV2
                    model = InceptionV2(n_classes=n_classes)
                case "torch-alexnet":
                    from torchvision.models.alexnet import alexnet
                    model = alexnet(num_classes=n_classes, pretrained=pretrained)
                case _:
                    raise ValueError(f"Model {model} not supported")
        case torch.nn.Module():
            ...
        case _:
            raise TypeError(f"Model of type {type(model)} not supported")

    device = get_device(disable_gpu)
    model = model.to(device)

    # CIFAR10 dataset
    train_loader, valid_loader = get_train_valid_loader(
        data_dir=cifar_location,
        batch_size=batch_size,
        augment=False,
        random_seed=seed,
        image_shape=image_shape
    )

    test_loader = get_test_loader(
        data_dir=cifar_location,
        batch_size=batch_size,
        image_shape=image_shape
    )

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.005,
        momentum=0.9
    )

    # Train the model
    total_step = len(train_loader)
    for epoch in range(n_epochs):
        total_loss = torch.tensor(0.0)
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            if type(outputs) is not torch.Tensor:
                outputs = outputs[0]

            loss = criterion(outputs, labels)
            total_loss += loss.cpu()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.info(f"Epoch [{epoch + 1}/{n_epochs}], Step [{i + 1}/{total_step}], "
                        f"Loss: {loss.item():.4f}")
            logger.tensorboard.add_scalar(
                "Loss/train/steps",
                loss.item(),
                epoch * total_step + i
            )

        logger.tensorboard.add_scalar(
            "Loss/train/epochs",
            total_loss.item() / total_step,
            epoch
        )

        # Validation
        validation_accuracy = compute_accuracy(model, device, valid_loader)
        logger.info(
            f"Accuracy of the network on the {len(valid_loader.dataset)} "
            f"validation images: {validation_accuracy:.2%}"
        )
        logger.tensorboard.add_scalar("Accuracy/validation", validation_accuracy, epoch)

    # Test the model
    test_accuracy = compute_accuracy(model, device, test_loader)
    logger.info(
        f"Accuracy of the network on the {len(test_loader.dataset)} "
        f"test images: {test_accuracy:.2%}"
    )


@hydra.main(
    version_base=None,
    config_path=get_env("CONFIGURATIONS_LOCATION"),
    config_name="scripts_train_network_CIFAR"
)
def main(cfg: Config):
    logger = TrainLogger(__name__, cfg.out_dirs)
    try:
        cfg = omegaconf.OmegaConf.to_object(cfg)

        train_model_CIFAR10(
            model=cfg.model,
            cifar_location=cfg.cifar_location,
            logger=logger,
            n_epochs=cfg.n_epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            disable_gpu=cfg.disable_gpu,
            pretrained=cfg.pretrained,
            seed=cfg.seed,
        )
    except Exception as e:
        logger.exception(e)
    # logger object must be deleted to close the tensorboard writer
    del logger


if __name__ == "__main__":
    resolve_results_location()
    config_store = OutputConfig.add_type_validation()
    config_store.store(name="_script_config_validation", node=Config)
    main()
