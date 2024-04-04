import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms, datasets

from utils import get_device


def get_train_valid_loader(data_dir,
                           batch_size,
                           augment,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    valid_transform = transforms.Compose([
        transforms.Resize((227, 227)),
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
            transforms.Resize((227, 227)),
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

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, valid_loader


def get_test_loader(data_dir,
                    batch_size,
                    shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transform
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    return data_loader


def compute_accuracy(model, device, data_loader):
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
        model,
        n_epochs=20,
        batch_size=64,
        learning_rate=0.005,
        disable_gpu=False
):
    CIFAR10_LOCATION = "../../data"
    n_classes = 10

    match model:
        case str():
            match model.lower():
                case "alexnet":
                    from model.alexnet import AlexNet
                    model = AlexNet(n_classes=n_classes)
                case "inceptionv2":
                    from model.inception_v2 import InceptionV2
                    model = InceptionV2(n_classes=n_classes)
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
        data_dir=CIFAR10_LOCATION,
        batch_size=batch_size,
        augment=False,
        random_seed=1
    )

    test_loader = get_test_loader(data_dir=CIFAR10_LOCATION,
                                  batch_size=batch_size)

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                weight_decay=0.005,
                                momentum=0.9)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(n_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            if type(outputs) is not torch.Tensor:
                outputs = outputs[0]

            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch + 1}/{n_epochs}], Step [{i + 1}/{total_step}], "
                  f"Loss: {loss.item():.4f}")

        # Validation
        validation_accuracy = compute_accuracy(model, device, valid_loader)
        print(f"Accuracy of the network on the {len(valid_loader.dataset)} validation "
              f"images: {validation_accuracy:.2%}")

    # Test the model
    test_accuracy = compute_accuracy(model, device, test_loader)
    print(f"Accuracy of the network on the {len(test_loader.dataset)} test images: {test_accuracy:.2%}")
