import torchvision
import torchvision.transforms as transforms
import torch


def get_cifar10(data_dir, batch_size, device, num_workers=2):

    print("==> Preparing data..", "CIFAR10")
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        # pin_memory=True if device != "cpu" else False,
        persistent_workers=True if num_workers > 0 else False,
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device != "cpu" else False,
        persistent_workers=True if num_workers > 0 else False,
    )

    # classes = (
    #     "plane",
    #     "car",
    #     "bird",
    #     "cat",
    #     "deer",
    #     "dog",
    #     "frog",
    #     "horse",
    #     "ship",
    #     "truck",
    # )
    return trainloader, testloader


def get_dataset(dataset_name, data_dir, batch_size, device, num_workers=2):
    if dataset_name == "cifar10":
        return get_cifar10(
            data_dir, batch_size, device=device, num_workers=num_workers
        )
    else:
        raise ValueError("Unknown dataset: ", dataset_name)
