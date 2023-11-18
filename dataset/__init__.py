from dataset.CIFAR import create_cifar10_dataset
from dataset.MNIST import create_mnist_dataset
from dataset.ImageData import create_custom_dataset


def create_dataset(name: str, **kwargs):
    if name.lower() == "cifar10":
        return create_cifar10_dataset(**kwargs)
    elif name.lower() == "mnist":
        return create_mnist_dataset(**kwargs)
    elif name.lower() == "custom":
        return create_custom_dataset(**kwargs)
    else:
        raise NotImplementedError(f"Dataset {name} not implemented")
