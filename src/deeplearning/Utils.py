import torch
from torch.utils.data.dataloader import DataLoader
import torchvision
from torchvision.transforms import Compose


# 1.- Preprocessing
#     Transform to apply for every input image. We want the same properties and dimensions.
#
#       1.1.- transforms.ToTensor() — converts the image into numbers, that are understandable by the system.
#       It separates the image into three color channels (separate images): red, green & blue. Then it converts
#       the pixels of each image to the brightness of their color between 0 and 255. These values are then scaled
#       down to a range between 0 and 1. The image is now a Torch Tensor.
#
#       1.2.- transforms.Normalize() — normalizes the tensor with a mean and standard deviation which goes as the two
#       parameters respectively.
#
#       TorchVision offers a lot of handy transformations, such as cropping or normalization.
#
def gen_transform() -> Compose:
    torchvision_tensor = torchvision.transforms.ToTensor()

    torchvision_normalize_global_mean = 0.1307
    torchvision_normalize_standard_deviation = 0.3081
    torchvision_normalize = torchvision.transforms.Normalize(
        (torchvision_normalize_global_mean,),
        (torchvision_normalize_standard_deviation,)
    )

    torchvision_transform = torchvision.transforms.Compose([
        torchvision_tensor,
        torchvision_normalize
    ])

    return torchvision_transform


# 2.- MNIST dataset loader
#     Transform to apply for every input image. We want the same properties and dimensions.
#
#       2.1.- batch_size - TODO Explicacion
#
#       2.2.- shuffle - TODO Explicacion
#
def load_mnist_dataset(transform: Compose, path: str, train: bool, batch_size: int) -> DataLoader:

    dataset_mnsit = torchvision.datasets.MNIST(root=path,
                                               train=train,
                                               transform=transform, download=True
                                               )
    loader: DataLoader = DataLoader(
        dataset=dataset_mnsit,
        batch_size=batch_size,
        shuffle=True
    )
    return loader


# 3.- Visualization
#
def visualize(loader, number_of_images, output_dir, output_file_name):
    examples = enumerate(loader)
    batch_idx, (example_data, example_targets) = next(examples)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    for i in range(number_of_images):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])

    plt.savefig(output_dir + "/" + output_file_name)

    return fig

