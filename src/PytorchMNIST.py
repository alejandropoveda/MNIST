from torch.optim import SGD
from src.deeplearning.Utils import *
from deeplearning.PytorchNet import Net
import argparse
from argparse import Namespace
from torch.optim.lr_scheduler import StepLR
from torch import device as TorchDevice

# Class Structure:
#
# 1.- Preprocessing. Transformer declaration.
# 2.- Data loading. Data loading and batch preprocessing.
# 3.-
# -----------------------------------------------------------------------


def visualization(test_loader: DataLoader):
    output_file_name = "numbers_sample.png"
    number_of_images = 6
    output_dir = './output/plots'
    fig = visualize(loader=test_loader, number_of_images=number_of_images, output_dir=output_dir, output_file_name=output_file_name)
    print(fig)


def runner(args: Namespace, with_visualization=False) -> int:

    result = 0

    # 0.- Set up
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device: TorchDevice = torch.device("cuda" if use_cuda else "cpu")
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {} TODO No cuda yet

    torch.manual_seed(args.seed)

    # 1.- Create transform
    transform = gen_transform()

    # 2.- Dataset load
    batch_size_train = args.batch_size
    batch_size_test = args.test_batch_size
    path = './data/'

    train_loader: DataLoader = load_mnist_dataset(
        transform=transform,
        path=path,
        train=True,
        batch_size=batch_size_train
    )
    test_loader: DataLoader = load_mnist_dataset(
        transform=transform,
        path=path,
        train=False,
        batch_size=batch_size_test
    )

    # 3.- Visualization
    if with_visualization:
        visualization(test_loader)

    # 4.- Network initialization
    network = Net().to(device)
    momentum = 0.5
    optimizer: SGD = SGD(network.parameters(), lr=args.lr, momentum=momentum)
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # 5 - Computation
    for epoch in range(1, args.epochs + 1):
        network.train_imp(train_loader, optimizer, epoch, args.log_interval)
        network.test_imp(test_loader, device)
        scheduler.step()

    if args.save_model:
        torch.save(network.state_dict(), "mnist_nn.pt")

    return result


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')

    # This package adds support for CUDA tensor types, that implement the same function as CPU tensors,
    # but they utilize GPUs for computation.
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    args: Namespace = parser.parse_args()

    result = runner(args)
    print("[apc] EXIT: {}".format(result))


if __name__ == "__main__":
    main()
