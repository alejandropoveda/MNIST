import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.optim import SGD
from torch import device as TorchDevice


# TODO
# TODO 1.- FullyConnected Neural network/Multilayer perceptron -> No convolutional
# TODO 2.- Activation -> RELU
# TODO 3.- Cost -> nll_loss
# TODO 4.- No dropout <- Para evitar overfitting

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.inputSize = 28 * 28
        self.hiddenLayer1Size = 128
        self.hiddenLayer2Size = 64

        self.outputSize = 10
        # weights
        self.W_HL_1: Tensor = nn.Parameter(torch.randn(self.inputSize, self.hiddenLayer1Size))
        self.W_HL_2: Tensor = nn.Parameter(torch.randn(self.hiddenLayer1Size, self.hiddenLayer2Size))
        self.W_OL: Tensor = nn.Parameter(torch.randn(self.hiddenLayer2Size, self.outputSize))

    def train_imp(self, train_loader: DataLoader, optimizer: SGD, epoch, log_interval):
        # TODO Set module in training mode. See: nn.Module.train
        self.train()
        for batch_idx, (data, target) in enumerate(train_loader):

            # TODO Clears the gradients of all optimized :class:`torch.Tensor`
            optimizer.zero_grad()

            # 0 - Flatten the input
            data = data.view(data.size(0), -1)

            # 1 - Forward
            output: Tensor = self.forward(data)

            # 2 - Loss
            loss: Tensor = self.loss(output, target)

            # 3 - Backward
            loss.backward()

            # 4 - Optimizer
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()
                    )
                )

    def test_imp(self, test_loader: DataLoader, device: TorchDevice):
        self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                # Input flat
                data = data.view(data.size(0), -1)

                output: Tensor = self(data)
                test_loss += self.loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)
            )

        )

    def forward(self, input_data: Tensor) -> Tensor:
        # Hidden Layer 1
        transform_2_hiddenlayer_1: Tensor = input_data.matmul(self.W_HL_1)
        output_hiddenlayer1: Tensor = F.relu(transform_2_hiddenlayer_1)

        # Hidden Layer 2
        transform_2_hiddenlayer_2: Tensor = output_hiddenlayer1.matmul(self.W_HL_2)
        output_hiddenlayer2: Tensor = F.relu(transform_2_hiddenlayer_2)

        # Output Layer
        transform_3_outputlayer: Tensor = output_hiddenlayer2.matmul(self.W_OL)
        output: Tensor = F.log_softmax(transform_3_outputlayer)

        return output

    @staticmethod
    def loss(output: Tensor, target: Tensor, reduction='mean') -> Tensor:
        result: Tensor = F.nll_loss(output, target, reduction=reduction)
        return result

    # def backpropagation(self):
