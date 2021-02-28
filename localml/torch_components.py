import torch
from torch.nn import Module, Conv2d, Linear, BatchNorm2d, MaxPool2d, ReLU

from torchvision import transforms

FINAL_SIZE = 32*75*75


class DogModel(Module):
    def __init__(self, num_classes=1):
        super(DogModel, self).__init__()
        # define all the layers to be used within the network

        # Output size after convolution filter
        # ((w-f+2P)/s) +1

        # Input shape= (256,3,150,150)
        self.conv1 = Conv2d(
            in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        #Shape= (256,12,150,150)
        self.bn1 = BatchNorm2d(num_features=12)
        #Shape= (256,12,150,150)
        self.relu1 = ReLU()
        #Shape= (256,12,150,150)

        self.pool = MaxPool2d(kernel_size=2)
        # Reduce the image size be factor 2
        #Shape= (256,12,75,75)

        self.conv2 = Conv2d(
            in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        #Shape= (256,20,75,75)
        self.relu2 = ReLU()
        #Shape= (256,20,75,75)

        self.conv3 = Conv2d(
            in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        #Shape= (256,32,75,75)
        self.bn3 = BatchNorm2d(num_features=32)
        #Shape= (256,32,75,75)
        self.relu3 = ReLU()
        #Shape= (256,32,75,75)

        self.fc = Linear(in_features=FINAL_SIZE, out_features=num_classes)

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.pool(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        # Above output will be in matrix form, with shape (256,32,75,75)

        output = output.view(-1, FINAL_SIZE)

        output = self.fc(output)

        return output


def generate_transformer(include_flip=False):
    opts = [transforms.Resize((150, 150))]

    if (include_flip is True):
        opts.append(transforms.RandomHorizontalFlip())

    opts.append(transforms.ToTensor())
    opts.append(transforms.Normalize([0.5, 0.5, 0.5],
                                     [0.5, 0.5, 0.5]))

    transformer = transforms.Compose(opts)

    return transformer
