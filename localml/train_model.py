#!/bin/python3
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch import device, save

from glob import glob

from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader

from localml.train_utils import train_model
from localml.torch_components import generate_transformer, DogModel


base_path = './images'
train_imgs = base_path + '/train'
test_imgs = base_path + '/test'

# training params
num_epochs = 10
device = device("cpu")
# import our own model already configured
model = DogModel(num_classes=2).to(device)
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
loss_function = CrossEntropyLoss()


def count_images(img_path):
    return len(glob(img_path + '/**/*.jpg'))


train_count = count_images(train_imgs)
test_count = count_images(test_imgs)

print('Image count for training [{}]'.format(train_count))
print('Image count for testing [{}]'.format(test_count))

# image formatter
train_transformer = generate_transformer(include_flip=True)
test_transformer = generate_transformer()

# data sets
train_dataset = ImageFolder(train_imgs, transform=train_transformer)
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_dataset = ImageFolder(test_imgs, transform=test_transformer)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=True
)

# assuming that both training and testing classes are the same
classes = train_dataset.classes
print('Classes [{}]'.format(classes))

# trianing the model will get the model instance
# updated by the reference
train_model(num_epochs,
            model,
            optimizer,
            loss_function,
            train_count,
            test_count,
            train_loader,
            test_loader)

print('Model {}', model)

checkpoint = {
    "classes": classes,
    "state": model.state_dict()
}

save(checkpoint, './model_checkpoint.pt')
