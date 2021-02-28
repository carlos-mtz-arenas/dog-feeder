import torch
import torchvision.transforms as transforms
from copy import deepcopy

from localml.torch_components import DogModel, generate_transformer


def train(model, optimizer, loss_function, train_loader, train_count):
    model.train()
    train_acc = 0.0
    train_loss = 0.0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        _, prediction = torch.max(outputs.data, 1)

        train_loss += loss.item() * images.size(0)
        train_acc += torch.sum(prediction == labels.data)

    train_acc = train_acc / train_count
    train_loss = train_loss / train_count
    return (train_acc, train_loss)


def validate(model, test_loader, test_count):
    model.eval()
    test_acc = 0.0
    for images, labels in test_loader:
        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        test_acc += int(torch.sum(prediction == labels.data))

    test_acc = test_acc / test_count
    return test_acc


def train_model(epochs, model, optimizer, loss_function, train_count, test_count, train_loader, test_loader):
    best_model = None
    best_acc = 0.0
    for epoch in range(epochs):

        (train_acc, train_loss) = train(model, optimizer,
                                        loss_function, train_loader, train_count)

        test_acc = validate(model, test_loader, test_count)

        print('Epoch=[{}]; Train loss=[{}]; Train acc=[{}]; Test acc=[{}]'.format(
            epoch, train_loss, train_acc, test_acc))

        if (test_acc > best_acc):
            best_acc = test_acc
            best_model = deepcopy(model.state_dict())

    model.load_state_dict(best_model)
