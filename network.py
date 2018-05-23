import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class MLPNet(nn.Module):
    def __init__(self, zero_init=True):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 80)
        self.fc2 = nn.Linear(80, 80)
        self.fc3 = nn.Linear(80, 80)
        self.fc4 = nn.Linear(80, 10)
        if zero_init:
            self.set_zero()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.tanh(self.fc1(x)) 
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)
        return x

    def set_zero(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            layer.weight.data.zero_()
            layer.bias.data.zero_()


def train_model(model, optimizer, criterion, epochs, train_loader, test_loader, print_progress=True):
    """ trains a model using a given optimizer and criterion. 
    """
    train_loss, test_loss = [], []
    train_acc, test_acc = [], []
    train_loss, test_error = [], []
    for epoch in range(1, epochs+1):
        sum_loss = 0
        n_correct = 0
        n_samples = 0
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            x, target = Variable(x), Variable(target)
            out = model(x)

            _, pred_label = torch.max(out.data, 1)
            n_samples += x.data.size()[0]
            n_correct += (pred_label == target.data).sum()

            loss = criterion(out, target)
            sum_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss.append(sum_loss / n_samples)
        train_acc.append(n_correct / n_samples)

        # testing
        sum_loss = 0
        n_correct = 0
        n_samples = 0
        for batch_idx, (x, target) in enumerate(test_loader):
            with torch.no_grad():
                x, target = Variable(x), Variable(target)
            out = model(x)

            _, pred_label = torch.max(out.data, 1)
            n_samples += x.data.size()[0]
            n_correct += (pred_label == target.data).sum()

            loss = criterion(out, target)
            sum_loss += loss.item()
        test_loss.append(sum_loss / n_samples)
        test_acc.append(n_correct / n_samples)

        if print_progress:
            lt, at = train_loss[-1], train_acc[-1]
            lte, ate = test_loss[-1], test_acc[-1]
            print('Epoch {e}: train: {lt} loss, {at} acc; test: {lte} loss, {ate} acc'
                  .format(e=epoch, lt=lt, at=at, lte=lte, ate=ate))

    metrics = {'train': {'loss': train_loss, 'acc': train_acc},
               'test': {'loss': test_loss, 'acc': test_acc}}
    return model, metrics

