import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

from optimizers import SGDl1, SGDFWl1, SGDFWNuclear

import numpy as np
from matplotlib import pyplot as plt


## load mnist dataset
use_cuda = torch.cuda.is_available()

root = './data'
if not os.path.exists(root):
    os.mkdir(root)
    
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
# if not exist, download mnist dataset
train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)

batch_size = 100

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

print('==>>> total trainning batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}'.format(len(test_loader)))

## network
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def name(self):
        return "MLP"

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def name(self):
        return "LeNet"


def train_model(model, optimizer, criterion, epochs, train_loader):
    train_loss = []
    test_error = []
    for epoch in range(epochs):
        # trainning
        ave_loss = 0
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            if use_cuda:
                x, target = x.cuda(), target.cuda()
            x, target = Variable(x), Variable(target)
            out = model(x)

            loss = criterion(out, target) 
            ave_loss = ave_loss * 0.9 + loss.item() * 0.1
            loss.backward()
            optimizer.step()
            if (batch_idx+1) % 100 == 0 or (batch_idx+1) == len(train_loader):
                print('==>>> epoch: {}, batch index: {}, train loss: {:.5f}'.format(
                    epoch, batch_idx+1, ave_loss))
        # testing
        correct_cnt, ave_loss = 0, 0
        total_cnt = 0
        for batch_idx, (x, target) in enumerate(test_loader):
            if use_cuda:
                x, target = x.cuda(), target.cuda()
            with torch.no_grad():
                x, target = Variable(x), Variable(target)
            out = model(x)

            loss = criterion(out, target) 
            _, pred_label = torch.max(out.data, 1)
            total_cnt += x.data.size()[0]
            correct_cnt += (pred_label == target.data).sum()
            # smooth average
            ave_loss = ave_loss * 0.9 + loss.item() * 0.1

        if(batch_idx+1) % 100 == 0 or (batch_idx+1) == len(test_loader):
            print('==>>> epoch: {}, batch index: {}, test loss: {:.5f}, acc: {}'.format(epoch, batch_idx+1, ave_loss, float(correct_cnt) * 1.0 / total_cnt))

        train_loss.append(ave_loss)
        test_error.append(float(correct_cnt) * 1.0 / total_cnt)

    return model, train_loss, test_error

## training
model = MLPNet()

epochs = 10
learning_rate = 0.01
momentum = 0.9
lmbd = 1e-10
kappa = 3000

if use_cuda:
    model = model.cuda()

#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
#optimizer = SGDl1(model.parameters(), lr=learning_rate, lambda_l1=lmbd, momentum=momentum)

optimizer = SGDFWNuclear(model.parameters(), kappa_l1=kappa)

criterion = nn.CrossEntropyLoss()

model, train_loss, test_error = train_model(model, optimizer, criterion, epochs, train_loader)


plt.plot(train_loss, label='train loss')
plt.plot(test_error, label='test error')
plt.xlabel("epochs"); 
plt.title("FWDL"); plt.legend();

plt.show()

#torch.save(model.state_dict(), model.name())