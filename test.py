import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

from optimizers import SGDl1, SGDFWl1, SGDFWNuclear

import network as net

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


## training

model = net.MLPNet()

epochs = 10
learning_rate = 0.01
momentum = 0.9
lmbd = 1e-10
kappa = 3000

if use_cuda:
    model = model.cuda()

## Choose optimizer

#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
#optimizer = SGDl1(model.parameters(), lr=learning_rate, lambda_l1=lmbd, momentum=momentum)

optimizer = SGDFWNuclear(model.parameters(), kappa_l1=kappa)
criterion = nn.CrossEntropyLoss()

model, train_loss, test_error = net.train_model(model, optimizer, \
                                                    criterion, epochs, train_loader, use_cuda)


plt.plot(train_loss, label='train loss')
plt.plot(test_error, label='test error')
plt.xlabel("epochs"); 
plt.title("FWDL"); plt.legend();

plt.show()

#torch.save(model.state_dict(), model.name())