import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from optimizers import SGDl1, SGDFWl1, SGDFWNuclear, PSGDl1
import network as net
import utils


use_cuda = torch.cuda.is_available()


batch_size = 100
## load mnist dataset
train_loader, test_loader = utils.load(batch_size=batch_size)


## Chose hyper-parameters

model = net.MLPNet()

epochs = 1
learning_rate = 0.01
momentum = 0.9
lmbd = 1e-10
kappa = 3000

if use_cuda:
    model = model.cuda()

## Choose optimizer

#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
#optimizer = SGDl1(model.parameters(), lr=learning_rate, lambda_l1=lmbd, momentum=momentum)
optimizer = PSGDl1(model.parameters(), lr=learning_rate, kappa_l1=kappa, momentum=momentum)
#optimizer = SGDFWNuclear(model.parameters(), kappa_l1=kappa)



## Choose loss

criterion = nn.CrossEntropyLoss()

## train model

model, train_loss, test_error = net.train_model(model, optimizer, criterion, epochs, \
                                                    train_loader, test_loader, use_cuda)


## plot results

utils.plot_loss_acc(train_loss, test_error, optimizer.name())

## save model

#torch.save(model.state_dict(), model.name())