import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

import numpy as np


## network
class MLPNet(nn.Module):

    def __init__(self):

        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):

        x = x.view(-1, 28*28)
        x = F.tanh(self.fc1(x)) 
        x = F.tanh(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x
    

def train_model(model, optimizer, criterion, epochs, train_loader, test_loader, use_cuda=False, print_progress=True):
    """ 
    """


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
            if print_progress and (batch_idx+1) % 100 == 0 or (batch_idx+1) == len(train_loader):
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

        if print_progress and (batch_idx+1) % 100 == 0 or (batch_idx+1) == len(test_loader):
            print('==>>> epoch: {}, batch index: {}, test loss: {:.5f}, acc: {}'.format(epoch, batch_idx+1, ave_loss, float(correct_cnt) * 1.0 / total_cnt))

        train_loss.append(ave_loss)
        test_error.append(float(correct_cnt) * 1.0 / total_cnt)

    return model, train_loss, test_error

