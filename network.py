import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

EPS = 1e-8  # sparsity threshold


# TODO: -0.5 & all zero does not converge (just like tanh)
# TODO: FW without -0.5 works with 0 init

class MLPNet(nn.Module):
    def __init__(self, zero_init=True):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 80)
        self.fc2 = nn.Linear(80, 80)
        self.fc3 = nn.Linear(80, 80)
        self.fc4 = nn.Linear(80, 10)
        self.layers = [self.fc1, self.fc2, self.fc3, self.fc4]
        if zero_init:
            self.set_zero()
        self._params = None
        self._nodes = None
        self._paths = None

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)
        return x

    def set_zero(self):
        for layer in self.layers:
            layer.weight.data.zero_()
            layer.bias.data.zero_()

    def get_weight_matrices(self):
        return [l.weight.detach().numpy() for l in self.layers]

    def get_bias_vectors(self):
        return [l.bias.detach().numpy() for l in self.layers]

    def paths(self):
        if self._paths is not None:
            return self._paths
        self._paths = np.product([l.weight.shape[0] for l in self.layers[:-1]]) * self.fc1.weight.shape[1]
        return self._paths

    def active_paths(self):
        n_active = 1
        for i in range(len(self.layers)):
            n_active *= self._comp_active_nodes(i)
        return n_active

    def nodes(self):
        if self._nodes is not None:
            return self._nodes
        self._nodes = np.sum([l.weight.shape[0] for l in self.layers[:-1]]) + self.fc1.weight.shape[1]
        return self._nodes

    def active_nodes(self):
        n_active = 0
        for i in range(len(self.layers)):
            n_active += self._comp_active_nodes(i)
        return n_active

    def _comp_active_nodes(self, layer):
        # count outgoing sparsity
        nextl = self.layers[layer]
        W = nextl.weight.detach().numpy()
        active_out = (np.sum(~np.isclose(W, 0, atol=EPS), axis=0) >= 1)
        """
        if layer > 0:
            # count incoming sparsity
            prevl = self.layers[layer - 1]
            W, b = prevl.weight.detach().numpy(), prevl.bias.detach().numpy()
            nodes = np.concatenate([W, b[:, np.newaxis]], axis=1)
            active_in = (np.sum(~np.isclose(nodes, 0, atol=EPS), axis=1) >= 1)
            active = active_in if active_out is None else (active_out | active_in)
        else:
            active = active_out"""
        return np.sum(active_out)

    def params(self):
        if self._params is not None:
            return self._params
        n_weights = np.sum([np.product(l.weight.shape) for l in self.layers])
        n_bias = np.sum([l.bias.shape[0] for l in self.layers])
        self._params = n_weights + n_bias
        return self._params

    def active_params(self):
        wvec = np.concatenate([l.weight.detach().numpy().reshape(-1) for l in self.layers])
        bvec = np.concatenate([l.bias.detach().numpy().reshape(-1) for l in self.layers])
        vec = np.concatenate([wvec, bvec])
        return np.sum(~np.isclose(vec, 0., atol=EPS))

    def f_active_paths(self):
        return self.active_paths() / self.paths()

    def f_active_nodes(self):
        return self.active_nodes() / self.nodes()

    def f_active_params(self):
        return self.active_params() / self.params()


def train_model(model, optimizer, criterion, epochs, train_loader, test_loader, print_progress=True):
    """ trains a model using a given optimizer and criterion. 
    """
    train_loss, test_loss = [], []
    train_acc, test_acc = [], []
    train_loss, test_error = [], []
    active_nodes = []
    active_paths = []
    active_params = []
    weight_matrices = []
    bias_vectors = []
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
            n_correct += (pred_label == target.data).sum().numpy()

            loss = criterion(out, target)
            sum_loss += loss.item()
            loss.backward()
            optimizer.step()
            # measure sparsity and paths
        active_nodes.append(model.f_active_nodes())
        active_paths.append(model.f_active_paths())
        active_params.append(model.f_active_params())
        weight_matrices.append(model.get_weight_matrices())
        bias_vectors.append(model.get_bias_vectors())
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
            n_correct += (pred_label == target.data).sum().numpy()

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
               'test': {'loss': test_loss, 'acc': test_acc},
               'sparsity': {'params': active_params, 'nodes': active_nodes, 'paths': active_paths},
               'params': {'weights': weight_matrices, 'biases': bias_vectors}}
    return model, metrics
