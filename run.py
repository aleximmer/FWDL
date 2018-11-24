import torch.nn as nn
from sacred import Experiment

from optimizers import SGDFWl1, PSGDl1, SGD
from network import MLPNet, train_model
from analysis import plot_sparse_network, plot_input_activations
import utils


ex = Experiment()
ex.observers.append(utils.get_ExpDB())


@ex.config
def configuration():
    kappa = None
    method = None
    epochs = 200
    batch_size = 128
    step_size = None
    momentum = None
    method_tag = None


@ex.named_config
def swell_frank_wolfe():
    kappa = 512
    method = 'SGDFWl1'
    method_tag = 'FW-NN'


@ex.named_config
def swell_projected_sgd():
    kappa = 512
    method = 'PSGDl1'
    method_tag = 'PSGD-NN'


@ex.named_config
def swell_sgd():
    method = 'SGD'
    method_tag = 'NN'


@ex.automain
def experiment(method, kappa, epochs, batch_size, step_size, momentum,
               method_tag, _run):
    print('Method Tag', method_tag)
    train_loader, test_loader = utils.load(batch_size=batch_size)
    model = MLPNet(zero_init=False)
    if method == 'SGD':
        optimizer = SGD(model.parameters(), lr=step_size, momentum=momentum)
    elif method == 'PSGDl1':
        optimizer = PSGDl1(model.parameters(), kappa_l1=kappa, lr=step_size, momentum=momentum)
    elif method == 'SGDFWl1':
        optimizer = SGDFWl1(model.parameters(), kappa_l1=kappa)
    else:
        raise ValueError('Invalid choice of method: ' + str(method))

    criterion = nn.CrossEntropyLoss(size_average=False)
    model, results, weights = train_model(model, optimizer, criterion, epochs, train_loader,
                                          test_loader, _run)
    plot_sparse_network(weights, _run)
    plot_input_activations(weights, 'max', _run)
    plot_input_activations(weights, 'mean', _run)
    plot_input_activations(weights, 'sum', _run)
    return results
