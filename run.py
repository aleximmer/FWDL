import pickle
import torch.nn as nn

from optimizers import SGDFWl1, PSGDl1, SGD
from network import MLPNet, train_model
import utils


def get_exp_name(*args):
    return '_'.join([str(e) for e in args])


def experiment(method, kappa, epochs, batch_size, lr, momentum, zero_init):
    train_loader, test_loader = utils.load(batch_size=batch_size)
    model = MLPNet(zero_init=zero_init)
    if method == 'SGD':
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    elif method == 'PSGDl1':
        optimizer = PSGDl1(model.parameters(), kappa_l1=kappa, lr=lr, momentum=momentum)
    elif method == 'SGDFWl1':
        optimizer = SGDFWl1(model.parameters(), kappa_l1=kappa)
    else:
        raise ValueError('Invalid choice of method: ' + str(method))

    criterion = nn.CrossEntropyLoss(size_average=False)
    model, metrics = train_model(model, optimizer, criterion, epochs, train_loader, test_loader)

    fname = get_exp_name(method, kappa, epochs, batch_size, zero_init)
    with open('results/' + fname + '.pkl', 'wb+') as f:
        pickle.dump(metrics, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('-k', '--kappa', type=float, help='kappa l1 threshold for parameters', default=None)
    parser.add_argument('-m', '--method', type=str, choices=['PSGDl1', 'SGDFWl1', 'SGD'],
                        help='method among PSGDl1, SGDFWl1, SGD', required=True)
    parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('-b', '--batchsize', type=int, default=256, help='batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='learning rate for SGD and PSGDl1')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD and PSGDlq')
    parser.add_argument('-z', '--zero_init', type=int, default=1, choices=[0, 1],
                        help='initialize the network with only zeros. 0 is False, 1 True')
    args = parser.parse_args()
    zinit = bool(args.zero_init)
    if args.kappa is None and args.method != 'SGD':
        raise ValueError('Requires kappa parameter for constrained optimization')
    experiment(args.method, args.kappa, args.epochs, args.batchsize, args.learning_rate, args.momentum, zinit)
