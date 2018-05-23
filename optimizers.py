import torch
from torch.optim import Optimizer
from oracles import LMO_nuclear, LMO_l1


class SGDl1(Optimizer):
    def __init__(self, params, lr, lambda_l1):
        assert lambda_l1 > 0
        defaults = dict(lr=lr, l1=lambda_l1)
        super(Optimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lam = group['l1']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                d_p.add_(lam, torch.sign(p.data))

                p.data.add_(-group['lr'], d_p)

        return loss


class SGDFWl1(Optimizer):
    def __init__(self, params, kappa_l1):
        assert kappa > 0
        defaults = dict(kappa=kappa_l1)
        super(Optimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            kappa = group['kappa_l1']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                s = LMO_l1(d_p, kappa)
                gamma = None

                #d_p.add_(lam, torch.sign(p.data))

                p.data.add_(-group['lr'], d_p)

        return loss
