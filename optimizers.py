import torch
from torch.optim import Optimizer
from oracles import LMO_nuclear, LMO_l1


class SGDl1(Optimizer):
    def __init__(self, params, lr, lambda_l1, momentum=0, dampening=0, nesterov=False):
        assert lambda_l1 > 0
        defaults = dict(lr=lr, l1=lambda_l1, momentum=momentum,
                        dampening=dampening, nesterov=nesterov)
        super(SGDl1, self).__init__(params, defaults)

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
            momentum = group['momentum']
            dampening = group['dampening']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                # l1 subgradient
                d_p.add_(lam, torch.sign(p.data))

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss


class SGDFWl1(Optimizer):
    """Stochastic Frank Wolfe with |vec(W_i)|_1 <= kappa_l1 where W_i are parameter sets
    """
    def __init__(self, params, kappa_l1):
        self.k = 0
        assert kappa_l1 > 0
        defaults = dict(kappa=kappa_l1)
        super(SGDFWl1, self).__init__(params, defaults)

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
            kappa = group['kappa']

            for p in group['params']:
                if p.grad is None:
                    continue
                s = LMO_l1(p.grad.data.numpy(), kappa)
                gamma = 2 / (self.k + 2)
                # x^(k+1) = x^(k) - g x^(k) + g s
                delta_p = torch.Tensor(gamma * s - gamma * p.data.numpy())
                p.data.add_(delta_p)

        self.k += 1
        return loss


class SGDFWNuclear(Optimizer):
    """Stochastic Frank Wolfe with |W_i|_* <= kappa_l1 where W_i are parameter sets
    if the gradient has dimensionality 1 or 3 and above, we use |vec(W_i|_1 <= kappa_l1
    """
    def __init__(self, params, kappa_l1):
        self.k = 0
        assert kappa_l1 > 0
        defaults = dict(kappa=kappa_l1)
        super(SGDFWNuclear, self).__init__(params, defaults)

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
            kappa = group['kappa']

            for p in group['params']:
                if p.grad is None:
                    continue

                print(p.grad.data.numpy().shape)
                s = LMO_nuclear(p.grad.data.numpy(), kappa)
                gamma = 2 / (self.k + 2)
                # x^(k+1) = x^(k) - g x^(k) + g s
                delta_p = torch.Tensor(gamma * s - gamma * p.data.numpy())
                p.data.add_(delta_p)

        self.k += 1
        return loss
