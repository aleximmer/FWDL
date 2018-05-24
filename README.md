# FWDL 
Frank Wolfe for Deep Learning

The FWDL repository contains code for training an MLP with Frank-Wolfe, SGD, and other popular optimizers. The network is implemented in torch, the Frank-Wolfe method, as well as projected SGD have been implemented so as to inherit from the torch optimizer framework.

Before running experiments, please run 

`mkdir results`

in your bash. Also you will need internet connection so as to download the mnist 
data before training which is done automatically.



#### FWDL files 

##### - Model and training

The model architecture can be found in `network.py`, a method to train said model ('network.train_model(model, optimizer, ...)') can be found in the same file. 

##### - Optimizers and Oracles

The optimizer functions can be found in `oracles.py`, respectively, they are:

* PSGDl1: Projected stochastic gradient descent onto the simplex.
* SGDFWl1: Stochastic Frank Wolfe with |vec(W_i)|_1 <= kappa_l1 where W_i are parameter sets
* SGDFWNuclear: Stochastic Frank Wolfe with |W_i|_* <= kappa_l1 where W_i are parameter sets, if the gradient has dimensionality 1 or 3 and above, we use |vec(W_i|_1 <= kappa_l1.

The optimizers use the following oracles (which can be found in `optimizers.py`):
* LMO_l1: Performs the linear maximum oracle on the simplex.
* P_l1: Given a vector and the radius of the simplex, returns a projection of the vector onto said simplex.


##### - Experiment

The experiment is done in the `run.py`.

##### - Auxiliary

To explore the hyper-parameters (mainly kappa, the radius of the simplex) grid search is used, which can be found in `grid_search.py`.

The loading of the MNIST data-set code can be found in `utils.py`.

Simple tests for the oracles can be found in `test_oracles.py`.









