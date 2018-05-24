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
* LMO_l1: 
* LMO_nuclear: 
* P_l1: Given a vector and the radius of the simplex, returns a projection of the vector onto said simplex.


##### - Experiment

The experiment is done in the `run.py`.

##### - Auxiliary

To explore the hyper-parameters (mainly kappa, the radius of the simplex) grid search is used, which can be found in `grid_search.py`.

The loading of the MNIST data-set code can be found in `utils.py`.

Simple tests for the oracles can be found in `test_oracles.py`. The code has also been tested
thoroughly but no unittests are provided.

##### - Reproducing Results and plots

To obtain the result of several different models, use `grid_search` with an array of kappa values. This is
very intensive and takes time so we will describe here the procedure to obtain the final models.
k corresponds to kappa, the l1 upper bound as described in the report. e corresponds to epochs and z determines
if the model is zero-initialized (1) or not (0). Zero-initialization only works with the Frank Wolfe method
as described in the report.

```bash
python run.py -m SGD -e 250 -z 0
python run.py -m PSGDl1 -e 250 -k 4096 -z 1
python run.py -m SGDFWl1 -e 250 -k 4096
```

This will save all the required files for analysis under `results/`. The analysis is done using
`analysis.py` with the following corresponding parameters that depend on the above chosen
parameters in turn:

```bash
python analysis.py --kappa_psgd 4096 --kappa_sgdfw 4096 -e 250
```
