# FWDL 
Frank Wolfe for Deep Learning

The FWDL repository contains code for training an MLP with Frank-Wolfe, SGD, and other popular optimizers.
The network is implemented in torch, the Frank-Wolfe method, as well as projected SGD have been implemented so as to
inherit from the torch optimizer framework.
The projection and Frank Wolfe are restricted to l1 norms on parameter sets, i.e. vectorized weight matrices or bias
vectors.

Before running experiments, please run 

`mkdir results`

in your bash. Also you will need internet connection so as to download the mnist 
data before training which is done automatically.


### - Model and training

The model architecture can be found in `network.py`, a method to train said model
('network.train_model(model, optimizer, ...)') can be found in the same file. The metrics described in
the report are implemented within the network and collected during training. Afterwards,
a dictionary of metrics is returned and saved when used in conjunction with `run.py`.

### - Optimizers and Oracles

The optimizer functions can be found in `oracles.py`, respectively, they are:

* PSGDl1: Projected stochastic gradient descent onto the l1 ball with given parameter k that applies to all sets.
* SGDFWl1: Stochastic Frank Wolfe with with the same interface as PSGDl1 using a parameter kappa. The learning rate is
computed according to the standard method without line search to minimize function evaluations.

The optimizers use the following oracles (which can be found in `optimizers.py`):
* LMO_l1: Performs the linear maximum oracle on the l1 ball.
* P_l1: Given a vector and the radius of the l1 ball, returns a projection of the vector onto said l1 ball.

### - Experiment

The experiment is done in the `run.py`. It will pickle and save all results obtained during training
in the results folder. To run several experiments at once, we have `grid_search.py`. Both provide a
command line interface and the parameters are straight-forward. `run.py` also shows help messages for all
parameters.

### - Auxiliary

The loading of the MNIST data-set code can be found in `utils.py`.

### - Reproducing Results and plots

To obtain the result of several different models, use `grid_search` with an array of kappa values. This is
very intensive and takes time so we will describe here the procedure to obtain the final models. To run this
grid_search we used a large array of parameters. For reproduction, run
```bash
python grid_search.py -m SGDFWl1 -e 250 -z 0 -k 16 32 64 128 256 512 1024 2048 4096 8192 ...
```
and do the same for `-m PSGDl1 -z 1` and without the kappa values for `-m SGD -z 1`.

k corresponds to kappa, the l1 upper bound as described in the report. e corresponds to epochs and z determines
if the model is zero-initialized (1) or not (0). Zero-initialization only works with the Frank Wolfe method
as described in the report.

To reproduce the results with final parameters used, run

```bash
python run.py -m SGD -e 250 -z 0
python run.py -m PSGDl1 -e 250 -k 4096 -z 0
python run.py -m SGDFWl1 -e 250 -k 4096 -z 1
```

This will save all the required files for analysis under `results/`. The analysis is done using
`analysis.py` with the following corresponding parameters that depend on the above chosen
parameters in turn:

```bash
python analysis.py --kappa_psgd 4096 --kappa_sgdfw 4096 -e 250
```

The resulting plots as used in the report will be saved in the working directory. Further, the sparsities
are logged to the terminal.

### Requirements

We use python 3.6, the whole code ist run with anaconda python so we recommend that.

package requirements:

```
numpy
scipy
pandas
pytorch
matplotlib
```
