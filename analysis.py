import pickle
import pandas as pd
import numpy as np
from run import get_exp_name
import matplotlib.pyplot as plt


def get_result_frame(method, kappa, epochs, batchsize, zero_init):
    kappa = float(kappa) if kappa is not None else kappa
    with open('results/' + get_exp_name(method, kappa, epochs, batchsize, zero_init) + '.pkl', 'rb') as f:
        res = pickle.load(f)
    df = pd.DataFrame(index=['acctest', 'acctrain', 'paths', 'nodes', 'params', 'weights', 'biases'],
                      columns=list(range(1, epochs+1)))
    df.loc['acctest'] = res['test']['acc']
    df.loc['acctrain'] = res['train']['acc']
    df.loc['paths'] = res['sparsity']['paths']
    df.loc['nodes'] = res['sparsity']['nodes']
    df.loc['params'] = res['sparsity']['params']
    df.loc['weights'] = res['params']['weights']
    df.loc['biases'] = res['params']['biases']
    return df


def plot_performances(dfs, dfw, dfp):
    plt.plot(dfs.T.acctest, label='SGD test')
    plt.plot(dfs.T.acctrain, label='SGD train')
    plt.plot(dfw.T.acctest, label='FW test')
    plt.plot(dfw.T.acctrain, label='FW train')
    plt.plot(dfp.T.acctest, label='PSGD test')
    plt.plot(dfp.T.acctrain, label='PSGD train')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig('performances.png')
    plt.show()


def plot_and_log_sparsity(dfs, dfw, dfp):
    plt.plot(dfp.T.paths, label='PSGD paths', alpha=0.7)
    plt.plot(dfw.T.paths, label='FW paths', alpha=0.7)
    plt.plot(dfp.T.nodes, label='PSGD nodes', alpha=0.7)
    plt.plot(dfw.T.nodes, label='FW nodes', alpha=0.7)
    plt.plot(dfp.T.params, label='PSGD params', alpha=0.7)
    plt.plot(dfw.T.params, label='FW params', alpha=0.7)
    plt.legend()
    plt.ylabel('Fraction')
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.savefig('sparsity.png')
    print('------------SPARSITIES-----------')
    log_sparsity('SGD', dfs)
    log_sparsity('FW', dfw)
    log_sparsity('PSGD', dfp)
    plt.show()


def log_sparsity(method, df):
    print(method + ': ', 'paths: ', list(df.T.paths)[-1], 'nodes: ', list(df.T.nodes)[-1],
          'params: ', list(df.T.params)[-1])


def plot_input_activations(dfs, dfw, dfp, agg):
    agg = {'sum': np.sum, 'mean': np.mean, 'max': np.max}[agg]
    W1s = dfs.loc['weights', np.argmin(dfs.T.acctest.values)+1][0]
    W1w = dfw.loc['weights', np.argmin(dfw.T.acctest.values)+1][0]
    W1p = dfp.loc['weights', np.argmin(dfp.T.acctest.values)+1][0]

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    ax1.imshow(agg(np.abs(W1s), axis=0).reshape(28, 28), cmap='gray')
    ax1.set_title('SGD')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.imshow(agg(np.abs(W1p), axis=0).reshape(28, 28), cmap='gray')
    ax2.set_title('PSGD')
    ax2.set_xticks([])
    ax3.imshow(agg(np.abs(W1w), axis=0).reshape(28, 28), cmap='gray')
    ax3.set_title('FW')
    ax3.set_xticks([])
    plt.tight_layout()
    plt.savefig('activations.png')
    plt.show()


def plot_sparse_network(dfw, epochs):
    weights = dfw.loc['weights', epochs][1:]
    f, ax = plt.subplots(figsize=(8, 8))
    for i, layer in enumerate(weights):
        M = layer.T
        active = ~np.isclose(np.sum(np.abs(M), axis=1), 0, atol=1e-8)
        for j, (x, y) in enumerate(zip(np.zeros(M.shape[0]) + i * 3, np.arange(M.shape[0]))):
            c = 'g' if active[j] else 'r'
            ax.plot(x, y, 'o', lw=1, markersize=1, c=c)
            y_off = 35 if i == len(weights) - 1 else 0
            for xp, yp in zip(np.zeros(M.shape[1]) + (i + 1) * 3, np.arange(M.shape[1]) + y_off):
                if np.isclose(M[j, yp - y_off], 0, atol=1e-8):
                    continue
                ax.arrow(x, y, 3, yp - y, alpha=0.2, lw=1, fc='gray')
    M = M.T
    for x, y in zip(np.zeros(M.shape[0]) + len(weights) * 3, np.arange(M.shape[0]) + y_off):
        ax.plot(x, y, 'o', lw=1, markersize=1, c='g')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('structure.png')
    plt.show()


def conduct_analysis(kappa_psgd, kappa_sgdfw, epochs, batchsize, agg):
    dfs = get_result_frame('SGD', None, epochs, batchsize, False)
    dfw = get_result_frame('SGDFWl1', kappa_sgdfw, epochs, batchsize, True)
    dfp = get_result_frame('PSGDl1', kappa_psgd, epochs, batchsize, False)
    plot_performances(dfs, dfw, dfp)
    plot_and_log_sparsity(dfs, dfw, dfp)
    plot_input_activations(dfs, dfw, dfp, agg)
    plot_sparse_network(dfw, epochs)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Experiment Analysis')
    parser.add_argument('--kappa_psgd', type=float, required=True)
    parser.add_argument('--kappa_sgdfw', type=float, required=True)
    parser.add_argument('-e', '--epochs', type=int, default=250)
    parser.add_argument('-b', '--batchsize', type=int, default=256)
    parser.add_argument('--pixel_agg', choices=['sum', 'mean', 'max'], default='max',
                        help='How to aggregate over outgoing activations per pixel')
    args = parser.parse_args()
    conduct_analysis(args.kappa_psgd, args.kappa_sgdfw, args.epochs, args.batchsize, args.pixel_agg)
