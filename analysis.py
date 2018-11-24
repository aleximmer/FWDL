import numpy as np
import matplotlib.pyplot as plt


def plot_input_activations(weights, aggst, run):
    weights = weights[0]
    agg = {'sum': np.sum, 'mean': np.mean, 'max': np.max}[aggst]
    f, ax1 = plt.subplots(1, 1)
    ax1.imshow(agg(np.abs(weights), axis=0).reshape(28, 28), cmap='gray')
    ax1.set_title('SGD')
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.tight_layout()
    plt.savefig('figure.png')
    run.add_artifact('figure.png', 'NetActivation-{aggst}.png'.format(aggst=aggst))
    plt.gcf().clear()


def plot_sparse_network(weights, run):
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
    plt.savefig('figure.png')
    run.add_artifact('figure.png', 'Sparsity.png')
    plt.gcf().clear()
