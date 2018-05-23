from run import experiment
from multiprocessing import Pool


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Model Training')
    # required args
    parser.add_argument('-k', '--kappas', type=float, nargs='+', required=True)
    parser.add_argument('-m', '--method', type=str, choices=['PSGDl1', 'SGDFWl1'], required=True)
    # default as used in main experiment (standard parameters from literature)
    parser.add_argument('-e', '--epochs', type=int, default=250)
    parser.add_argument('-b', '--batchsize', type=int, default=256)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    args = parser.parse_args()
    m, e, b, lr, mo = args.method, args.epochs, args.batchsize, args.learning_rate, args.momentum
    prms = [(m, kappa, e, b, lr, mo) for kappa in args.kappas]
    with Pool(processes=8) as pool:
        pool.starmap(experiment, prms)
