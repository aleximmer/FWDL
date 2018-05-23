from run import experiment
from multiprocessing import Pool


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Model Training')
    # required args
    parser.add_argument('-k', '--kappas', type=float, nargs='+', required=True)
    parser.add_argument('-m', '--method', type=str, choices=['PSGDl1', 'SGDFWl1'], required=True)
    # default as used in main experiment (standard parameters from literature)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-b', '--batchsize', type=int, default=256)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('-p', '--processes', type=int, default=1)
    parser.add_argument('-z', '--zero_init', type=bool, default=True)
    args = parser.parse_args()
    m, e, b, lr, mo, z = args.method, args.epochs, args.batchsize, args.learning_rate, args.momentum, args.zero_init
    prms = [(m, kappa, e, b, lr, mo, z) for kappa in args.kappas]
    if args.processes > 1:
        with Pool(processes=args.processes) as pool:
            pool.starmap(experiment, prms)
    else:
        for param in prms:
            experiment(*param)
