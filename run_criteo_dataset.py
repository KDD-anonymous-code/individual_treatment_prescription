import sys
import itertools
import os
import multiprocessing
import numpy as np
import pandas as pd
import argparse
import warnings
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from methods import ite_2m, ite_sdr, mite_2m, mite_sdr
from metrics import metric_auuc_random, metric_auuc
warnings.filterwarnings('ignore')


class Dataset:
    _dataset = None

    def __init__(self, filepath, outcome, seed):
        self.filepath = filepath
        self.outcome = outcome
        self.seed = seed
        self.features = [f'f{i}' for i in range(12)]

    def get_dataframe(self):
        if Dataset._dataset is None:
            if self.filepath.endswith('.csv') or self.filepath.endswith('.csv.gz'):
                df = pd.read_csv(self.filepath)
            else:
                df = pd.read_pickle(self.filepath)
            df[self.features] = normalize(df[self.features], axis=0, norm='l2')
            Dataset._dataset = df
        return Dataset._dataset

    def get_split(self, i):
        return train_test_split(self.get_dataframe(), test_size=0.5, random_state=self.seed + i)


def compute_metrics(test_df, u_pred_test, params, outcome):
    auuc_rand = metric_auuc_random(test_df, outcome, 'treatment')
    auuc = metric_auuc(test_df, outcome, 'treatment', u_pred_test)

    return {
        'penalty': params.penalty,
        'C': params.c,
        'split': params.split,
        'method': params.method,
        'AUUC': auuc,
        'AUUC_random': auuc_rand
    }


def eval_ite_2m(dataset, params):
    np.random.seed(0)
    clf_t = params.new_logistic_regression()
    clf_c = params.new_logistic_regression()
    train_df, test_df = dataset.get_split(params.split)

    u_pred_test = ite_2m(train_df, test_df, dataset.features, dataset.outcome, 'treatment', clf_t, clf_c)
    return compute_metrics(test_df, u_pred_test, params, dataset.outcome)


def eval_ite_sdr(dataset, params):
    np.random.seed(0)
    clf = params.new_logistic_regression()
    train_df, test_df = dataset.get_split(params.split)
    u_pred_test = ite_sdr(train_df, test_df, dataset.features, dataset.outcome, 'treatment', clf)
    return compute_metrics(test_df, u_pred_test, params, dataset.outcome)


def eval_mite_2m(dataset, params):
    np.random.seed(0)
    clf_er = LogisticRegression(C=100.,
                                penalty='l1',
                                fit_intercept=False,
                                solver='saga',
                                class_weight='balanced')
    clf_t = params.new_logistic_regression()
    clf_c = params.new_logistic_regression()
    train_df, test_df = dataset.get_split(params.split)
    u_pred_test = mite_2m(train_df, test_df, dataset.features, dataset.outcome,
                          'treatment', 'exposure', clf_t, clf_c, clf_er)
    return compute_metrics(test_df, u_pred_test, params, dataset.outcome)


def eval_mite_sdr(dataset, params):
    np.random.seed(0)
    clf_er = LogisticRegression(C=100.,
                                penalty='l1',
                                fit_intercept=False,
                                solver='saga',
                                class_weight='balanced')
    clf = params.new_logistic_regression()
    train_df, test_df = dataset.get_split(params.split)
    u_pred_test = mite_sdr(train_df, test_df, dataset.features, dataset.outcome, 'treatment', 'exposure', clf, clf_er)
    return compute_metrics(test_df, u_pred_test, params, dataset.outcome)


def evaluate_model(args):
    dataset, params = args
    fun = {
        'ITE_2M': eval_ite_2m,
        'ITE_SDR': eval_ite_sdr,
        'MITE_2M': eval_mite_2m,
        'MITE_SDR': eval_mite_sdr,
    }[params.method]
    result = fun(dataset, params)
    return result


class Hyperparams:
    def __init__(self, penalty, c, split, method):
        self.penalty = penalty
        self.c = c
        self.split = split
        self.method = method

    def new_logistic_regression(self):
        return LogisticRegression(C=self.c, penalty=self.penalty, solver='saga')

    def __str__(self):
        return f"penalty={self.penalty} c={self.c} split={self.split} method={self.method}"

    @staticmethod
    def grid(full):
        n_splits = 100
        split_list = range(n_splits)
        if full:
            penal_list = ['l1', 'l2']
            c_list = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
        else:
            penal_list = ['l2']
            c_list = [10]
        method_list = ['ITE_2M', 'MITE_2M', 'ITE_SDR', 'MITE_SDR']

        return [Hyperparams(penalty, c, split, method) for penalty, c, split, method in
                itertools.product(penal_list, c_list, split_list, method_list)]


def main(args):
    parser = argparse.ArgumentParser(prog=args[0])
    subparsers = parser.add_subparsers(dest="step")
    convert_parser = subparsers.add_parser('convert')
    convert_parser.add_argument("input", help="Path to the input csv file: /path/input.csv.gz")
    convert_parser.add_argument("output", help="Path to the output pickle file: /path/ouput.pkl.gz")
    run_parser = subparsers.add_parser('run')
    run_parser.add_argument("input", help="Path to the input file (csv or pickle)")
    run_parser.add_argument("output", help="Path to the output directory")
    run_parser.add_argument("-s", "--seed", type=int, default=0, help="Seed for train/test split")
    run_parser.add_argument("-n", "--ncpus", type=int, default=1, help="Number of CPUs to be used")
    run_parser.add_argument("-f", "--fullgrid", action='store_true',
                            help="To explore the complete grid of hyperparameters")
    run_parser.add_argument("-r", "--resumefrom")

    args = parser.parse_args(args[1:])
    if args.step is None:
        parser.print_usage()
        return 1

    if args.step == "convert":
        # to convert into pickle format
        print(f"Convert '{args.input}' into '{args.output}'")
        pd.read_csv(args.input).to_pickle(args.output)
        print("Done")
        return 0

    input_filepath = os.path.abspath(args.input)
    output_directory = os.path.abspath(args.output)
    n_cpus = args.ncpus

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    filename = f'result_criteo{"_fullgrid" if args.fullgrid else ""}.csv'

    def save_results(results_, i_=None):
        out_filename = filename if i_ is None else f"_tmp_{i_}_{filename}"
        res = pd.DataFrame(results_)
        res.reindex(sorted(res.columns), axis=1) \
            .sort_values(by=['penalty', 'C', 'split', 'method']) \
            .to_csv(os.path.join(output_directory, out_filename), index=False)

    dataset = Dataset(filepath=input_filepath, outcome="visit", seed=args.seed)
    tasks = [(dataset, params) for params in Hyperparams.grid(full=args.fullgrid)]

    results = []
    if args.resumefrom is not None:
        num_tasks = len(tasks)
        df = pd.read_csv(args.resumefrom)
        results.extend(df.T.to_dict().values())

        def in_df(prms):
            red = df[(df['split'] == prms.split) & (df['penalty'] == prms.penalty)
                     & (df['C'] == prms.c) & (df['method'] == prms.method)]
            assert len(red) <= 1
            return len(red) == 1
        tasks = [(dataset, params) for dataset, params in tasks if not in_df(params)]
        print(f"Remaining tasks: {len(tasks)} / {num_tasks} ({len(df)} completed tasks in file)")

    if n_cpus <= 1:
        for i, task in tqdm(enumerate(tasks), total=len(tasks)):
            results.append(evaluate_model(task))
            if i % 10 == 0:
                save_results(results, i)
        save_results(results)
    else:
        with multiprocessing.Pool(n_cpus) as pool:
            for i, result in tqdm(enumerate(pool.imap_unordered(evaluate_model, tasks)), total=len(tasks)):
                results.append(result)
                if i % 10 == 0:
                    save_results(results, i)
            save_results(results)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
