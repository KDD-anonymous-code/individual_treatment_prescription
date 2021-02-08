import os
import sys
import collections
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from methods import ite_2m, ite_sdr, mite_2m, mite_sdr, ite_best
from metrics import metric_rmse, metric_auuc, metric_auuc_random
from utils import ProgressBar
from synthetic_data_generator import SyntheticDataGenerator, ComplianceType


def put_metric_results(res_df, method, split, rmse, auuc, auuc_thout):
    res_df.loc[len(res_df)] = {'method': method, 'split': split, 'RMSE': rmse, 'AUUC': auuc, 'AUUC_thout': auuc_thout}


def compute_metrics(test_df, u_pred_test, res_df, method, split):
    rmse = metric_rmse(test_df, 'uplift', u_pred_test)
    auuc = metric_auuc(test_df, 'Y', 'T', u_pred_test, weight='weight')
    auuc_thout = metric_auuc(test_df, 'th_Y', 'T', u_pred_test, weight='th_weight')
    put_metric_results(res_df, method, split, rmse, auuc, auuc_thout)


def eval_ite_best(train_df, test_df, res_df, split, features):
    u_pred_test = ite_best(train_df, test_df, features, 'Y', 'T')
    compute_metrics(test_df, u_pred_test, res_df, 'ITE_best', split)


def eval_ite_2m(train_df, test_df, res_df, split, features):
    np.random.seed(0)
    clf_t = LogisticRegression(solver='liblinear')
    clf_c = LogisticRegression(solver='liblinear')

    u_pred_test = ite_2m(train_df, test_df, features, 'Y', 'T', clf_t, clf_c)
    compute_metrics(test_df, u_pred_test, res_df, 'ITE_2M', split)


def eval_ite_sdr(train_df, test_df, res_df, split, features):
    np.random.seed(0)
    clf = LogisticRegression(solver='liblinear')

    u_pred_test = ite_sdr(train_df, test_df, features, 'Y', 'T', clf)
    compute_metrics(test_df, u_pred_test, res_df, 'ITE_SDR', split)


def eval_mite_2m(train_df, test_df, res_df, split, features):
    np.random.seed(0)
    clf_t = LogisticRegression(solver='liblinear')
    clf_c = LogisticRegression(solver='liblinear')
    clf_er = LogisticRegression(solver='liblinear')

    u_pred_test = mite_2m(train_df, test_df, features, 'Y', 'T', 'M', clf_t, clf_c, clf_er)
    compute_metrics(test_df, u_pred_test, res_df, 'MITE_2M', split)


def eval_mite_sdr(train_df, test_df, res_df, split, features):
    np.random.seed(0)
    clf_er = LogisticRegression(solver='liblinear')
    clf = LogisticRegression(solver='liblinear')

    u_pred_test = mite_sdr(train_df, test_df, features, 'Y', 'T', 'M', clf, clf_er)
    compute_metrics(test_df, u_pred_test, res_df, 'MITE_SDR', split)


method_funcs = [eval_ite_best, eval_ite_2m, eval_ite_sdr, eval_mite_2m, eval_mite_sdr]


def eval_models(splitter, splits=None):
    if splits is None:
        splits = list(range(len(splitter)))

    res_df = pd.DataFrame(columns=['method', 'split', 'RMSE', 'AUUC', 'AUUC_thout'])

    for split in splits:
        train_df, test_df = splitter[split]

        auuc_random = metric_auuc_random(test_df, 'Y', 'T', weight='weight')
        auuc_random_th = metric_auuc_random(test_df, 'th_Y', 'T', weight='th_weight')

        auuc_perfect = metric_auuc(test_df, 'Y', 'T', test_df['uplift'].values, weight='weight')
        auuc_perfect_th = metric_auuc(test_df, 'th_Y', 'T', test_df['uplift'].values, weight='th_weight')

        put_metric_results(res_df, 'RAND', split, None, auuc_random, auuc_random_th)
        for func in method_funcs:
            func(train_df=train_df, test_df=test_df, res_df=res_df, split=split, features=splitter.get_features())
            ProgressBar.incr()
        put_metric_results(res_df, 'Oracle', split, None, auuc_perfect, auuc_perfect_th)
    return res_df


def eval_models_multiprocess(args):
    data_gen, i_dataset, i_split_group = args
    id_, splitter = data_gen[i_dataset]
    return id_, eval_models(splitter, splits=i_split_group)


def main(args):
    data_gens = {
        'single': SyntheticDataGenerator(
            size=int(2e6),
            compliances=[0.99, 0.5, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009,
                         0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001],
            compliance_type=ComplianceType.Single,
            ctrl_outcome_rate=1e-1,
            treatment_uplifts=[-1e-1, 0, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1],
            n_splits=51,
            seed=338
        ),
        'multi': SyntheticDataGenerator(
            size=int(2e6),
            compliances=[0.005, 0.005, 0.005, 0.005, 0.005, 0.01, 0.01, 0.01, 0.01, 0.01],
            compliance_type=ComplianceType.Multi,
            ctrl_outcome_rate=1e-1,
            treatment_uplifts=[-1e-1, 1e-1, 3e-1, 5e-1, 7e-1, -1e-1, 1e-1, 3e-1, 5e-1, 7e-1],
            n_splits=51,
            seed=338
        )
    }

    parser = argparse.ArgumentParser(prog=args[0])
    parser.add_argument("input", choices=sorted(data_gens.keys()), help="Id of the dataset config to be used")
    parser.add_argument("output", help="Path to the output directory")
    parser.add_argument("-n", "--ncpus", type=int, default=1, help="Number of CPUs to be used")
    args = parser.parse_args(args[1:])
    data_gen = data_gens[args.input]
    output_directory = os.path.abspath(args.output)
    n_cpus = args.ncpus

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    prefix_output_filename = os.path.join(output_directory, f"result_{args.input}")

    n_steps = data_gen.get_total_n_splits() * len(method_funcs)

    if n_cpus <= 1:
        ProgressBar.init_monop(total=n_steps)
        for i in range(len(data_gen)):
            id_, splitter = data_gen[i]
            res_df = eval_models(splitter)
            res_df.to_csv(path_or_buf=f"{prefix_output_filename}{id_}.csv", index=False)
    else:
        tasks = [(data_gen, i_dataset, i_split_group) for i_dataset, i_split_group in data_gen.split_in_tasks(n_cpus)]
        res_by_id = collections.defaultdict(list)
        with ProgressBar.init_pool(n_cpus, total=n_steps) as pool:
            for id_, res in pool.imap_unordered(eval_models_multiprocess, tasks):
                res_by_id[id_].append(res)

        for id_, res_df in res_by_id.items():
            res_df = pd.concat(res_df, ignore_index=True).copy(deep=True)
            res_df = res_df \
                .reset_index() \
                .sort_values(by=['split', 'index']) \
                .drop(columns=['index'])
            res_df.to_csv(path_or_buf=f"{prefix_output_filename}{id_}.csv", index=False)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
