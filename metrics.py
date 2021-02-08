import numpy as np
import pandas as pd


def metric_auuc_random(df, outcome, treatment, weight=None):
    weight = df[weight] if weight is not None else 1
    r_t = (df[outcome] * weight * df[treatment]).sum()
    r_c = (df[outcome] * weight * (1 - df[treatment])).sum()
    n_t = (weight * df[treatment]).sum()
    n_c = (weight * (1 - df[treatment])).sum()
    return (r_t - r_c * n_t / n_c) / 2


def metric_auuc(df, outcome, treatment, u_pred, weight=None):
    assert type(u_pred) is np.ndarray
    assert len(u_pred.shape) == 1
    assert len(u_pred) == len(df)

    if weight is None:
        weight = 'loc_weight'
        sub_df = pd.DataFrame(np.ones(len(df)), index=df.index, columns=[weight])
    else:
        sub_df = df[[weight]].copy()

    tot_weights = sub_df[weight].sum()

    sub_df['r_t'] = df[outcome] * sub_df[weight] * df[treatment]
    sub_df['r_c'] = df[outcome] * sub_df[weight] * (1 - df[treatment])
    sub_df['n_t'] = sub_df[weight] * df[treatment]
    sub_df['n_c'] = sub_df[weight] * (1 - df[treatment])
    sub_df['u_pred'] = u_pred

    group_df = sub_df.groupby('u_pred')[[weight, 'r_t', 'r_c', 'n_t', 'n_c']].sum()
    group_df.sort_values(by='u_pred', ascending=False, inplace=True)

    ratios = (group_df['n_t'].cumsum() / group_df['n_c'].cumsum()).replace(np.inf, 0)
    bars = ((group_df['r_t'].cumsum() - group_df['r_c'].cumsum() * ratios) * group_df[weight]).sum()
    triangles = ((group_df['r_t'] - group_df['r_c'] * ratios) * group_df[weight]).sum() / 2
    return (bars - triangles) / tot_weights


def metric_rmse(df, uplift, u_pred, weight='weight'):
    assert type(u_pred) is np.ndarray
    assert len(u_pred.shape) == 1
    weights = df[weight].values
    gt = df[uplift].values
    assert u_pred.shape == gt.shape
    return np.sqrt((weights * (u_pred - gt)**2).sum() / weights.sum())
