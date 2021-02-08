import numpy as np
from utils import split_treatment_control


def ite_best(train_df, test_df, features, outcome, treatment):
    """
    Best possible ITE learnt model (i.e. without exploiting the observable interference).
    Only available for synthetic datasets.
    """
    train_t_df, train_c_df = split_treatment_control(train_df, treatment)

    by_feat_t = train_t_df.groupby(features)[outcome].mean()
    by_feat_c = train_c_df.groupby(features)[outcome].mean()
    by_feat = by_feat_t - by_feat_c

    return test_df[features].join(by_feat, on=features)[outcome].values


def ite_2m(train_df, test_df, features, outcome, treatment, clf_t, clf_c):
    """
    Individual Treatment Effects with Two Models (2T)
    """
    np.random.seed(0)

    train_t_df, train_c_df = split_treatment_control(train_df, treatment)

    clf_t_trained = clf_t.fit(train_t_df[features], train_t_df[outcome])
    clf_c_trained = clf_c.fit(train_c_df[features], train_c_df[outcome])

    test_f_df = test_df[features]
    return clf_t_trained.predict_proba(test_f_df)[:, 1] - clf_c_trained.predict_proba(test_f_df)[:, 1]


def mite_2m(train_df, test_df, features, outcome, treatment, exposure, clf_t, clf_c, clf_er):
    """
    Post-Mediation Individual Treatment Effects with Two Models (2T)
    """
    np.random.seed(0)

    train_exposed_df, train_not_exposed_df = split_treatment_control(train_df, exposure)
    train_t_df, _ = split_treatment_control(train_df, treatment)

    clf_t_trained = clf_t.fit(train_exposed_df[features], train_exposed_df[outcome])
    clf_c_trained = clf_c.fit(train_not_exposed_df[features], train_not_exposed_df[outcome])
    clf_er_trained = clf_er.fit(train_t_df[features], train_t_df[exposure])

    test_f_df = test_df[features]
    return clf_er_trained.predict_proba(test_f_df)[:, 1] * \
        (clf_t_trained.predict_proba(test_f_df)[:, 1] - clf_c_trained.predict_proba(test_f_df)[:, 1])


def _make_sdr(train_df, features, outcome, treatment):
    train_t_df, train_c_df = split_treatment_control(train_df, treatment)
    train_t_f_df = train_t_df[features]
    train_c_f_df = train_c_df[features]

    data_t = np.hstack((train_t_f_df, train_t_f_df, np.zeros(train_t_f_df.shape)))
    data_c = np.hstack((train_c_f_df, np.zeros(train_c_f_df.shape), train_c_f_df))
    data_full = np.concatenate((data_t, data_c))

    data_full_y = np.concatenate((train_t_df[outcome], train_c_df[outcome]))
    return data_full, data_full_y


def ite_sdr(train_df, test_df, features, outcome, treatment, clf):
    """
    Individual Treatment Effects with Shared Data Representation (SDR)
    """
    np.random.seed(0)

    train_f, train_y = _make_sdr(train_df, features, outcome, treatment)

    clf_trained = clf.fit(train_f, train_y)

    test_f_df = test_df[features]
    y_pred_t = clf_trained.predict_proba(np.hstack((test_f_df, test_f_df, np.zeros(test_f_df.shape))))[:, 1]
    y_pred_c = clf_trained.predict_proba(np.hstack((test_f_df, np.zeros(test_f_df.shape), test_f_df)))[:, 1]
    return y_pred_t - y_pred_c


def mite_sdr(train_df, test_df, features, outcome, treatment, exposure, clf, clf_er):
    """
    Post-Mediation Individual Treatment Effects with Shared Data Representation (SDR)
    """
    np.random.seed(0)

    train_f, train_y = _make_sdr(train_df, features, outcome, exposure)
    train_t_df, _ = split_treatment_control(train_df, treatment)

    clf_trained = clf.fit(train_f, train_y)
    clf_er_trained = clf_er.fit(train_t_df[features], train_t_df[exposure])

    test_f_df = test_df[features]
    y_pred_t = clf_trained.predict_proba(np.hstack((test_f_df, test_f_df, np.zeros(test_f_df.shape))))[:, 1]
    y_pred_c = clf_trained.predict_proba(np.hstack((test_f_df, np.zeros(test_f_df.shape), test_f_df)))[:, 1]
    return clf_er_trained.predict_proba(test_f_df)[:, 1] * (y_pred_t - y_pred_c)
