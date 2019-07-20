from typing import Dict, List

import hyperopt
import lightgbm as lgb
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from util import Config, log, timeit
import random


@timeit
def train(X: pd.DataFrame, y: pd.Series, config: Config):
    train_lightgbm(X, y, config)


@timeit
def predict(X: pd.DataFrame, config: Config) -> List:
    #feature = config["importance"]
    #X = X.iloc[:, feature]
    preds = predict_lightgbm(X, config)
    return preds


@timeit
def validate(preds, y_path) -> np.float64:
    score = roc_auc_score(pd.read_csv(y_path)['label'].values, preds)
    log("Score: {:0.4f}".format(score))
    return score


@timeit
def train_lightgbm(X: pd.DataFrame, y: pd.Series, config: Config):

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "seed": 1,
        "num_threads": 4
    }

    X_sample, y_sample = data_sample_v2(X, y, 30000)
    hyperparams = hyperopt_lightgbm(X_sample, y_sample, params, config)

    # 样本采样，节约时间
    pos_len=len(y[y==1])
    neg_len=len(y[y==0])
    if pos_len < neg_len:
        ratio= neg_len/pos_len
        mark = 1
    else:
        mark = 0
        ratio = pos_len / neg_len
    if ratio > 30.0:
        config["ensemble"] = True
    else:
        config["ensemble"] = False

    if config["ensemble"] is True:
        mark_index = y[y == mark].index.values.tolist()
        nomark_index = y[y == (1-mark)].index.values.tolist()
        len_nomark = len(nomark_index)
        random.shuffle(nomark_index)
        d1_index = mark_index + nomark_index[:int(0.33*len_nomark)]
        d2_index = mark_index + nomark_index[int(0.33 * len_nomark):int(0.67 * len_nomark)]
        d3_index = mark_index + nomark_index[int(0.67 * len_nomark):]

        ratio1 = max(5, int(ratio // 5))
        X1, y1 = X.loc[d1_index], y.loc[d1_index]
        X1, y1 = data_sample_v2(X1, y1, len(mark_index) * ratio1)
        X1, X_val1, y1, y_val1 = data_split(X1, y1)
        train_data1 = lgb.Dataset(X1, label=y1)
        valid_data1 = lgb.Dataset(X_val1, label=y_val1)
        del X1, X_val1, y1, y_val1
        config["model1"] = lgb.train({**params, **hyperparams},
                                    train_data1,
                                    500,
                                    valid_data1,
                                    early_stopping_rounds=30,
                                    verbose_eval=100)

        ratio2 = max(5, int(ratio // 5))
        X2, y2 = X.loc[d2_index], y.loc[d2_index]
        X2, y2 = data_sample_v2(X2, y2, len(mark_index) * ratio2)
        X2, X_val2, y2, y_val2 = data_split(X2, y2)
        train_data2 = lgb.Dataset(X2, label=y2)
        valid_data2 = lgb.Dataset(X_val2, label=y_val2)
        del X2, X_val2, y2, y_val2
        config["model2"] = lgb.train({**params, **hyperparams},
                                     train_data2,
                                     500,
                                     valid_data2,
                                     early_stopping_rounds=30,
                                     verbose_eval=100)

        ratio3 = max(5, int(ratio // 5))
        X3, y3 = X.loc[d3_index], y.loc[d3_index]
        X3, y3 = data_sample_v2(X3, y3, len(mark_index) * ratio3)
        X3, X_val3, y3, y_val3 = data_split(X3, y3)
        train_data3 = lgb.Dataset(X3, label=y3)
        valid_data3 = lgb.Dataset(X_val3, label=y_val3)
        del X3, X_val3, y3, y_val3
        config["model3"] = lgb.train({**params, **hyperparams},
                                     train_data3,
                                     500,
                                     valid_data3,
                                     early_stopping_rounds=30,
                                     verbose_eval=100)
    else:
        ratio = max(5, int(ratio // 5))
        X, y = data_sample_v2(X, y, len(y[y == mark]) * ratio)

        X, X_val, y, y_val = data_split(X, y, 0.1)
        train_data = lgb.Dataset(X, label=y)
        valid_data = lgb.Dataset(X_val, label=y_val)
        '''
        # 预训练，筛选特征
        feature_clf = lgb.train({**params, **hyperparams},
                  train_data,
                  500,
                  valid_data,
                  early_stopping_rounds=30,
                  verbose_eval=100)
        f_imp=feature_clf.feature_importance()
        print(f_imp)
        print(X.columns[np.argsort(f_imp)])  # 从小到大
        score = feature_clf.feature_importance() / feature_clf.feature_importance().sum()
        if (len(feature_clf.feature_importance())) > 170:
            feature = list(np.where(score > np.percentile(score, 60))[0])
        else:
            feature = list(np.where(score > np.percentile(score, 50))[0])
    
        config["importance"] = feature
        X = X.iloc[:, feature]
        print("Important Feature:")
        print(X.columns)
    
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=2)
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)
        '''

        config["model"] = lgb.train({**params, **hyperparams},
                                    train_data,
                                    700,
                                    valid_data,
                                    early_stopping_rounds=30,
                                    verbose_eval=100)


@timeit
def predict_lightgbm(X: pd.DataFrame, config: Config) -> List:
    if config["ensemble"] is True:
        pre = (np.array(config["model1"].predict(X))+np.array(config["model2"].predict(X))+np.array(config["model3"].predict(X)))/3.0
        return pre.tolist()
    return config["model"].predict(X)


@timeit
def hyperopt_lightgbm(X: pd.DataFrame, y: pd.Series, params: Dict, config: Config):
    X_train, X_val, y_train, y_val = data_split(X, y, test_size=0.5)
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)

    space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.5)),
        "max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6]),
        "num_leaves": hp.choice("num_leaves", np.linspace(10, 200, 50, dtype=int)),
        "feature_fraction": hp.quniform("feature_fraction", 0.5, 1.0, 0.1),
        "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 1.0, 0.1),
        "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 50, 10, dtype=int)),
        "reg_alpha": hp.uniform("reg_alpha", 0, 2),
        "reg_lambda": hp.uniform("reg_lambda", 0, 2),
        "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
    }

    def objective(hyperparams):
        model = lgb.train({**params, **hyperparams}, train_data, 300,
                          valid_data, early_stopping_rounds=30, verbose_eval=0)

        score = model.best_score["valid_0"][params["metric"]]

        # in classification, less is better
        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                         algo=tpe.suggest, max_evals=30, verbose=1,
                         rstate=np.random.RandomState(1))

    hyperparams = space_eval(space, best)
    log(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")
    return hyperparams


def data_split(X: pd.DataFrame, y: pd.Series, test_size: float=0.2):
    #  -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    return train_test_split(X, y, test_size=test_size, random_state=1)

def data_split_v2(X: pd.DataFrame, y: pd.Series, test_size: float=0.2):  # 按时间切分版本
    #  -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    samples_num = X.shape[0]
    # X["sort_index"]=range(samples_num)
    split_num = int((1.0-test_size)*samples_num)
    '''
    X_train, X_val, y_train, y_val = train_test_split(X.iloc[0:split_num, :], y.iloc[0:split_num], test_size=test_size, random_state=1)
    X_val = pd.concat([X_val, X.iloc[split_num:, :]])
    X_train.sort_values("sort_index", inplace=True)
    y_train = y_train.loc[X_train.index]
    X_train.drop("sort_index", axis=1, inplace=True)
    X_val.drop("sort_index", axis=1, inplace=True)
    X.drop("sort_index", axis=1, inplace=True)
    '''
    X_train, X_val, y_train, y_val = X.iloc[0:split_num, :], X.iloc[split_num:, :], y.iloc[0:split_num], y.iloc[split_num:]
    del X
    del y
    return X_train, X_val, y_train, y_val


def data_sample_v2(X: pd.DataFrame, y: pd.Series, nrows: int=5000):
    # -> (pd.DataFrame, pd.Series):
    if len(X) > nrows:
        pos_len=len(y[y==1])
        neg_len=len(y[y==0])
        if pos_len < neg_len:
            mark = 1
        else:
            mark = 0
        y_1 = y[y == mark]
        y_0 = y[y == (1-mark)]
        half_num = nrows // 2
        if len(y_1) >= half_num:
            y_1_sample = y_1.sample(half_num)
            y_0_sample = y_0.sample(half_num)
        else:
            y_1_sample = y_1
            y_0_sample = y_0.sample(2 * half_num - len(y_1))

        y_sample = pd.concat([y_0_sample, y_1_sample])
        X_sample = X.loc[y_sample.index]
    else:
        X_sample = X
        y_sample = y

    return X_sample, y_sample

