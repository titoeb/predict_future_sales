from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import pickle
import xgboost


space = {
    'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
    # A problem with max_depth casted to float instead of int with
    # the hp.quniform method.
    'max_depth':  hp.choice('max_depth', np.arange(1, 25, dtype=int)),
    'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
    'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
    'objective': 'reg:squarederror', 
    'booster': 'gbtree',
    'tree_method': 'exact',
    'silent': 1,
    'seed': 1993
}

# load data

global y_train, X_train, X_pre_val, y_pre_val, X_val, y_val

y_train = np.load("data/y_train.npy")
X_train = np.load("data/X_train.npy")

X_pre_val = np.load("data/X_pre_val.npy")
y_pre_val = np.load("data/y_pre_val.npy")

X_val = np.load("data/X_val.npy")
y_val = np.load("data/y_val.npy")


def score(params):
    print(f"Fit model with params: {params}")

    xgb = XGBRegressor(n_estimators=1000, **params)

    xgb.fit(X_train, y_train, eval_metric="rmse", eval_set=[(X_train, y_train), (X_pre_val, y_pre_val)], verbose=True, early_stopping_rounds=5)

    predictions_y_val = np.clip(xgb.predict(X_val), 0, 20)

    error = np.sqrt(mean_squared_error(predictions_y_val, y_val))

    print(f"error was: {error}")

    return {'loss': error, 'status': STATUS_OK}

trials = Trials()
best = fmin(score, space, algo=tpe.suggest, 
                trials=trials, 
                max_evals=150)

pickle.dump(trials, open("data/trials_2.p", "wb"))