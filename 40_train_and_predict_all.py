import numpy as np
from xgboost import XGBRegressor

params = {'colsample_bytree': 0.5, 'eta': 0.025, 'gamma': 0.9, 'max_depth': 11, 'min_child_weight': 5.0, 'subsample': 1.0,  'seed': 1993,    'objective': 'reg:squarederror', 
    'booster': 'gbtree',
    'tree_method': 'exact',
    'silent': 1,}

xgb = XGBRegressor(n_estimators=116, **params)

y_train = np.load("data/y_train.npy")
print(y_train.dtype)

X_train = np.load("data/X_train.npy")
print(X_train.dtype)

X_pre_val = np.load("data/X_pre_val.npy")
print(X_pre_val.dtype)

y_pre_val = np.load("data/y_pre_val.npy")
print(y_pre_val.dtype)


X_val = np.load("data/X_val.npy")
print(X_val.dtype)

y_val = np.load("data/y_val.npy")
print(y_val.dtype)

X_train = np.concatenate([X_train, X_pre_val, X_val], axis=0)
y_train = np.concatenate([y_train, y_pre_val, y_val])

xgb.fit(X_train, y_train)

predictions_y_train = np.clip(xgb.predict(X_train), 0, 20)

predictions_y_val = np.clip(xgb.predict(X_val), 0, 20)

del X_train, y_train

X_test = np.load("data/X_test.npy")
predictions_y_test = np.clip(xgb.predict(X_test), 0, 20)

np.save("data/predictions_y_train.npy", predictions_y_train)
np.save("data/predictions_y_test.npy", predictions_y_test)
np.save("data/predictions_y_val.npy", predictions_y_val)