import numpy as np
from xgboost import XGBRegressor


params = {}

xgb = XGBRegressor(n_estimators=1000, **params)

y_train = np.load("data/y_train.npy")
print(y_train.dtype)

X_train = np.load("data/X_train.npy")
print(X_train.dtype)

X_pre_val = np.load("data/X_pre_val.npy")
print(X_val.dtype)

y_pre_val = np.load("data/y_pre_val.npy")
print(y_val.dtype)

X_train = np.stack([X_train, X_pre_val])
y_train = np.stack([y_train, y_pre_val])


X_val = np.load("data/X_val.npy")
print(X_val.dtype)

y_val = np.load("data/y_val.npy")
print(y_val.dtype)

xgb.fit(X_train, y_train, eval_metric="rmse", eval_set=[(X_train, y_train), (X_val, y_val)], verbose=True, early_stopping_rounds=3)

predictions_y_train = np.clip(xgb.predict(X_train), 0, 20)

predictions_y_val = np.clip(xgb.predict(X_val), 0, 20)

del X_train, y_train

X_test = np.load("data/X_test.npy")
predictions_y_test = np.clip(xgb.predict(X_test), 0, 20)

np.save("data/predictions_y_train.npy", predictions_y_train)
np.save("data/predictions_y_test.npy", predictions_y_test)
np.save("data/predictions_y_val.npy", predictions_y_val)