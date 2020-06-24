import numpy as np
from xgboost import XGBRegressor


xgb = XGBRegressor(n_estimators=1000)

y_train = np.load("data/y_train.npy")
print(y_train.dtype)

X_train = np.load("data/X_train.npy")
print(X_train.dtype)

X_val = np.load("data/X_val.npy")
print(X_val.dtype)

y_val = np.load("data/y_val.npy")
print(y_val.dtype)

xgb.fit(X_train, y_train, eval_metric="rmse", eval_set=[(X_val, y_val)], verbose=True, early_stopping_rounds=5)
predictions_y_train = np.clip(xgb.predict(X_train), 0, 20)

del X_train, y_train

X_test = np.load("data/X_test.npy")
predictions_y_test = np.clip(xgb.predict(X_test), 0, 20)

np.save("data/predictions_y_train.npy", predictions_y_train)
np.save("data/predictions_y_test.npy", predictions_y_test)