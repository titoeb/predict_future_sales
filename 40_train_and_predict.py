import numpy as np
from xgboost import XGBRegressor


xgb = XGBRegressor()

y_train = np.load("data/y_train.npy")
print(y_train.dtype)

X_train = np.load("data/X_train.npy")
print(X_train.dtype)

xgb.fit(X_train, y_train)
predictions_y_train = np.clip(xgb.predict(X_train), 0, 20)

del X_train, y_train

X_test = np.load("data/X_test.npy")
predictions_y_test = np.clip(xgb.predict(X_test), 0, 20)

np.save("data/predictions_y_train.npy", predictions_y_train)
np.save("data/predictions_y_test.npy", predictions_y_test)