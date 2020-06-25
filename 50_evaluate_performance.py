import numpy as np
from sklearn.metrics import mean_squared_error

predictions_y_train = np.load("data/predictions_y_train.npy")
predictions_y_test = np.load("data/predictions_y_test.npy")
predictions_y_val = np.load("data/predictions_y_val.npy")

y_train = np.load("data/y_train.npy")
y_test = np.load("data/y_test.npy")
y_val = np.load("data/y_val.npy")

train_error = np.sqrt(mean_squared_error(predictions_y_train, y_train))

y_val_nan = np.isnan(y_val)
if y_val_nan.all():
    print("All values in y_val are nan, the error is set to np.NaN")
    eval_error = np.NaN
else:
    print(f"{y_val.sum()} out of {len(y_val)} values are nan in the test set. They will be removed." )
    eval_error = np.sqrt(mean_squared_error(predictions_y_val[~y_val_nan], y_val[~y_val_nan]))


y_test_nan = np.isnan(y_test)
if y_test_nan.all():
    print("All values in y_test are nan, the error is set to np.NaN")
    test_error = np.NaN
else:
    print(f"{y_test.sum()} out of {len(y_test)} values are nan in the test set. They will be removed." )
    test_error = np.sqrt(mean_squared_error(predictions_y_test[~y_test_nan], y_test[~y_test_nan]))


print(f"Train errror was: {train_error}, eval_error was {eval_error}, test error was: {test_error}")