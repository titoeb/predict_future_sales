import pandas as pd
import numpy as np

train_start = 2
train_end = 33
test_end  = 34


# Load data
data = pd.read_parquet("data/data.parquet")

# Create the target columns that is the value of the target next month:
#index_next_month = data.index.get_level_values("date").shift(freq="MS")

target = data.groupby(level=["shop_id", "item_id"])["item_cnt_day_sum"].shift(-1)

data["target"] = np.clip(target.values, 0, 20)

# Split train and test based on time.
train = data.loc[(data.index.get_level_values("date_block_num") <= train_end) & (data.index.get_level_values("date_block_num") >= train_start) , :]
test = data.loc[(data.index.get_level_values("date_block_num") > train_end) & (data.index.get_level_values("date_block_num") <= test_end), :]

del data

# Now we need to treat train and test differently
# For train create the target column, which is the value of the target the next month, we can only keep data points that have a target
y_train = train["target"].values.astype(np.float32)
X_train = train.drop(columns=["target"]).values.astype(np.float32)

del train

# For train we will exclude nas
y_train_is_na = np.isnan(y_train)

print(f"The target in the train set contains {y_train_is_na.sum()} NaN values out of {len(y_train)} values.")

y_train = y_train[~y_train_is_na]
X_train = X_train[~y_train_is_na, :]

y_test= test["target"].values.astype(np.float32)
X_test = test.drop(columns=["target"]).values.astype(np.float32)

del test

print(f"The target in the test set contains {np.isnan(y_test).sum()} NaN values out of {len(y_test)} values.")

# We will store both datasets as numpy arrays.
np.save("data/y_train.npy", y_train)
np.save("data/X_train.npy", X_train)
np.save("data/y_test.npy", y_test)
np.save("data/X_test.npy", X_test)