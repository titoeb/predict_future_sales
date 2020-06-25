import pandas as pd
import numpy as np
from helpers import create_mean_encoded_feature, create_lag_feature


# Load the dataset
data = pd.read_parquet("data/train.parquet")
test = pd.read_parquet("data/test.parquet")

data = data.append(test)

lags_to_create = [1, 2, 3, 4]

del test

# Step 1: Create features from other datasets

# Get the item categories.
item_categories = pd.read_csv("data/items.csv")[["item_id", "item_category_id"]]

item_categories = item_categories.astype({"item_category_id": np.int16})

save_index = data.index

data = pd.merge(data, item_categories, how='left', on="item_id")

data.index = save_index

del data["item_id"]

# Get the city names!
shops = pd.read_parquet("data/shops.parquet")

save_index = data.index

data = pd.merge(data, shops, how='left', on="shop_id")

data.index = save_index

del data["shop_id"]

data = data.reset_index()

# Step 2: Create temporal features (lagged target)
data = create_lag_feature(df=data, column2lag="item_cnt_day_sum", lags_to_create=lags_to_create)

data = create_lag_feature(df=data, column2lag="item_cnt_day_mean", lags_to_create=lags_to_create)

data = create_lag_feature(df=data, column2lag="item_price_nanmean", lags_to_create=lags_to_create)

data = create_lag_feature(df=data, column2lag="item_price_nanmedian", lags_to_create=lags_to_create)

data = create_lag_feature(df=data, column2lag="revenue_sum", lags_to_create=lags_to_create)

data = create_lag_feature(df=data, column2lag="revenue_median", lags_to_create=lags_to_create)

# Step 3: Create features based on the item and the shop.
data["shop_id"] = data["shop_id"].astype(np.int32)
data["item_id"] = data["item_id"].astype(np.int32)

# Step 4: Mean encoding
# Date
data = create_mean_encoded_feature(df=data, vars_to_group=["date_block_num"], lags_to_create=lags_to_create)

# Date vs. item
data = create_mean_encoded_feature(df=data, vars_to_group=["date_block_num", "item_id"], lags_to_create=lags_to_create)

# Date vs. shop
data = create_mean_encoded_feature(df=data, vars_to_group=["date_block_num", "shop_id"], lags_to_create=lags_to_create)

# Date vs. category
data = create_mean_encoded_feature(df=data, vars_to_group=['date_block_num', 'item_category_id'], lags_to_create=lags_to_create)

# Date vs. Category vs item
data = create_mean_encoded_feature(df=data, vars_to_group=['date_block_num', 'shop_id', 'item_category_id'], lags_to_create=lags_to_create)

# Date vs. City vs. Item
data = create_mean_encoded_feature(df=data, vars_to_group=['date_block_num', 'item_id', 'city_code'], lags_to_create=lags_to_create)


# Step 5: temporal features:
data['month'] = (data['date_block_num'] % 12).astype(np.int8)

days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])
data['days'] = data['month'].map(days).astype(np.int8)

# Finally create the target!
data["target"] = np.clip(data["item_cnt_day_sum"], 0, 20)

del data["item_cnt_day_sum"]
del data["item_cnt_day_mean"]
del data["item_price_nanmean"]
del data["item_price_nanmedian"]
del data["revenue_sum"]
del data["revenue_median"]

print(data.dtypes)


# Set date as index
data = data.set_index(["date_block_num", "shop_id", "item_id"])

# Store dataset
data.to_parquet("data/data.parquet")