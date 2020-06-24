import pandas as pd
import numpy as np
from helpers import create_mean_encoded_feature


# Load the dataset
data = pd.read_parquet("data/train.parquet")
test = pd.read_parquet("data/test.parquet")

data = data.append(test)

del test

# Step 1: Create features from other datasets

# Get the item categories.
item_categories = pd.read_csv("data/items.csv")[["item_id", "item_category_id"]]

item_categories = item_categories.astype({"item_category_id": np.int16})

save_index = data.index

data = pd.merge(data, item_categories, how='left', on="item_id")

data.index = save_index

# ????? Does this make sense?
#data.drop(columns=["item_id"], inplace=True)

# Get the city names!
shops = pd.read_parquet("data/shops.parquet")

save_index = data.index

data = pd.merge(data, shops, how='left', on="shop_id")

data.index = save_index

# Step 2: Create temporal features (lagged target)
data["item_cnt_day_sum_lag_1"] = data.groupby(level=["shop_id", "item_id"])["item_cnt_day_sum"].shift(1)
data["item_cnt_day_sum_lag_2"] = data.groupby(level=["shop_id", "item_id"])["item_cnt_day_sum"].shift(2)

data["item_cnt_day_mean_lag_1"] = data.groupby(level=["shop_id", "item_id"])["item_cnt_day_mean"].shift(1)
data["item_cnt_day_mean_lag_2"] = data.groupby(level=["shop_id", "item_id"])["item_cnt_day_mean"].shift(2)

data["item_price_nanmedian_lag_1"] = data.groupby(level=["shop_id", "item_id"])["item_price_nanmedian"].shift(1)
data["item_price_nanmedian_lag_2"] = data.groupby(level=["shop_id", "item_id"])["item_price_nanmedian"].shift(2)

data["item_price_nanmean_lag_1"] = data.groupby(level=["shop_id", "item_id"])["item_price_nanmean"].shift(1)
data["item_price_nanmean_lag_2"] = data.groupby(level=["shop_id", "item_id"])["item_price_nanmean"].shift(2)

data["revenue_sum_lag_1"] = data.groupby(level=["shop_id", "item_id"])["revenue_sum"].shift(1)
data["revenue_sum_lag_2"] = data.groupby(level=["shop_id", "item_id"])["revenue_sum"].shift(2)

data["revenue_median_lag_1"] = data.groupby(level=["shop_id", "item_id"])["revenue_median"].shift(1)
data["revenue_median_lag_2"] = data.groupby(level=["shop_id", "item_id"])["revenue_median"].shift(2)

# Step 3: Create features based on the item and the shop.
data["shop_id"] = data.index.get_level_values("shop_id").astype(np.int32)
data["item_id"] = data.index.get_level_values("item_id").astype(np.int32)

# Step 4: Mean encoding
# Date
data = create_mean_encoded_feature(df=data, vars_to_group=["date_block_num"], lags_to_create=[1, 2])

# Date vs. item
data = create_mean_encoded_feature(df=data, vars_to_group=["date_block_num", "item_id"], lags_to_create=[1, 2])

# Date vs. shop
data = create_mean_encoded_feature(df=data, vars_to_group=["date_block_num", "shop_id"], lags_to_create=[1, 2])

# Date vs. category
data = create_mean_encoded_feature(df=data, vars_to_group=['date_block_num', 'item_category_id'], lags_to_create=[1, 2])

# Date vs. Category vs item
data = create_mean_encoded_feature(df=data, vars_to_group=['date_block_num', 'shop_id', 'item_category_id'], lags_to_create=[1, 2])

# Date vs. City vs. Item
data = create_mean_encoded_feature(df=data, vars_to_group=['date_block_num', 'item_id', 'city_code'], lags_to_create=[1, 2])

# Step 5: temporal features:
matrix['month'] = matrix['date_block_num'] % 12

days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])
matrix['days'] = matrix['month'].map(days).astype(np.int8)

print(data.dtypes)

# Store dataset
data.to_parquet("data/data.parquet")