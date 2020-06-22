import pandas as pd
import numpy as np


# Load the dataset
data = pd.read_parquet("data/train.parquet")
test = pd.read_parquet("data/test.parquet")

data = data.append(test)

del test

# Step 2: Create features from other datasets
item_categories = pd.read_csv("data/items.csv")[["item_id", "item_category_id"]]

item_categories = item_categories.astype({"item_category_id": np.int32})

save_index = data.index

data = pd.merge(data, item_categories, how='left', on="item_id")

data.index = save_index

data.drop(columns=["item_id"], inplace=True)

print(data.dtypes)

# Step 3: Create temporal features (lagged target)
data["item_price_mean_lag_1"] = data.groupby(level=["shop_id", "item_id"])["item_cnt_day_sum"].shift(1)
data["item_price_mean_lag_2"] = data.groupby(level=["shop_id", "item_id"])["item_cnt_day_sum"].shift(2)
data["item_price_mean_lag_3"] = data.groupby(level=["shop_id", "item_id"])["item_cnt_day_sum"].shift(3)

data["item_cnt_day_mean_lag_1"] = data.groupby(level=["shop_id", "item_id"])["item_cnt_day_mean"].shift(1)
data["item_cnt_day_mean_lag_2"] = data.groupby(level=["shop_id", "item_id"])["item_cnt_day_mean"].shift(2)
data["item_cnt_day_mean_lag_3"] = data.groupby(level=["shop_id", "item_id"])["item_cnt_day_mean"].shift(3)

data["item_price_median_lag_1"] = data.groupby(level=["shop_id", "item_id"])["item_price_nanmedian"].shift(1)
data["item_price_median_lag_2"] = data.groupby(level=["shop_id", "item_id"])["item_price_nanmedian"].shift(2)
data["item_price_median_lag_3"] = data.groupby(level=["shop_id", "item_id"])["item_price_nanmedian"].shift(3)

data["item_price_nanmean_lag_1"] = data.groupby(level=["shop_id", "item_id"])["item_price_nanmean"].shift(1)
data["item_price_nanmean_lag_2"] = data.groupby(level=["shop_id", "item_id"])["item_price_nanmean"].shift(2)
data["item_price_nanmean_lag_3"] = data.groupby(level=["shop_id", "item_id"])["item_price_nanmean"].shift(3)

data["revenue_sum_lag_1"] = data.groupby(level=["shop_id", "item_id"])["revenue_sum"].shift(1)
data["revenue_sum_lag_2"] = data.groupby(level=["shop_id", "item_id"])["revenue_sum"].shift(2)
data["revenue_sum_lag_3"] = data.groupby(level=["shop_id", "item_id"])["revenue_sum"].shift(3)

data["revenue_median_lag_1"] = data.groupby(level=["shop_id", "item_id"])["revenue_median"].shift(1)
data["revenue_median_lag_2"] = data.groupby(level=["shop_id", "item_id"])["revenue_median"].shift(2)
data["revenue_median_lag_3"] = data.groupby(level=["shop_id", "item_id"])["revenue_median"].shift(3)

# Step 5: Add periodic sales features (when was the item last bought)

# Step 6: Create features based on the item and the shop.
data["shop_id"] = data.index.get_level_values("shop_id").astype(np.int32)
data["item_id"] = data.index.get_level_values("item_id").astype(np.int32)

# Step 7: Extract city name and code from shop name

# Store dataset
data.to_parquet("data/data.parquet")