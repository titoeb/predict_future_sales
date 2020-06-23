import pandas as pd
import numpy as np


# PREPARE SALES_TRAIN
# Load the dataset
train = pd.read_csv("data/sales_train.csv")

# CLEANING
# A few data cleaning steps here.
# Only select observations with fewer than 1000 daily counts

train = train.loc[train.item_cnt_day<1000, :]

# Only select prices up to 100000
train = train.loc[train.item_price < 80000, :]

# Replace negative prices with zero.
train.loc[train.item_price < 0, :] = 0.0

train["revenue"] = train["item_cnt_day"] * train["item_price"]

train.loc[train.shop_id == 0, "shop_id"] = 57
train.loc[train.shop_id == 1, "shop_id"] = 58
train.loc[train.shop_id == 10, "shop_id"] = 11

train.drop(columns = ["date"], inplace=True)

train.loc[:, "date_block_num"] = train.loc[:, "date_block_num"].astype(np.int32)
train.loc[:, "shop_id"] = train.loc[:, "shop_id"].astype(np.int32)
train.loc[:, "item_id"] = train.loc[:, "item_id"].astype(np.int32)

# Store final data
train.to_parquet("data/sales_train.parquet")


# PREPARE TEST 
# Step 1: Append the test data
test = pd.read_csv("data/test.csv")

test.drop(columns = ["ID"], inplace=True)

test["date_block_num"] = train.date_block_num.max() + 1

# Duplicated shops: 0 -> 57, 1 -> 58, 10 -> 11
test.loc[test.shop_id == 0, "shop_id"] = 57
test.loc[test.shop_id == 1, "shop_id"] = 58
test.loc[test.shop_id == 10, "shop_id"] = 11

# Add columns that will exist in train
needed_colums = ['item_price_nanmean', 'item_price_nanmedian', 'revenue_sum',
       'revenue_median', 'item_cnt_day_sum', 'item_cnt_day_mean']

for column in needed_colums:
    test[column] = np.NaN

# Set index for test
test.set_index(["date_block_num", "shop_id", "item_id"], inplace=True)

test = test.astype({col: np.float32 for col in test.columns})
print(test.dtypes)

# Store test
test.to_parquet("data/test.parquet")

# PREPARE SHOPS
shops = pd.read_csv("data/shops.csv")
shops = shops.loc[(shops.shop_id != 0) & (shops.shop_id != 1) & (shops.shop_id != 10), :]

# Create City from shop name
shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])
shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'
shops['city_code'] = LabelEncoder().fit_transform(shops['city']).astype(np.int16)
shops = shops[['shop_id','city_code']]

# Extract the name of the city from the beginning of the shops.
shops.to_parquet("data/shops.parquet")


# PREPARE ITEMS
items = pd.read_csv("data/items.csv")
items.to_parquet("data/items.parquet")