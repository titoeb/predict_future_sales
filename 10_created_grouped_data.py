import pandas as pd
import numpy as np

# Load the dataset
train = pd.read_parquet("data/sales_train.parquet")

# Append all possible shops.
all_shops = np.unique(pd.read_parquet("data/shops.parquet").shop_id)
all_items = np.unique(pd.read_parquet("data/items.parquet").item_id)

# Create all potential shop-item combinations to append to the data.
all_items_shop_shops, all_items_shop_items = np.meshgrid(all_shops, all_items)
all_items_shop_shops = all_items_shop_shops.flatten()
all_items_shop_items = all_items_shop_items.flatten()


all_date_block_num = np.unique(train.date_block_num)

for date_block_num in all_date_block_num:
    # Create dataframe to append.
        print(f"Appending date_block_num: {int(date_block_num)}")
        this_df = pd.DataFrame({
            'shop_id': all_items_shop_shops,
            'item_id': all_items_shop_items,
            'item_price': np.NaN,
            'item_cnt_day': 0,
            'date_block_num': int(date_block_num),
            'revenue': 0
        })
        n_rows = train.shape[0]
        train = train.append(this_df)
        print(f"Appended {train.shape[0] - n_rows} observations to train.")

grouped_train = train.groupby(["date_block_num", "shop_id", "item_id"])
train_monthly = grouped_train.agg({"item_price": [np.nanmean, np.nanmedian], "revenue": [np.sum, np.median], "item_cnt_day": [np.sum, np.mean]})
train_monthly.columns = ["_".join(x) for x in train_monthly.columns.ravel()]


train_monthly = train_monthly.astype({col: np.float32 for col in train_monthly.columns})
print(train_monthly.dtypes)

# Recreate date column
train_monthly = train_monthly.reset_index()

# Set date as index
train_monthly = train_monthly.set_index(["date_block_num", "shop_id", "item_id"])

# Store data
train_monthly.to_parquet("data/train.parquet")