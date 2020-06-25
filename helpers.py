import pandas as pd
from typing import List, Optional

def create_mean_encoded_feature(df: pd.DataFrame, vars_to_group: List[str], lags_to_create: List[int]) -> pd.DataFrame:
    grouped_df = df.groupby(vars_to_group).agg({'item_cnt_day_sum': ['mean']})

    new_var_name = f"{'_'.join(vars_to_group)}_avg_item_cnt"

    grouped_df.columns=[new_var_name]
    grouped_df = grouped_df.reset_index()

    for lag in lags_to_create:
        this_grouped = grouped_df.copy()
        this_grouped["date_block_num"] = this_grouped["date_block_num"] + lag
        this_grouped = this_grouped.rename(columns={new_var_name:f"{new_var_name}_lag_{lag}"})

        df = pd.merge(df, this_grouped, how="left", on=vars_to_group)

        # Finally fill missing values with 0.0
        df.loc[:, f"{new_var_name}_lag_{lag}"] = df.loc[:, f"{new_var_name}_lag_{lag}"].fillna(0.0)

    return df

def create_lag_feature(df: pd.DataFrame, column2lag: str, lags_to_create: List[int], col_name: Optional[str]=None) -> pd.DataFrame:
    data = df.loc[:, ['date_block_num','shop_id','item_id', column2lag]]
    for lag in lags_to_create:
        rel_data = data.copy()
        rel_data["date_block_num"] = rel_data["date_block_num"] + lag
        if col_name is None:
            rel_data = rel_data.rename(columns={column2lag: f"{column2lag}_lag_{lag}"})
        else:
            rel_data = rel_data.rename(columns={column2lag: col_name})
        df = pd.merge(df, rel_data, how="left", on=['date_block_num','shop_id','item_id'])

        # Finally fill missing values with 0.0
        if col_name is None:
            df.loc[:, f"{column2lag}_lag_{lag}"] = df.loc[:, f"{column2lag}_lag_{lag}"].fillna(0.0)
        else:
            df.loc[:, col_name] = df.loc[:, col_name].fillna(0.0)

    return df