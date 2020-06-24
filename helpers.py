import pandas as pd

from typing import List

def create_mean_encoded_feature(df: pd.DataFrame, vars_to_group: List[str], lags_to_create: List[int]) -> pd.DataFrame:
    grouped_df = df.groupby(vars_to_group).agg({'item_cnt_day_sum': ['mean']})

    new_var_name = f"{'_'.join(vars_to_group)}_avg_item_cnt"

    grouped_df.columns=[new_var_name]
    grouped_df.reset_index()

    for lag in lags_to_create:
        this_grouped = grouped_df.copy()
        this_grouped["date_block_num"] = this_grouped["date_block_num"] + lag
        this_grouped = this_grouped.rename(columns={new_var_name: f"new_var_name_lag_{lag}"})

        df = df.merge(df, this_grouped, how="left", on=vars_to_group)
    return df


