import polars as pl
import pandas as pd
import os
from sklearn.model_selection import train_test_split


# helper function from contest creator starter notebook
def set_table_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    # implement here all desired dtypes for tables
    # the following is just an example
    for col in df.columns:
        # last letter of column name will help you determine the type
        if col[-1] in ("P", "A"):
            df = df.with_columns(pl.col(col).cast(pl.Float64).alias(col))

    return df



# helper function to create a list of formatted aggregations given features and aggregation types
def create_agg_list(features, aggs):
    agg_features = list(filter(lambda x: x not in ['target','case_id', "num_group1", "num_group2"], features))
    nested_lists = [
        [pl.max(f).name.suffix("_max") if "agg_max" in aggs else None, 
          pl.min(f).name.suffix("_min") if "agg_min" in aggs else None, 
          pl.median(f).name.suffix("_median") if "agg_median" in aggs else None
         ] for f in agg_features]
    agg_list = []
    for l in nested_lists:
        agg_list.extend(l)
    agg_list = list(filter(lambda x: x is not None, agg_list))
    return agg_list



# this function loads a given dataset, performs aggregation if depth >0, and returns train and test sets
def load_df(name, depth=0, features=None, feature_types=None, aggs=["agg_max"], description=None):
    # TODO: cater to another agg type: selecting only the first (i.e., group1=0 and group2=0)
    dataPath = "/kaggle/input/home-credit-credit-risk-model-stability/csv_files/"
    if features is not None:
        if "case_id" not in features:
            features = ['case_id']+features
    
    results = []
    for split in ["train", "test"]:
        # load file; it may have been partitioned into multiple csvs
        filenames = os.listdir(dataPath + f"{split}")
        matching_filenames = [f for f in filenames if f.startswith(f"{split}_{name}")]
        # load all partitions
        df_list = []
        for file in matching_filenames:
            df_list.append(pl.read_csv(dataPath+split+"/"+file).pipe(set_table_dtypes))
        df = pl.concat(df_list, how="vertical_relaxed")
        
        # select the columns specified by features and feature_types
        if features is None:
            features = df.columns
        if split == "test":
            features= list(filter(lambda x: x != "target", features))
        if feature_types is not None:
            features = ['case_id']+[f for f in features if f[-1] in feature_types]
        df = df.select(features)
        
        # if depth > 0, aggregate
        if depth > 0:
            # determine the aggregations to perform
            agg_list = create_agg_list(features, aggs)
            # groupby and aggregate
            df = df.group_by("case_id").agg(agg_list)
        results.append(df)
        
    return results



# given a dict of dataset info, this calls load_df to pull data for each and then joins all.
def load_all_dfs(datasets):
    train = {}
    test = {}
    for name in datasets:
        train_df, test_df = load_df(**datasets[name])
        train[name]=train_df
        test[name]=test_df
    
    train_base = train.pop('base')
    test_base = test.pop('base')
    
#     # special case: merge tax registry a, b and c
#     tax_registries = [k for k in train.keys() if k[:12]=="tax_registry"]
#     if len(tax_registries) >1:
#         name_map ={"M":"employerM", "A":"taxdeductionA","D":"processdateD"}
#         for k in tax_registries:
#             # standardize their names to allow concat
#             cols = train[k].columns
#             standard_cols = [name_map[c.split("_")[-2][-1]]+c.split("_")[-1] for c in cols]
#             train[k].columns = standard_cols
#             train[k] = train[k].select(sorted(train[k].columns))
#             test[k].columns = standard_cols
#             test[k] = test[k].select(sorted(train[k].columns))
        
#         train_tr_df = pl.concat([train[k] for k in tax_registries], how="vertical_relaxed")
#         # TODO fill features below
#         train_tr_agg_list = create_agg_list(___, aggs)
#         test_tr_df = pl.concat([test[k] for k in tax_registries], how="vertical_relaxed")
        
    
    # join all sets
    for dataset in train:
        train_base = train_base.join(train[dataset], how='left', on='case_id')
        test_base = test_base.join(test[dataset], how='left', on='case_id')
    return train_base, test_base



# from contest creator starter notebook
def convert_strings(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:  
        if df[col].dtype.name in ['object', 'string']:
            df[col] = df[col].astype("string").astype('category')
            current_categories = df[col].cat.categories
            new_categories = current_categories.to_list() + ["Unknown"]
            new_dtype = pd.CategoricalDtype(categories=new_categories, ordered=True)
            df[col] = df[col].astype(new_dtype)
    return df


# from contest creator starter notebook
def from_polars_to_pandas(case_ids: pl.DataFrame, df) -> pl.DataFrame:
    cols_pred = []
    for col in df.columns:
        if col[-1].isupper() and col[:-1].islower():
            cols_pred.append(col)
    return (
        df.filter(pl.col("case_id").is_in(case_ids))[["case_id", "WEEK_NUM", "target"]].to_pandas(),
        df.filter(pl.col("case_id").is_in(case_ids))[cols_pred].to_pandas(),
        df.filter(pl.col("case_id").is_in(case_ids))["target"].to_pandas()
    )

def train_val_test_split(train_df, train_split=0.9, val_split=0.5):
    # the following code is mostly copied from contest creator starter notebook
    # although it has been changed to facilitate functional programming
    case_ids = train_df["case_id"].unique().shuffle(seed=1)
    case_ids_train, case_ids_test = train_test_split(case_ids, train_size=train_split, random_state=1)
    case_ids_val, case_ids_test = train_test_split(case_ids_test, train_size=val_split, random_state=1)
    
    
    base_train, X_train, y_train = from_polars_to_pandas(case_ids_train, train_df)
    base_val, X_val, y_val = from_polars_to_pandas(case_ids_val, train_df)
    base_test, X_test, y_test = from_polars_to_pandas(case_ids_test, train_df)
    
    for df in [X_train, X_val, X_test]:
        df = convert_strings(df)
    
    return (
        (base_train, X_train, y_train), 
        (base_val, X_val, y_val), 
        (base_test, X_test, y_test)
    )