import polars as pl
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


# helper function modified from code in contest creator starter notebook
def set_table_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    for col in df.columns:
        # last letter of column name will help you determine the type
        if col[-1] in ("P", "A"):
            df = df.with_columns(pl.col(col).cast(pl.Float64).alias(col))
        elif col[-1] == "D":
            # for dates, we want both the year as a feature, and the month encoded trigonometrically to capture its cyclic nature
            date_split = df[col].str.split(by="-")
            df = df.with_columns(
                (2*np.pi*date_split.list.get(1).cast(pl.Float32)/12).sin().alias(col[:-1]+"X"),
                (2*np.pi*date_split.list.get(1).cast(pl.Float32)/12).cos().alias(col[:-1]+"Y"),
                date_split.list.get(0).cast(pl.Int16).alias(col)
            )

    return df



# helper function to create a list of formatted aggregations given features and aggregation types
def create_agg_list(features, aggs):
    agg_features = list(filter(lambda x: x not in ['target','case_id', "num_group1", "num_group2"], features))
    nested_lists = [
        [pl.max(f).name.prefix("max_") if "agg_max" in aggs else None, 
          pl.min(f).name.prefix("min_") if "agg_min" in aggs else None, 
          pl.median(f).name.prefix("median_") if "agg_median" in aggs else None
         ] for f in agg_features]
    agg_list = []
    for l in nested_lists:
        agg_list.extend(l)
    agg_list = list(filter(lambda x: x is not None, agg_list))
    return agg_list



# this function loads a given dataset, performs aggregation if depth >0, and returns train and test sets
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
            
        # special case: tax registries need feature names standardized
        if name == "tax_registry":
            name_map ={"M":"employerM", "A":"taxdeductionA","D":"processdateD", "X":"processdateX","Y":"processdateY"}
            for i in range(len(df_list)):
                # standardize their names to allow concat
                standard_cols = [name_map[c[-1]] if c not in ['case_id', "num_group1"] else c for c in df_list[i].columns]
                df_list[i].columns = standard_cols
                df_list[i] = df_list[i].select(sorted(standard_cols))
            
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


# modified from code in contest creator starter notebook
def from_polars_to_pandas(case_ids: pl.DataFrame, df, is_submit=False) -> pl.DataFrame:
    cols_pred = []
    for col in df.columns:
        if col[-1].isupper() and col[:-1].islower():
            cols_pred.append(col)
    base_cols = ["case_id", "WEEK_NUM"] if is_submit else ["case_id", "WEEK_NUM", "target"]
    return (
        df.filter(pl.col("case_id").is_in(case_ids))[base_cols].to_pandas(),
        df.filter(pl.col("case_id").is_in(case_ids))[cols_pred].to_pandas(),
        None if is_submit else df.filter(pl.col("case_id").is_in(case_ids))["target"].to_pandas()
    )



def train_val_test_split(train_df, submit_df, train_split=0.9, val_split=0.5):
    # the following code is mostly copied from contest creator starter notebook
    # although it has been changed to facilitate functional programming
    case_ids = train_df["case_id"].unique().shuffle(seed=1)
    case_ids_train, case_ids_test = train_test_split(case_ids, train_size=train_split, random_state=1)
    case_ids_val, case_ids_test = train_test_split(case_ids_test, train_size=val_split, random_state=1)
    case_ids_submit = submit_df["case_id"].unique()
    
    base_train, X_train, y_train = from_polars_to_pandas(case_ids_train, train_df)
    base_val, X_val, y_val = from_polars_to_pandas(case_ids_val, train_df)
    base_test, X_test, y_test = from_polars_to_pandas(case_ids_test, train_df)
    base_submit, X_submit, y_submit = from_polars_to_pandas(case_ids_submit, submit_df, is_submit=True)
    
    for df in [X_train, X_val, X_test, X_submit]:
        df = convert_strings(df)
    
    return (
        (base_train, X_train, y_train), 
        (base_val, X_val, y_val), 
        (base_test, X_test, y_test),
        (base_submit, X_submit, y_submit)
    )



def cat_to_dummies(X_train, other_dfs = [], max_categories=5):
    # select categorical columns
    cols_cat = X_train.select_dtypes(include="category").columns
    cols_non_cat = X_train.select_dtypes(exclude="category").columns
    X_train_cat = X_train[cols_cat]
        
    # condense least common categories to "Unknown"
    top_n = {}
    for col in cols_cat:
        categories = X_train_cat[col].dtype.categories
        if len(categories) > max_categories:
            # find most common in train set
            top_n[col] = X_train_cat[col].value_counts().index[:max_categories]
            X_train_cat.loc[:,col] = X_train_cat.loc[:,col].apply(lambda x: x if x in top_n[col] else "Unknown").astype(X_train_cat.loc[:,col].dtype)
        else:
            top_n[col] = categories
    
    # create dummies
    X_train_dummies = pd.get_dummies(X_train_cat.astype("object"), dummy_na=True)
    
    # join to non categorical
    X_train_non_cat = X_train[cols_non_cat]
    X_train_combined = pd.concat([X_train_dummies, X_train_non_cat], axis=1)
    
    
    # repeat for other dfs, using the same categories
    other_combined = []
    if other_dfs:
        for df in other_dfs:
            df_cat = df[cols_cat]
            
            # reduce categories using top_n from X_train
            for col in cols_cat:
                df_cat.loc[:,col] = df_cat.loc[:,col].apply(lambda x: x if x in top_n[col] else "Unknown").astype(df_cat.loc[:,col].dtype)
                
            # create dummies
            df_dummies = pd.get_dummies(df_cat.astype("object"), dummy_na=True)
            
            # join to non categorical
            df_non_cat = df[cols_non_cat]
            other_combined.append(pd.concat([df_dummies, df_non_cat], axis=1))
                
    
    return X_train_combined, *other_combined