import polars as pl
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer


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


class split:
    def __init__(self, name):
        self.name = name
        self.base = None
        self.X = None
        self.y = None
        
class data_splits:
    
    def __init__(self):
        self.train = None
        self.val = None
        self.test = None
        self.submit =None
        
    def __iter__(self):
        for attr in ["train", "val", "test", "submit"]:
            val = self.__dict__[attr]
            yield attr, val
    
    def add_split(self, split):
        match split.name:
            case "train":
                self.train = split
            case "val":
                self.val = split
            case "test":
                self.test = split
            case "submit":
                self.submit = split
                
                
# modified from code in contest creator starter notebook
def create_pandas_split(case_ids: pl.DataFrame, df, name):
    cols_pred = []
    for col in df.columns:
        if col[-1].isupper() and col[:-1].islower():
            cols_pred.append(col)
    base_cols = ["case_id", "WEEK_NUM"] if (name == 'submit') else ["case_id", "WEEK_NUM", "target"]
    
    new_split = split(name)
    new_split.base = df.filter(pl.col("case_id").is_in(case_ids))[base_cols].to_pandas()
    new_split.X = convert_strings(df.filter(pl.col("case_id").is_in(case_ids))[cols_pred].to_pandas())
    
    if name != 'submit':
        new_split.y = df.filter(pl.col("case_id").is_in(case_ids))["target"].to_pandas()
    return new_split



def train_val_test_split(train_df, submit_df, train_split=0.9, val_split=0.5):
    # the following code is altered from code in contest creator starter notebook
    case_ids = train_df["case_id"].unique().shuffle(seed=1)
    case_ids_train, case_ids_test = train_test_split(case_ids, train_size=train_split, random_state=1)
    case_ids_val, case_ids_test = train_test_split(case_ids_test, train_size=val_split, random_state=1)
    case_ids_submit = submit_df["case_id"].unique()
    
    data = data_splits()
    data.add_split(create_pandas_split(case_ids_train, train_df, "train"))
    data.add_split(create_pandas_split(case_ids_val, train_df, "val"))
    data.add_split(create_pandas_split(case_ids_test, train_df, "test"))
    data.add_split(create_pandas_split(case_ids_submit, submit_df, "submit"))
    
    return data



def cat_to_dummies(data, max_categories=5):
    
    data_with_dummies = copy.deepcopy(data)
    top_n = {} # will save the top categories for each categorical column
    for split_name, split_data in data_with_dummies: 
        X = split_data.X
        is_train = (split_name=="train")
        
        # select categorical columns
        if is_train:
            cols_cat = X.select_dtypes(include="category").columns
            cols_non_cat = X.select_dtypes(exclude="category").columns
        X_cat = X[cols_cat]

        # condense least common categories to "Unknown"
        for col in cols_cat:
            if is_train:
                categories = X_cat[col].dtype.categories
                if len(categories) > max_categories:
                    # find most common in train set
                    top_n[col] = X_cat[col].value_counts().index[:max_categories]
                else:
                    top_n[col] = categories
            
            X_cat.loc[:,col] = X_cat.loc[:,col].apply(lambda x: x if x in top_n[col] else "Unknown").astype(X_cat.loc[:,col].dtype)

        # create dummies
        X_dummies = pd.get_dummies(X_cat.astype("object"), dummy_na=True)

        # join to non categorical
        X_non_cat = X[cols_non_cat]
        X_combined = pd.concat([X_dummies, X_non_cat], axis=1)
        
        match split_name:
            case "train":
                data_with_dummies.train.X = X_combined
            case "val":
                data_with_dummies.val.X = X_combined
            case "test":
                data_with_dummies.test.X = X_combined
            case "submit":
                data_with_dummies.submit.X = X_combined

    return data_with_dummies



def handle_missing_data(data, max_missing_feature=0.5, max_missing_instance=0.9, missing_indicator_threshold=None, imputer=None, string_imputer=None):
    # imputers from: (None, sklearn.impute.SimpleImputer, sklearn.impute.IterativeImputer)
    # if no string_imputer provided, imputer should be able to handle both numeric and string
    data_no_na = copy.deepcopy(data)
    for split_name, split_data in data_no_na:
        X = split_data.X
        is_train = (split_name=="train")
        
        # first, remove any features which exceed the max_missing threshold
        if is_train:
            features_to_keep = X.isna().sum()/len(X)<max_missing_feature
        X=X.loc[:, features_to_keep]

        # for features with missing counts above this threshold, add a dummy column indicating which entries were imputed
        if missing_indicator_threshold:
            if is_train:
                features_to_indicate = X.isna().sum()/len(X) > missing_indicator_threshold
                dummy_cols = X.loc[:, features_to_indicate].columns
            dummies = X[dummy_cols].isna()
            X = pd.concat([X, dummies.add_prefix("ismissing_").astype(int)], axis=1) 

        # remove any instances (rows) which have insufficient data
        instances_to_keep = X.isna().sum(1)/len(X.columns)< max_missing_instance
        X = X.loc[instances_to_keep,:]

        # finally, impute features if required
        if imputer:
            if string_imputer:
                # impute strings seperately
                X_strs = X.select_dtypes(include=['object', 'category'])
                if len(X_strs.columns) > 0:
                    if is_train:
                        string_imputer.fit(X_strs)
                    X_strs = pd.DataFrame(string_imputer.transform(X_strs), index=X_strs.index, columns=X_strs.columns).astype(X_strs.dtypes.to_dict())
                X_nums = X.select_dtypes(exclude=['object', 'category'])
                if len(X_nums.columns) > 0:
                    if is_train:
                        imputer.fit(X_nums)
                    X_nums = pd.DataFrame(imputer.transform(X_nums), index=X_nums.index, columns=X_nums.columns).astype(X_nums.dtypes.to_dict())

                X = pd.concat([X_strs, X_nums], axis=1)

            else:
                # imputer can handle strings, or you think there will be none
                if is_train:
                    imputer.fit(X)
                X = pd.DataFrame(imputer.transform(X), index=X.index, columns=X.columns).astype(X.dtypes.to_dict())

        match split_name:
            case "train":
                data_no_na.train.X = X
            case "val":
                data_no_na.val.X = X
            case "test":
                data_no_na.test.X = X
            case "submit":
                data_no_na.submit.X = X
    
    return data_no_na