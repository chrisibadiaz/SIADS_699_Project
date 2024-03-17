####################################################
# stores dataset info, arguments for load_df
#    description: notes to self. Ignored by load functions
#    name: from the actual name of the file, ignoring extra info (e.g., train/train_{NAME}_1.csv)
#    features: specify columns to keep (ignore all others)
#    feature_types: from kept features, select only those ending with these tags
#    depth: from kaggle description. If .0, aggregation will be performed
#    agg_max (default True): if depth>0, return the max for each case_id for each a feature
#    agg_min (default False): if depth>0, return the min for each case_id for each a feature
#    agg_median (default False): if depth>0, return the max for each case_id for each a feature
#####################################################

# Note this will not run as is. credit_bureau_a_2 exceeds memory limits.
dataset_full = {
    "base":{
        "description": "internal/0.04GB/100%: links case_id to WEEK_NUM and target",
        "name":"base",
        'depth':0
    },
    "static_0":{
        "description":"internal/0.36GB/75%: contains transaction history for each case_id (late payments, total debt, etc)",
        "name":"static_0",
        'depth':0
    },
    "static_cb":{
        "description":"external/0.24GB/38%: demographic data, risk assessment, number of credit checks",
        "name":"static_cb",
        'depth':0
    },
    "applprev_1":{
        "description":"internal/1.3GB/70%: info from clients previous loan applications",
        "name":"applprev_1",
        "depth":1,
    },
    "other_1":{
        "description":"internal/0.002GB/100%: transaction history of client debit account",
        "name":"other_1",
        "depth":1,
    },
    "tax_registry_a_1":{
        "description":"external/0.12GB/100%: amount and date of tax deductions claimed, along with hashed employer name (from provider A)",
        "name":"tax_registry_a_1",
        "depth":1,
    },
    "tax_registry_b_1":{
        "description":"external/0.04GB/100%: amount and date of tax deductions claimed, along with hashed employer name (from provider B)",
        "name":"tax_registry_b_1",
        "depth":1,
    },
    "tax_registry_c_1":{
        "description":"external/0.12GB/100%: amount and date of tax deductions claimed, along with hashed employer name (from provider B)",
        "name":"tax_registry_c_1",
        "depth":1,
    },
    "credit_bureau_a_1":{
        "description":"external/3.2GB/35%: detailed history for client loan contracts",
        "name":"credit_bureau_a_1",
        "depth":1,
    },
    "credit_bureau_b_1":{
        "description":"external/0.02GB/80%: detailed history for client loan contracts",
        "name":"credit_bureau_b_1",
        "depth":1,
    },
    "deposit_1":{
        "description":"internal/0.005GB/90%: deposit account history",
        "name":"deposit_1",
        "depth":1,
    },
    "person_1":{
        "description":" internal/0.5GB/55%: demographic information: zip code, marital status, gender etc (all hashed)",
        "name":"person_1",
        "depth":1,
    },
    "debitcard_1":{
        "description":"internal/0.004GB/55%: information about debit card usage",
        "name":"debitcard_1",
        "depth":1,
    },
    "applprev_2":{
        "description":"internal/0.45GB/81%: info about card blockage on previous credit accounts",
        "name":"applprev_2",
        "depth":2,
    },
    "person_2":{
        "description":"internal/0.1GB/75%: info about clients contacts and their employment",
        "name":"person_2",
        "depth":2,
    },
    "credit_bureau_a_2":{
        "description":"external/15.6GB/66%: info about collateral and payment dates",
        "name":"credit_bureau_a_2",
        "depth":2,
    },
    "credit_bureau_b_2":{
        "description":"external/0.04GB/100%: num and value of overdue payments",
        "name":"credit_bureau_b_2",
        "depth":2,
    }
}

# focus on A and M features, excludes some sets
dataset_am = {
    "base":{
        "description": "internal/0.04GB/100%: links case_id to WEEK_NUM and target",
        "name":"base",
        'depth':0
    },
    "static_0":{
        "description":"internal/0.36GB/75%: contains transaction history for each case_id (late payments, total debt, etc)",
        "name":"static_0",
        "feature_types":["A", "M"],
        'depth':0
    },
    "static_cb":{
        "description":"external/0.24GB/38%: demographic data, risk assessment, number of credit checks",
        "name":"static_cb",
        "feature_types":["A", "M"],
        'depth':0
    },
    "applprev_1":{
        "description":"internal/1.3GB/70%: info from clients previous loan applications",
        "name":"applprev_1",
        "feature_types":["A", "M"],
        "depth":1,
    },
    "tax_registry_a_1":{
        "description":"external/0.12GB/100%: amount and date of tax deductions claimed, along with hashed employer name (from provider A)",
        "name":"tax_registry_a_1",
        "feature_types":["A", "M"],
        "depth":1,
    },
    "tax_registry_b_1":{
        "description":"external/0.04GB/100%: amount and date of tax deductions claimed, along with hashed employer name (from provider B)",
        "name":"tax_registry_b_1",
        "depth":1,
    },
    "tax_registry_c_1":{
        "description":"external/0.12GB/100%: amount and date of tax deductions claimed, along with hashed employer name (from provider B)",
        "name":"tax_registry_c_1",
        "depth":1,
    },
    "credit_bureau_a_1":{
        "description":"external/3.2GB/35%: detailed history for client loan contracts",
        "name":"credit_bureau_a_1",
        "depth":1,
    },
    "credit_bureau_b_1":{
        "description":"external/0.02GB/80%: detailed history for client loan contracts",
        "name":"credit_bureau_b_1",
        "depth":1,
    },
    "deposit_1":{
        "description":"internal/0.005GB/90%: deposit account history",
        "name":"deposit_1",
        "depth":1,
    },
    "person_1":{
        "description":" internal/0.5GB/55%: demographic information: zip code, marital status, gender etc (all hashed)",
        "name":"person_1",
        "depth":1,
    },
    "debitcard_1":{
        "description":"internal/0.004GB/55%: information about debit card usage",
        "name":"debitcard_1",
        "depth":1,
    },
    "applprev_2":{
        "description":"internal/0.45GB/81%: info about card blockage on previous credit accounts",
        "name":"applprev_2",
        "depth":2,
    },
    "person_2":{
        "description":"internal/0.1GB/75%: info about clients contacts and their employment",
        "name":"person_2",
        "depth":2,
    },
    "credit_bureau_a_2":{
        "description":"external/15.6GB/66%: info about collateral and payment dates",
        "name":"credit_bureau_a_2",
        "depth":2,
    },
    "credit_bureau_b_2":{
        "description":"external/0.04GB/100%: num and value of overdue payments",
        "name":"credit_bureau_b_2",
        "depth":2,
    }
}


# this is almost exactly the same dataset used in the starter notebook
dataset_example = {
    "base":{
        "description": "links case_id to WEEK_NUM and target",
        "name":"base",
    },
    "static_0":{
        "description":"contains transaction history for each case_id (late payments, total debt, etc)",
        "name":"static_0",
        "feature_types":["A", "M"],
    },
    "static_cb":{
        "description":"data from an external cb: demographic data, risk assessment, number of credit checks",
        "name":"static_cb",
        "feature_types":["A", "M"],
    },
    "person_1_feats_1":{
        "description":" internal demographic information: zip code, marital status, gender etc (all hashed)",
        "name":"person_1",
        "features":["mainoccupationinc_384A", "incometype_1044T"],
        "depth":1,
    },
    "person_1_feats_2":{
        "description":" internal demographic information: zip code, marital status, gender etc (all hashed)",
        "name":"person_1",
        "features":["housetype_905L"],
        "depth":1,
    },
    "credit_bureau_b_2":{
        "description":"historical data from an external source, num and value of overdue payments",
        "name":"credit_bureau_b_2",
        "features":["pmts_pmtsoverdue_635A","pmts_dpdvalue_108P"],
        "depth":2,
    }
}