import json,pickle
from category_encoders import TargetEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import combinations


def pre_process_and_feature_engineer(data,target_col='ln_fpd',input_file='input.json'):
    df = data.copy()
    df.columns = df.columns.str.replace(r'[^\w]+', '_', regex=True)
    categorical_columns = df.select_dtypes(exclude = np.number).columns
    with open('input.json') as f:
        input = json.load(f)
        lb_months_at_address_median = input['lb_months_at_address_median']
        final_features = input['final_features']
    for col in categorical_columns:
        if df[col].isnull().sum()>0:
            df[col] = df[col].fillna("Missing")
    days_features = [col for col in df.columns if 'day_first_seen' in col or 'day_last_seen' in col]

    df = df.drop(columns=days_features,axis=1, errors='ignore')
    df['lb_months_at_address_missing'] = df['lb_months_at_address'].isnull().astype(bool)
    df['lb_months_at_address'] = df['lb_months_at_address'].fillna(lb_months_at_address_median)
    top_exts = df['VarianceTable_variance_table_email_extension'].value_counts().nlargest(20).index
    df['email_ext_grouped'] = df['VarianceTable_variance_table_email_extension'].apply(lambda x: x if x in top_exts else 'Other')
    df = df.drop('VarianceTable_variance_table_email_extension',axis=1)
    col_TE = ['email_ext_grouped','VarianceTable_variance_table_device_parent','PricingTool_predictions_min','PricingTool_predictions_ranked','PricingTool_predictions_min_max','PricingTool_predictions_max']
    with open("TE.pkl", "rb") as f:
        TE = pickle.load(f)
    temp1 = TE.transform(df[col_TE])
    for col in temp1.columns:
          temp1.rename(columns={col:col + '_TE'}, inplace=True)

    df = pd.concat([df,temp1], axis = 1)
    df = df.drop('VarianceTable_variance_table_device_parent',axis=1)
    df['lead_datetime'] = pd.to_datetime(df['lead_datetime'], errors='coerce')
    
    
    redundant_columns = [col for col in df.columns if df[col].nunique() <= 1]

    df = df.drop(redundant_columns,axis=1, errors='ignore')
    eps = 1.e-6

    day_count_cols = sorted([col for col in df.columns if '_count_' in col and '_days' in col])

    for col in day_count_cols:

        window = int(col.split('_days')[0].split('_')[-1])
        base = col.split('count_')[0]
        all_count_cols = [i for i in day_count_cols if base in i]
        for a, b in combinations(df[all_count_cols].columns,2):
            df[f'{a}/{b}'] = df[a].div(df[b]+eps)

        rolling_mean = df[col].rolling(window).mean().shift(1)  # shift to exclude current row

        # Velocity feature = current value / rolling mean
        df[f'{col}_velocity'] = df[col] / (rolling_mean+eps)
        df[f'{col}_velocity'] = df[f'{col}_velocity'].fillna(0)
 
    scale_cols = [col for col in df.columns if 'scale_scale_between' in col]
    short_term  = [col for col in scale_cols if '24_hours_and_60_minutes' in col or '6_hours_and_60_minutes' in col]
    mid_term    = [col for col in scale_cols if '7_days_and_24_hours' in col or '30_days_and_7_days' in col]
    long_term   = [col for col in scale_cols if '90_days_and_30_days' in col or 
                   '365_days_and_90_days' in col or 'all_time' in col]
    
    df['scale_mean_short_term'] = df[short_term].mean(axis=1)
    df['scale_mean_mid_term']   = df[mid_term].mean(axis=1)
    df['scale_mean_long_term']  = df[long_term].mean(axis=1)
    

    df['day_of_week'] = df['lead_datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6])


    df = df.drop(['lead_datetime'],axis=1)
    
    categorical_columns = df.select_dtypes(exclude = np.number).columns
    df2 = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    df2.columns = df2.columns.str.replace(r'[^\w]+', '_', regex=True)
    return df2[final_features] 
    
    



class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return pre_process_and_feature_engineer(X)




    