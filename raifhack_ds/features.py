import pandas as pd
from raifhack_ds.utils import UNKNOWN_VALUE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder



class FeaturesPreparation:
    def __init__(self, config):
        self.config = config
        self.categorical_features = config.get('categorical_features')

    def prepare_train(d) -> pd.DataFrame:
        df_new = df.copy()
        
        
        # add nans
        replace_nans = config.get('replace_nans', 'unknown_value')
        for col in df_new.columns():
            if df_new[col].isna().sum() != 0:
                if col in self.categorical_features:
                    df_new[col].fillna(UNKNOWN_VALUE)
                elif col in self.numerical_features:
                    df_new[col].fillna(df[!df_new[col].isna()].mean())

        # features interaction
        features_to_interact = self.config.get(features_to_interact, [])
        for f1 in features_to_interact:
            for f2 in features_to_interact:
                if f1 != f2:
                    df_new[f'{f1}_{f2}'] = df_new[f1] + df_new[f2] # ADD NORM


        return df_new

    def prepare_test(d) -> pd.DataFrame:
        df_new = df.copy()
        
        
        # add nans
        replace_nans = config.get('replace_nans', 'unknown_value')
        for col in df_new.columns():
            if df_new[col].isna().sum() != 0:
                if col in self.categorical_features:
                    df_new[col].fillna(UNKNOWN_VALUE)
                elif col in self.numerical_features:
                    df_new[col].fillna(df[!df_new[col].isna()].mean())

        # features interaction
        features_to_interact = self.config.get(features_to_interact, [])
        for f1 in features_to_interact:
            for f2 in features_to_interact:
                if f1 != f2:
                    df_new[f'{f1}_{f2}'] = df_new[f1] + df_new[f2] # ADD NORM


        return df_new