import pandas as pd
from raifhack_ds.utils import UNKNOWN_VALUE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder



class FeaturesPreparation:
    def __init__(self, config):
        self.config = config
        self.categorical_ordinal = config.get('categorical_ordinal')
        self.categorical_ohe = config.get('categorical_ohe')
        self.categorical_tomodel = config.get('categorical_tomodel')
        self.categorical_features = (self.categorical_ordinal +
                                    self.categorical_ohe +
                                    self.categorical_tomodel)
        self.numerical_features = config.get('num_features')
        self.ohe_encoder = ColumnTransformer(transformers=[('ohe', OneHotEncoder(), self.categorical_ohe)])
        transformers = []
        nums_scaler = config.get('nums_scaler', 'standart')
        if nums_scaler == 'standart':
            transformers.append(('num', StandardScaler(), self.numerical_features))
        transformers.append(('ord', OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),
                                           self.categorical_ordinal))
        self.other_encoders = ColumnTransformer(transformers=transformers, remainder='passthrough')
        self.features_names = self.numerical_features + self.categorical_tomodel + self.categorical_ordinal
        self.categorical_to_model = self.categorical_tomodel + self.categorical_ordinal
        self.target_name = config['target_column']

    def prepare(self, df, mode) -> pd.DataFrame:
        df_new = df.copy()


        # add nans
        replace_nans = self.config.get('replace_nans', 'unknown_value')
        for col in df_new.columns:
            print(col, df_new[col].isna().sum())
            if df_new[col].isna().sum() != 0:
                if col in self.categorical_features:
                    print("HERE")
                    print(col)
                    df_new[col].fillna(UNKNOWN_VALUE)
                # elif col in self.numerical_features:
                #     df_new[col].fillna(df[~df_new[col].isna()][col].mean())

        # features interaction
        features_to_interact = self.config.get('features_to_sum', [])
        for f1 in features_to_interact:
            for f2 in features_to_interact:
                if f1 != f2:
                    df_new[f'{f1}_{f2}'] = df_new[f1] + df_new[f2] # ADD NORM
                    if train:
                        self.numerical_features.append(f'{f1}_{f2}')

        if mode in ['train', 'val']:
            target = df_new[self.target_name]
            df_new.drop(columns=[self.target_name], inplace=True)

        if mode in ['train']:
            transformed = self.ohe_encoder.fit_transform(df_new)
            df_new[self.ohe_encoder.get_feature_names()] = transformed
            self.features_names += self.ohe_encoder.get_feature_names()
            df_matrix = self.other_encoders.fit_transform(df_new[self.features_names])
        else:
            transformed = self.ohe_encoder.transform(df_new)
            df_new[self.ohe_encoder.get_feature_names()] = transformed
            df_matrix = self.other_encoders.transform(df_new[self.features_names])
        new_df = pd.DataFrame(data=df_matrix, columns=self.features_names)
        if mode in ['train', 'val']:
            return new_df, target
        return new_df
