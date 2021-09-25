import pandas as pd
from raifhack_ds.utils import UNKNOWN_VALUE
from imblearn.over_sampling import RandomOverSampler

def preprocess(dataframe, oversample_rate='auto'):
    df1 = dataframe.copy()

    ros = RandomOverSampler(random_state=42, sampling_strategy=oversample_rate)
    df_X_resampled, df_y_resampled = ros.fit_resample(df1.drop(columns='price_type'), df1['price_type'])
    df1 = df_X_resampled
    df1['price_type'] = df_y_resampled
    print(df1.head(10))
    return df1

def prepare_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заполняет пропущенные категориальные переменные
    :param df: dataframe, обучающая выборка
    :return: dataframe
    """
    df_new = df.copy()
    fillna_cols = ['region','city','street','realty_type']
    df_new[fillna_cols] = df_new[fillna_cols].fillna(UNKNOWN_VALUE)
    return df_new