import pandas as pd
from raifhack_ds.utils import UNKNOWN_VALUE

def preprocess(dataframe):
    df1 = dataframe.copy()

    import math
    def dirty_floor_to_num(a):
        if not isinstance(a, str) and math.isnan(a):
            return 0
        if isinstance(a, int) or isinstance(a, float):
            return a
        if a == 'цоколь' or a == '1, цоколь' or a == 'подвал':
            return -1
        if a.isnumeric():
            return float(a)
        else:
            return 0

    df1['real_floor'] = df1.floor.apply(dirty_floor_to_num)
    df1['floor_isna'] = df1.floor.isna().astype(int)
    df1['high_floor'] = (df1.real_floor > 3).astype(int)
    df1['underground_floor'] = (df1.real_floor <= 0).astype(int)
    df1['very_high_floor'] = (df1.real_floor > 10).astype(int)
    df1.drop(['id', 'floor'], axis=1, inplace=True)
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