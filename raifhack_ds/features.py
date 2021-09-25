import pandas as pd
import numpy as np
from raifhack_ds.utils import UNKNOWN_VALUE

big_cities_set = set()
middle_cities_set = set()
df4_ = pd.DataFrame()

# def preprocess(df, val=False):
#     global df4_
#     alpha=2
#     beta=1
#     gamma=1
#     df1 = df.copy()
#     if not val:
#         df2 = df1[['lat', 'lng', 'region']].groupby('region').mean().reset_index()
#         df3 = df1[['osm_city_nearest_population', 'region']].groupby('region').max().reset_index()
#         df4 = df2.merge(df3, on='region')
#         df4.columns = ['region', 'lat_center', 'lng_center', 'pop']
#         df4_ = df4

#         def func(a):
#             x = a.lat
#             y = a.lng
#             return np.sum(np.exp(-alpha * ((df4.lat_center - x) ** 2 + (df4.lng_center - y) ** 2) ** (gamma / 2)) * df4['pop'] ** beta)

#         df1['ne_v_jope'] = df1.apply(func, axis=1)
#         return df1
#     else:
#         df4 = df4_
#         def func(a):
#             x = a.lat
#             y = a.lng
#             return np.sum(np.exp(-alpha * ((df4.lat_center - x) ** 2 + (df4.lng_center - y) ** 2) ** (gamma / 2)) * df4['pop'] ** beta)

#         df1['ne_v_jope'] = df1.apply(func, axis=1)
#         return df1

def preprocess(dataframe, val=False):
    global big_cities_set
    global middle_cities_set
    df0 = dataframe.copy()
    if not val:
        df1 = df0[df0.price_type == 1]
        df_big_cities = pd.DataFrame(df1.city.value_counts()[:10])
        df_big_cities['big_city'] = True
        df_middle_cities = pd.DataFrame(df1.city.value_counts()[10:50])
        df_middle_cities['middle_city'] = True
        df2 = df0.merge(df_big_cities.drop('city', axis=1), how='left', left_on='city', right_index=True)
        df3 = df2.merge(df_middle_cities.drop('city', axis=1), how='left', left_on='city', right_index=True)
        df3[['big_city', 'middle_city']] = df3[['big_city', 'middle_city']].fillna(False)
        df3['top_city'] = 'rare'
        df3.loc[df3.big_city | df3.middle_city, 'top_city'] = df3.loc[df3.big_city | df3.middle_city]['city']
        df3[['big_city', 'middle_city']] = df3[['big_city', 'middle_city']].astype(int)

        big_cities_set = set(pd.unique(df3.city[df3.big_city == 1]))
        middle_cities_set = set(pd.unique(df3.city[df3.middle_city == 1]))
        return df3.drop('city', axis=1)
    else:
        df1 = df0
        df1.loc[df1.city.isin(big_cities_set), 'big_city'] = True
        # df1.city[df1.city.isin(big_cities_set)]
        df1.loc[df1.city.isin(middle_cities_set), 'middle_city'] = True
        df1[['big_city', 'middle_city']] = df1[['big_city', 'middle_city']].fillna(False)
        df1['top_city'] = 'rare'
        df1.loc[(df1.city.isin(big_cities_set) | df1.city.isin(middle_cities_set)), 'top_city'] = df1.city[(df1.city.isin(big_cities_set) | df1.city.isin(middle_cities_set))]

        return df1.drop('city', axis=1)
        


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