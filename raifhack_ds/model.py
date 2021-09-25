import typing
import pickle
import pandas as pd
import numpy as np
import logging

from lightgbm import LGBMRegressor

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.exceptions import NotFittedError
from raifhack_ds.metrics import metrics_stat

from raifhack_ds.data_transformers import SmoothedTargetEncoding

logger = logging.getLogger(__name__)

TARGET = 'per_square_meter_price'
# признаки (или набор признаков), для которых применяем smoothed target encoding
CATEGORICAL_STE_FEATURES = ['region', 'realty_type']

# признаки, для которых применяем one hot encoding
CATEGORICAL_OHE_FEATURES = []

def cnt_ohe_ft(df):
    res = 0
    for ft in CATEGORICAL_OHE_FEATURES:
        res += len(df[ft].value_counts())
    return res

# численные признаки
NUM_FEATURES = ['lat', 'lng', 'osm_amenity_points_in_0.001',
       'osm_amenity_points_in_0.005', 'osm_amenity_points_in_0.0075',
       'osm_amenity_points_in_0.01', 'osm_building_points_in_0.001',
       'osm_building_points_in_0.005', 'osm_building_points_in_0.0075',
       'osm_building_points_in_0.01', 'osm_catering_points_in_0.001',
       'osm_catering_points_in_0.005', 'osm_catering_points_in_0.0075',
       'osm_catering_points_in_0.01', 'osm_city_closest_dist',
      'osm_city_nearest_population',
       'osm_crossing_closest_dist', 'osm_crossing_points_in_0.001',
       'osm_crossing_points_in_0.005', 'osm_crossing_points_in_0.0075',
       'osm_crossing_points_in_0.01', 'osm_culture_points_in_0.001',
       'osm_culture_points_in_0.005', 'osm_culture_points_in_0.0075',
       'osm_culture_points_in_0.01', 'osm_finance_points_in_0.001',
       'osm_finance_points_in_0.005', 'osm_finance_points_in_0.0075',
       'osm_finance_points_in_0.01', 'osm_healthcare_points_in_0.005',
       'osm_healthcare_points_in_0.0075', 'osm_healthcare_points_in_0.01',
       'osm_historic_points_in_0.005', 'osm_historic_points_in_0.0075',
       'osm_historic_points_in_0.01', 'osm_hotels_points_in_0.005',
       'osm_hotels_points_in_0.0075', 'osm_hotels_points_in_0.01',
       'osm_leisure_points_in_0.005', 'osm_leisure_points_in_0.0075',
       'osm_leisure_points_in_0.01', 'osm_offices_points_in_0.001',
       'osm_offices_points_in_0.005', 'osm_offices_points_in_0.0075',
       'osm_offices_points_in_0.01', 'osm_shops_points_in_0.001',
       'osm_shops_points_in_0.005', 'osm_shops_points_in_0.0075',
       'osm_shops_points_in_0.01', 'osm_subway_closest_dist',
       'osm_train_stop_closest_dist', 'osm_train_stop_points_in_0.005',
       'osm_train_stop_points_in_0.0075', 'osm_train_stop_points_in_0.01',
       'osm_transport_stop_closest_dist', 'osm_transport_stop_points_in_0.005',
       'osm_transport_stop_points_in_0.0075',
       'osm_transport_stop_points_in_0.01',
       'reform_count_of_houses_1000', 'reform_count_of_houses_500',
       'reform_house_population_1000', 'reform_house_population_500',
       'reform_mean_floor_count_1000', 'reform_mean_floor_count_500',
       'reform_mean_year_building_1000', 'reform_mean_year_building_500','total_square',
        'price_type'
                # 'real_floor', 'floor_isna', 'high_floor', 'underground_floor', 'very_high_floor'
]

MODEL_PARAMS = dict(
            n_estimators=2000,
            learning_rate=0.01,
            reg_alpha=1,
            num_leaves=40,
            min_child_samples=5,
            importance_type="gain",
            n_jobs=1,
            random_state=563
        )

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"},
    },
    "handlers": {
        "file_handler": {
            "level": "INFO",
            "formatter": "default",
            "class": "logging.FileHandler",
            "filename": 'train.log',
            "mode": "a",
        },
    },
    "loggers": {
        "": {"handlers": ["file_handler"], "level": "INFO", "propagate": False},
    },
}

class BenchmarkModel():
    """
    Модель представляет из себя sklearn pipeline. Пошаговый алгоритм:
      1) в качестве обучения выбираются все данные с price_type=0
      1) все фичи делятся на три типа (numerical_features, ohe_categorical_features, ste_categorical_features):
          1.1) numerical_features - применяется StandardScaler
          1.2) ohe_categorical_featires - кодируются через one hot encoding
          1.3) ste_categorical_features - кодируются через SmoothedTargetEncoder
      2) после этого все полученные фичи конкатенируются в одно пространство фичей и подаются на вход модели Lightgbm
      3) делаем предикт на данных с price_type=1, считаем среднее отклонение реальных значений от предикта. Вычитаем это отклонение на финальном шаге (чтобы сместить отклонение к 0)

    :param numerical_features: list, список численных признаков из датафрейма
    :param ohe_categorical_features: list, список категориальных признаков для one hot encoding
    :param ste_categorical_features, list, список категориальных признаков для smoothed target encoding.
                                     Можно кодировать сразу несколько полей (например объединять категориальные признаки)
    :
    """

    def __init__(self, numerical_features: typing.List[str],
                 ohe_categorical_features: typing.List[str],
                 ste_categorical_features: typing.List[typing.Union[str, typing.List[str]]],
                 model_params: typing.Dict[str, typing.Union[str,int,float]]):
        self.num_features = numerical_features
        self.ohe_cat_features = ohe_categorical_features
        self.ste_cat_features = ste_categorical_features

        self.preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), self.num_features),
            ('ohe', OneHotEncoder(), self.ohe_cat_features),
            ('ste', OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),
             self.ste_cat_features)])

        self.model = LGBMRegressor(**model_params)

        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('model', self.model)])

        self._is_fitted = False
        self.corr_coef = 0

    def _find_corr_coefficient(self, X_manual: pd.DataFrame, y_manual: pd.Series):
        """Вычисление корректирующего коэффициента

        :param X_manual: pd.DataFrame с ручными оценками
        :param y_manual: pd.Series - цены ручника
        """
        predictions = self.pipeline.predict(X_manual)
        print(predictions)
        best_metrics = 10
        ans = -1
        for deviation in np.linspace(-0.5, 0.5, num=100):
            print('trying deviation:', deviation)
            y_preds = pd.Series(np.array(predictions) * (1 + deviation))
            new_metrics = metrics_stat(y_manual.values, y_preds)['raif_metric']
            if new_metrics < best_metrics:
                best_metrics = new_metrics
                ans = deviation
        print('Best corr_coef:', ans)
        self.corr_coef = ans

    def fit(self, X_offer: pd.DataFrame, y_offer: pd.Series,
            X_manual: pd.DataFrame, y_manual: pd.Series, X_all, y_all):
        """Обучение модели.
        ML модель обучается на данных по предложениям на рынке (цены из объявления)
        Затем вычисляется среднее отклонение между руяными оценками и предиктами для корректировки стоимости

        :param X_offer: pd.DataFrame с объявлениями
        :param y_offer: pd.Series - цена предложения (в объявлениях)
        :param X_manual: pd.DataFrame с ручными оценками
        :param y_manual: pd.Series - цены ручника
        """
        logger.info('Fit lightgbm')
        print(X_offer.columns)
        ohe_features = cnt_ohe_ft(X_offer)
        self.pipeline.fit(X_all, y_all, model__feature_name=[f'{i}' for i in range(len(NUM_FEATURES) + ohe_features + len(CATEGORICAL_STE_FEATURES))],model__categorical_feature=[f'{i}' for i in range(len(NUM_FEATURES) + ohe_features, len(NUM_FEATURES) + ohe_features + len(CATEGORICAL_STE_FEATURES))], model__eval_metric='mape')
        logger.info('Find corr coefficient')
        self._find_corr_coefficient(X_manual, y_manual)
        logger.info(f'Corr coef: {self.corr_coef:.2f}')
        self.__is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.array:
        """Предсказание модели Предсказываем преобразованный таргет, затем конвертируем в обычную цену через обратное
        преобразование.

        :param X: pd.DataFrame
        :return: np.array, предсказания (цены на коммерческую недвижимость)
        """
        if self.__is_fitted:
            predictions = self.pipeline.predict(X)
            corrected_price = predictions * (1 + self.corr_coef)
            return corrected_price
        else:
            raise NotFittedError(
                "This {} instance is not fitted yet! Call 'fit' with appropriate arguments before predict".format(
                    type(self).__name__
                )
            )

    def save(self, path: str):
        """Сериализует модель в pickle.

        :param path: str, путь до файла
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, path: str):
        """Сериализует модель в pickle.

        :param path: str, путь до файла
        :return: Модель
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model