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
from raifhack_ds.data_transformers import SmoothedTargetEncoding

logger = logging.getLogger(__name__)


class BaseModel():

    def __init__(self, feature_names, categorical_names, model_params, separate_training):
        self.feature_names = feature_names
        self.categorical_names = categorical_names
        self.model = None
        self.separate_training = separate_training
        self._is_fitted = False
        self.corr_coef = 0

    def _find_corr_coefficient(self, X_manual: pd.DataFrame, y_manual: pd.Series):
        """Вычисление корректирующего коэффициента

        :param X_manual: pd.DataFrame с ручными оценками
        :param y_manual: pd.Series - цены ручника
        """
        predictions = self.model.predict(X_manual)
        deviation = ((y_manual - predictions)/predictions).median()
        self.corr_coef = deviation

    def fit(self, X_offer_dict, y_offer, X_manual_dict, y_manual):
            pass

    def predict(self, X: pd.DataFrame) -> np.array:
        """Предсказание модели Предсказываем преобразованный таргет, затем конвертируем в обычную цену через обратное
        преобразование.

        :param X: pd.DataFrame
        :return: np.array, предсказания (цены на коммерческую недвижимость)
        """
        X_numpy = X.to_numpy()
        if self.__is_fitted:
            predictions = self.model.predict(X_numpy)
            if not self.separate_training:
                return predictions
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


class LightGBMModel(BaseModel):
    def __init__(self, feature_names, categorical_names, model_params, separate_training):
        super().__init__(feature_names, categorical_names, model_params, separate_training)
        self.model = LGBMRegressor(**model_params)
        self._is_fitted = False
        self.corr_coef = 0

    def fit(self, X_offer, y_offer,
            X_manual, y_manual):
        X_offer_numpy = X_offer.to_numpy()
        y_offer_numpy  = y_offer.to_numpy()
        X_manual_numpy  = X_manual.to_numpy()
        y_manual_numpy  = y_manual.to_numpy()
        logger.info('Fit')
        if self.separate_training:
            self.model.fit(X_offer_numpy, y_offer_numpy,
                           feature_name=self.feature_names,
                           categorical_feature=self.categorical_names)
            logger.info('Find corr coefficient')
            self._find_corr_coefficient(X_manual, y_manual)
            logger.info(f'Corr coef: {self.corr_coef:.2f}')
            self.__is_fitted = True
        else:
            concatted_matrix = np.hstack([X_offer_numpy, X_manual_numpy])
            self.model.fit(concatted_matrix, y_offer_numpy,
                           feature_name=self.feature_names,
                           categorical_feature=self.categorical_names)
            logger.info('Find corr coefficient')
            self._find_corr_coefficient(X_manual, y_manual)
            logger.info(f'Corr coef: {self.corr_coef:.2f}')
            self.__is_fitted = True
            self.__is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.array:
            """Предсказание модели Предсказываем преобразованный таргет, затем конвертируем в обычную цену через обратное
            преобразование.

            :param X: pd.DataFrame
            :return: np.array, предсказания (цены на коммерческую недвижимость)
            """
            X_numpy = X.to_numpy()
            if self.__is_fitted:
                predictions = self.model.predict(X_numpy)
                if not self.separate_training:
                    return predictions
                corrected_price = predictions * (1 + self.corr_coef)
                return corrected_price
            else:
                raise NotFittedError(
                    "This {} instance is not fitted yet! Call 'fit' with appropriate arguments before predict".format(
                        type(self).__name__
                    )
                )
