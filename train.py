import argparse
import logging.config
import pandas as pd
from traceback import format_exc
import json

from raifhack_ds.models import LightGBMModel
from raifhack_ds.utils import PriceTypeEnum
from raifhack_ds.metrics import metrics_stat
from raifhack_ds.features import FeaturesPreparation
import os


os.mkdir(os.path.join('experiments', config['exp_name']))

with open('config.json') as f:
    config = json.load(f)

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
            "filename": os.path.join('experiments', config['exp_name'], 'logs'),
            "mode": "a",
        },
    },
    "loggers": {
        "": {"handlers": ["file_handler"], "level": "INFO", "propagate": False},
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser(
        description="""
    Бенчмарк для хакатона по предсказанию стоимости коммерческой недвижимости от "Райффайзенбанк"
    Скрипт для обучения модели

     Примеры:
        1) с poetry - poetry run python3 train.py --train_data /path/to/train/data --model_path /path/to/model
        2) без poetry - python3 train.py --train_data /path/to/train/data --model_path /path/to/model
    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--train_data", "-d", type=str, dest="d", required=True, help="Путь до обучающего датасета")
    parser.add_argument("--val_data", "-v", type=str, dest="v", required=True, help="Путь до валидационного датасета")
    parser.add_argument("--test_data", "-t", type=str, dest="t", required=True, help="Путь до отложенной выборки")
    return parser.parse_args()

if __name__ == "__main__":

    try:
        logger.info('START train.py')
        args = vars(parse_args())
        logger.info('Load train df')
        train_df = pd.read_csv(args['d'])
        train_df = train_df.reindex(sorted(train_df.columns), axis=1)
        logger.info('Load validation df')
        val_df = pd.read_csv(args['v'])
        val_df = val_df.reindex(sorted(val_df.columns), axis=1)
        logger.info('Load Test df')
        test_df = pd.read_csv(args['t'])
        test_df = test_df.reindex(sorted(test_df.columns), axis=1)
        logger.info(f'Input shape: {train_df.shape}')

        FP = FeaturesPreparation(config)

        X_train, y_train = FP.prepare(train_df, 'train')
        X_val, y_val = FP.prepare(val_df, 'val')
        X_test = FP.prepare(test_df, 'test')

        X_offer = X_train[train_df.price_type == PriceTypeEnum.OFFER_PRICE]
        y_offer = y_train[train_df.price_type == PriceTypeEnum.OFFER_PRICE]
        X_manual = X_train[train_df.price_type == PriceTypeEnum.MANUAL_PRICE]
        y_manual = y_train[train_df.price_type == PriceTypeEnum.MANUAL_PRICE]
        logger.info(f'X_offer {X_offer.shape}  y_offer {y_offer.shape}\tX_manual {X_manual.shape} y_manual {y_manual.shape}')

        X_val_offer = X_val[val_df.price_type == PriceTypeEnum.OFFER_PRICE]
        y_val_offer = y_val[val_df.price_type == PriceTypeEnum.OFFER_PRICE]
        X_val_manual = X_val[val_df.price_type == PriceTypeEnum.MANUAL_PRICE]
        y_val_manual = y_val[val_df.price_type == PriceTypeEnum.MANUAL_PRICE]

        model = LightGBMModel(feature_names=FP.features_names,
                              categorical_names=FP.categorical_to_model,
                              model_params=config['model']['params'],
                              separate_training=config['separate_training'])
        logger.info('Fit model')
        model.fit(X_offer, y_offer, X_manual, y_manual)
        logger.info('Save model')
        model.save(os.path.join('experiments', config['exp_name'], 'model.pkl'))

        predictions_offer = model.predict(X_offer)
        metrics = metrics_stat(y_offer.values, predictions_offer/(1+model.corr_coef)) # для обучающей выборки с ценами из объявлений смотрим качество без коэффициента
        logger.info(f'Metrics stat for training data with offers prices: {metrics}')

        predictions_manual = model.predict(X_manual)
        metrics = metrics_stat(y_manual.values, predictions_manual)
        logger.info(f'Metrics stat for training data with manual prices: {metrics}')

        # VALIDATION
        predictions_val_offer = model.predict(X_val_offer)
        metrics_val = metrics_stat(y_val_offer.values, predictions_val_offer / (
                    1 + model.corr_coef))  # для обучающей выборки с ценами из объявлений смотрим качество без коэффициента
        logger.info(f'Metrics stat for validation data with offers prices: {metrics_val}')

        predictions_val_manual = model.predict(X_val_manual)
        metrics_val = metrics_stat(y_val_manual.values, predictions_val_manual)
        logger.info(f'Metrics stat for validation data with manual prices: {metrics_val}')

        # TEST
        logger.info('Predict')
        test_df['per_square_meter_price'] = model.predict(X_test)
        logger.info('Save results')
        test_df[['id','per_square_meter_price']].to_csv(os.path.join('experiments', config['exp_name'], 'results.csv'), index=False)
        with open(os.path.join('experiments', config['exp_name'], 'config.json'), 'w') as f:
            json.dump(config, f)


    except Exception as e:
        err = format_exc()
        logger.error(err)
        raise(e)
    logger.info('END train.py')
