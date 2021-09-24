import argparse
import logging.config
import pandas as pd
from traceback import format_exc
import json

from raifhack_ds.model import BenchmarkModel
from raifhack_ds.settings import LOGGING_CONFIG
from raifhack_ds.utils import PriceTypeEnum
from raifhack_ds.metrics import metrics_stat
from raifhack_ds.features import FeaturesPreparation

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
    parser.add_argument("--model_path", "-mp", type=str, dest="mp", required=True, help="Куда сохранить обученную ML модель")
    parser.add_argument("--val_data", "-v", type=str, dest="v", required=True, help="Путь до валидационного датасета")
    parser.add_argument("--config", "-c", type=str, dest="v", required=True, help="Конфиг препроцесса")
    parser.add_argument("--test_data", "-t", type=str, dest="d", required=True, help="Путь до отложенной выборки")
    parser.add_argument("--output", "-o", type=str, dest="o", required=True, help="Путь до выходного файла")
    return parser.parse_args()

if __name__ == "__main__":

    try:
        logger.info('START train.py')
        args = vars(parse_args())
        logger.info('Load train df')
        train_df = pd.read_csv(args['d'])
        logger.info('Load validation df')
        val_df = pd.read_csv(args['v'])
        logger.info('Load Test df')
        test_df = pd.read_csv(args['t'])
        logger.info(f'Input shape: {train_df.shape}')

        with open(args[c]) as f:
            config = json.load(f)

        FP = FeaturesPreparation(config)

        train_df = FP.prepare_train(train_df)
        val_df = FP.prepare_test(val_df)
        test_df = FP.prepare_test(test_df)

        NUM_FEATURES = config.get('num_features')
        CATEGORICAL_OHE = config.get('categorical_ohe')
        CATEGORICAL_ORDINAL = config.get('categorical_ordinal')
        CATEGORICAL_TOMODEL = config.get('categorical_tomodel')

        X_offer = train_df[train_df.price_type == PriceTypeEnum.OFFER_PRICE][NUM_FEATURES+CATEGORICAL_OHE+CATEGORICAL_ORDINAL+CATEGORICAL_TOMODEL]
        y_offer = train_df[train_df.price_type == PriceTypeEnum.OFFER_PRICE][TARGET]
        X_manual = train_df[train_df.price_type == PriceTypeEnum.MANUAL_PRICE][NUM_FEATURES+CATEGORICAL_OHE+CATEGORICAL_ORDINAL+CATEGORICAL_TOMODEL]
        y_manual = train_df[train_df.price_type == PriceTypeEnum.MANUAL_PRICE][TARGET]
        logger.info(f'X_offer {X_offer.shape}  y_offer {y_offer.shape}\tX_manual {X_manual.shape} y_manual {y_manual.shape}')

        X_val_offer = val_df[val_df.price_type == PriceTypeEnum.OFFER_PRICE][NUM_FEATURES+CATEGORICAL_OHE+CATEGORICAL_ORDINAL+CATEGORICAL_TOMODEL]
        y_val_offer = val_df[val_df.price_type == PriceTypeEnum.OFFER_PRICE][TARGET]
        X_val_manual = val_df[val_df.price_type == PriceTypeEnum.MANUAL_PRICE][NUM_FEATURES+CATEGORICAL_OHE+CATEGORICAL_ORDINAL+CATEGORICAL_TOMODEL]
        y_val_manual = val_df[val_df.price_type == PriceTypeEnum.MANUAL_PRICE][TARGET]

        model = BenchmarkModel(numerical_features=NUM_FEATURES, ohe_categorical_features=CATEGORICAL_OHE_FEATURES,
                                  ste_categorical_features=CATEGORICAL_STE_FEATURES, model_params=MODEL_PARAMS)
        logger.info('Fit model')
        model.fit(X_offer, y_offer, X_manual, y_manual)
        logger.info('Save model')
        model.save(args['mp'])

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
        test_df['per_square_meter_price'] = model.predict(test_df[NUM_FEATURES+CATEGORICAL_OHE+CATEGORICAL_ORDINAL+CATEGORICAL_TOMODEL])
        logger.info('Save results')
        test_df[['id','per_square_meter_price']].to_csv(args['o'], index=False)


    except Exception as e:
        err = format_exc()
        logger.error(err)
        raise(e)
    logger.info('END train.py')