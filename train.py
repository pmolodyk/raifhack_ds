import argparse
import logging.config
import pandas as pd
from traceback import format_exc

from raifhack_ds.model import BenchmarkModel
from raifhack_ds.model import MODEL_PARAMS, LOGGING_CONFIG, NUM_FEATURES, CATEGORICAL_OHE_FEATURES,CATEGORICAL_STE_FEATURES,TARGET, BAD_REGIONS
from raifhack_ds.utils import PriceTypeEnum
from raifhack_ds.metrics import metrics_stat
from raifhack_ds.features import prepare_categorical, preprocess

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

def bin_cities(df):
    df1 = df.copy()
    df_big_cities = pd.DataFrame(df1.city.value_counts()[:20])
    df_big_cities['big_city'] = 1.0
    df_middle_cities = pd.DataFrame(df1.city.value_counts()[20:100])
    df_middle_cities['middle_city'] = 1.0
    df2 = df1.merge(df_big_cities.drop('city', axis=1), how='left', left_on='city', right_index=True)
    df3 = df2.merge(df_middle_cities.drop('city', axis=1), how='left', left_on='city', right_index=True).drop('city', axis=1)
    return df3
    # df3[['big_city', 'middle_city']] = df3[['big_city', 'middle_city']].fillna(0)

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

    return parser.parse_args()

if __name__ == "__main__":

    try:
        logger.info('START train.py')
        args = vars(parse_args())
        logger.info('Load train df')
        train_df = pd.read_csv(args['d'])
        logger.info('Load validation df')
        val_df = pd.read_csv(args['v'])
        logger.info(f'Input shape: {train_df.shape}')
        train_df = prepare_categorical(train_df)
        val_df = prepare_categorical(val_df)
        train_df = preprocess(train_df)
        X_offer = train_df[train_df.price_type == PriceTypeEnum.OFFER_PRICE][NUM_FEATURES+CATEGORICAL_OHE_FEATURES+CATEGORICAL_STE_FEATURES]
        y_offer = train_df[train_df.price_type == PriceTypeEnum.OFFER_PRICE][TARGET]
        X_manual = train_df[train_df.price_type == PriceTypeEnum.MANUAL_PRICE][NUM_FEATURES+CATEGORICAL_OHE_FEATURES+CATEGORICAL_STE_FEATURES]
        y_manual = train_df[train_df.price_type == PriceTypeEnum.MANUAL_PRICE][TARGET]
        X_all = train_df[NUM_FEATURES+CATEGORICAL_OHE_FEATURES+CATEGORICAL_STE_FEATURES]
        y_all = train_df[TARGET]
        logger.info(f'X_offer {X_offer.shape}  y_offer {y_offer.shape}\tX_manual {X_manual.shape} y_manual {y_manual.shape}')

        X_val_offer = val_df[val_df.price_type == PriceTypeEnum.OFFER_PRICE][
        NUM_FEATURES + CATEGORICAL_OHE_FEATURES + CATEGORICAL_STE_FEATURES]
        y_val_offer = val_df[val_df.price_type == PriceTypeEnum.OFFER_PRICE][TARGET]
        X_val_manual = val_df[val_df.price_type == PriceTypeEnum.MANUAL_PRICE][
        NUM_FEATURES + CATEGORICAL_OHE_FEATURES + CATEGORICAL_STE_FEATURES]
        y_val_manual = val_df[val_df.price_type == PriceTypeEnum.MANUAL_PRICE][TARGET]
        models = {}
        for s in BAD_REGIONS:
            models[s] = BenchmarkModel(numerical_features=NUM_FEATURES, ohe_categorical_features=CATEGORICAL_OHE_FEATURES,
                                  ste_categorical_features=CATEGORICAL_STE_FEATURES, model_params=MODEL_PARAMS)
            logger.info('Fit {} model'.format(s))
            models[s].fit(X_manual[X_manual.region == s], y_manual[X_manual.region == s], X_all[X_all.region == s], y_all[X_all.region == s])
            logger.info('Save model')
            models[s].save('model' + s)

        model = BenchmarkModel(numerical_features=NUM_FEATURES,
                                      ohe_categorical_features=CATEGORICAL_OHE_FEATURES,
                                      ste_categorical_features=CATEGORICAL_STE_FEATURES, model_params=MODEL_PARAMS)
        logger.info('Fit model')
        model.fit(X_manual[~X_manual.region.isin(BAD_REGIONS)], y_manual[~X_manual.region.isin(BAD_REGIONS)],
                         X_all[~X_all.region.isin(BAD_REGIONS)], y_all[~X_all.region.isin(BAD_REGIONS)])
        logger.info('Save model')
        model.save('model')

        #VALIDATION
        for s in BAD_REGIONS:
            predictions_val_manual_city = models[s].predict(X_val_manual[X_val_manual.region == s][NUM_FEATURES + CATEGORICAL_OHE_FEATURES + CATEGORICAL_STE_FEATURES])
            X_val_manual.loc[X_val_manual.region == s, 'prediction'] = predictions_val_manual_city

        predictions_val_manual_other = model.predict(X_val_manual[~X_val_manual.region.isin(BAD_REGIONS)][NUM_FEATURES + CATEGORICAL_OHE_FEATURES + CATEGORICAL_STE_FEATURES])
        X_val_manual.loc[~X_val_manual.region.isin(BAD_REGIONS), 'prediction'] = predictions_val_manual_other
        metrics_val = metrics_stat(y_val_manual.values, X_val_manual['prediction'].values)
        logger.info(f'Metrics stat for validation data with manual prices: {metrics_val}')


    except Exception as e:
        err = format_exc()
        logger.error(err)
        raise(e)
    logger.info('END train.py')