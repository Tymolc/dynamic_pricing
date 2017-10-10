import argparse
import base64
import hashlib
import os
import sys

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

sys.path.append('./')
sys.path.append('../')
from merchant_sdk.api import KafkaApi, PricewarsRequester

'''
    Input
'''
merchant_token = 'z35jXmfpJaK3KnpQpEV3DGQwBZocVgVVjZFHMv7fWRiqFYH5mm8z3YwE8lqeSMAB'
# merchant_token = '2ZnJAUNCcv8l2ILULiCwANo7LGEsHCRJlFdvj18MvG8yYTTtCfqN3fTOuhGCthWf'
merchant_id = None
kafka_api = None

default_host = os.getenv('PRICEWARS_KAFKA_REVERSE_PROXY_URL', 'http://vm-mpws2016hp1-05.eaalab.hpi.uni-potsdam.de:8001')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Machine learning on PriceWars simulation data')
    parser.add_argument('-k', '--kafka_host', metavar='kafka_host', type=str, default=default_host,
                        help='endpoint of kafka reverse proxy', required=True)
    parser.add_argument('-t', '--merchant_token', metavar='merchant_token', type=str, default=merchant_token,
                        help='merchant token', required=True)
    return parser.parse_args()


'''
    Output
'''
market_situation_df = None
buy_offer_df = None
data_products = {}
model_products = {}


def make_relative_path(path):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(script_dir, path)


def match_timestamps(continuous_timestamps, point_timestamps):
    t_ms = pd.DataFrame({
        'timestamp': continuous_timestamps,
        'origin': np.zeros((len(continuous_timestamps)))
    })
    t_bo = pd.DataFrame({
        'timestamp': point_timestamps,
        'origin': np.ones((len(point_timestamps)))
    })

    t_combined = pd.concat([t_ms, t_bo], axis=0).sort_values(by='timestamp')
    original_locs = t_combined['origin'] == 1

    t_combined.loc[original_locs, 'timestamp'] = np.nan
    # pad: propagates last marketSituation timestamp to all following (NaN) buyOffers
    t_padded = t_combined.fillna(method='pad')

    return t_padded[original_locs]['timestamp']


def download():
    global market_situation_df, buy_offer_df

    market_situation_csv_url = kafka_api.request_csv_export_for_topic('marketSituation')
    market_situation_df = pd.read_csv(market_situation_csv_url)
    buy_offer_csv_url = kafka_api.request_csv_export_for_topic('buyOffer')
    buy_offer_df = pd.read_csv(buy_offer_csv_url)

    # market_situation_df.to_csv('data/ms.csv')
    # buy_offer_df.to_csv('data/buyOffer.csv')


def load_offline():
    global market_situation_df, buy_offer_df
    market_situation_df = pd.read_csv('../../marketSituation.csv')
    buy_offer_df = pd.read_csv('../../buyOffer.csv')


def extract_features_from_offer_snapshot(offers_df, merchant_id, product_id=None):
    if product_id:
        offers_df = offers_df[offers_df['product_id'] == product_id]
    competitors = offers_df[offers_df['merchant_id'] != merchant_id]
    own_situation = offers_df[offers_df['merchant_id'] == merchant_id]
    has_offer = len(own_situation) > 0
    has_competitors = len(competitors) > 0

    if has_offer:
        own_offer = own_situation.sort_values(by='price').iloc[0]
        own_price = own_offer['price']
        own_quality = own_offer['quality']
        price_rank = 1 + (offers_df['price'] < own_price).sum() + ((offers_df['price'] == own_price).sum()/2)
        distance_to_cheapest_competitor = float(own_price - competitors['price'].min()) if has_competitors else np.nan
        quality_rank = (offers_df['quality'] < own_quality).sum() + 1
    else:
        own_price = np.nan
        price_rank = np.nan
        distance_to_cheapest_competitor = np.nan
        quality_rank = np.nan

    amount_of_all_competitors = len(competitors)
    average_price_on_market = offers_df['price'].mean()
    return {
        'own_price': own_price,
        'price_rank': price_rank,
        'distance_to_cheapest_competitor': distance_to_cheapest_competitor,
        'quality_rank': quality_rank,
        'amount_of_all_competitors': amount_of_all_competitors,
        'average_price_on_market': average_price_on_market
    }


def aggregate():
    """
    aggregate is going to transform the downloaded two csv it into a suitable data format, based on:
        $timestamp_1, $merchant_id_1, $product_id, $quality, $price
        $timestamp_1, $product_id, $sku, $price

        $timestamp_1, $sold_yes_no, $own_price, $own_price_rank, $cheapest_competitor, $best_competitor_quality
    :return:
    """
    global merchant_id, data_products, buy_offer_df, market_situation_df

    own_ms_view = market_situation_df
    own_sales = buy_offer_df[buy_offer_df['http_code'] == 200].copy()
    own_sales.loc[:, 'timestamp'] = match_timestamps(own_ms_view['timestamp'], own_sales['timestamp'])

    for product_id in np.unique(own_ms_view['product_id']):
        ms_df_prod = own_ms_view[own_ms_view['product_id'] == product_id]

        dict_array = []
        for timestamp, group in ms_df_prod.groupby('timestamp'):
            features = extract_features_from_offer_snapshot(group, merchant_id)
            features.update({
                'timestamp': timestamp,
                'sold': own_sales[own_sales['timestamp'] == timestamp]['amount'].sum(),
            })
            dict_array.append(features)

        data_products[product_id] = pd.DataFrame(dict_array)
        filename = 'data/product_{}_data_ML.csv'.format(product_id)
        data_products[product_id].to_csv(make_relative_path(filename))


def train():
    global data_products, model_products

    for product_id in data_products:
        data = data_products[product_id].dropna()
        X = data[['amount_of_all_competitors',
                  'average_price_on_market',
                  'distance_to_cheapest_competitor',
                  'price_rank',
                  'quality_rank',
                  ]]
        y = data['sold'].copy()
        y[y > 1] = 1

        model = LogisticRegression()
        model.fit(X, y)

        model_products[product_id] = model


def save_as_txt(model, filename):
    lines = []
    # prepend header if file is created newly
    if not os.path.isfile(filename):
        lines.append(','.join([
            'amount_of_all_competitors',
            'average_price_on_market',
            'distance_to_cheapest_competitor',
            'price_rank',
            'quality_rank',
            ]))
    lines.append(','.join(['{:f}'.format(coef) for coef in np.ndarray.flatten(model.coef_)]))
    open(filename, 'w+').writelines(lines)


def export_models():
    global model_products
    for product_id in model_products:
        model = model_products[product_id]
        filename = 'models/{}.pkl'.format(product_id)
        joblib.dump(model, make_relative_path(filename))
        save_as_txt(model, make_relative_path(filename.split('.')[0] + '.csv'))


if __name__ == '__main__':
    print('start learning')
    args = parse_arguments()

    merchant_token = args.merchant_token
    merchant_id = base64.b64encode(hashlib.sha256(merchant_token.encode('utf-8')).digest()).decode('utf-8')
    merchant_id = 'DaywOe3qbtT3C8wBBSV+zBOH55DVz40L6PH1/1p9xCM='
    PricewarsRequester.add_api_token(merchant_token)

    kafka_host = args.kafka_host
    kafka_api = KafkaApi(host=kafka_host)
    print('params:', merchant_token, kafka_host)

    print('download')
    # download()
    load_offline()
    print('aggregate')
    aggregate()
    print('train')
    train()
    print('export')
    export_models()
