#!/usr/bin/env python3
import argparse
import base64
import hashlib
import os
import sys

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from skopt import gp_minimize

sys.path.append('./')
sys.path.append('../')
from merchant_sdk.api import KafkaApi, PricewarsRequester

'''
    Input
'''
# merchant_token = 'z35jXmfpJaK3KnpQpEV3DGQwBZocVgVVjZFHMv7fWRiqFYH5mm8z3YwE8lqeSMAB'
# merchant_token = 'vfOxgR0UsXlnvEg6HoDV6ybPdjes0ERvtpsaHt5ARrxN5eTMTgrYfSxJCcBCqc7k'
merchant_token = None
merchant_id = None
kafka_api = None

default_host = os.getenv('PRICEWARS_KAFKA_REVERSE_PROXY_URL', 'http://kafka-reverse-proxy:8001/')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Machine learning on PriceWars simulation data')
    parser.add_argument('-k', '--kafka_host', metavar='kafka_host', type=str, default=default_host,
                        help='endpoint of kafka reverse proxy', required=False)
    parser.add_argument('-m', '--merchant', metavar='merchant_id', type=str, default=None,
                        help='merchant ID', required=False)
    parser.add_argument('--token', metavar='merchant_token', type=str, default=None,
                        help='merchant Token', required=False)
    parser.add_argument('-t', '--train', metavar='market_situation', type=str, help = 'market situation', required=False)
    parser.add_argument('-b', '--buy', metavar='buy_offers', type=str, help = 'buy offers', required=False)
    parser.add_argument('--test', metavar='test_offers', type=str, help = 'test offers', required=False)
    parser.add_argument('-o', '--output', metavar='output', type=str, help = 'output file', required=False)
    return parser.parse_args()


'''
    Output
'''
market_situation_df = None
buy_offer_df = None
test_df = None
data_products = {}
test_data_products = {}
model_products = {}
result = []


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
    print(market_situation_csv_url)
    market_situation_df = pd.read_csv(market_situation_csv_url)
    buy_offer_csv_url = kafka_api.request_csv_export_for_topic('buyOffer')
    buy_offer_df = pd.read_csv(buy_offer_csv_url)

    # market_situation_df.to_csv('data/ms.csv')
    # buy_offer_df.to_csv('data/buyOffer.csv')


def load_offline():
    global market_situation_df, buy_offer_df, test_df
    market_situation_df = pd.read_csv(args.train)
    buy_offer_df = pd.read_csv(args.buy)
    if args.test is not None:
        test_df = pd.read_csv(args.test)

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
    global merchant_id, data_products, test_data_products, buy_offer_df, market_situation_df, test_df

    own_ms_view = market_situation_df
    own_sales = buy_offer_df#[buy_offer_df['http_code'] == 200].copy()
    test_data = test_df
    own_sales.loc[:, 'timestamp'] = match_timestamps(own_ms_view['timestamp'], own_sales['timestamp'])

    if test_df is None:
        X_train, X_test = train_test_split(
            own_ms_view, test_size=0.4, random_state=0)
        test_data = X_test
        own_ms_view = X_train

    # Train data
    print("Aggregating Training Data")
    for product_id in np.unique(own_ms_view['product_id']):
        ms_df_prod = own_ms_view[own_ms_view['product_id'] == product_id]

        dict_array = []
        for timestamp, group in ms_df_prod.groupby('timestamp'):
            features = extract_features_from_offer_snapshot(group, merchant_id)
            features.update({
                'timestamp': timestamp,
                'sold': own_sales[own_sales['timestamp'] == timestamp]['amount'].sum()
            })
            dict_array.append(features)

        data_products[product_id] = pd.DataFrame(dict_array)
        filename = 'data/product_{}_data.csv'.format(product_id)
        data_products[product_id].to_csv(make_relative_path(filename))

    # Test data
    print("Aggregating Test Data")
    for product_id in np.unique(test_data['product_id']):
        test_data_prod = test_data[test_data['product_id'] == product_id]

        dict_array = []
        for timestamp, group in test_data_prod.groupby('timestamp'):
            features = extract_features_from_offer_snapshot(group, merchant_id)
            features.update({
                'timestamp': timestamp,
                'sold': own_sales[own_sales['timestamp'] == timestamp]['amount'].sum()
            })
            dict_array.append(features)

        test_data_products[product_id] = pd.DataFrame(dict_array)

def log_likelihood(y_true, y_pred, eps=1e-4):
    v_adjust = np.vectorize(lambda pred: min(max(pred, eps), 1 - eps))
    v_adjust(y_pred)
    y_pred[y_pred <= 0] = eps
    ones = np.full(len(y_pred), 1)
    return sum(y_true * np.log(y_pred) + (ones - y_true) * np.log(ones - y_pred))

def aic(num_features, y_true, y_pred):
    return 2 * num_features - 2 * log_likelihood(y_true, y_pred)

def mcFadden_R2(y_true, y_pred):
    constant_feature = pd.DataFrame(np.full(len(y_true), 1))
    logistic_regression = PassiveAggressiveRegressor()
    logistic_regression.fit(constant_feature, y_true)
    null_model_prediction = logistic_regression.predict(constant_feature)
    print('avg log-likelihood null-model: {}'.format(log_likelihood(y_true, null_model_prediction)))

    L = log_likelihood(y_true, y_pred)
    L_null = log_likelihood(y_true, null_model_prediction)
    return 1 - L / L_null

def train(params = [1.0, 0.001]):
    global data_products, model_products
    _C, _epsilon= params

    for product_id in data_products:
        data = data_products[product_id].dropna()
        if len(data.index) <= 0:
            return
        X = data[['amount_of_all_competitors',
                  'average_price_on_market',
                  'distance_to_cheapest_competitor',
                  'price_rank',
                  'quality_rank'
                  ]]
        y = data['sold'].copy()
        y[y > 1] = 1

        model = PassiveAggressiveRegressor(n_iter = 1000)
        model.set_params(C=_C,
                         epsilon=_epsilon)

        model.fit(X, y)

        model_products[product_id] = model

def classify():
    global data_products, model_products, result
    result = []
    y_predicted_sale = []
    y_true = []
    for product_id in data_products:
        data = data_products[product_id].dropna()
        if len(data.index) <= 0:
            return
        X = data[['amount_of_all_competitors',
                  'average_price_on_market',
                  'distance_to_cheapest_competitor',
                  'price_rank',
                  'quality_rank'
                ]]
        y = data['sold'].copy()
        y[y > 1] = 1

        model = model_products[product_id]
        y_predicted = model.predict(X)
        print(y_predicted)

        y_predicted_sale = y_predicted.copy()
        y_true = y
        result.append((product_id, y_predicted.copy()))

    _aic = aic(5, y_true, y_predicted_sale)
    print("AIC: ", _aic)
    _mcFadden_R2 = mcFadden_R2(y_true, y_predicted_sale)
    print("McFadden: ", _mcFadden_R2)
    print("McFadden2: ", r2_score(y_true, y_predicted_sale))

def cross_validate(params):
    global test_data_products, model_products

    _C, _epsilon = params

    data = test_data_products[1].dropna()
    if len(data.index) <= 0:
        return 0
    X = data[['amount_of_all_competitors',
              'average_price_on_market',
              'distance_to_cheapest_competitor',
              'price_rank',
              'quality_rank'
              ]]
    y = data['sold'].copy()
    y[y > 1] = 1
    model = PassiveAggressiveRegressor(n_iter = 1000)
    model.set_params(C=_C,
                     epsilon=_epsilon)

    score = -np.mean(cross_val_score(model, X, y, cv=3, scoring='r2'))
    return score

def optimize():
    space  = [(10**-1, 10**1, "log-uniform"),   # C
              (10**-6, 10**-1, "log-uniform")]   # epsilon

    res_gp = gp_minimize(cross_validate, space, n_calls=100, random_state=0)

    print("Best score= ", res_gp.fun)
    params = [res_gp.x[0], res_gp.x[1]]
    train(params)
    classify()

def save_as_txt(model, filename):
    lines = []
    # append header if file is created newly
    if not os.path.isfile(filename):
        lines.append(','.join([
              'amount_of_all_competitors',
              'average_price_on_market',
              'distance_to_cheapest_competitor',
              'price_rank',
              'quality_rank'
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

def output():
    global result
    np.set_printoptions(threshold=np.inf)
    if(args.output is None):
        return

    output = open(args.output, "w")
    output.write(str(result))
    output.close()


if __name__ == '__main__':
    print('\n\n---------------------------------------')
    sys.stdout.flush()
    print('start learning')
    args = parse_arguments()

    if(args.token is not None):
        merchant_token = args.token
        PricewarsRequester.add_api_token(merchant_token)
        merchant_id = base64.b64encode(hashlib.sha256(merchant_token.encode('utf-8')).digest()).decode('utf-8')

    if(args.merchant is not None):
        merchant_id = args.merchant

    kafka_host = args.kafka_host
    kafka_api = KafkaApi(host=kafka_host)

    print('load')
    if((args.buy is not None) and args.train is not None):
        load_offline()
    else:
        try:
            download()
        except:
            print('No files available to download for learning. Skipping learning.')
            exit()
    print('aggregate ... This takes a while')
    aggregate()
    print('train')
    train()
    print('classify')
    classify()
    print('validate')
    optimize()
    print('export')
    export_models()
    output()
