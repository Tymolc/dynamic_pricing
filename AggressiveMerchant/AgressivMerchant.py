import argparse
import sys
import os
import numpy as np
import pandas as pd
import datetime
from sklearn.externals import joblib

sys.path.append('./')
sys.path.append('../')
from merchant_sdk import MerchantBaseLogic, MerchantServer
from merchant_sdk.api import PricewarsRequester, MarketplaceApi, ProducerApi
from merchant_sdk.models import Offer

from machine_learning.market_learning import extract_features_from_offer_snapshot

merchant_token = "DaywOe3qbtT3C8wBBSV+zBOH55DVz40L6PH1/1p9xCM="
parser = argparse.ArgumentParser(description='PriceWars Merchant')
parser.add_argument('--port', type=int,
                    help='port to bind flask App to')
parser.add_argument('-k', '--kafka_host', metavar='kafka_host', type=str,
                    help='endpoint of kafka reverse proxy', required=False)
parser.add_argument('-m', '--merchant', metavar='merchant_id', type=str, default=None,
                    help='merchant ID', required=False)
parser.add_argument('-t', '--train', metavar='market_situation', type=str, help = 'market situation', required=False)
parser.add_argument('-b', '--buy', metavar='buy_offers', type=str, help = 'buy offers', required=False)
parser.add_argument('--test', metavar='test_offers', type=str, help = 'test offers', required=False)
parser.add_argument('-o', '--output', metavar='output', type=str, help = 'output file', required=False)

settings = {
    'merchant_id': MerchantBaseLogic.calculate_id(merchant_token),
    'marketplace_url': MerchantBaseLogic.get_marketplace_url(),
    'producer_url': MerchantBaseLogic.get_producer_url(),
    'kafka_reverse_proxy_url': MerchantBaseLogic.get_kafka_reverse_proxy_url(),
    'debug': True,
    'max_amount_of_offers': 10,
    'shipping': 5,
    'primeShipping': 1,
    'max_req_per_sec': 10.0,
    'minutes_between_learnings': 5.0,
}


def make_relative_path(path):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(script_dir, path)


def trigger_learning(merchant_token, kafka_host):
    args = parser.parse_args()
    fixed_path = make_relative_path("demand_learning.py")
    old_dir = os.getcwd()
    os.chdir(os.path.dirname(fixed_path))
    os.system('python3 "{:s}" -k "{:s}" --merchant "{:s}" --train "{:s}" --buy "{:s}" --test "{:s}" -o "{:s}" >> learning.log &'
        .format(fixed_path, kafka_host, merchant_token, args.train, args.buy, args.test, args.output))
    os.chdir(old_dir)


class AgressivMerchant(MerchantBaseLogic):
    def __init__(self):
        MerchantBaseLogic.__init__(self)
        global settings
        self.settings = settings

        '''
            Predefined API token
        '''
        self.merchant_id = settings['merchant_id']
        self.merchant_token = merchant_token

        '''
            Setup API
        '''
        PricewarsRequester.add_api_token(self.merchant_token)
        self.marketplace_api = MarketplaceApi(host=self.settings['marketplace_url'])
        self.producer_api = ProducerApi(host=self.settings['producer_url'])

        '''
            Setup ML model
        '''
        self.models_per_product = self.load_models_from_filesystem()
        self.last_learning = datetime.datetime.now()
        trigger_learning(self.merchant_token, settings['kafka_reverse_proxy_url'])

        '''
            Start Logic Loop
        '''
        self.run_logic_loop()

    @staticmethod
    def load_models_from_filesystem(folder='models'):
        result = {}
        for root, dirs, files in os.walk(make_relative_path(folder)):
            pkl_files = [f for f in files if f.endswith('.pkl')]
            for pkl_file in pkl_files:
                complete_path = os.path.join(root, pkl_file)
                try:
                    product_id = int(pkl_file.split('.')[0])
                    result[product_id] = joblib.load(complete_path)
                    # print(result[product_id].coef_)
                except ValueError:
                    # do not load model files, that don't have the naming scheme
                    pass
            break
        return result

    def update_api_endpoints(self):
        """
        Updated settings may contain new endpoints, so they need to be set in the api client as well.
        However, changing the endpoint (after simulation start) may lead to an inconsistent state
        :return: None
        """
        self.marketplace_api.host = self.settings['marketplace_url']
        self.producer_api.host = self.settings['producer_url']

    '''
        Implement Abstract methods / Interface
    '''

    def update_settings(self, new_settings):
        MerchantBaseLogic.update_settings(self, new_settings)
        self.update_api_endpoints()
        return self.settings

    def sold_offer(self, offer):
        pass

    '''
        Merchant Logic
    '''

    def price_product(self, product_or_offer, product_prices_by_uid, current_offers=None):
        """
        Computes a price for a product based on trained models or (exponential) random fallback
        :param product_or_offer: product object that is to be priced
        :param current_offers: list of offers
        :return:
        """
        price = product_prices_by_uid[product_or_offer.uid]
        try:
            model = self.models_per_product[product_or_offer.product_id]

            offer_df = pd.DataFrame([o.to_dict() for o in current_offers])
            offer_df = offer_df[offer_df['product_id'] == product_or_offer.product_id]
            own_offers_mask = offer_df['merchant_id'] == self.merchant_id

            features = []
            for potential_price in range(1, 100, 1):
                potential_price_candidate = potential_price / 10.0
                potential_price = price + potential_price_candidate #product_or_offer.price + potential_price_candidate
                offer_df.loc[own_offers_mask, 'price'] = potential_price
                features.append(extract_features_from_offer_snapshot(offer_df, self.merchant_id,
                                                                     product_id=product_or_offer.product_id))
            data = pd.DataFrame(features).dropna()

            try:
                filtered = data[['amount_of_all_competitors',
                                 'average_price_on_market',
                                 'distance_to_cheapest_competitor',
                                 'price_rank',
                                 'quality_rank',
                                 ]]
                data['sell_prob'] = model.predict(filtered)#model.predict_proba(filtered)[:,1]
                data['expected_profit'] = data['sell_prob'] * (data['own_price'] - price)
                print("set price as ", data['own_price'][data['expected_profit'].argmax()])
            except Exception as e:
                print(e)

            return data['own_price'][data['expected_profit'].argmax()]
        except (KeyError, ValueError) as e:
            return price * (np.random.exponential() + 0.99)
        except Exception as e:
            pass

    def execute_logic(self):
        next_training_session = self.last_learning \
                                + datetime.timedelta(minutes=self.settings['minutes_between_learnings'])
        if next_training_session <= datetime.datetime.now():
            self.last_learning = datetime.datetime.now()
            trigger_learning(self.merchant_token, self.settings['kafka_reverse_proxy_url'])

        request_count = 0

        self.models_per_product = self.load_models_from_filesystem()

        try:
            offers = self.marketplace_api.get_offers(include_empty_offers=True)
        except Exception as e:
            print('error on getting offers:', e)
        own_offers = [offer for offer in offers if offer.merchant_id == self.merchant_id]
        own_offers_by_uid = {offer.uid: offer for offer in own_offers}
        missing_offers = settings['max_amount_of_offers'] - sum(offer.amount for offer in own_offers)

        new_products = []
        for _ in range(missing_offers):
            try:
                prod = self.producer_api.buy_product()
                new_products.append(prod)
            except:
                pass

        products = self.producer_api.get_products()
        product_prices_by_uid = {product.uid: product.price for product in products}

        for own_offer in own_offers:
            if own_offer.amount > 0:
                own_offer.price = self.price_product(own_offer, product_prices_by_uid, current_offers=offers)
                try:
                    self.marketplace_api.update_offer(own_offer)
                    request_count += 1
                except Exception as e:
                    print('error on updating offer:', e)

        for product in new_products:
            try:
                if product.uid in own_offers_by_uid:
                    offer = own_offers_by_uid[product.uid]
                    offer.amount += product.amount
                    offer.signature = product.signature
                    try:
                        self.marketplace_api.restock(offer.offer_id, amount=product.amount, signature=product.signature)
                    except Exception as e:
                        print('error on restocking an offer:', e)
                    offer.price = self.price_product(product, product_prices_by_uid, current_offers=offers)
                    try:
                        self.marketplace_api.update_offer(offer)
                        request_count += 1
                    except Exception as e:
                        print('error on updating an offer:', e)
                else:
                    offer = Offer.from_product(product)
                    offer.prime = True
                    offer.shipping_time['standard'] = self.settings['shipping']
                    offer.shipping_time['prime'] = self.settings['primeShipping']
                    offer.merchant_id = self.merchant_id
                    offer.price = self.price_product(product, product_prices_by_uid, current_offers=offers+[offer])
                    try:
                        self.marketplace_api.add_offer(offer)
                    except Exception as e:
                        print('error on adding an offer to the marketplace:', e)
            except Exception as e:
                print('could not handle product:', product, e)

        return max(1.0, request_count) / settings['max_req_per_sec']


merchant_logic = AgressivMerchant()
merchant_server = MerchantServer(merchant_logic)
app = merchant_server.app

if __name__ == "__main__":
    args = parser.parse_args()

    app.run(host='0.0.0.0', port=args.port)
