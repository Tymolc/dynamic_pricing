import sys
import pandas as pd

sys.path.append('./')
sys.path.append('../')
from merchant_sdk.api import KafkaApi, PricewarsRequester

'''
    Input
'''
merchant_token = '2ZnJAUNCcv8l2ILULiCwANo7LGEsHCRJlFdvj18MvG8yYTTtCfqN3fTOuhGCthWf'
merchant_token = 'bTEXsl4wJJomq5h1BaDEWCstSPbcGmIqFWO8IS5bltOcy6eBgrOD3H7Vgh8wUQnk'
PricewarsRequester.add_api_token(merchant_token)
kafka_api = KafkaApi(host='http://vm-mpws2016hp1-05.eaalab.hpi.uni-potsdam.de:8001')
kafka_api = KafkaApi(host='http://127.0.0.1:8001')

topics = ['marketSituation', 'buyOffer']

for topic in topics:
    try:
        url = kafka_api.request_csv_export_for_topic(topic)
        df = pd.read_csv(url)
        df.to_csv(topic + '.csv')
    except:
        print('error on', topic)
