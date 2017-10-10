import numpy as np
from sklearn.externals import joblib

'''
values:

amount_of_all_competitors, average_price_on_market, distance_to_cheapest_competitor, price_rank, quality_rank
'''

values = np.array([3, 10, 1, 2, 1])
model_file = 'bla.pkl'


def predict(model_file, values):
    model = joblib.load(model_file)
    print('model {:s} with coefficients: {} and values {} --> P = {:f}'.format(
        model_file,
        model.coef_[:,1],
        values,
        model.predict_proba(values)[:,1]
    ))


if __name__ == '__main__':
    predict(model_file, values)
