import ast
import json
import time
import pickle
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from rest_framework.response import Response
from rest_framework.views import APIView
from pyramid.arima import auto_arima
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime as dat
from dateutil import relativedelta
from .models import univarientdata
from django_pandas.io import read_frame
from .serializers import ArimaSerializer


def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    mins = np.amin(np.hstack([forecast[:, None],
                              actual[:, None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:, None],
                              actual[:, None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'mape': mape, 'me': me, 'mae': mae,
            'mpe': mpe, 'minmax': minmax})


class ArimaUnivarient(APIView):
    serializer_class = ArimaSerializer

    def get(self, request, *args, **kwargs):
        start_date = self.request.query_params.get('startdate', '1990-01-31')
        end_date = self.request.query_params.get('enddate', '2018-06-30')

        data = read_frame(univarientdata.objects.all())
        data['date'] = pd.to_datetime(data['date'])
        data = data.drop('id', axis=1)
        data = data.set_index('date')

        startdate = dat.strptime(start_date, '%Y-%m-%d')
        enddate = dat.strptime(end_date, '%Y-%m-%d')

        nextmonth = enddate + relativedelta.relativedelta(months=1)

        train, test = data[startdate:nextmonth], data[nextmonth:]
        arima = auto_arima(train, error_action='ignore', trace=1,
                           seasonal=True, m=12)
        predict = arima.predict(n_periods=test.shape[0])

        filename = 'arimamodel.sav'
        pickle.dump(arima, open(filename, 'wb'))
        predictdata = pd.DataFrame(
            predict, index=test.index, columns=['predictprice'])
        metrics = forecast_accuracy(predictdata.values, test.values)
        predictdata['actual'] = test.values
        predictdata['date'] = predictdata.index.astype('str')
        
        actual_data = predictdata[['date', 'actual']].values.tolist()
        predicted_data = predictdata[['date', 'predictprice']].values.tolist()

        return Response({'actual_data': actual_data, 'predicted_data': predicted_data, 'mape': metrics.get('mape', 0)})


class ArimaForeCast(APIView):

    def get(self, request, *args, **kwargs):
        n_steps = self.request.query_params.get('nsteps', 10)
        model = pickle.load(open('arimamodel.sav', 'rb'))

        data = read_frame(univarientdata.objects.all())
        data['date'] = pd.to_datetime(data['date'])
        data = data.drop('id', axis=1)
        data = data.set_index('date')

        arima = auto_arima(data, error_action='ignore', trace=1,
                           seasonal=True, m=12)
        date_index = pd.date_range(start='1/1/2019', periods=n_steps, freq='M')
        data = pd.DataFrame()
        data['prediction'] = arima.predict(n_periods=n_steps)
        data['date'] = date_index

        data['date'] = data['date']
        predicted_data = data[['date', 'prediction']].values.tolist()
        print(predicted_data)

        return Response({'predicted_data': predicted_data})
