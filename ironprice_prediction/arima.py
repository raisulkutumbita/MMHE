import ast
import pickle
import pyramid
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


class arima_univarient(APIView):
    def post(self, request, *args, **kwargs):
        body_data = request.data
        data = read_frame(univarientdata.objects.all())
        data['date'] = pd.to_datetime(data['date'])
        data = data.drop('id', axis=1)
        data = data.set_index('date')
        startdate = dat.strptime(body_data['startdate'], '%Y-%m-%d')
        enddate = dat.strptime(body_data['enddate'], '%Y-%m-%d')
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
        print(predictdata)
        predictdata.index = predictdata.index.astype("str")
        json = predictdata.to_json()
        json = ast.literal_eval(json)
        json['mape'] = metrics['mape']
        train.index = train.index.astype("str")
        train_json = train.to_json()
        train_json = ast.literal_eval(train_json)

        return Response({'test_json': json, 'train_json': train_json})


class arimaforecast(APIView):
    def post(self, request, *args, **kwargs):
        body_data = request.data
        n_steps = int(body_data['nsteps'])
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
        data.index = date_index
        data.index = data.index.astype("str")
        json = data.to_json()
        json = ast.literal_eval(json)
        return Response(json)
