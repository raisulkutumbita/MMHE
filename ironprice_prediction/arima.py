import numpy as np
import pandas as pd
from datetime import datetime as dat
from dateutil import relativedelta
from django_pandas.io import read_frame
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import UnivarientData
from .serializers import ArimaSerializer
from statsmodels.tsa.statespace.sarimax import SARIMAX
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
        start_date = self.request.query_params.get('startdate', '1970-01-30')
        end_date = self.request.query_params.get('enddate', '2018-01-01')
        data = read_frame(UnivarientData.objects.all())
        data['date'] = pd.to_datetime(data['date'])
        data = data.drop('id', axis=1)
        data = data.set_index('date')
        startdate = dat.strptime(start_date, '%Y-%m-%d')
        enddate = dat.strptime(end_date, '%Y-%m-%d')
        maxdate=data.index.max()
        if enddate > maxdate:
            end_date='2018-01-01'
            enddate = dat.strptime(end_date, '%Y-%m-%d')

        nextmonth = enddate + relativedelta.relativedelta(months=1)
        train, test = data[startdate:nextmonth], data[nextmonth:]
        arima = SARIMAX(train,order=(1, 0, 2),freq='M',seasonal_order=(1, 1, 2, 6), trend='t', enforce_stationarity=False, enforce_invertibility=False).fit()
        predict=arima.predict(test.index.min(),test.index.max())
        predictdata = pd.DataFrame(
            predict, index=test.index, columns=['predictprice'])
        metrics = forecast_accuracy(predictdata.values, test.values)

        predictdata['actual'] = test.values
        predictdata['date'] = predictdata.index.astype('str')

        actual_data = predictdata[['date', 'actual']].values.tolist()
        predicted_data = predictdata[['date', 'predictprice']].values.tolist()

        return Response({'actual_data': actual_data, 'predicted_data': predicted_data, 'mape': metrics.get('mape', 0)*100})


class ArimaForecast(APIView):

    def get(self, request, *args, **kwargs):
        n_steps = int(self.request.query_params.get('nsteps', 10))

        data = read_frame(UnivarientData.objects.all())
        data['date'] = pd.to_datetime(data['date'])
        data = data.drop('id', axis=1)
        data = data.set_index('date')
        arima = SARIMAX(data, order=(1, 0, 2), freq='M', seasonal_order=(1, 2, 1, 6),
                        enforce_stationarity=False, enforce_invertibility=False, ).fit()
        date_index = pd.date_range(start='7/1/2019', periods=n_steps, freq='M')
        data = pd.DataFrame()
        data['prediction']  = arima.predict(date_index.min(),date_index.max())
        data['date'] = date_index
        data['date'] = data['date']
        predicted_data = data[['date', 'prediction']].values.tolist()
        return Response({'predicted_data': predicted_data})
