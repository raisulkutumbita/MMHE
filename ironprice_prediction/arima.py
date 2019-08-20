import datetime
import numpy as np
import pandas as pd
from datetime import datetime as dat
from dateutil import relativedelta
from django_pandas.io import read_frame
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import UnivarientData, MultivarientData
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


class Arimamultivarient(APIView):
    serializer_class = ArimaSerializer

    def get(self, request, *args, **kwargs):
        first_date = str(MultivarientData.objects.earliest('date').date)
        last_date = str(MultivarientData.objects.latest(
            'date').date - datetime.timedelta(days=250))

        start_date = self.request.query_params.get('startdate', first_date)
        end_date = self.request.query_params.get('enddate', last_date)

        if end_date > last_date:
            end_date = last_date

        date_valid = MultivarientData.objects.exclude(
            date__gt=end_date).exclude(date__lt=start_date)
        if not date_valid:
            start_date = first_date
            end_date = last_date

        data = read_frame(MultivarientData.objects.all())
        data['date'] = pd.to_datetime(data['date'])
        data = data.drop('id', axis=1)
        data = data.set_index('date')

        startdate = dat.strptime(start_date, '%Y-%m-%d')
        enddate = dat.strptime(end_date, '%Y-%m-%d')

        nextmonth = enddate + relativedelta.relativedelta(months=1)
        train, test = data[startdate:nextmonth], data[nextmonth:]
        oiltrain = train['oil_price']
        arima = SARIMAX(endog=oiltrain, order=(3, 0, 4), freq='M', seasonal_order=(0, 1, 1, 6), trend='t',
                        enforce_stationarity=False, enforce_invertibility=False).fit()
        test['oilpriceprediction'] = arima.predict(
            test.index.min(), test.index.max())
        arima = SARIMAX(endog=train['iron_price'], exog=train[['oil_price']], order=(1, 0, 0), freq='M',
                        seasonal_order=(0, 1, 1, 6), trend='t', enforce_stationarity=False,
                        enforce_invertibility=False).fit()

        test['ironpriceprediction'] = arima.predict(test.index.min(), test.index.max(),
                                                    exog=test[['oilpriceprediction']])
        test['date'] = test.index.astype('str')

        ironactual_data = test[['date', 'iron_price']].values.tolist()
        ironpredicted_data = test[[
            'date', 'ironpriceprediction']].values.tolist()
        oilactual_data = test[['date', 'oil_price']].values.tolist()
        oilpredicted_data = test[[
            'date', 'oilpriceprediction']].values.tolist()
        ironmetrics = forecast_accuracy(
            test['iron_price'], test['ironpriceprediction'])
        oilmetrics = forecast_accuracy(
            test['oil_price'], test['oilpriceprediction'])
        return Response({
            'actual_irondata': ironactual_data, 
            'predicted_irondata': ironpredicted_data, 
            'actual_oildata': oilactual_data, 
            'predicted_oildata': oilpredicted_data, 
            'ironmape': ironmetrics.get('mape', 0)*100, 
            'oilmape': oilmetrics.get('mape', 0)*100}
        )


class ArimaMultivarientForecast(APIView):

    def get(self, request, *args, **kwargs):
        n_steps = int(self.request.query_params.get('nsteps', 10))

        last_date = MultivarientData.objects.latest(
            'date').date + datetime.timedelta(days=30)

        data = read_frame(MultivarientData.objects.all())
        data['date'] = pd.to_datetime(data['date'])
        data = data.drop('id', axis=1)
        data = data.set_index('date')
        oildf = data['oil_price']
        date_index = pd.date_range(start=last_date, periods=n_steps, freq='M')
        df = pd.DataFrame()
        arima = SARIMAX(endog=oildf, order=(3, 0, 4), freq='M', seasonal_order=(0, 1, 1, 6), trend='t',
                        enforce_stationarity=False, enforce_invertibility=False).fit()
        df['oilpriceprediction'] = arima.predict(
            date_index.min(), date_index.max())
        arima = SARIMAX(endog=data['iron_price'], exog=data[['oil_price']], order=(1, 0, 0), freq='M',
                        seasonal_order=(0, 1, 1, 6), trend='t', enforce_stationarity=False,
                        enforce_invertibility=False).fit()

        df['ironpriceprediction'] = arima.predict(df.index.min(), df.index.max(),
                                                  exog=df[['oilpriceprediction']])
        df['date'] = df.index
        predicted_data = df[['date', 'ironpriceprediction',
                             'oilpriceprediction']].values.tolist()
        return Response({'predicted_data': predicted_data})


class ArimaUnivarient(APIView):
    serializer_class = ArimaSerializer

    def get(self, request, *args, **kwargs):
        first_date = str(UnivarientData.objects.earliest('date').date)
        last_date = str(UnivarientData.objects.latest(
            'date').date - datetime.timedelta(days=250))

        start_date = self.request.query_params.get('startdate', first_date)
        end_date = self.request.query_params.get('enddate', last_date)

        if end_date > last_date:
            end_date = last_date

        date_valid = UnivarientData.objects.exclude(
            date__gt=end_date).exclude(date__lt=start_date)

        if not date_valid:
            start_date = first_date
            end_date = last_date

        data = read_frame(UnivarientData.objects.all())
        data['date'] = pd.to_datetime(data['date'])
        data = data.drop('id', axis=1)
        data = data.set_index('date')

        startdate = dat.strptime(start_date, '%Y-%m-%d')
        enddate = dat.strptime(end_date, '%Y-%m-%d')

        nextmonth = enddate + relativedelta.relativedelta(months=1)

        train, test = data[startdate:nextmonth], data[nextmonth:]

        arima = SARIMAX(train, order=(1, 0, 2), freq='M', seasonal_order=(
            1, 1, 2, 6), trend='t', enforce_stationarity=False, enforce_invertibility=False).fit()
        predict = arima.predict(test.index.min(), test.index.max())
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

        last_date = UnivarientData.objects.latest(
            'date').date + datetime.timedelta(days=30)

        data = read_frame(UnivarientData.objects.all())
        data['date'] = pd.to_datetime(data['date'])
        data = data.drop('id', axis=1)
        data = data.set_index('date')
        arima = SARIMAX(data, order=(1, 0, 2), freq='M', seasonal_order=(1, 2, 1, 6),
                        enforce_stationarity=False, enforce_invertibility=False, ).fit()
        date_index = pd.date_range(start=last_date, periods=n_steps, freq='M')
        data = pd.DataFrame()
        data['prediction'] = arima.predict(date_index.min(), date_index.max())
        data['date'] = date_index
        data['date'] = data['date']
        predicted_data = data[['date', 'prediction']].values.tolist()
        return Response({'predicted_data': predicted_data})
