import ast
import pandas as pd
import numpy as np
from datetime import datetime as dat
from dateutil import relativedelta
from sklearn.linear_model import Lasso
from django_pandas.io import read_frame
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import PriceProduction
from .arima import forecast_accuracy
from .serializers import LassoSerializer
from .models import UnivarientData
class LassoUnivarient(APIView):
    serializer_class = LassoSerializer

    def get(self, request, *args, **kwargs):
        start_date = self.request.query_params.get('startdate', '1970-01-30')
        end_date = self.request.query_params.get('enddate', '2018-01-01')
        data = read_frame(UnivarientData.objects.all())
        data['date'] = pd.to_datetime(data['date'])
        data = data.drop('id', axis=1)
        data = data.set_index('date')
        startdate = dat.strptime(start_date, '%Y-%m-%d')
        enddate = dat.strptime(end_date, '%Y-%m-%d')
        nextmonth = enddate + relativedelta.relativedelta(months=1)
        train, test = data[startdate:nextmonth], data[nextmonth:]
        print(train )
        train['Price_lag'] = train['price'].shift(1)
        train['rolling_mean_price'] = train['Price_lag'].rolling(2, min_periods=1).sum()
        train = train.dropna()
        X = train.drop(["price"], axis=1)
        y = train['price']
        X_train = X[:-1]
        y_train = y[:-1]
        reg = Lasso().fit(X_train, y_train)
        reg.score(X_train, y_train)
        prediction = []
        test_forecast = X[-1:]
        t = reg.predict(test_forecast)
        n = test_forecast.values.tolist()[0]
        t = test_forecast.values
        t = t[0][0]
        for i in range(1, len(test) + 1):
            m = reg.predict(test_forecast)
            forecastdf = pd.DataFrame(columns=['Price_lag'])
            forecastdf['Price_lag'] = [t, m.tolist()[0]]
            forecastdf['rolling_mean_price'] = forecastdf['Price_lag'].rolling(2, min_periods=1).sum()
            print(forecastdf)
            test_forecast = forecastdf[-1:]
            t = m.tolist()[0]
            prediction.append(m.tolist()[0])
        test['prediction'] = prediction
        print(test)
        metrics=forecast_accuracy(test['price'], test['prediction'])
        test['date']=test.index.astype('str')
        actual_data = test[['date', 'price']].values.tolist()
        predicted_data = test[['date', 'prediction']].values.tolist()
        return Response({'actual_data': actual_data, 'predicted_data': predicted_data, 'mape': metrics.get('mape', 0)*100})


class lasso_univarientForecast(APIView):

    def get(self, request, *args, **kwargs):
        n_steps = int(self.request.query_params.get('nsteps', 10))

        data = read_frame(UnivarientData.objects.all())
        data['date'] = pd.to_datetime(data['date'])
        data = data.drop('id', axis=1)
        data = data.set_index('date')
        data['Price_lag'] = data['price'].shift(1)
        data['rolling_mean_price'] = data['Price_lag'].rolling(2, min_periods=1).sum()
        data = data.dropna()
        X = data.drop(["price"], axis=1)
        y = data['price']
        X_train = X[:-1]
        y_train = y[:-1]
        reg = Lasso().fit(X_train, y_train)
        reg.score(X_train, y_train)
        prediction = []
        test_forecast = X[-1:]
        t = reg.predict(test_forecast)
        n = test_forecast.values.tolist()[0]
        t = test_forecast.values
        t = t[0][0]
        for i in range(1, n_steps + 1):
            m = reg.predict(test_forecast)
            forecastdf = pd.DataFrame(columns=['Price_lag'])
            forecastdf['Price_lag'] = [t, m.tolist()[0]]
            forecastdf['rolling_mean_price'] = forecastdf['Price_lag'].rolling(2, min_periods=1).sum()
            print(forecastdf)
            test_forecast = forecastdf[-1:]
            t = m.tolist()[0]
            prediction.append(m.tolist()[0])
        date_index = pd.date_range(start='1/1/2019', periods=n_steps, freq='M')
        data = pd.DataFrame()
        data['prediction'] = prediction
        data['date'] = date_index

        data['date'] = data['date']
        predicted_data = data[['date', 'prediction']].values.tolist()

        return Response({'predicted_data': predicted_data})

class LassoView(APIView):
    def get(self, request, *args, **kwargs):
        body_data = request.data
        data = read_frame(PriceProduction.objects.all())
        data['date'] = pd.to_datetime(data['date'])
        data = data.drop('id', axis=1)
        data = data.set_index('date')
        startdate = dat.strptime(body_data['startdate'], '%Y-%m-%d')
        enddate = dat.strptime(body_data['enddate'], '%Y-%m-%d')
        nextmonth = enddate + relativedelta.relativedelta(months=1)
        train, test = data[startdate:nextmonth], data[nextmonth:]
        train['Price_lag'] = train['price'].shift(1)
        train['rolling_mean_price'] = train['Price_lag'].rolling(
            2, min_periods=1).sum()
        train = train.dropna()
        X = train.drop(["price"], axis=1)
        y = train['price']
        X_train = X[:-1]
        y_train = y[:-1]
        reg = Lasso().fit(X_train, y_train)
        reg.score(X_train, y_train)
        actual_price = pd.DataFrame()
        actual_price = test['price']
        test_data = test.drop(["price"], axis=1)
        prediction = []
        test_forecast = X[-1:]
        t = reg.predict(test_forecast)
        print(test_forecast)
        test_forecast = test_forecast[['production', 'Price_lag']]
        n = test_forecast.values.tolist()[0]
        for i in range(1, len(test_data) + 1):
            forecastdata = test_data[i - 1:i]
            print(forecastdata)
            o = forecastdata.values
            print(y)
            y = list(np.append(o, t))
            forecast = pd.DataFrame(columns=['production', 'Price_lag'])
            forecast.loc[0] = n
            forecast.loc[1] = y
            forecast['rolling_mean_price'] = forecast['Price_lag'].rolling(
                2, min_periods=1).sum()
            t = reg.predict(forecast[-1:].values)
            n = y
            print(forecast)
            print(t)
            prediction.append(t)
        predictdata = pd.DataFrame(
            prediction, index=test_data.index, columns=['price'])
        predictdata['actual'] = actual_price
        metrics = forecast_accuracy(
            predictdata['price'], predictdata['actual'])
        predictdata.index = predictdata.index.astype("str")
        json = predictdata.to_json()
        json = ast.literal_eval(json)
        json['mape'] = metrics['mape']
        return Response(json)
