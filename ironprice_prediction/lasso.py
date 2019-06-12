import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from .models import price_production
from django_pandas.io import read_frame
from rest_framework.response import Response
from datetime import datetime as dat
from rest_framework.views import APIView
import ast
from .arima import forecast_accuracy
from dateutil import relativedelta


class lasso(APIView):
    def post(self, request, *args, **kwargs):
        body_data = request.data
        data = read_frame(price_production.objects.all())
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
            # forecast['Price_diff']=forecast['Price_lag'].diff(periods=1)
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
