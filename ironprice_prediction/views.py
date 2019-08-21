import ast
import time
import datetime
import json
import pandas as pd
import numpy as np
from django.shortcuts import render
from django_pandas.io import read_frame

from rest_framework.response import Response
from rest_framework.views import APIView

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from .models import UnivarientData, CycloneData,MultivarientData


class CycloneClimateView(APIView):

    def get(self, request, *args, **kwargs):
        queryset = CycloneData.objects.values()
        cyclone = []
        
        for item in queryset:
            date = time.mktime(time.strptime(str(item['date']), "%Y-%m-%d"))
            arr = [date, item['climate']]
            cyclone.append(arr)

        return Response({'results': cyclone})


class datahead(APIView):

    def get(self, request, *args, **kwargs):
        data = pd.read_csv('/home/blueschemeai/Downloads/final_iron.csv')
        data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')
        df = data.sort_values('date')
        df = df.set_index('date')
        df = df.dropna()
        df.index = df.index.astype('str')
        json = df.to_json()
        json = ast.literal_eval(json)
        return Response(json)


class MultivarientDataPreview(APIView):

    def get(self, request, *args, **kwargs):
        queryset = MultivarientData.objects.all()
        oil_data = [[item.date, item.iron_price] for item in queryset]
        iron_data = [[item.date, item.oil_price] for item in queryset]

        return Response({'oil_data': oil_data, 'iron_data': iron_data})


class UnivarientDataPreview(APIView):

    def get(self, request, *args, **kwargs):
        queryset = UnivarientData.objects.values()
        results = []
        
        for item in queryset:
            arr = [item['date'], item['price']]
            results.append(arr)

        return Response({'results': results})


class random_rf(APIView):

    def get(self, request, *args, **kwargs):
        data = pd.read_csv('/home/blueschemeai/Downloads/final_iron.csv')
        data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')
        df = data.sort_values('date')
        df['Month'] = df['date'].dt.month
        df['Price_lag'] = df['price'].shift(1)
        df['Price_diff'] = df['Price_lag'].diff(periods=1)
        df = df.set_index('date')
        df['rolling_mean_price'] = df['Price_lag'].rolling(
            2, min_periods=1).sum()
        df = df.dropna()
        X = df.drop(["price"], axis=1)
        y = df['price']
        X_train = X[:-1]
        y_train = y[:-1]
        # reg=RandomForestRegressor(max_depth=5,random_state=0,n_estimators=100,max_features=1)
        param_grid = {
            'min_samples_split': (5, 10, 20),
            'min_samples_leaf': (5, 10, 20),
            'max_depth': np.arange(3, 10, 20),
            "n_estimators": (50, 100, 300, 500),
        }

        rf_regression = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
        reg = rf_regression.fit(X_train, y_train)
        reg.score(X_train, y_train)
        test_data = pd.read_csv(
            '/home/blueschemeai/Downloads/testironmonth.csv')
        test_data['date'] = pd.to_datetime(
            test_data['date'], format='%d/%m/%Y')
        test_data = test_data.sort_values('date')
        test_data['Month'] = test_data['date'].dt.month
        test_data = test_data.set_index('date')
        actual_price = pd.DataFrame()
        actual_price = test_data['price']
        test_data = test_data.drop('price', axis=1)
        prediction = []
        test_forecast = X[-1:]
        t = reg.predict(test_forecast.values)
        test_forecast = test_forecast[
            ['climatic', 'iron_prod', 'iron_metallics', 'Shipments', 'Reserve_crudeore', 'Reserve_ironcontent',
             'demand', 'rio_tinosprod', 'Month', 'Price_lag']]
        n = test_forecast.values.tolist()[0]
        for i in range(1, len(test_data) + 1):
            forecastdata = test_data[i - 1:i]
            o = forecastdata.values
            y = list(np.append(o, t))
            forecast = pd.DataFrame(columns=['climatic', 'iron_prod', 'iron_metallics', 'Shipments', 'Reserve_crudeore',
                                             'Reserve_ironcontent', 'demand', 'rio_tinosprod', 'Month', 'Price_lag'])
            forecast.loc[0] = n
            forecast.loc[1] = y
            forecast['Price_diff'] = forecast['Price_lag'].diff(periods=1)
            forecast['rolling_mean_price'] = forecast['Price_lag'].rolling(
                2, min_periods=1).sum()
            t = reg.predict(forecast[-1:].values)
            n = y
            prediction.append(t)
        predictdata = pd.DataFrame(
            prediction, index=test_data.index, columns=['price'])
        rmse = mean_squared_error(actual_price, prediction)
        predictdata['actual'] = actual_price
        predictdata.index = predictdata.index.astype("str")
        json = predictdata.to_json()
        json = ast.literal_eval(json)
        json['rmse'] = rmse
        return Response(json)
