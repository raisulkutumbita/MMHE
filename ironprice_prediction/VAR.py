import pandas as pd
from statsmodels.tsa.statespace.varmax import VARMAX
from django_pandas.io import read_frame
from rest_framework.response import Response
from datetime import datetime as dat
from rest_framework.views import APIView
import ast
from .arima import forecast_accuracy
from .models import price_production
from dateutil import relativedelta


class vectormodel(APIView):
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
        model = VARMAX(train, order=(1, 1, 1))
        model_fit = model.fit(disp=False)
        yhat = model_fit.forecast(len(test)-1)
        yhat['actual'] = test['price']
        predictdata = yhat.drop("production", axis=1)
        metrics = forecast_accuracy(
            predictdata['price'], predictdata['actual'])
        predictdata.index = predictdata.index.astype("str")
        print(predictdata)
        json = predictdata.to_json()
        json = ast.literal_eval(json)
        json['mape'] = metrics['mape']
        return Response(json)


class VARforecast(APIView):
    def post(self, request, *args, **kwargs):
        body_data = request.data
        n_steps = int(body_data['nsteps'])
        data = read_frame(price_production.objects.all())
        data['date'] = pd.to_datetime(data['date'])
        data = data.drop('id', axis=1)
        data = data.set_index('date')
        model = VARMAX(data, order=(1, 1, 1))
        model_fit = model.fit(disp=False)
        yhat = model_fit.forecast(n_steps)
        yhat = yhat['price']
        yhat.index = yhat.index.astype("str")
        json = yhat.to_json()
        json = ast.literal_eval(json)
        return Response(json)
