import pandas as pd
from statsmodels.tsa.statespace.varmax import VARMAX
from django_pandas.io import read_frame
from rest_framework.response import Response
from datetime import datetime as dat
from rest_framework.views import APIView
import ast
from .arima import forecast_accuracy
from .models import PriceProduction
from dateutil import relativedelta


class VectorModel(APIView):

    def get(self, request, *args, **kwargs):
        start_date = self.request.query_params.get('startdate', '1970-01-30')
        end_date = self.request.query_params.get('enddate', '2018-01-01')

        data = read_frame(PriceProduction.objects.all())
        data['date'] = pd.to_datetime(data['date'])
        data = data.drop('id', axis=1)
        data = data.set_index('date')

        startdate = dat.strptime(start_date, '%Y-%m-%d')
        enddate = dat.strptime(end_date, '%Y-%m-%d')

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


class VectorForecast(APIView):

    def get(self, request, *args, **kwargs):
        n_steps = int(self.request.query_params.get('nsteps', 10))
        
        data = read_frame(PriceProduction.objects.all())
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
