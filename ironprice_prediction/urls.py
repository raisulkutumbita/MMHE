from django.urls import path
from .views import random_rf, datahead, UnivarientDataPreview, CycloneClimateView
from .arima import ArimaUnivarient, ArimaForecast
from .lasso import LassoView,LassoUnivarient,lasso_univarientForecast
from .vector_model import VectorModel, VectorForecast


urlpatterns = [
    path('rf-data', random_rf.as_view(), name='rf-data'),
    path('datahead', datahead.as_view(), name='datahead'),
    path('cycloneclimate', CycloneClimateView.as_view(), name='cyclone-data'),
    path('univarientdata', UnivarientDataPreview.as_view(), name='univarient-data'),
    path('arimamodel', ArimaUnivarient.as_view(), name='aimamodel'),
    path('arimaforecast', ArimaForecast.as_view(), name='arima-focast'),
    path('lasso', LassoView.as_view()),
    path('lasso_univarient', LassoUnivarient.as_view(),name='lasso-uni'),
    path('lasso_uni_forecast', lasso_univarientForecast.as_view(),name='lasso-uni-forecast'),
    path('vectormodel', VectorModel.as_view()),
    path('varforecast', VectorForecast.as_view())
]