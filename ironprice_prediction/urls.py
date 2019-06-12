from django.urls import path
from .views import random_rf, datahead, UnivarientDataPreview
from .arima import ArimaUnivarient, ArimaForeCast


urlpatterns = [
    path('rf-data', random_rf.as_view(), name='rf-data'),
    path('datahead', datahead.as_view(), name='datahead'),
    path('univarientdata', UnivarientDataPreview.as_view(), name='univarient-data'),
    path('arimamodel', ArimaUnivarient.as_view(), name='aimamodel'),
    path('arimaforecast', ArimaForeCast.as_view(), name='arima-focast')
]