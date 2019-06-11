from django.urls import path
from .views import random_rf, datahead, UnivarientDataPreview
# from .arima import arima_univarient, arimaforecast


urlpatterns = [
    path('rf-data', random_rf.as_view(), name='rf-data'),
    path('datahead', datahead.as_view(), name='datahead'),
    path('univarientdata', UnivarientDataPreview.as_view(), name='univarient-data'),
    # path('arimamodel', arima_univarient.as_view(), name='aimamodel'),
    # path('arimaforecast', arimaforecast.as_view(), name='arima-focast')
]