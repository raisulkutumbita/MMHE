from django.contrib import admin
from .models import UnivarientData, PriceProduction, CycloneData

# Register your models here.
admin.site.register(UnivarientData)
admin.site.register(PriceProduction)
admin.site.register(CycloneData)

