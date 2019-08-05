from django.contrib import admin
from .models import UnivarientData, PriceProduction, CycloneData


class UnivarientDataAdmin(admin.ModelAdmin):
    list_display = ['date', 'price']


admin.site.register(UnivarientData, UnivarientDataAdmin)

# Register your models here.
admin.site.register(PriceProduction)
admin.site.register(CycloneData)

