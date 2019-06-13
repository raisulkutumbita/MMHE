from django.db import models


class UnivarientData(models.Model):
    price = models.FloatField(blank=True, null=True)
    date = models.DateField()

    class Meta:
        db_table = "univarient_data"


class PriceProduction(models.Model):
    date = models.DateField()
    price = models.FloatField(blank=True, null=True)
    production = models.FloatField(blank=True, null=True)

    class Meta:
        db_table = "price_production"
