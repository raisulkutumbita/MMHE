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


class CycloneData(models.Model):
    date = models.DateField()
    climate = models.IntegerField(default=0)

    class Meta:
        db_table = "cylone_data"


class MultivarientData(models.Model):
    date = models.DateField()
    iron_price = models.FloatField(blank=True, null=True)
    oil_price = models.FloatField(blank=True, null=True)

    class Meta:
        db_table = "multivarient_data"