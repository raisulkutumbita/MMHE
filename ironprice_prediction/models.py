from django.db import models


class univarientdata(models.Model):
    price = models.FloatField(blank=True, null=True)
    date = models.DateField()


class price_production(models.Model):
    date = models.DateField()
    price = models.FloatField(blank=True, null=True)
    production = models.FloatField(blank=True, null=True)
