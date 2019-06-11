from django.db import models


class univarientdata(models.Model):
    price = models.FloatField(blank=True, null=True)
    date = models.DateField()

