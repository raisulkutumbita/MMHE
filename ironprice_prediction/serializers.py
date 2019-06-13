from rest_framework import serializers


class ArimaSerializer(serializers.Serializer):
    startdate = serializers.DateField()
    enddate = serializers.DateField()


class LassoSerializer(serializers.Serializer):
    startdate = serializers.DateField()
    enddate = serializers.DateField()
