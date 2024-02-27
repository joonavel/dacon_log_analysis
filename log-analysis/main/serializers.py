from rest_framework import serializers

class LogDataSerializer(serializers.Serializer):
    data = serializers.CharField()