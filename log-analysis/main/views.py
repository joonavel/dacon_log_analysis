from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import requests
from .serializers import LogDataSerializer


def index(request):
    return render(request, 'main/index.html')

class PredictLogData(APIView):
    def post(self, request):
        serializer = LogDataSerializer(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data['data']
            
            response = {'result': 6}
            # Flask 모델 서버로 요청을 보냅니다.
            # response = requests.post('http://localhost:5001/predict', json={'data': data})
            if response:
                return Response(response, status=status.HTTP_200_OK)
        #     else:
        #         return Response({'error': 'Model server error'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


