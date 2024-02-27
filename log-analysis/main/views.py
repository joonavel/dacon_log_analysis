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
            # Flask 모델 서버로 요청 > 결과 받아서 Front로.
            response = requests.post('http://localhost:5000/prediction/', json={'input_data': data})
            print(response.json())
            
            if response.status_code == 200:
                return Response(response.json(), status=status.HTTP_200_OK)
            else:
                return Response({'error': 'Model server error'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


