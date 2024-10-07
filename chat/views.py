from rest_framework import generics, serializers, status
from rest_framework.response import Response
from .models import Room, Chat
from .serializers import RoomSerializer, ChatSerializer
from rest_framework.exceptions import ValidationError
from django.http import Http404
from django.http import JsonResponse
from django.conf import settings
from diary.auth import validate_token
from diary.models import Member


class ImmediateResponseException(Exception):
    def __init__(self, response):
        self.response = response

class RoomListCreateView(generics.ListCreateAPIView):
    serializer_class = RoomSerializer

    def get_queryset(self):
        try:
            user = self.request.user
            validate_token(user)
            queryset = Room.objects.filter(counselor=user) | Room.objects.filter(client=user)
            return queryset

        except Exception as e:
            content = {'detail': str(e)}
            return Response(content, status=status.HTTP_400_BAD_REQUEST)

    def get_serializer_context(self):
        context = super(RoomListCreateView, self).get_serializer_context()
        context['request'] = self.request
        return context
    
    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            room = self.perform_create(serializer)
            headers = self.get_success_headers(serializer.data)

            return Response(RoomSerializer(room, context={'request': self.request}).data, status=status.HTTP_201_CREATED, headers=headers)
        except ImmediateResponseException as e:
            return e.response

    def perform_create(self, serializer):
        user = self.request.user
        validate_token(user)

        email = self.request.data.get('opponent_email')
        opponent = Member.objects.get(email=email)
        is_counselor = opponent.is_counselor

        if user.email == email:
            raise ImmediateResponseException(Response("동일한 유저 간 채팅은 불가능함.", status=status.HTTP_400_BAD_REQUEST))
        
        existing_chatroom = (
            Room.objects.filter(client=opponent, counselor=user) | Room.objects.filter(counselor=opponent, client=user)
        ).first()

        if existing_chatroom:
            serialized_data = RoomSerializer(existing_chatroom, context={'request': self.request}).data
            raise ImmediateResponseException(Response(serialized_data, status=status.HTTP_200_OK))
        
        else:
            if not is_counselor:
                serializer.save(client=opponent, counselor=user)
            else:
                serializer.save(counselor=opponent, client=user)


class ChatListView(generics.ListAPIView):
    serializer_class = ChatSerializer

    def get_queryset(self):
        room_id = self.kwargs.get('room_id')
        
        if not room_id:
            content = {'detail': 'room_id 파라미터가 필요합니다.'}
            return Response(content, status=status.HTTP_400_BAD_REQUEST)

        queryset = Chat.objects.filter(room_id=room_id)
        
        if not queryset.exists():
            raise Http404('해당 room_id로 메시지를 찾을 수 없습니다.')

        return queryset


class CurrentUserView(generics.ListAPIView):
    def get(self, request):
        user = request.user
        validate_token(user)
        return Response(user.email, status=status.HTTP_200_OK)
    

class ClientListView(generics.ListAPIView):
    def get(self, request):
        user = request.user
        validate_token(user)
        # Match List 보내기
        return Response(status=status.HTTP_200_OK)


class CounselorDiaryView(generics.ListAPIView):
    def get(self, request):
        # 상담사가 내담자의 일기 조회 (공개 여부에 따라)
        return Response(status=status.HTTP_200_OK)