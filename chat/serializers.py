from rest_framework import serializers
from .models import Room, Chat

class ChatSerializer(serializers.ModelSerializer):
    class Meta:
        model = Chat
        fields = "__all__"

class RoomSerializer(serializers.ModelSerializer):
    latest_message = serializers.SerializerMethodField()
    latest_message_time = serializers.SerializerMethodField()
    opponent_email = serializers.SerializerMethodField()
    counselor_email = serializers.SerializerMethodField()
    client_email = serializers.SerializerMethodField()
    chats = ChatSerializer(many=True, read_only=True, source="chat.all")

    class Meta:
        model = Room
        fields = ['id', 'counselor_email', 'client_email', 'latest_message', 'latest_message_time', 'opponent_email', 'chats']

    def get_latest_message(self, obj):
        latest_msg = Chat.objects.filter(room=obj).order_by('-send_date').first()
        if latest_msg:
            return latest_msg.msg
        return None
    
    def get_latest_message_time(self, obj):
        latest_msg = Chat.objects.filter(room=obj).order_by('-send_date').first()
        if latest_msg:
            return latest_msg.send_date
        return None

    def get_opponent_email(self, obj):
        request_user_email = self.context['request'].user.email

        if request_user_email == obj.counselor.email:
            return obj.client.email
        return obj.counselor.email

    def get_client_email(self, obj):  
        return obj.client.email
    
    def get_counselor_email(self, obj):
        print(obj)
        return obj.counselor.email