from channels.generic.websocket import AsyncJsonWebsocketConsumer
from channels.db import database_sync_to_async

class ChatConsumer(AsyncJsonWebsocketConsumer):

    async def connect(self):
        try:          
            self.room_id = self.scope['url_route']['kwargs']['room_id']

            if not await self.check_room_exists(self.room_id):
                raise ValueError('채팅방이 존재하지 않습니다.')
                        
            group_name = self.get_group_name(self.room_id)

            await self.channel_layer.group_add(group_name, self.channel_name)                 
            await self.accept()

        except ValueError as e:       
            await self.send_json({'error': str(e)})
            await self.close()

    async def disconnect(self, close_code):
        try:            
            group_name = self.get_group_name(self.room_id)
            await self.channel_layer.group_discard(group_name, self.channel_name)

        except Exception as e:
            pass

    async def receive_json(self, content):
        try:
            message = content['msg']
            sender_email = content['sender_email']
            opponent_email = content.get('opponent_email')

            if not sender_email or not opponent_email:
                raise ValueError("송수신자 누락")

            counselor = await self.check_sender_is_counselor(sender_email, opponent_email, 'counselor')
            client = await self.check_sender_is_counselor(sender_email, opponent_email, 'client')
            
            room = await self.get_or_create_room(counselor = counselor, client = client)
            
            self.room_id = str(room.id)
            
            group_name = self.get_group_name(self.room_id)
            
            await self.save_message(room, sender_email, message)

            await self.send_push_notification(sender_email, opponent_email, message)

            await self.channel_layer.group_send(group_name, {
                'type': 'chat_message',
                'msg': message,
                'sender_email': sender_email
            })

        except ValueError as e:
            await self.send_json({'error': str(e)})

    async def chat_message(self, event):
        try:
            message = event['msg']
            sender_email = event['sender_email']
            
            await self.send_json({'msg': message, 'sender_email': sender_email})
        except Exception as e:
            await self.send_json({'error': '메시지 전송 실패'})

    @staticmethod
    def get_group_name(room_id):
        return f"chat_room_{room_id}"
     
    @database_sync_to_async
    def get_or_create_room(self, counselor, client):
        from .models import Room

        room, created = Room.objects.get_or_create(
        	counselor=counselor,
        	client=client
    	)
        return room

    @database_sync_to_async
    def save_message(self, room, sender, message):
        from .models import Chat
        if not sender or not message:
            raise ValueError("발신자 이메일 및 메시지가 필요합니다.")

        Chat.objects.create(room=room, sender_email=sender, msg=message)

    @database_sync_to_async
    def check_room_exists(self, room_id):
        from .models import Room
        return Room.objects.filter(id=room_id).exists()
    
    @database_sync_to_async
    def check_sender_is_counselor(self, sender_email, opponent_email, role):
        from .models import Member
        sender = Member.objects.filter(email=sender_email).first()
        opponent = Member.objects.filter(email=opponent_email).first()

        if role == 'counselor':
            if sender.is_counselor:
                return sender
            elif opponent.is_counselor:
                return opponent
            else:
                raise ValueError("상담사 누락")
        else:
            if not sender.is_counselor:
                return sender
            elif not opponent.is_counselor:
                return opponent
            else:
                raise ValueError("내담자 누락")
            
    @database_sync_to_async
    def send_push_notification(self, sender_email, recipient_email, message):
        from .models import Member
        from diary.fcm_push import send_to_firebase_cloud_messaging

        member = Member.objects.filter(email=recipient_email).first()
        opponent_name = Member.objects.filter(email=sender_email).first().name

        alert_name = "상담 도착!"
        alert_msg = f"{opponent_name} : {message}"
        send_to_firebase_cloud_messaging(member, alert_name, alert_msg)