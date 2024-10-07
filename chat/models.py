from django.db import models

class Room(models.Model):
    counselor = models.ForeignKey('diary.Member', related_name='counselor_room', on_delete=models.CASCADE)
    client = models.ForeignKey('diary.Member', related_name='client_room', on_delete=models.CASCADE)
    is_opened = models.BooleanField(default = False)

    is_active = models.BooleanField(default = True)

    def __str__(self) :
        return f"{self.client.name} / {self.counselor.name}"


class CounselorInfo(models.Model):
    member = models.ForeignKey('diary.Member', on_delete = models.CASCADE)
    propensity = models.BooleanField(default = False)
    summary = models.TextField(null = False)

    class Meta : 
        db_table = "counselor_info"

    def __str__(self) :
        return f"{self.member.name}'s counselor info"

    
class Chat(models.Model):
    msg = models.TextField(null = False)
    read = models.BooleanField(default=False)
    send_date = models.DateTimeField(auto_now_add=True)
    room = models.ForeignKey(Room, related_name='chat', on_delete=models.CASCADE)
    # is_clients = models.BooleanField(default = False)
    sender_email = models.EmailField()

    def __str__(self):
        return f"{self.room.id} / {self.sender_email} / {self.msg}"
    

class Match(models.Model):
    counselor = models.ForeignKey('diary.Member', related_name='counselor', on_delete=models.CASCADE)
    client = models.ForeignKey('diary.Member', related_name='client', on_delete=models.CASCADE)
    # 우울감이 존재하는 사람과 상담사를 매칭
    # 상담사 입장에서 우울감을 느끼는 사용자 LIST 출력 > 상담사가 채팅을 시작하면 MATCH에서 사라지고 ROOM으로 변경
    # 사용자 입장에서 우울감 느낄 시 상담사 1인 매칭 > 채팅 시작하면 동일하게 ROOM으로