from django.dispatch import receiver
from .models import Achievement, Collection, Member, Diary, MemberInfo
from django.db.models.signals import post_save

from .model_task  import process_diary

# @receiver(post_save, sender = Member)
# def achv_join(sender, instance, created, **kwargs):
#     if created:
#         achievement = Achievement.objects.get('')
#         Collection.objects.create(user=instance, achievement=achievement)


#sender 신호를 보낸 객체, instance 신호 인스턴스 객체, **kwargs 추가적 인수
# @receiver(post_save, sender=Diary)
# def save_diary(sender, instance, created, **kwargs):
#     if created:
#         process_diary.delay(instance.id)

