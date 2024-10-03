from .models import Member, Friend, Diary, Result, Achievement, Collection, Alert, MemberInfo
from chat.models import Match

from .fcm_push import send_to_firebase_cloud_messaging
from django.db.models import Count
import random
from chat.views import RoomListCreateView

    
def collect_achivement(user, type):
    achievement = Achievement.objects.get(name = type)
    try :
        Collection.objects.get(member = user, achievement = achievement)
    except Collection.DoesNotExist:
        collection = Collection.objects.create(member = user, achievement = achievement)
        collection.save()
        alert_name = "컬렉션 획득!"
        alert_msg = type + " 컬렉션을 획득했어요. 앱에서 확인하세요!"
        send_to_firebase_cloud_messaging(user, alert_name, alert_msg)

def check_depression(request):
    user = request.user
    recent_5days_of_diarys = Diary.objects.get(writer = user).order_by('-created_date')[:5]
    results = []

    for diary in recent_5days_of_diarys:
        results.append(diary.result)

    count_depression = 0

    for result in results:
        mixed_emotion = result.emotions
        for emotion in mixed_emotion:
            if emotion.emotion.name == "슬픔" and emotion.emotion.rate > 30:
                count_depression = count_depression + 1
    
    if count_depression > 3 :
        counselors = Member.objects.filter(
            is_counselor=True
        ).annotate(
            room_count=Count('counselor_room')
        ).filter(
            room_count__lte=3
        )

        if counselors.exists():
            # 조건에 맞는 상담사 중에서 랜덤으로 1명 선택
            selected_counselor = random.choice(counselors)
        else:
            # 조건에 맞는 상담사가 없을 경우, 아무 상담사나 랜덤으로 선택
            selected_counselor = random.choice(Member.objects.filter(is_counselor=True))

        # Room 생성 요청에 필요한 데이터 구성
        data = {
            'opponent_email': selected_counselor.email,  # 선택된 상담사의 이메일
        }

        # RoomListCreateView를 통해 Room 생성 호출
        view = RoomListCreateView.as_view({'post': 'create'})

        alert_name = "상담 대상자 추가"
        alert_msg = type + " 우울감을 느끼는 상담 대상자가 추가되었어요. 앱에서 확인하세요!"
        send_to_firebase_cloud_messaging(selected_counselor, alert_name, alert_msg)

        alert_name = "우울감 발견"
        alert_msg = type + " 최근 작성된 일기에서 우울한 감정이 자주 발견되었어요. 상담사를 매칭해드릴게요!"
        send_to_firebase_cloud_messaging(user, alert_name, alert_msg)

        
        # request.data에 선택된 상담사 데이터를 넣어서 Room 생성 요청을 보냄
        view(request._request, data=data)


