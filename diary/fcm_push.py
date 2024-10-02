from firebase_admin import messaging
from .models import MemberInfo, Alert

def send_to_firebase_cloud_messaging(user, title, body):
    # This registration token comes from the client FCM SDKs.
    user_info = MemberInfo.objects.get(member = user)

    registration_token = user_info.fcm_token

    # See documentation on defining a message payload.
    message = messaging.Message(
    notification=messaging.Notification(
        title=title,
        body=body,
    ),
    token=registration_token,
    )

    try:
        response = messaging.send(message)
        # Response is a message ID string.
        print('Successfully sent message:', response)
        
        alert = Alert.objects.create(member = user, alert_contents = body)
        alert.save()
    except Exception as e:
        print('예외가 발생했습니다.', e)