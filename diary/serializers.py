from rest_framework import serializers
from .models import Member, Friend, Diary, Emotion, Result, Achievement, Collection, Alert, MixedEmotion, MemberInfo

from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework import serializers
from rest_framework.validators import UniqueValidator
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer

    # fields = [''] = 포함할 필드 기입
    # fields = '__all__' = 전체 필드 포함
    # exclude = [''] = 제외할 필드 기입

class MemberDataSerializer(serializers.ModelSerializer):
    is_public = serializers.SerializerMethodField()
    title = serializers.SerializerMethodField()

    class Meta:
        model = Member
        fields = ['id', 'name', 'email', 'is_public', 'title']

    def get_is_public(self, obj):
        user_info = MemberInfo.objects.get(member = obj)
        if user_info:
            return user_info.is_public
        return None
    
    def get_title(self, obj):
        user_info = MemberInfo.objects.get(member = obj)
        if user_info and user_info.collection:
            return user_info.collection.achievement.name
        return None

class MemberSerializer(serializers.ModelSerializer):
    class Meta:
        model = Member
        fields = ['id', 'name', 'email']

class DiarySerializer(serializers.ModelSerializer):
    class Meta:
        model = Diary
        fields = '__all__'

class EmotionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Emotion
        fields = ['name', 'hex']


class FriendSerializer(serializers.ModelSerializer):
    sender = MemberSerializer()
    receiver = MemberSerializer()

    class Meta:
        model = Friend
        fields = '__all__'

class AchievementSerializer(serializers.ModelSerializer):
    class Meta:
        model = Achievement
        fields = '__all__'

class CollectionSerializer(serializers.ModelSerializer):
    achievement = AchievementSerializer()

    class Meta:
        model = Collection
        fields = ['id', 'achievement', 'collect_date']

class AlertSerializer(serializers.ModelSerializer):
    class Meta:
        model = Alert
        fields = ['alert_contents', 'alert_date']

class MixedEmotionSerializer(serializers.ModelSerializer):
    emotion = EmotionSerializer()

    class Meta:
        model = MixedEmotion
        fields = ['emotion', 'rate']

class ResultSerializer(serializers.ModelSerializer):
    diary = DiarySerializer()
    emotions = EmotionSerializer(many = True)
    
    class Meta:
        model = Result
        fields = '__all__'

class MemberInfoSerializer(serializers.ModelSerializer):
    member = MemberSerializer()
    collection = CollectionSerializer()

    class Meta:
        model = MemberInfo
        fields = '__all__'

""" custom serializer """

class SignUpSerializer(serializers.ModelSerializer):
    class Meta:
        model = Member
        fields = ['email', 'password']

class ResultDataSerializer(serializers.ModelSerializer):
    mixed_emotion = serializers.SerializerMethodField()

    class Meta:
        model = Result
        fields = ['id', 'mixed_emotion', 'answer']

    def get_mixed_emotion(self, obj):
        mixed_emotions = MixedEmotion.objects.filter(result=obj)
        return MixedEmotionSerializer(mixed_emotions, many=True).data

class DiaryResultSerializer(serializers.ModelSerializer):
    result = ResultDataSerializer()
    
    class Meta:
        model = Diary
        fields = '__all__'


class DiaryListSerializer(serializers.ModelSerializer):
    hex = serializers.SerializerMethodField()

    class Meta:
        model = Diary
        fields = ['id', 'created_date', 'hex']

    def get_hex(self, obj):
        result = Result.objects.filter(diary=obj).first()
        if result:
            mixed_emotion = MixedEmotion.objects.filter(result=result).order_by('-rate').first()
            if mixed_emotion:
                return mixed_emotion.emotion.hex
        return None
    
class TitleSerializer(serializers.ModelSerializer):
    title = serializers.SerializerMethodField()
    name = serializers.SerializerMethodField()
    is_public = serializers.SerializerMethodField()
    last_color = serializers.SerializerMethodField()

    class Meta:
        model = Member
        fields = ['title', 'name', 'is_public', 'last_color']

    def get_name(self, obj):
        return obj.name

    def get_title(self, obj):
        member_info = MemberInfo.objects.filter(member=obj).first()
        if member_info.collection:
            if member_info.collection.achievement:
                return member_info.collection.achievement.name
            return None
        return None
    
    def get_is_public(self, obj):
        member_info = MemberInfo.objects.filter(member=obj).first()
        return member_info.is_public
    
    def get_last_color(self, obj):
        diary = Diary.objects.filter(writer = obj).order_by('-created_date').first()
        if diary:
            result = Result.objects.filter(diary = diary).first()
            if result:
                mixed_emotion = MixedEmotion.objects.filter(result=result).order_by('-rate').first()
                if mixed_emotion:
                    return mixed_emotion.emotion.hex
        else:
            return None

class FriendInfoSerializer(serializers.ModelSerializer):
    friend_id = serializers.IntegerField()
    title = serializers.SerializerMethodField()
    name = serializers.SerializerMethodField()
    is_public = serializers.SerializerMethodField()
    last_color = serializers.SerializerMethodField()
    friended = serializers.BooleanField()

    class Meta:
        model = Member
        fields = ['friend_id', 'title', 'name', 'is_public', 'last_color', 'friended']

    def get_name(self, obj):
        return obj.name

    def get_title(self, obj):
        member_info = MemberInfo.objects.filter(member=obj).first()
        if member_info.collection and member_info.collection.achievement:
            return member_info.collection.achievement.name
        return None

    def get_is_public(self, obj):
        member_info = MemberInfo.objects.filter(member=obj).first()
        return member_info.is_public

    def get_last_color(self, obj):
        diary = Diary.objects.filter(writer = obj).order_by('-created_date').first()
        if diary:
            result = Result.objects.filter(diary = diary).first()
            if result:
                mixed_emotion = MixedEmotion.objects.filter(result=result).order_by('-rate').first()
                if mixed_emotion:
                    return mixed_emotion.emotion.hex
        return None



""" Swagger Test Serializer """


class LoginSerializer(serializers.ModelSerializer):
    class Meta:
        model = Member
        fields = ['email', 'password']

class ChangePasswordSerializer(serializers.Serializer):
    password = serializers.CharField(help_text="비밀번호")
    new_password = serializers.CharField(help_text="새 비밀번호")

class FriendRequestSerializer(serializers.Serializer):
    receiver_email = serializers.CharField(help_text="상대방 이메일")

class FriendAcceptSerializer(serializers.Serializer):
    friend_id = serializers.CharField(help_text="친구 id")