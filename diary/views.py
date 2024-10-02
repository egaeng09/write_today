from django.shortcuts import render
from rest_framework import generics
from rest_framework.response import Response
from datetime import datetime

from django.contrib.auth import authenticate, login, logout
from rest_framework.authtoken.models import Token
from rest_framework.permissions import IsAuthenticated

from django.core.exceptions import ObjectDoesNotExist

from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from .models import Member, Friend, Diary, Result, Achievement, Collection, Alert, MemberInfo
from .serializers import MemberSerializer, MemberDataSerializer, DiarySerializer, ResultSerializer, SignUpSerializer, FriendRequestSerializer, FriendAcceptSerializer, ChangePasswordSerializer, DiaryResultSerializer, DiaryListSerializer, CollectionSerializer, TitleSerializer, FriendInfoSerializer, AlertSerializer
from .model_task import process_diary, process_modified_diary
from .task import admin_check, superuser_check, validate_token, collect_achivement, check_depression


""" 임시 """   
class MemberList(generics.ListAPIView):
    queryset = Member.objects.all()
    serializer_class = MemberSerializer

class MemberDetailEmail(generics.RetrieveAPIView):
    queryset = Member.objects.all()
    serializer_class = MemberSerializer
    lookup_field = "email"

    
class DiaryDetailPk(generics.RetrieveAPIView):
    queryset = Diary.objects.all()
    serializer_class = DiaryResultSerializer


""" 회원 관련 로직 """

class SignUp(generics.CreateAPIView):
    serializer_class = SignUpSerializer
    def post(self, request):
        email = request.data.get("email")
        name = request.data.get("name")
        password = request.data.get("password")
        password_check = request.data.get("password_check")
        phone_number = request.data.get("phone_number")

        # 휴대폰 인증 OR 이메일 인증 추가할 것?

        # 상담사 계정 구분

        if password != password_check:
            return Response({"error": "비밀번호 확인이 일치하지 않음."}, status=400)

        try:
            user = Member.objects.create_user(email=email, name=name, phone_number=phone_number, password=password)
            user.save()
            user_info = MemberInfo.objects.create(member = user)
            user_info.save()
            collect_achivement(user, "병아리")
        except Exception as e:
            return Response({"error": str(e)}, status=400)

        return Response({"message": "회원가입 성공.", "user": MemberSerializer(user).data}, status=201)

class Login(generics.CreateAPIView):
    serializer_class = MemberDataSerializer
    
    def post(self, request):
        email = request.data.get("email")
        password = request.data.get("password")
        user = authenticate(email=email, password=password)
        
        if user:
            if not user.is_active:
                return Response({"error": "탈퇴된 회원."}, status=401)
            
            login(request, user)
            token, _ = Token.objects.get_or_create(user=user)

            fcm_token = request.data.get("fcm_token")
            user_info = MemberInfo.objects.get(member = user)
            user_info.fcm_token = fcm_token
            user_info.save()

            serializer = self.get_serializer(user)
            return Response({"token": token.key, "member" : serializer.data}, status=201)
        else:
            return Response({"error": "로그인 실패."}, status=401)

class Logout(generics.GenericAPIView):
    swagger_schema = None
    
    def post(self, request):
        user = request.user
        user_info = MemberInfo.objects.get(member = user)
        user_info.fcm_token = None
        user_info.save()
    
        logout(request)
        # 토큰 만료시키기
        token = Token.objects.filter(key = request.auth)
        token.delete()
        return Response({"message": "로그아웃 성공."}, status=200)
    
class TokenTest(generics.GenericAPIView):
    swagger_schema = None
    permission_classes = [IsAuthenticated] # 토큰 인증 필요

    def get(self, request):
        user = request.user
        validate_token(user)
        user_data = {
            "name": user.name,
            "email": user.email,
            # 필요한 다른 사용자 정보 추가 가능
        }
        return Response(user_data, status=200)


class MemberDetail(generics.GenericAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = MemberSerializer

    def get(self, request):
        user = request.user
        validate_token(user)
        serializer = self.get_serializer(user) # JSON으로 직렬화
        return Response(serializer.data, status=200)

class MemberExist(generics.GenericAPIView):
    serializer_class = MemberSerializer

    def get(self, request):
        email = request.query_params.get('email')
        if email:
            user = Member.objects.filter(email=email).first()
            if user:
                serializer = self.get_serializer(user)
                return Response(serializer.data, status=200)
            else:
                return Response({"error": "존재하지 않는 회원."}, status=404)
        else:
            return Response({"error": "이메일 누락."}, status=400)

class ChangeMemberState(generics.GenericAPIView):
    swagger_schema = None
    permission_classes = [IsAuthenticated] # 토큰 인증 필요

    def put(self, request):
        user = request.user
        validate_token(user)
        user_info = MemberInfo.objects.get(member = user)
        if user_info:
            state = user_info.is_public
            user_info.is_public = not state
            user_info.save()
            return Response({"message : " : str(not state)}, status=200)
        else :
            return Response({"error : " : "비정상적인 접근. "}, status=400)

class ChangePassword(generics.GenericAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = ChangePasswordSerializer

    def put(self, request):
        user = request.user
        password = request.data.get("password")
        new_password = request.data.get("new_password")
        user = authenticate(email=user.email, password=password)
        validate_token(user)
        user = Member.objects.change_password(user, new_password)
        user.save()
        return Response({"message : " : "비밀번호 변경 성공."}, status=200)
    

class FindPassword(generics.GenericAPIView):
    serializer_class = ChangePasswordSerializer

    def put(self, request):
        email = request.data.get("email")
        phone_number = request.data.get("phone_number")
        new_password = request.data.get("new_password")
        user = Member.objects.filter(email=email, phone_number=phone_number).first()
        user.change_password(user, new_password)
        user.save()
        return Response({"message : " : "비밀번호 변경 성공."}, status=200)


class MemberQuit(generics.GenericAPIView):
    swagger_schema = None
    permission_classes = [IsAuthenticated]

    def put(self, request):
        user = request.user

        validate_token(user)
        user.is_active = False
        user.save()
        logout(request)
        return Response({"message": "회원 탈퇴 성공."}, status=200)
        

""" 친구 관련 로직 """

class RequestFriend(generics.GenericAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = FriendRequestSerializer

    def post(self, request):
        user = request.user
        validate_token(user)
        receiver_email = request.data.get("receiver_email")
        target = Member.objects.filter(email=receiver_email).first()
        if isinstance(target, Member):
            if Friend.objects.filter(sender=user, receiver=target).exists() | Friend.objects.filter(sender=target, receiver=user).exists():
                if Friend.friended:
                    return Response({"error": "이미 친구 관계가 존재함."}, status=400)
                else:
                    return Response({"error": "친구 신청이 존재함."}, status=400)
            if user.email == target.email :
                return Response({"error": "자신에게는 친구 신청을 할 수 없음."}, status=400)
            friend = Friend.objects.create(
                sender=user,
                receiver=target,
                friended=False
            )
            friend.save()
            return Response({"message": "친구 신청 성공."}, status=201)
        else:
            return Response({"error": "상대방 회원 정보가 존재하지 않음."}, status=400)


class AcceptFriend(generics.GenericAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = FriendAcceptSerializer

    def put(self, request):
        user = request.user
        validate_token(user)
        friend_id = request.data.get("friend_id")
        friend = Friend.objects.filter(id=friend_id, receiver = user).first()
        if isinstance(friend, Friend):
            if friend.friended:
                return Response({"error": "이미 친구 관계임."}, status=400)
            friend.friended = True
            friend.save()

            """ 업적 로직 """
            try :
                friend_sender = Friend.objects.filter(sender=user)
                friend_receiver = Friend.objects.filter(receiver=user)

                friends = []
                for friend in friend_sender:
                    friends.append(friend.receiver)

                for friend in friend_receiver:
                    friends.append(friend.sender)


                if (len(friends) >= 1) :
                    collect_achivement(user, "솔로 탈출")

                if (len(friends) >= 10) :
                    collect_achivement(user, "사교적인")

                if (len(friends) >= 30) :
                    collect_achivement(user, "마당발")

                if (len(friends) >= 100) :
                    collect_achivement(user, "마을 이장")

            except Friend.DoesNotExist:
                pass

            collect_achivement(user, "친절한")

            """ ### """

            return Response({"message": "친구 신청 수락 성공."}, status=200)
        else:
            return Response({"error": "친구 신청 정보 존재하지 않음."}, status=400)


    def delete(self, request):
        user = request.user
        validate_token(user)
        friend_id = request.data.get("friend_id")
        friend = Friend.objects.filter(id=friend_id).first()
        if isinstance(friend, Friend):
            if friend.friended:
                friend.delete()
                return Response({"message": "친구 삭제 성공."}, status=200)
            else:
                friend.delete()

                """ 업적 로직 """
                collect_achivement(user, "가차없는")

                return Response({"message": "친구 신청 삭제 성공."}, status=200)
        else:
            return Response({"error": "친구 신청 정보 존재하지 않음."}, status=400)


class FriendList(generics.GenericAPIView):
    serializer_class = FriendInfoSerializer
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        validate_token(user)

        friend_sender = Friend.objects.filter(sender=user, friended = True)
        friend_receiver = Friend.objects.filter(receiver=user)

        friends = []
        for friend in friend_sender:
            friend.receiver.friend_id = friend.pk
            friend.receiver.friended = friend.friended
            friends.append(friend.receiver)

        for friend in friend_receiver:
            friend.sender.friend_id = friend.pk
            friend.sender.friended = friend.friended
            friends.append(friend.sender)

        serializer = self.get_serializer(friends, many=True)
        
        if friends:
            return Response(serializer.data, status=200)
        return Response({"error": "친구 정보 존재하지 않음."}, status=400)


""" 일기 관련 로직 """

class DiaryList(generics.ListAPIView):
    serializer_class = DiaryListSerializer
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        validate_token(user)
        diarys = Diary.objects.filter(writer = user)
        if diarys:
            serializer = self.get_serializer(diarys, many = True)
            return Response(serializer.data, status=200)
        else:
            return Response({"error": "일기 정보 존재하지 않음."}, status=400)


class DiaryDetail(generics.GenericAPIView):
    serializer_class = DiaryResultSerializer
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        validate_token(user)
        created_date = datetime.strptime(request.query_params.get("created_date"), '%Y-%m-%d').date()
        diary = Diary.objects.filter(writer = user, created_date = created_date).first()
        if isinstance(diary, Diary):
            serializer = self.get_serializer(diary)
            return Response(serializer.data, status=200)
        else:
            return Response({"error": "일기 정보 존재하지 않음."}, status=400)
        


class StatisticsDiarys(generics.ListAPIView):
    serializer_class = DiaryResultSerializer
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        validate_token(user)
        start_date = datetime.strptime(request.query_params.get("start_date"), '%Y-%m-%d').date()
        end_date = datetime.strptime(request.query_params.get("end_date"), '%Y-%m-%d').date()
        diarys = Diary.objects.filter(writer = user, created_date__range=(start_date, end_date))
        if diarys:
            serializer = self.get_serializer(diarys, many = True)
            return Response(serializer.data, status=200)
        else:
            return Response({"error": "일기 정보 존재하지 않음."}, status=400)


class WriteDiary(generics.CreateAPIView):
    serializer_class = DiarySerializer
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user
        validate_token(user)
        contents = request.data.get("contents")
        created_date = datetime.strptime(request.data.get("created_date"), '%Y-%m-%d').date()
        nowDate = datetime.now().date()
        if created_date > nowDate:
            return Response({"error": "미래 일기 작성 불가능."}, status=400)
        
        diary = Diary.objects.create(writer=user, contents=contents, created_date=created_date)
        diary.contents = contents
        diary.save()

        serializer = self.get_serializer(diary)
        self.perform_create(serializer)

        """ 업적 로직 """
        # 일기 컬렉션 로직
        # if 일기 개수 check + collection 유무 => 컬렉션 획득
        
        try :
            diarys = Diary.objects.filter(writer = user)

            if (len(diarys) >= 1) :
                collect_achivement(user, "아장아장")

            if (len(diarys) >= 10) :
                collect_achivement(user, "성실한")
                
            if (len(diarys) >= 30) :
                collect_achivement(user, "습관갖춘")

            if (len(diarys) >= 100) :
                collect_achivement(user, "기록자")

            if (len(diarys) >= 1000) :
                collect_achivement(user, "기록관리자")

        except Diary.DoesNotExist:
            pass

        check_depression(request)
     
        return Response(serializer.data, status=201)
    
    def perform_create(self, serializer):
        process_diary.delay(serializer.instance.id)


    
    def put(self, request):
        user = request.user
        validate_token(user)
        contents = request.data.get("contents")

        created_date = datetime.strptime(request.data.get("created_date"), '%Y-%m-%d').date()
        nowDate = datetime.now().date()
        if (created_date > nowDate) :
            return Response({"error": "미래 일기 작성 불가능."}, status=400)
        
        diary, _ = Diary.objects.get_or_create(writer=user, created_date=created_date)
        diary.contents = contents
        diary.save()

        serializer = self.get_serializer(diary)
        self.perform_update(serializer)
        
        check_depression(request)

        return Response(serializer.data, status=201)
    
    def perform_update(self, serializer):
        process_modified_diary.delay(serializer.instance.id)


class ResultDetail(generics.RetrieveAPIView):
    queryset = Result.objects.select_related("diary").all()
    serializer_class = ResultSerializer
    # 일기 상세 조회 = 결과도 조회
    """
    만약 결과를 조회하는데 같은 날짜에 일기는 존재하고 결과는 존재하지 않다면 즉시 재요청하기
    """

""" 컬렉션 관련 로직 """

class CollectionList(generics.GenericAPIView):
    serializer_class = CollectionSerializer
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        user = request.user
        validate_token(user)
        collections = Collection.objects.filter(member = user)
        if collections:
            serializer = self.get_serializer(collections, many = True)
            return Response(serializer.data, status=201)
        else:
            return Response({"error": "컬렉션 정보 존재하지 않음."}, status=400)
        

class SetTitle(generics.GenericAPIView):
    serializer_class = TitleSerializer
    permission_classes = [IsAuthenticated]
    
    def put(self, request, collection_id):
        user = request.user
        validate_token(user)
        collection = Collection.objects.get(id = collection_id)
        if collection:
            user_info = MemberInfo.objects.get(member = user)
            if user_info:
                user_info.collection = collection
                user_info.save()
                serializer = self.get_serializer(user)
                return Response(serializer.data, status=201)
            else :
                return Response({"error": "사용자 정보 존재하지 않음."}, status=400)
        else:
            return Response({"error": "컬렉션 정보 존재하지 않음."}, status=400)


""" 알림 관련 로직 """

class AlertList(generics.GenericAPIView):
    serializer_class = AlertSerializer
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        user = request.user
        validate_token(user)
        alerts = Alert.objects.filter(member = user)
        if alerts:
            serializer = self.get_serializer(alerts, many = True)
            return Response(serializer.data, status=201)
        else:
            return Response({"error": "알람 정보 존재하지 않음."}, status=400)