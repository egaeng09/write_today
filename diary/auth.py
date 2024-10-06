from rest_framework.response import Response
from .models import Member

def admin_check(user):
    if not user.is_staff:
        return Response({"error": "관리자 권한 없음."}, status=403)
    
def superuser_check(user):
    if not user.is_superuser:
        return Response({"error": "관리자 권한 없음."}, status=403)
    
def validate_token(user):
    if not isinstance(user, Member):  # 유저가 인증된 경우
        return Response({"error": "회원 검증 실패."}, status=401)