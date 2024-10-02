from django.urls import path

from . import views

urlpatterns = [
    path('signup', views.SignUp.as_view()),
    path('login', views.Login.as_view()),
    path('logout', views.Logout.as_view()),
    path('member/exist/', views.MemberExist.as_view()),

    path('member/change_password/', views.ChangePassword.as_view()),
    path('member/find_password/', views.FindPassword.as_view()),
    path('member/withdrawal', views.MemberQuit.as_view()),

    path('members', views.MemberList.as_view()),
    path('member', views.MemberDetail.as_view()),
    path('member/<email>', views.MemberDetailEmail.as_view()), # PathParam
    path('member/change_state/', views.ChangeMemberState.as_view()),

    path('friends', views.FriendList.as_view()),
    path('friend/request', views.RequestFriend.as_view()),
    path('friend/accept/', views.AcceptFriend.as_view()),

    path('diarys', views.DiaryList.as_view()),
    path('diary/', views.DiaryDetail.as_view()),
    path('diary/<pk>', views.DiaryDetailPk.as_view()),
    path('write', views.WriteDiary.as_view()),
    path('result/<pk>', views.ResultDetail.as_view()),

    path('collections', views.CollectionList.as_view()),
    path('title/<int:collection_id>', views.SetTitle.as_view()),

    path('statistics/', views.StatisticsDiarys.as_view()),
    path('alerts', views.AlertList.as_view()),

    path('tokentest', views.TokenTest.as_view()),

    
]