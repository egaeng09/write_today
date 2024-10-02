from django.urls import path
from . import views

urlpatterns = [
    path('rooms/', views.RoomListCreateView.as_view(), name='room'),
    path('<int:room_id>/messages', views.ChatListView.as_view(), name='chat_messages'),
    path('email/', views.CurrentUserView.as_view(), name='email'),
]