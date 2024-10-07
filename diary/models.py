from django.db import models
from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin, BaseUserManager
from django.core.validators import RegexValidator

class UserManager(BaseUserManager):
    def create_user(self, email, name, phone_number, password):
        if not email:
            raise ValueError(('Users must have an email address'))

        member = self.model(
            email=self.normalize_email(email),
            name=name,
            phone_number = phone_number,
        )

        member.set_password(password)
        member.save(using=self._db)
        return member
    

    def create_superuser(self, email, name, phone_number, password):
        member = self.create_user(
            email=email,
            name=name,
            phone_number = phone_number,
            password = password,
        )

        #member.is_admin = True
        member.is_superuser = True
        member.is_staff = True
        member.save(using=self._db)
        return member
    
    def change_password(self, member, new_password):
        member.set_password(new_password)
        member.save(using=self._db)
        return member


class Member(AbstractBaseUser, PermissionsMixin):
    name = models.CharField(max_length = 100, null = False)
    #password = models.CharField(max_length = 100, null = False)
    #phone_number = PhoneNumberField(region = 'KR', unique = True)
    phone_number = models.CharField(
        max_length=16,
        blank=False,
        null=False,
        unique=True,
        validators=[
        RegexValidator(
            regex=r'^\+?1?\d{9,15}$',
            message="Phone number must be entered in the format '+123456789'. Up to 15 digits allowed."
        ),
        ],
    )
    email = models.EmailField(max_length = 254, unique = True)
    # is_public = models.BooleanField(default = True)
    created_date = models.DateTimeField(auto_now_add = True)
    
    is_active = models.BooleanField(default = True) # 탈퇴 여부
    is_staff = models.BooleanField(default = False) # 장고 관리자 페이지에서의 관리자

    is_counselor = models.BooleanField(default = False) # 상담사

    objects = UserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['name', 'phone_number']

    def __str__(self) :
        return f"{self.name} / {self.email}"
    
    def get_full_name(self):
        return '{} ({})'.format(
            self.email,
            self.name,
        )

    def get_short_name(self):
        return self.name
    

class Friend(models.Model):
    sender = models.ForeignKey(Member, related_name = 'sent_friend_requests', on_delete = models.CASCADE)
    receiver = models.ForeignKey(Member, related_name = 'received_friend_requests', on_delete = models.CASCADE)
    friended = models.BooleanField(default = False)
    
    def __str__(self) :
        return self.sender.name + " + " + self.receiver.name
    

class Diary(models.Model):
    writer = models.ForeignKey(Member, on_delete = models.CASCADE)
    contents = models.TextField(null = False)
    created_date = models.DateField(null = False)

    def __str__(self) :
        return '{} / {}'.format(
            self.writer.name,
            self.created_date.strftime("%Y년 %m월 %d일 %H시 %M분"),
        )


class Emotion(models.Model):
    name = models.CharField(max_length = 100, null = False)
    hex = models.CharField(max_length = 100, null = False)

    def __str__(self) :
        return f"{self.name} / {self.hex}"


class Result(models.Model):
    diary = models.OneToOneField(Diary, on_delete = models.CASCADE, related_name="result")
    emotions = models.ManyToManyField(Emotion, through='MixedEmotion') # 다대다로 변경
    # color = models.OneToOneField(Color, on_delete = models.SET_NULL, null=True)
    answer = models.TextField(null = False)

    def __str__(self) :
        return '{} / {}'.format(
            self.diary.writer.name,
            self.diary.created_date.strftime("%Y년 %m월 %d일 %H시 %M분")
        )


class MixedEmotion(models.Model):
    result  = models.ForeignKey(Result, on_delete=models.CASCADE)
    emotion  = models.ForeignKey(Emotion, on_delete=models.CASCADE)
    rate = models.FloatField()

    class Meta : 
        db_table = "mixed_emotion"
    
    def __str__(self) :
        return '{} / {} / {}'.format(
            self.result.diary,
            self.emotion.name,
            self.rate,
        )
    

class Achievement(models.Model):
    name = models.CharField(max_length = 100, null = False)
    summary = models.TextField(null = False)

    def __str__(self) :
        return self.name
    

class Collection(models.Model):
    member = models.ForeignKey(Member, on_delete = models.CASCADE)
    achievement = models.ForeignKey(Achievement, on_delete = models.CASCADE)
    collect_date = models.DateField(auto_now_add = True)

    def __str__(self) :
        return f"{self.member.email} / {self.achivement.name}"
    

class MemberInfo(models.Model):
    is_public = models.BooleanField(default = True, null = False)
    member = models.OneToOneField(Member, related_name = 'infos', on_delete = models.CASCADE, null = False)
    collection = models.OneToOneField(Collection, null = True, on_delete=models.SET_NULL)

    fcm_token = models.CharField(blank = True, max_length = 500, null = True)

    class Meta : 
        db_table = "member_info"
    
    def __str__(self) :
        return f"{self.member.name} / {self.is_public} / {self.collection.achievement.name if self.collection else 'None'}"


class Alert(models.Model):
    member = models.ForeignKey(Member, on_delete = models.CASCADE)
    alert_contents = models.TextField(null = False)
    alert_date = models.DateTimeField(auto_now_add = True)

    def __str__(self) :
        return f"{self.member.email} / {self.alert_contents}"