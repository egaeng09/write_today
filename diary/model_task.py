from celery import shared_task
# from celery.signals import worker_process_init
from celery.signals import worker_init

from .models import Diary, Result, Emotion, MixedEmotion

from .kobert_utils import predict_emotions, kobert_load_model
from .llm_utils import generate_empathy_message, llm_load_model


###################################### Model Version  ###################################### 
###################################### Model Version  ######################################
# from .firebase_utils import send_notification

# global kobert_model, llm_model, tokenizer
# kobert_model = None
# llm_model = None
# tokenizer = None

## @worker_process_init.connect
# @worker_init.connect
# def init_worker(**kwargs):
#     print("================= init_worker 함수 호출 시작 =================")
#     global kobert_model, llm_model, tokenizer
#     if kobert_model is None:
#         print("================= kobert_model 로드 시작 =================")
#         kobert_model = kobert_load_model()
#         print("================= kobert_model 로드 완료 =================")
#     if llm_model is None or tokenizer is None:
#         print("================= llm_model 및 tokenizer 로드 시작 =================")
#         lm_model, tokenizer = llm_load_model()
#         print("================= llm_model 및 tokenizer 로드 완료 =================")
#     print("================= init_worker 함수 호출 완료 =================")

# @shared_task
# def process_diary(diary_id):
#     diary = Diary.objects.get(id=diary_id)
#     contents = diary.contents

#     print("================= predict_emotions 시작 =================")
#     emotion_ratio_list , emotion_one_ratio_list = predict_emotions(contents, kobert_model)
#     print("================= predict_emotions 끝 =================")

#     emotion_colors = {
#         "공포": "#3F8A5F",
#         "놀람": "#E87F50",
#         "분노": "#CD5070",
#         "슬픔": "#3C63EB",
#         "행복": "#F1BD47",
#         "중립": "#FFFFFF",
#         "혐오": "#488C97",
#     }

#     emotion_str = emotion_ratio_list
#     emotion_one_str = emotion_one_ratio_list

#     diary_entry = f"{contents} [{emotion_one_str}]"

#     print("================= empathy_message 생성 시작 =================")
#     empathy_message = generate_empathy_message(diary_entry)
#     print("================= empathy_message 생성 끝 =================")

#     result = Result.objects.create(diary=diary, answer=empathy_message)
#     result.save()

#     for emotion_name, emotion_rate in emotion_ratio_list :
#         emotion, _ = Emotion.objects.get_or_create(name=emotion_name, hex=emotion_colors.get(emotion_name))
#         mixed_emotion = MixedEmotion.objects.create(result=result, emotion=emotion, rate=float(emotion_rate.strip('%')))
#         mixed_emotion.save()

#     print("================= 결과 생성 완료 =================")
#     print("일기 내용:", contents)
#     print("감정 분석 결과:", emotion_str)
#     print("공감 메시지:", empathy_message)


# @shared_task
# def process_modified_diary(diary_id):
#     diary = Diary.objects.get(id=diary_id)
#     contents = diary.contents
#     print("================= diary 수정 시작 =================")
#     print("================= predict_emotions  수정 시작 =================")
#     emotion_ratio_list , emotion_one_ratio_list = predict_emotions(contents, kobert_model)
#     print("================= predict_emotions 수정 끝 =================")

#     emotion_colors = {
#         "공포": "#3F8A5F",
#         "놀람": "#E87F50",
#         "분노": "#CD5070",
#         "슬픔": "#3C63EB",
#         "중립": "#FFFFFF",
#         "행복": "#F1BD47",
#         "혐오": "#488C97",
#     }


#     emotion_str = emotion_ratio_list
#     emotion_one_str = emotion_one_ratio_list

#     diary_entry = f"{contents} [{emotion_one_str}]"

#     print("================= empathy_message 수정 시작 =================")
#     empathy_message = generate_empathy_message(diary_entry)
#     print("================= empathy_message 수정 끝 =================")

#     Result.objects.filter(diary=diary).delete()
#     print("================= 기존 결과 삭제 완료 =================")
    
#     result = Result.objects.create(diary=diary, answer=empathy_message)
#     result.save()

#     for emotion_name, emotion_rate in emotion_ratio_list :
#         emotion, _ = Emotion.objects.get_or_create(name=emotion_name, hex=emotion_colors.get(emotion_name))
#         mixed_emotion = MixedEmotion.objects.create(result=result, emotion=emotion, rate=float(emotion_rate.strip('%')))
#         mixed_emotion.save()

#     print("================= 수정된 결과 생성 완료 =================")
#     print("일기 내용:", contents)
#     print("감정 분석 결과:", emotion_str)
#     print("공감 메시지:", empathy_message)

###################################### API Version  ###################################### 
###################################### API Version  ######################################
global kobert_model, tokenizer
kobert_model = None
tokenizer = None

@worker_init.connect
def init_worker(**kwargs):
    print("================= init_worker 함수 호출 시작 =================")
    global kobert_model, tokenizer
    if kobert_model is None:
        print("================= kobert_model 로드 시작 =================")
        kobert_model = kobert_load_model()
        print("================= kobert_model 로드 완료 =================")
    # if  tokenizer is None:
    #     print("================= tokenizer 로드 시작 =================")
    #     tokenizer = llm_load_model()
    #     print("================= tokenizer 로드 완료 =================")
    print("================= init_worker 함수 호출 완료 =================")

@shared_task
def process_diary(diary_id):
    diary = Diary.objects.get(id=diary_id)
    contents = diary.contents

    print("================= predict_emotions 시작 =================")
    emotion_ratio_list , emotion_one_ratio_list = predict_emotions(contents, kobert_model)
    print("================= predict_emotions 끝 =================")

    emotion_colors = {
        "공포": "#3F8A5F",
        "놀람": "#E87F50",
        "분노": "#CD5070",
        "슬픔": "#3C63EB",
        "행복": "#F1BD47",
        "중립": "#FFFFFF",
        "혐오": "#488C97",
    }

    emotion_str = emotion_ratio_list
    emotion_one_str = emotion_one_ratio_list

    diary_entry = f"{contents} [{emotion_one_str}]"

    print("================= empathy_message 생성 시작 =================")
    empathy_message = generate_empathy_message(contents,emotion_one_str)
    print("================= empathy_message 생성 끝 =================")

    result = Result.objects.create(diary=diary, answer=empathy_message)
    result.save()

    for emotion_name, emotion_rate in emotion_ratio_list :
        emotion, _ = Emotion.objects.get_or_create(name=emotion_name, hex=emotion_colors.get(emotion_name))
        mixed_emotion = MixedEmotion.objects.create(result=result, emotion=emotion, rate=float(emotion_rate.strip('%')))
        mixed_emotion.save()

    print("================= 결과 생성 완료 =================")
    print("일기 내용:", contents)
    print("감정 분석 결과:", emotion_str)
    print("공감 메시지:", empathy_message)


@shared_task
def process_modified_diary(diary_id):
    diary = Diary.objects.get(id=diary_id)
    contents = diary.contents
    print("================= diary 수정 시작 =================")
    print("================= predict_emotions  수정 시작 =================")
    emotion_ratio_list , emotion_one_ratio_list = predict_emotions(contents, kobert_model)
    print("================= predict_emotions 수정 끝 =================")

    emotion_colors = {
        "공포": "#3F8A5F",
        "놀람": "#E87F50",
        "분노": "#CD5070",
        "슬픔": "#3C63EB",
        "중립": "#FFFFFF",
        "행복": "#F1BD47",
        "혐오": "#488C97",
    }


    emotion_str = emotion_ratio_list
    emotion_one_str = emotion_one_ratio_list

    diary_entry = f"{contents} [{emotion_one_str}]"

    print("================= empathy_message 수정 시작 =================")
    empathy_message = generate_empathy_message(contents,emotion_one_str)
    print("================= empathy_message 수정 끝 =================")

    Result.objects.filter(diary=diary).delete()
    print("================= 기존 결과 삭제 완료 =================")
    
    result = Result.objects.create(diary=diary, answer=empathy_message)
    result.save()

    for emotion_name, emotion_rate in emotion_ratio_list :
        emotion, _ = Emotion.objects.get_or_create(name=emotion_name, hex=emotion_colors.get(emotion_name))
        mixed_emotion = MixedEmotion.objects.create(result=result, emotion=emotion, rate=float(emotion_rate.strip('%')))
        mixed_emotion.save()

    print("================= 수정된 결과 생성 완료 =================")
    print("일기 내용:", contents)
    print("감정 분석 결과:", emotion_str)
    print("공감 메시지:", empathy_message)
