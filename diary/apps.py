from django.apps import AppConfig


from django.apps import AppConfig
# from .kobert_utils import kobert_load_model
# from .llm_utils import llm_load_model


class DiaryConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'diary'
    
    # def kobert_ready(self):
    #     kobert_load_model()
    
    # def llm_ready(self):
    #     llm_load_model()

    def ready(self):
        import diary.signals