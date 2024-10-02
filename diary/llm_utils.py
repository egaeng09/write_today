from threading import Thread
from typing import Iterator
import time
import torch
import re
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)

import json
import time
from openai import OpenAI
client = OpenAI(api_key="")

from write_today.settings import get_secret

SECRET_KEY = get_secret("GPT_API_KEY")


# from huggingface_hub import login
# login(token="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")
# login(token="47ffc38e-900b-4547-ab2e-ca04c0650d5b")

device = torch.device("cpu")

model_id = "hwan1/ohss-polyglot-ko-empathy-message-friend-1.3b"
# bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16,
# )    

# def llm_load_model():
#     global llm_model,tokenizer
    
#     if torch.cuda.is_available():
#         llm_model = AutoModelForCausalLM.from_pretrained(
#         model_id, quantization_config=bnb_config, device_map=device
#     )
#     else:
#         llm_model = None
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     return llm_model, tokenizer




system_prompt = "일기와 일기의 감정에 대해 공감 메시지를 작성해주세요."
def get_prompt(diary_entry: str, system_prompt: str) -> str:
    return f"{system_prompt}\n\n일기: {diary_entry}\n공감 메시지:"

def llm_load_model():
    global llm_model
    llm_model = AutoModelForCausalLM.from_pretrained(model_id)
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return llm_model, tokenizer

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop for stop in stops]
        self.sentence_ender = re.compile(r'[. ?̊̈ !̆̈ ]$')

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(그만) :])).item():
                return True

        decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        if self.sentence_ender.search(decoded):
            return False

        return False

# class StoppingCriteriaSub(StoppingCriteria):
#     def __init__(self, stops=[], encounters=1):
#         super().__init__()
#         self.stops = [stop for stop in stops]
#         self.sentence_ender = re.compile(r'[. ?̊̈ !̆̈ ]$')

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
#         for stop in self.stops:
#             if torch.all((stop == input_ids[0][-len(그만) :])).item():
#                 return True

#         decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True)
#         if self.sentence_ender.search(decoded):
#             return False

#         return False
###########################################################################API Version###############################################################################
def generate_empathy_message(text, emotions):
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": "당신은 사용자의 친한 친구입니다. 주어진 텍스트와 감정에 기반하여 친구의 입장에서 공감 메시지를 생성하고 JSON 형식으로 출력하세요. 반말을 사용하고, 친구끼리 대화하는 것처럼 자연스럽고 친근한 말투로 응답하세요."},
                {"role": "user", "content": f"다음 내용에 대해 친구의 입장에서 공감하는 메시지를 JSON 형식으로 생성해줘: {text} (감정: {emotions})"}
            ]
        )
        result = json.loads(response.choices[0].message.content)
        message = result.get('message', '야, 네 말 들었어. 어떤 기분인지 알 것 같아.')
    except Exception as e:
        print(f"공감 메시지 생성 중 오류 발생: {e}")
        message = "야, 네 말 들었어. 어떤 기분인지 알 것 같아."
    finally:
        end_time = time.time()
        return message
####################################################################################################################################################################

######################################################################LLM Version####################################################################################

def generate_empathy_message_1(
    diary_entry: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
) -> str:
    prompt = get_prompt(diary_entry, system_prompt)
    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        add_special_tokens=False,
        return_token_type_ids=False,
    ).to("cpu")
    stop_words = ["</s>"]
    stop_words_ids = [
        tokenizer(stop_word, return_tensors="pt").to("cpu")["input_ids"].squeeze()
        for stop_word in stop_words
    ]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    generate_kwargs = dict(
        input_ids=inputs["input_ids"],
        max_length=500,  # 적절한 최대 길이로 설정
        temperature=temperature,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        num_beams=1,
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.eos_token_id,
    )

    outputs = llm_model.generate(**generate_kwargs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    empathy_message = response.split("공감 메시지:")[1].strip()
    empathy_message = empathy_message.split("</끝>")[0].strip()

    return empathy_message
################################################################################################################################################################



# def generate_empathy_message(
#     diary_entry: str,
#     temperature: float = 0.8,
#     top_p: float = 0.95,
#     top_k: int = 50,
#     repetition_penalty: float = 1.2,
# ) -> str:
#     prompt = get_prompt(diary_entry, system_prompt)
#     inputs = tokenizer(
#         [prompt],
#         return_tensors="pt",
#         add_special_tokens=False,
#         return_token_type_ids=False,
#     ).to("cuda")
#     stop_words = ["</s>"]
#     stop_words_ids = [
#         tokenizer(stop_word, return_tensors="pt").to("cuda")["input_ids"].squeeze()
#         for stop_word in stop_words
#     ]
#     stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

#     generate_kwargs = dict(
#         input_ids=inputs["input_ids"],
#         max_length=500,  # 적절한 최대 길이로 설정
#         temperature=temperature,
#         do_sample=True,
#         top_p=top_p,
#         top_k=top_k,
#         repetition_penalty=repetition_penalty,
#         num_beams=1,
#         stopping_criteria=stopping_criteria,
#         pad_token_id=tokenizer.eos_token_id,
#     )

#     # max = 200
#     outputs = llm_model.generate(**generate_kwargs)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     # if len(response.split("공감 메시지:")[1].strip()) > max:
#     #     empathy_message = re.search(r'공감 메시지:(.+?)[.?!]', response).group(1) + "."
#     # else:

#     empathy_message = response.split("공감 메시지:")[1].strip()
#     empathy_message = empathy_message.split("</끝>")[0].strip()


#     return empathy_message

# from threading import Thread
# from typing import Iterator
# import time
# import torch
# import re
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     StoppingCriteria,
#     StoppingCriteriaList,
#     pipeline,
# )

# from langchain_core.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain_community.llms import HuggingFacePipeline

# device = torch.device('mps')

# from huggingface_hub import login
# login(token="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")

# model_id = "hwan1/ohss-polyglot-ko-empathy-message-friend-1.3b"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

# system_prompt = "일기에 대해 공감 메시지를 작성해주세요."

# prompt = PromptTemplate(
#    input_variables=["diary_entry"],
#    template=f"{system_prompt}\n\n일기: {{diary_entry}}\n공감 메시지:",
# )

# def llm_load_model():
#     global llm
#     llm = HuggingFacePipeline(model=model, tokenizer=tokenizer)
#     global empathy_chain
#     empathy_chain = LLMChain(llm=llm, prompt=prompt)


# def get_prompt(diary_entry: str, system_prompt: str) -> str:
#     return f"{system_prompt}\n\n일기: {diary_entry}\n공감 메시지:"


# class StoppingCriteriaSub(StoppingCriteria):
#     def __init__(self, stops=[], encounters=1):
#         super().__init__()
#         self.stops = [stop for stop in stops]
#         self.sentence_ender = re.compile(r'[.?!]$')

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
#         for stop in self.stops:
#             if torch.all((stop == input_ids[0][-len(그만) :])).item():
#                 return True

#         decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True)
#         if self.sentence_ender.search(decoded):
#             return False

#         return False
# def generate_empathy_message(
#     diary_entry: str,
#     temperature: float = 0.8,
#     top_p: float = 0.95,
#     top_k: int = 50,
#     repetition_penalty: float = 1.2,
# ) -> str:
#     prompt_string = prompt.format(diary_entry=diary_entry)
#     inputs = tokenizer(
#         [prompt_string],
#         return_tensors="pt",
#         add_special_tokens=False,
#         return_token_type_ids=False,
#     ).to("cpu")
#     stop_words = ["</s>"]
#     stop_words_ids = [
#         tokenizer(stop_word, return_tensors="pt").to("cpu")["input_ids"].squeeze()
#         for stop_word in stop_words
#     ]
#     stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

#     kwargs = dict(
#         input_ids=inputs["input_ids"],
#         max_length=200,  # 적절한 최대 길이로 설정
#         temperature=temperature,
#         do_sample=True,
#         top_p=top_p,
#         top_k=top_k,
#         repetition_penalty=repetition_penalty,
#         num_beams=1,
#         stopping_criteria=stopping_criteria,
#         pad_token_id=tokenizer.eos_token_id,
#     )

#     max = 200
#     outputs = empathy_chain.run(diary_entry, **kwargs)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     if len(response.split("공감 메시지:")[1].strip()) > max:
#         empathy_message = re.search(r'공감 메시지:(.+?)[.?!]', response).group(1) + "."
#     else:
#         empathy_message = response.split("공감 메시지:")[1].strip()
#         empathy_message = empathy_message.split("</끝>")[0].strip()


#     return empathy_message
