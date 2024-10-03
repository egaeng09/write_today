import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
np.bool = bool
import gluonnlp as nlp
import pandas as pd

from torch.optim import AdamW
#transformer
#from transformers import AdamW # 곧 종료될 예정 이여서 torch.optim의 AdamW 사용하기
from transformers import XLNetTokenizer
#KoBERT
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

device = torch.device("cpu:0")

tokenizer = XLNetTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, mask_token=None,padding_token='[PAD]')
tokenizer = XLNetTokenizer.from_pretrained('skt/kobert-base-v1')
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower = False)

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    # def preprocess_labels(self, label):
    #     label_tensor = np.zeros(len(emotion_map))
    #     label_tensor[label] = 1
    #     return label_tensor
    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i],))
    def __len__(self):
        return len(self.labels)
    
max_len = 100
batch_size = 64
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

# BERTDataset : 각 데이터가 BERT 모델의 입력으로 들어갈 수 있도록 tokenization, int encoding, padding하는 함수
tok = tokenizer.tokenize

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes = 7,   # 감정 클래스 수로 조정
                 dr_rate = None,
                 params = None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size , num_classes)

        if dr_rate:
            self.dropout = nn.Dropout(p = dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device),return_dict = False)

        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

#BERT  모델 불러오기
model1 = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

# optimizer와 schedule 설정
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model1.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model1.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr = learning_rate)
loss_fn = nn.CrossEntropyLoss() # 다중분류를 위한 loss function

model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

# model.load_state_dict(torch.load('/Users/jeonghwan-il/Desktop/write_today/koBERT_model_mulit_state_dict(0506).pt'))
# checkpoint = torch.load('/Users/jeonghwan-il/Desktop/write_today/koBERT_model_mulit_all(0506).tar')
# model.load_state_dict(checkpoint['model'])
# print("================== KoBERT 모델 로드 완료 ==================")

def kobert_load_model():
    global model
    print("================= kobert_model 로드 중 =================")
    model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
    model.load_state_dict(torch.load('/home/ubuntu/write_today-backend/write_today/models/koBERT_model_mulit_state_dict(0506).pt', map_location=torch.device('cpu:0')))
    checkpoint = torch.load('/home/ubuntu/write_today-backend/write_today/models/koBERT_model_mulit_all(0506).tar', map_location=torch.device('cpu:0'))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("================== kobert_model 모델 로드 완료 ==================")
    return model

# /write_today-backend/
# def kobert_load_model():
#     global model1, optimizer
#     model1 = torch.load('/Users/jeonghwan-il/Desktop/models/koBERT_model_mulit(0505).pt')
#     model1 = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
#     model1.load_state_dict(torch.load('/Users/jeonghwan-il/Desktop/models/koBERT_model_mulit_state_dict(0505).pt'))
#     checkpoint = torch.load('/Users/jeonghwan-il/Desktop/models/koBERT_model_mulit_all(0505).tar')
#     model1.load_state_dict(checkpoint['model'])
#     optimizer = optim.AdamW(model1.parameters())
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     print("================== KoBERT 모델 로드 완료 ==================")


# 감정 라벨 정의
emotion_labels = ["공포", "놀람", "분노", "슬픔", "중립", "행복", "혐오"]

def predict_emotions(predict_sentence, model):
    data = [predict_sentence, '0']
    dataset_another = [data]
    another_test = BERTDataset(dataset_another, 0, 1, tok, vocab, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5, multiprocessing_context='spawn')
    
    model.eval()
    emotion_ratio_list = []
    
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)
        
        out = model(token_ids, valid_length, segment_ids)
        
        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()
            exp_logits = np.exp(logits)  # 지수 값 계산
            probs = exp_logits / np.sum(exp_logits)  # 확률 값 계산 (전체 합이 1이 되도록 정규화)
            
            descending_probs = np.argsort(-probs)  # 내림차순 정렬된 인덱스 배열
            top_7_indices = descending_probs[:7]  # 상위 7개 인덱스
            top_7_emotions = [emotion_labels[idx] for idx in top_7_indices]
            top_7_probs = [probs[idx] * 100 for idx in top_7_indices]  # 100%로 정규화
            
            emotion_ratio_pairs = [[emotion, f"{prob:.2f}%"] for emotion, prob in zip(top_7_emotions, top_7_probs)]
            emotion_ratio_list.extend(emotion_ratio_pairs)

            top1_index = np.argmax(probs)  # 가장 높은 확률의 인덱스
            top1_emotion = emotion_labels[top1_index]
            top1_prob = probs[top1_index] * 100  # 100%로 정규화
            
            emotion_one_ratio_list = []
            emotion_ratio_pair = [top1_emotion, f"{top1_prob:.2f}%"]
            emotion_one_ratio_list.append(emotion_ratio_pair)
    
    return emotion_ratio_list , emotion_one_ratio_list