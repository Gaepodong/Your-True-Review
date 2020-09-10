import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
import time

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import WarmupLinearSchedule

##GPU 사용 시
# device = torch.device("cuda:0")
device = torch.device("cpu")

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        np_classifier = self.classifier(out)
        return out, np_classifier

class new_bert(nn.Module):
    def __init__(self, model, hidden_size = 768):
        super(new_bert, self).__init__()
        self.main_model = model
        self.fc1 = nn.Linear(hidden_size, 512)
        self.fc2 = nn.Linear(512, 11) # output is rating.

    def forward(self, token_ids, valid_length, segment_ids):
        x, np_cls = self.main_model(token_ids, valid_length, segment_ids)
        x = self.fc1(x)
        
        x = F.gelu(x)
        rating_pred = self.fc2(x)
        return rating_pred, np_cls

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, rating_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.ratings = [np.int32(i[rating_idx]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]
       

    def __getitem__(self, i):
        return (self.sentences[i] + (self.ratings[i], ) + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 30
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

class get_model():
    def __init__(self):
        self.bertmodel, self.vocab = get_pytorch_kobert_model()
        self.tokenizer = get_tokenizer()
        self.tok = nlp.data.BERTSPTokenizer(self.tokenizer, self.vocab, lower=False)
        self.old_model = BERTClassifier(self.bertmodel,  dr_rate=0.5).to(device)
        self.to_model = new_bert(self.old_model).to(device)
        load_model = True
        start_point = 24
        if load_model:
            self.to_model.load_state_dict(torch.load('/Users/yohan/Downloads/jungwlee/model/model_'+str(start_point), map_location=device))
            print("load model")

    def calc_accuracy(self, X, Y):
        max_vals, max_indices = torch.max(X, 1)
        train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
        return train_acc
    
    def inference(self, input_x):
        transform = nlp.data.BERTSentenceTransform(self.tok, max_seq_length=max_len, pad=True, pair=False)
        #input_set = nlp.data.TSVDataset(input_x, field_indices=[0])
        
        query = transform([input_x])
        
        query_2 = np.zeros_like([1])
        query_2[0] = query[1]
     
        token_ids = torch.LongTensor(query[0]).unsqueeze(0).to(device)
        length = torch.LongTensor(query_2)
        segment_ids = torch.LongTensor(query[2]).unsqueeze(0).to(device)
   
        rating_pred, np_cls = self.to_model(token_ids, length, segment_ids)
        return rating_pred, np_cls
