import pandas as pd
import torch.nn.functional as F
import torch
import torch.nn as nn
from konlpy.tag import Mecab
import csv
import glob
import get_model as m
from textrank import KeysentenceSummarizer

def sampling_texts(texts, length):
    samples=[]
    for i in range(len(texts)):
        if str(texts[i]).endswith('..'):
            continue
        if len(str(texts[i])) > length:
            samples.append(str(texts[i]))
    return samples

def mecab_tokenizer(sent):
    words = sent.split()
    words = [w for w in words if ('/NN' in w or '/XR' in w or '/VA' in w or '/VV' in w)]
    return words

test2 = m
test = test2.get_model()
soft = nn.Softmax(dim=1)

files = glob.glob('/Users/yohan/automatic_tagging_review_ratings/sanam/data/애매한영화/*.csv')

for f in files:
    data = pd.read_csv(f, encoding='euc-kr')
    sentences = data['review']
    pos_sents = []
    neg_sents = []
    count = 0
    mean = 0

    print(len(sentences))
    length = 130
    if len(sentences) > 1000:
        sents = []
        while len(sents) < 1000:
            sents = sampling_texts(sentences, length)
            length -= 3
    else:
        sents = []
        for i in range(len(sentences)):
            sents.append(sentences[i])
    print(len(sents))

    for sentence in sents:
        rating, posneg = test.inference(sentence)
        posneg_soft = soft(posneg)
        rating_soft = soft(rating)

        mean += torch.argmax(rating_soft).item()

        if posneg_soft.argmax().item() == 0:
            neg_sents.append(sentence)
        else:
            pos_sents.append(sentence)
        count += 1
        print(count)

    print(len(pos_sents), ' ', len(neg_sents))

    m = Mecab()

    pos_samples = pos_sents
    neg_samples = neg_sents
    pos_samples_sents = []
    neg_samples_sents = []

    for i in range(len(pos_samples)):
        sent = ''
        try:
            tmp = m.pos(pos_samples[i])
        except:
            continue
        for j in range(len(tmp)):
            sent += tmp[j][0] + '/' + tmp[j][1] + ' '
        pos_samples_sents.append(sent)

    for i in range(len(neg_samples)):
        sent = ''
        try:
            tmp = m.pos(neg_samples[i])
        except:
            continue
        for j in range(len(tmp)):
            sent += tmp[j][0] + '/' + tmp[j][1] + ' '
        neg_samples_sents.append(sent)

    pos_samples_sents = [sent.strip() for sent in pos_samples_sents]
    pos_samples_texts = [sent.strip() for sent in pos_samples]
    neg_samples_sents = [sent.strip() for sent in neg_samples_sents]
    neg_samples_texts = [sent.strip() for sent in neg_samples]

    summarizer = KeysentenceSummarizer(
        tokenize = mecab_tokenizer,
        min_sim = 0.3
    )

    pos_keysents = summarizer.summarize(pos_samples_sents, topk=10)
    neg_keysents = summarizer.summarize(neg_samples_sents, topk=10)

    pos_ks = []
    neg_ks = []
    for i in range(len(pos_keysents)):
        pos_tmp = []
        pos_tmp.append(f[63:-3])
        pos_tmp.append(pos_keysents[i][1])
        pos_tmp.append(pos_samples_texts[pos_keysents[i][0]])
        pos_tmp.append(1)
        pos_ks.append(pos_tmp)
        neg_tmp = []
        neg_tmp.append(f[63:-3])
        neg_tmp.append(neg_keysents[i][1])
        neg_tmp.append(neg_samples_texts[neg_keysents[i][0]])
        neg_tmp.append(0)
        neg_ks.append(neg_tmp)

    with open(f[63:], 'w', encoding='euc-kr', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['movie_id','text_rank','content','emotion'])
        for i in pos_ks:
            wr.writerow(i)
        for i in neg_ks:
            wr.writerow(i)

