import pandas as pd
import torch.nn.functional as F
import torch
import torch.nn as nn
from konlpy.tag import Mecab
import csv
import glob
import get_model as m
from textrank import KeysentenceSummarizer
from textrank import KeywordSummarizer
from tqdm import trange
from wordcloud import WordCloud
import matplotlib.pyplot as plt

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

def mecab_noun_tokenizer(sent):
    words = sent.split()
    words = [w for w in words if ('/NN' in w)]
    return words

def __array__(self):
    return self.to_array()

def to_array(self):
    return np.array(self.to_image())

test2 = m
test = test2.get_model()
soft = nn.Softmax(dim=1)

wordcloud = WordCloud(
    font_path = '/Library/fonts/NanumBarunpenR.ttf',
    width = 540,
    height = 250,
    background_color='white'
)

files = glob.glob('/Users/yohan/Your-True-Review/rawdata/애매한영화/*.csv')

for f in trange(len(files)):
    data = pd.read_csv(files[f], encoding='euc-kr')
    sentences = data['review']
    total_len = len(sentences)
    pos_sents = []
    neg_sents = []
    mean = 0
    name = files[f][44:]

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

    for sentence in trange(len(sents)):
        rating, posneg = test.inference(sents[sentence])
        posneg_soft = soft(posneg)
        rating_soft = soft(rating)

        mean += torch.argmax(rating_soft).item()

        if posneg_soft.argmax().item() == 0:
            neg_sents.append(sents[sentence])
        else:
            pos_sents.append(sents[sentence])

    mean /= len(sents)
    pos_total_len = len(pos_sents)
    neg_total_len = len(neg_sents)

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

    pos_keysents_rating = []
    neg_keysents_rating = []

    for i in range(len(pos_keysents)):
        rating, posneg = test.inference(pos_samples_texts[pos_keysents[i][0]])
        rating_soft = soft(rating)
        pred = torch.argmax(rating_soft).item()
        pos_keysents_rating.append(pred)
    
    for i in range(len(neg_keysents)):
        rating, negneg = test.inference(neg_samples_texts[neg_keysents[i][0]])
        rating_soft = soft(rating)
        pred = torch.argmax(rating_soft).item()
        neg_keysents_rating.append(pred)

    pos_ks = []
    neg_ks = []
    for i in range(len(pos_keysents)):
        pos_tmp = []
        pos_tmp.append(files[f][44:-4])
        pos_tmp.append(pos_keysents[i][1])
        pos_tmp.append(pos_samples_texts[pos_keysents[i][0]])
        pos_tmp.append(1)
        pos_tmp.append(pos_keysents_rating[i])
        pos_tmp.append(mean)
        pos_tmp.append(pos_total_len)
        pos_tmp.append(total_len)
        pos_ks.append(pos_tmp)
        neg_tmp = []
        neg_tmp.append(files[f][44:-4])
        neg_tmp.append(neg_keysents[i][1])
        neg_tmp.append(neg_samples_texts[neg_keysents[i][0]])
        neg_tmp.append(0)
        neg_tmp.append(neg_keysents_rating[i])
        neg_tmp.append(mean)
        neg_tmp.append(neg_total_len)
        neg_tmp.append(total_len)
        neg_ks.append(neg_tmp)

    with open('/Users/yohan/Your-True-Review/data/애매한영화/' + name, 'w', encoding='euc-kr', newline='') as f2:
        wr = csv.writer(f2)
        wr.writerow(['movie_id','text_rank','content','emotion', 'rating', 'total_rank_pred', 'posneg_len', 'total_len'])
        for i in pos_ks:
            wr.writerow(i)
        for i in neg_ks:
            wr.writerow(i)

    keyword_extractor = KeywordSummarizer(
        tokenize = mecab_noun_tokenizer,
        window = 2,
        verbose = True
    )
    pos_keywords = keyword_extractor.summarize(pos_samples_sents, topk=100)
    neg_keywords = keyword_extractor.summarize(neg_samples_sents, topk=100)
    pos_wordrank = {}
    neg_wordrank = {}
    for i in range(len(pos_keywords)):
        pos_wordrank[pos_keywords[i][0][:pos_keywords[i][0].find('/')]] = pos_keywords[i][1]
    for i in range(len(neg_keywords)):
        neg_wordrank[neg_keywords[i][0][:neg_keywords[i][0].find('/')]] = neg_keywords[i][1]

    if '영화' in pos_wordrank.keys():
        del pos_wordrank['영화']
    if '것' in pos_wordrank.keys():
        del pos_wordrank['것']
    if '듯' in pos_wordrank.keys():
        del pos_wordrank['듯']
    if '관람객' in pos_wordrank.keys():
        del pos_wordrank['관람객']
    if '자' in pos_wordrank.keys():
        del pos_wordrank['자']
    if '후' in pos_wordrank.keys():
        del pos_wordrank['후']
    if '때' in pos_wordrank.keys():
        del pos_wordrank['때']
    if '중' in pos_wordrank.keys():
        del pos_wordrank['중']
    if '평점' in pos_wordrank.keys():
        del pos_wordrank['평점']
    if '점' in pos_wordrank.keys():
        del pos_wordrank['점']

    if '영화' in neg_wordrank.keys():
        del neg_wordrank['영화']
    if '것' in neg_wordrank.keys():
        del neg_wordrank['것']
    if '듯' in neg_wordrank.keys():
        del neg_wordrank['듯']
    if '관람객' in neg_wordrank.keys():
        del neg_wordrank['관람객']
    if '자' in neg_wordrank.keys():
        del neg_wordrank['자']
    if '후' in neg_wordrank.keys():
        del neg_wordrank['후']
    if '때' in neg_wordrank.keys():
        del neg_wordrank['때']
    if '중' in neg_wordrank.keys():
        del neg_wordrank['중']
    if '평점' in neg_wordrank.keys():
        del neg_wordrank['평점']
    if '점' in neg_wordrank.keys():
        del neg_wordrank['점']

    pos_wordcloud = wordcloud.generate_from_frequencies(pos_wordrank)
    pos_array = pos_wordcloud.to_array()
    neg_wordcloud = wordcloud.generate_from_frequencies(neg_wordrank)
    neg_array = neg_wordcloud.to_array()

    pos_fig = plt.figure(figsize=(10, 10))
    plt.imshow(pos_array, interpolation='bilinear')
    plt.axis('off')
    pos_fig.savefig('/Users/yohan/Your-True-Review/data/wordcloud/' + name[:-4] + '_pos')

    neg_fig = plt.figure(figsize=(10, 10))
    plt.imshow(neg_array, interpolation='bilinear')
    plt.axis('off')
    neg_fig.savefig('/Users/yohan/Your-True-Review/data/wordcloud/' + name[:-4] + '_neg')
