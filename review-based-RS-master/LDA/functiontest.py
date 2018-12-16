import numpy as np
from gensim import corpora, models, similarities
import time
import matplotlib.pyplot as plt
from pylab import *

def GibbsSampling(docs,n_dz,n_zw,n_z):
    for d, doc in enumerate(docs):
        for index, w in enumerate(doc):
            z = Z[d][index]
            # 将当前文档当前单词原topic相关计数减去1
            n_dz[d, z] -= 1
            n_zw[z, w] -= 1
            n_z[z] -= 1
            # 重新计算当前文档当前单词属于每个topic的概率
            pz = np.divide(np.multiply(n_dz[d, :], n_zw[:, w]), n_z)
            # 按照计算出的分布进行采样
            z = np.random.multinomial(1, pz / pz.sum()).argmax()
            Z[d][index] = z
            # 将当前文档当前单词新采样的topic相关计数加上1
            n_dz[d, z] += 1
            n_zw[z, w] += 1
            n_z[z] += 1

def StartApp():
    with open('sw.txt') as f_stop:
        sw = [line.strip() for line in f_stop]
    # remove the stop words,数据清洗与预处理
    with open('news_cn.dat', encoding='utf-8') as file:
        texts = [[word for word in line.strip().lower().split() if word not in sw] for line in file]
    M = len(texts)#text total number
    print('语料库载入完成，据统计，一共有 %d 篇文档' % M)

    # build the dictionary for texts
    dictionary = corpora.Dictionary(texts)
    id2word = {}
    for word,idnum in dictionary.token2id.items():
        id2word[idnum] = word
    dict_len = len(dictionary)#dictionary total number
    # transform the whole texts to sparse vector
    corpus = [dictionary.doc2bow(text) for text in texts]
    # create a transformation, from initial model to tf-idf model
    corpus_tfidf = models.TfidfModel(corpus)[corpus]
    topic_num = 9
    alpha = 0.1
    beta = 0.1
    n_dz = np.zeros(M,topic_num)+alpha
    n_zw = np.zeros(topic_num,dict_len)+beta
    n_z = np.zeros(topic_num)+dict_len*beta


if __name__ == '__main__':
    StartApp()
