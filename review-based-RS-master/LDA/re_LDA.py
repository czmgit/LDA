import numpy as np
from gensim import corpora, models, similarities
import time
import matplotlib.pyplot as plt
from pylab import *

def GibbsSampling(docs,n_dz,n_zw,n_z,Z):
    for d, doc in enumerate(docs):
        for index, w in enumerate(doc):
            z = Z[d][index]
            # 将当前文档当前单词原topic相关计数减去1
            n_dz[d, z] -= 1
            n_zw[z, w] -= 1
            n_z[z] -= 1
            # 重新计算当前文档当前单词属于每个topic的概率
            multi = np.multiply(n_dz[d, :], n_zw[:, w])
            pz = np.divide(multi, n_z)
            # 按照计算出的分布进行采样
            temp = pz / pz.sum()
            z1 = np.random.multinomial(1, temp)
            z = z1.argmax()
            Z[d][index] = z
            # 将当前文档当前单词新采样的topic相关计数加上1
            n_dz[d, z] += 1
            n_zw[z, w] += 1
            n_z[z] += 1

def RandomInit(docs,n_dz,n_zw,n_z,Z):
    for d ,doc in enumerate(docs):
        zCurrentword = []
        for w in doc:
            pz = np.divide(np.multiply(n_dz[d,:],n_zw[:,w]),n_z)
            z = np.random.multinomial(1,pz/pz.sum()).argmax()
            zCurrentword.append(z)
            n_dz[d,z] += 1
            n_zw[z,w] += 1
            n_z[z] += 1
        Z.append(zCurrentword)

def Perplexity(docs,n_dz,n_zw,n_z):
    nd = np.sum(n_dz,1)
    pw = 0.0
    n = 0
    for d,doc in enumerate(docs):
        for w in doc:
            pw += np.log(((n_zw[:,w]/n_z)*(n_dz[d,:]/nd[d])).sum())
            n += 1
    return np.exp(pw/(-n))#n=847036

def main():
    with open('sw.txt') as f_stop:
        sw = [line.strip() for line in f_stop]
    # remove the stop words,数据清洗与预处理
    with open('news_cn.dat', encoding='utf-8') as file:
        texts = [[word for word in line.strip().lower().split() if (word not in sw) and (len(word)>1)] for line in file]
    M = len(texts)#text total number
    print('语料库载入完成，据统计，一共有 %d 篇文档' % M)

    # build the dictionary for texts
    dictionary = corpora.Dictionary(texts)
    id2word = {}
    for word,idnum in dictionary.token2id.items():
        id2word[idnum] = word
    dict_len = len(dictionary)#dictionary total number
    currentword = []
    docs = []
    for text in texts:
        for word in text:
            currentword.append(dictionary.token2id[word])
        docs.append(currentword)
        currentword = []

    # transform the whole texts to sparse vector
    #corpus = [dictionary.doc2bow(text) for text in texts]
    # create a transformation, from initial model to tf-idf model
    # corpus_tfidf = models.TfidfModel(corpus)[corpus]

    topic_num = 9
    alpha = 1
    beta = 0.1
    Z =[]
    iteratnum = 50
    n_dz = np.zeros([M,topic_num]) + alpha#文档-主题分布矩阵
    n_zw = np.zeros([topic_num,dict_len]) + beta#主题-词分布矩阵
    n_z = np.zeros([topic_num]) + dict_len*beta#为什么要这么乘以dict_len后再相加？

    RandomInit(docs,n_dz,n_zw,n_z,Z)
    perplexity ={}
    for i in range(iteratnum):
        GibbsSampling(docs,n_dz,n_zw,n_z,Z)
        perplexity[i] = Perplexity(docs,n_dz,n_zw,n_z)
        print(time.strftime("%X"),"Iteration",i,"completed!","Perplexity:",perplexity[i])

    plt.plot(perplexity.keys(),perplexity.values(),linewidth = 3)
    plt.title("Topic Number %d" % topic_num)
    plt.xlabel("Iteration Count")
    plt.ylabel("Perplexity")
    plt.show()

    topicwords = []
    maxtopinum = 9
    for z in range(maxtopinum):
        ids = n_zw[z,:].argsort()
        topicword =[]
        for j in ids:#range(1,100):
            #index_num = ids[j]
            topicword.insert(0,id2word[j])
        topicwords.append(topicword[0:min(maxtopinum,len(topicword))])
        print("topicwords:",topicwords[z])
    print("total topic:",topicwords)

if __name__ == '__main__':
    main()
