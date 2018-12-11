from gensim import corpora, models, similarities

sw = [ '宝瓶座', '流星雨', '会津']

stoplist=list('for a of the and to in'.split())
print(stoplist)
with open('news_cn_1.dat',encoding='utf-8') as file:
    list = [[word for word in line.strip().lower().split() if word not in sw] for line in file]
    print(list)
dictionary = corpora.Dictionary(list)
dict_len = len(dictionary)
corpus = [dictionary.doc2bow(text) for text in list]
print(corpus)
