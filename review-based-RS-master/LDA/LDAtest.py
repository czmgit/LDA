from gensim.test.utils import common_texts,datapath
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel

common_dictionary = Dictionary(common_texts)
common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
lda = LdaModel(common_corpus, num_topics=10)
temp_file = datapath("model")
lda.save(temp_file)
lda = LdaModel.load(temp_file)
other_texts = [
    ['computer', 'time', 'graph'],
    ['survey', 'response', 'eps'],
    ['human', 'system', 'computer']
]
other_corpus = [common_dictionary.doc2bow(text) for text in other_texts]
unseen_doc = other_corpus[0]
vector = lda[unseen_doc]
print("参数更新前：",vector)
lda.update(other_corpus)
vector = lda[unseen_doc]
print("参数更新后：",vector)
