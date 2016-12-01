from gensim.models import Word2Vec
from gensim.corpora.dictionary import Dictionary
from data_preprocessing import data
import numpy as np
from misc import wd


# Import w2v's dictionary to a bag-of-words model
w2v_model = Word2Vec.load(wd +'/w2v/w2v_allwiki_nkjp300_200.model')
w2v_vocabulary = Dictionary()
w2v_vocabulary.doc2bow(w2v_model.vocab.keys(), allow_update=True)

# Initialize dicts for representing w2v's dictionary as indices and 200-dim vectors
w2indx = {v: k+1 for k, v in w2v_vocabulary.items()}
w2vec = {word: w2v_model[word] for word in w2indx.keys()}

w2v_vocabulary_size = len(w2indx) + 1
w2v_vocabulary_dimension = len(w2vec.values()[0])


def map_treebank_words_to_w2v_indices(treebank_data, w2indx):
    treebank_data_vec = []
    for sentence in treebank_data:
        vectorized_sentence = []
        for word in sentence:
            try:
                vectorized_sentence.append(w2indx[word])
            except KeyError:  # words absent in w2v model will be indexed as 0s
                vectorized_sentence.append(0)
        treebank_data_vec.append(vectorized_sentence)
    return treebank_data_vec

data = map_treebank_words_to_w2v_indices(data, w2indx)

embedding_weights = np.zeros((w2v_vocabulary_size, w2v_vocabulary_dimension))
for word, index in w2indx.items():
    embedding_weights[index, :] = w2vec[word]
