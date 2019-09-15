import gensim
from lib.utils import stems
import numpy as np

class TfidfModel(object):
    '''
    TFIDFモデル、辞書などを保持

    Attributes
    ----------
    self.dictionary : dict
        辞書
    self.corpus : list
        コーパス
    self.model : model
        tfidfモデル
    '''

    def __init__(self, docs, *, no_below=10, no_above=0.05, keep_n=10000):
        '''
        Parameters
        ----------
        docs : list
            対象の文章
        no_below : int
            XX回以下しか出てこない単語は無視
        no_above : int
            頻出単語も無視
        keep_n : int
            使用単語数に上限設定
        '''

        self.docs = []
        for doc in docs:
            self.docs.append(stems(doc))

        print(self.docs)

        self.dictionary = gensim.corpora.Dictionary(self.docs)
        self.dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
        self.corpus = list(map(self.dictionary.doc2bow, self.docs))
        self.model = gensim.models.TfidfModel(self.corpus)

        # corpusへのモデル適用
        # corpus_tfidf = tfidf.model[tfidf.corpus]
        # print(len(tfidf.dictionary.token2id))
        # for doc in corpus_tfidf:
        #     print(doc)

    # GensimのTFIDFモデルを用いた文のベクトル化
    def toVector(self, docs):
        sparse = []
        sent_vecs = [self.model[self.dictionary.doc2bow(stems(doc))] for doc in docs]
        for vec in sent_vecs:
            tmp = np.zeros(len(self.dictionary))
            for key, val in vec:
                tmp[key] = val
            sparse.append(tmp)

        return sparse
