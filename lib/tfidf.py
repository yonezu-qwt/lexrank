import gensim
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

    def __init__(self, *, no_below=10, no_above=0.05, keep_n=10000, train=False):
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

        self.no_below = no_below  # XX回以下しか出てこない単語は無視
        self.no_above = no_above  # 頻出単語も無視
        self.keep_n = keep_n  # 使用単語数に上限設定
        self.dictionary = None
        self.model = None
        self.corpus = None
        if not(train):
            self.dictionary = gensim.corpora.Dictionary.load_from_text('./model/text.dict')
            self.corpus = gensim.corpora.MmCorpus('./model/text.mm')
            self.model = gensim.models.TfidfModel(self.corpus)

        # corpusへのモデル適用
        # corpus_tfidf = tfidf.model[tfidf.corpus]
        # print(len(tfidf.dictionary.token2id))
        # for doc in corpus_tfidf:
        #     print(doc)

    # モデル生成
    def train(self, docs):
        self.dictionary = gensim.corpora.Dictionary(docs)
        self.dictionary.filter_extremes(no_below=self.no_below, no_above=self.no_above, keep_n=self.keep_n)
        self.corpus = list(map(self.dictionary.doc2bow, docs))
        self.model = gensim.models.TfidfModel(self.corpus)
        self.dictionary.save_as_text('./model/text.dict')  # 保存
        gensim.corpora.MmCorpus.serialize('./model/text.mm', self.corpus)  # 保存

    # GensimのTFIDFモデルを用いた文のベクトル化
    def toVector(self, docs):
        sparse = []
        sent_vecs = [self.model[self.dictionary.doc2bow(doc)] for doc in docs]
        for vec in sent_vecs:
            tmp = np.zeros(len(self.dictionary))
            for key, val in vec:
                tmp[key] = val
            sparse.append(tmp.tolist())

        return sparse
