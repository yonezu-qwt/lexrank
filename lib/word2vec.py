# ライブラリ読み込み
from gensim import models
from lib import utils
import numpy as np


class Word2Vec:
    def __init__(self, *, alpha=0.025, min_alpha=0.00025, min_count=5, sample=1e-5, vector_size=100, epochs=20, workers=4):
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.min_count = min_count
        self.sample = sample
        self.vector_size = vector_size
        self.epochs = epochs
        self.workers = workers
        self.label = None
        self.model = None

    def load_model(self, path):
        # モデルをロード
        self.model = models.Word2Vec.load(path)

    def update(self, docs):
        self.model.build_vocab(docs, update=True)
        self.model.train(docs, total_examples=self.model.corpus_count, epochs=self.model.iter)
        self.model.save('./model/word2vec/updated_word2vec_' + str(self.vector_size) + '.model')

    def train(self, docs):
        # word2vec の学習条件設定
        # alpha: 学習率 / min_count: X回未満しか出てこない単語は無視
        # size: ベクトルの次元数 / iter: 反復回数 / workers: 並列実行数
        self.model = models.Word2Vec(alpha=self.alpha, min_alpha=self.min_count, min_count=self.min_count, sample=self.sample, size=self.vector_size, iter=self.epochs, workers=self.workers)

        # word2vec の学習前準備(単語リスト構築)
        self.model.build_vocab(docs)

        # 学習実行
        self.model.train(docs, total_examples=self.model.corpus_count, epochs=self.model.iter)
        # training = 10
        # for epoch in range(training):
        #     print('epoch ' + str(epoch + 1))

        # セー ブ
        self.model.save('./model/word2vec/word2vec_' + str(self.vector_size) + '.model')

    def to_vector(self, docs):
        feature_vecs = []
        for doc in docs:
            count = 0
            sum = np.zeros(len(self.model.wv.syn0[0]), dtype="float32") # 特徴ベクトルの入れ物を初期化
            for word in doc:
                if word in self.model.wv.vocab:
                    count += 1
                    sum = np.add(sum, self.model.wv[word])

            if count != 0:
                feature_vecs.append(np.divide(sum, count))

        return feature_vecs
