# ライブラリ読み込み
from gensim import models
from lib import utils


class Doc2Vec:
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
        self.model = models.Doc2Vec.load(path)

    def update(self, docs, label):
        sentences = utils.LabeledListSentence(docs, label)
        self.model.build_vocab(sentences, update=True)
        self.model.train(sentences, total_examples=self.model.corpus_count, epochs=self.model.iter)
        self.model.save('./model/doc2vec/updated_doc2vec_' + str(self.vector_size) + '.model')

    def train(self, docs, label):
        sentences = utils.LabeledListSentence(docs, label)

        # doc2vec の学習条件設定
        # alpha: 学習率 / min_count: X回未満しか出てこない単語は無視
        # size: ベクトルの次元数 / iter: 反復回数 / workers: 並列実行数
        self.model = models.Doc2Vec(alpha=self.alpha, min_alpha=self.min_count, min_count=self.min_count, sample=self.sample, vector_size=self.vector_size, epochs=self.epochs, workers=self.workers)

        # doc2vec の学習前準備(単語リスト構築)
        self.model.build_vocab(sentences)

        # Wikipedia から学習させた単語ベクトルを無理やり適用して利用することも出来ます
        # self.model.intersect_word2vec_format('./model/word2vec/entity_vector.model.bin', binary=True)

        # 学習実行
        self.model.train(sentences, total_examples=self.model.corpus_count, epochs=self.model.iter)
        # training = 10
        # for epoch in range(training):
        #     print('epoch ' + str(epoch + 1))

        # セーブ
        self.model.save('./model/doc2vec/doc2vec_' + str(self.vector_size) + '.model')

        # 順番が変わってしまうことがあるので会社リストは学習後に再呼び出し
        self.label = self.model.docvecs.offset2doctag

    def to_vector(self, docs):
        sent_vecs = []
        for doc in docs:
            sent_vecs.append(self.model.infer_vector(doc))

        return sent_vecs
