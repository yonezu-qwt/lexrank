# 自作のデータ読み込み&前処理用ライブラリ
from lib import scraping
from lib.tfidf import TfidfModel
from lib.doc2vec import Doc2Vec
from lib.summarize import summarize
from lib.utils import stems
from lib import utils
import datetime
import sys


def validate_args(args):
    if 2 <= len(args):
        if not(args[1] == 'tfidf' or args[1] == 'doc2vec'):
            print('Argument is invalid')
            exit()

        if not(args[2] == 'sentence' or args[2] == 'docs'):
            print('Argument is invalid')
            exit()

        if not(args[3] == 'mmr' or args[3] == 'normal'):
            print('Argument is invalid')
            exit()
    else:
        print('Arguments are too sort')
        exit()

    return args[1], args[2], args[3]


if __name__ == '__main__':

    model, vec_type, sort_type = validate_args(sys.argv)
    train = True

    # docs: インタビュー全体
    print('Load data')
    path = './data/test.txt'
    # path = './data/interview-text_01-26_all.txt'
    data = utils.load_data(path)

    # モデルを訓練する場合
    if train:
        # docs
        # for train
        data_for_train = utils.to_sentence(data)
        label = [row[0] for row in data_for_train]
        docs = [row[1] for row in data_for_train]
        label_docs = list(range(len(docs)))

        # max_characters: XX文字以上の単文は要約対象外
        # docs = utils.polish_docs(docs, max_characters=1000)
        docs_for_train = [stems(doc) for doc in docs]
        print(docs_for_train[:1])

    # 要約する単位 文 or 発言
    # to sentence
    if vec_type == 'sentence':
        data_for_summarization = utils.to_sentence(data)
    else:
        data_for_summarization = data
    # for sum
    docs_for_summarization = [row[1] for row in data_for_summarization]
    docs_for_vec = [stems(doc) for doc in docs_for_summarization]
    print(docs_for_summarization[:1])
    print('Done')

    if model == 'tfidf':
        # TFIDFモデル生成
        # GensimのTFIDFモデルを用いた文のベクトル化
        print('===TFIDFモデル生成===')
        tfidf = TfidfModel(no_below=10, no_above=0.05, keep_n=100000, train=train)
        if train:
            tfidf.train(docs_for_train)
        sent_vecs = tfidf.to_vector(docs_for_vec)
    elif model == 'doc2vec':
        print('===Doc2Vec===')
        doc2vec = Doc2Vec(alpha=0.025, min_count=10, size=100, iter=50, workers=4, train=train)
        if train:
            doc2vec.train(docs_for_train, label_docs)
        sent_vecs = doc2vec.to_vector(docs_for_vec)
    else:
        exit()

    # 表示
    print('===要約===')
    # 要約
    docs_summary = summarize(docs_for_summarization, sent_vecs, sort_type=sort_type, sent_limit=50, threshold=0.1)

    with open('./result/summary_' + model + '_' + vec_type + '_' + sort_type + '_' + str(datetime.date.today()) + '.txt', 'w') as f:
        for i, docs in enumerate(docs_summary):
            print(str(i) + ': ' + docs.strip(), file=f)
