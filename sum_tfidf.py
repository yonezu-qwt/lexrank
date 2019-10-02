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
        if not(args[1] == 'sentence' or args[1] == 'docs'):
            print('Argument is invalid')
            exit()

        if not(args[2] == 'mmr' or args[2] == 'normal'):
            print('Argument is invalid')
            exit()
    else:
        print('Arguments are too sort')
        exit()

    return args[1], args[2]


if __name__ == '__main__':

    sum_type, sort_type = validate_args(sys.argv)

    # ハイパーパラメータ
    no_below = 10
    no_above=0.1
    keep_n=100000
    threshold = 0

    # docs: インタビュー全体
    print('Load data')
    # path = './data/test.txt'
    path = './data/interview-text_01-26_all.txt'
    data = utils.load_data(path)
    print('Done')

    # 要約する単位 文 or 発言
    # to sentence
    if sum_type == 'sentence':
        data = utils.to_sentence(data)

    # for sum
    docs = [row[1] for row in data]
    print(docs[:1])

    # GensimのTFIDFモデルを用いた文のベクトル化
    tfidf = TfidfModel(no_below=no_below, no_above=no_above, keep_n=keep_n)
    tfidf.load_model()
    sent_vecs = tfidf.to_vector([stems(doc) for doc in docs])

    # 表示
    print('===要約===')
    # 要約
    docs_summary = summarize(docs, sent_vecs, sort_type=sort_type, sent_limit=50, threshold=threshold)

    with open('./result/summary/tfidf/' + sum_type + '_' + sort_type + '_' + str(datetime.date.today()) + '.txt', 'w') as f:
        print("no_below: " + str(no_below) + ", no_above: " + str(no_above) + ", keep_n: " + str(keep_n) + ", threshold: " + str(threshold), file=f)
        for i, docs in enumerate(docs_summary):
            print(str(i) + ': ' + docs.strip(), file=f)
