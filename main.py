# 自作のデータ読み込み&前処理用ライブラリ
from lib.tfidf import TfidfModel
from lib.summarize import summarize
from lib import utils
import sys


if __name__ == '__main__':
    args = sys.argv
    if 2 <= len(args):
        if args[1] == 'tfidf' or args[1] == 'doc2vec':
            model = args[1]
        else:
            print('Argument is invalid')
            exit()
    else:
        print('Arguments are too sort')
        exit()


    # docs: インタビュー全体
    # 文章単位ではなくやりとり単位で分割
    path = './data/test.txt'
    # path = './data/interview-text_01-26_all.txt'
    docs = utils.load_data(path)
    # max_characters: XX文字以上の単文は要約対象外
    docs = utils.polish_data(docs, max_characters=1000)

    if model == 'tfidf':
        # TFIDFモデル生成
        # GensimのTFIDFモデルを用いた文のベクトル化
        print('===TFIDFモデル生成===')
        tfidf = TfidfModel(docs, no_below=10, no_above=0.1, keep_n=100000)
        sent_vecs = tfidf.toVector(docs)
    elif model == 'doc2vec':
        print('===Doc2Vec===')
        exit()
        # TODO: doc2vec
        # doc2vec = Doc2Vec(docs, no_below=10, no_above=0.1, keep_n=100000)
        # sent_vecs = doc2vec.tofVector(docs)
    else:
        exit()

    # 表示
    print('===要約===')

    # インタビュー要約
    docs_summary = summarize(docs, sent_vecs, use_mmr=True, sent_limit=50)

    with open('out_tmp.txt', 'w') as f:
        for i, docs in enumerate(docs_summary):
            print(str(i) + ': ' + docs.strip(), file=f)
