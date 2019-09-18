# 自作のデータ読み込み&前処理用ライブラリ
from lib.tfidf import TfidfModel
from lib.doc2vec import Doc2Vec
from lib.summarize import summarize
from lib.utils import stems
from lib import utils
import datetime
import sys


if __name__ == '__main__':
    args = sys.argv
    if 2 <= len(args):
        if args[1] == 'tfidf' or args[1] == 'doc2vec':
            model = args[1]
        else:
            print('Argument is invalid')
            exit()

        if args[2] == 'sentence' or args[2] == 'conversation':
            sum_type = args[2]
        else:
            print('Argument is invalid')
            exit()

        if args[3] == 'mmr' or args[3] == 'normal':
            sort_type = args[3]
        else:
            print('Argument is invalid')
            exit()
    else:
        print('Arguments are too sort')
        exit()


    # docs: インタビュー全体
    # 文章単位ではなくやりとり単位で分割
    # path = './data/test.txt'
    path = './data/interview-text_01-26_all.txt'

    print('Load data')
    data = utils.load_data(path)
    # to sentence
    if sum_type == 'sentence':
        data = utils.to_sentence(data)

    # docs
    label = [row[0] for row in data]
    docs = [row[1] for row in data]
    label_docs = list(range(len(docs)))
    print('Done')

    # max_characters: XX文字以上の単文は要約対象外
    docs = utils.polish_docs(docs, max_characters=1000)
    docs_for_model = [stems(doc) for doc in docs]

    if model == 'tfidf':
        # TFIDFモデル生成
        # GensimのTFIDFモデルを用いた文のベクトル化
        print('===TFIDFモデル生成===')
        tfidf = TfidfModel(docs_for_model, no_below=10, no_above=0.1, keep_n=100000)
        sent_vecs = tfidf.toVector(docs_for_model)
    elif model == 'doc2vec':
        print('===Doc2Vec===')
        doc2vec = Doc2Vec(alpha=0.025, min_count=10, size=200, iter=50, workers=4)
        doc2vec.train(docs_for_model, label_docs)
        sent_vecs = doc2vec.model.docvecs.doctag_syn0.tolist()
    else:
        exit()

    # 表示
    print('===要約===')

    # インタビュー要約
    docs_summary = summarize(docs, sent_vecs, sort_type=sort_type, sent_limit=50, threshold=0.1)

    with open('./result/summary_' + model + '_' + sum_type + '_' + sort_type + '_' + str(datetime.date.today()) + '.txt', 'w') as f:
        for i, docs in enumerate(docs_summary):
            print(str(i) + ': ' + docs.strip(), file=f)
