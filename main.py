# 自作のデータ読み込み&前処理用ライブラリ
from lib import scraping
from lib.tfidf import TfidfModel
from lib.doc2vec import Doc2Vec
from lib.summarize import summarize
from lib.utils import stems
from lib import utils
import datetime
import sys


if __name__ == '__main__':
    args = sys.argv
    train = False
    if 2 <= len(args):
        if args[1] == 'tfidf' or args[1] == 'doc2vec':
            model = args[1]
        else:
            print('Argument is invalid')
            exit()

        if args[2] == 'sentence' or args[2] == 'docs':
            vec_type = args[2]
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

    print('Load data')
    # モデルを君れする場合
    if train:
        data = scraping.scraping(1)
        # to sentence
        if vec_type == 'sentence':
            data = utils.to_sentence(data)

        # docs
        label = [row[0] for row in data]
        docs = [row[1] for row in data]
        label_docs = list(range(len(docs)))
        print('Done')

        # max_characters: XX文字以上の単文は要約対象外
        # docs = utils.polish_docs(docs, max_characters=1000)
        docs_for_train = [stems(doc) for doc in docs]
        print(docs_for_train[:3])

    data_for_summarization = utils.to_sentence([(scraping.get_doc(59))])
    docs_for_summarization = [row[1] for row in data_for_summarization]
    print(docs_for_summarization[:3])

    if model == 'tfidf':
        # TFIDFモデル生成
        # GensimのTFIDFモデルを用いた文のベクトル化
        print('===TFIDFモデル生成===')
        tfidf = TfidfModel(no_below=5, no_above=0.1, keep_n=100000, train=train)
        if train:
            tfidf.train(docs_for_train)
        sent_vecs = tfidf.toVector([stems(doc) for doc in docs_for_summarization])
    elif model == 'doc2vec':
        print('===Doc2Vec===')
        # doc2vec = Doc2Vec(alpha=0.025, min_count=10, size=200, iter=50, workers=4)
        # doc2vec.train(docs_for_train, label_docs)
        # sent_vecs = doc2vec.model.docvecs.doctag_syn0.tolist()
        exit()
    else:
        exit()

    # 表示
    print('===要約===')

    # インタビュー要約
    docs_summary = summarize(docs_for_summarization, sent_vecs, sort_type=sort_type, sent_limit=5, threshold=0)

    with open('./result/summary_' + model + '_' + vec_type + '_' + sort_type + '_' + str(datetime.date.today()) + '.txt', 'w') as f:
        for i, docs in enumerate(docs_summary):
            print(str(i) + ': ' + docs.strip(), file=f)
