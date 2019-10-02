from lib import scraping
from lib.tfidf import TfidfModel
from lib.doc2vec import Doc2Vec
from lib.utils import stems
from lib import utils
import sys


if __name__ == '__main__':
    args = sys.argv
    if 2 <= len(args):
        if not(args[1] == 'tfidf' or args[1] == 'doc2vec'):
            print('Argument is invalid')
            exit()
    else:
        print('Arguments are too sort')
        exit()

    model_type = args[1]

    # docs: インタビュー全体
    print('Load data')
    path = './data/interview-text_01-26_all.txt'
    # モデルを訓練する
    data = utils.to_sentence(utils.load_data(path))
    docs = [row[1] for row in data]

    # max_characters: XX文字以上の単文は要約対象外
    # docs = utils.polish_docs(docs, max_characters=1000)
    docs_for_train = [stems(doc) for doc in docs]
    """
    以下のようなデータを作っています
    edocs_for_train = [
    ['出身は', 'どこ', 'ですか' ...
    ['好き', 'な', '食べもの', ...
    ...
    ]
    """
    print(docs[:1])
    print('Done')

    if model_type == 'tfidf':
        # TFIDFモデル生成
        # GensimのTFIDFモデルを用いた文のベクトル化
        print('===TFIDFモデル生成===')
        print('Train tfidf model')
        tfidf = TfidfModel(no_below=10, no_above=0.1, keep_n=100000)
        tfidf.train(docs_for_train)
        print('Done')

    elif model_type == 'doc2vec':
        label = [row[0] for row in data]
        label_docs = list(range(len(docs)))
        print('===Doc2Vec===')
        doc2vec = Doc2Vec(alpha=0.025, min_count=10, size=100, iter=50, workers=4)
        doc2vec.train(docs, label_docs)
    else:
        print('Invalid model type')
        exit()
