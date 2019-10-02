from lib.utils import stems
from lib.tfidf import TfidfModel
from lib.text_tiling import TextTiling
from lib import utils
import datetime
import sys


if __name__ == '__main__':

    # ハイパーパラメータ
    no_below = 10
    no_above=0.1
    keep_n=100000

    tfidf = TfidfModel(no_below=no_below, no_above=no_above, keep_n=keep_n)
    tfidf.load_model()

    # docs: インタビュー全体
    print('Load data')
    path = './data/test.txt'
    # path = './data/interview-text_01-26_all.txt'
    data = utils.load_data(path)
    print('Done')

    # 発言単位
    docs = [row[1] for row in data]
    print(docs[:1])

    # print('===セグメンテーション===')
    # TODO
    # コサイン類似度
    # 可視化
    text_tiling = TextTiling(window_size=2, model=tfidf)
    res = text_tiling.segment([stems(doc) for doc in docs])
    print(res)


    # with open('./result/segmentation/tfidf/' + str(datetime.date.today()) + '.txt', 'w') as f:
    #     print("no_below: " + str(no_below) + ", no_above: " + str(no_above) + ", keep_n: " + str(keep_n) + ", threshold: " + str(threshold), file=f)
    #     for i, docs in enumerate(docs_summary):
    #         print(str(i) + ': ' + docs.strip(), file=f)
