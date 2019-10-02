from sklearn.metrics.pairwise import cosine_similarity
import copy

class TextTiling(object):
    def __init__(self, *, window_size=2, model=False):
        self.window_size = window_size
        self.model = model

    def segment(self, docs):
        # 窓を動かしながらコサイン類似度を計算する
        sim_mat = []
        for i in range(self.window_size, len(docs)):
            print(i)
            if not(docs[i+self.window_size]):
                break

            left_start = i-self.window_size
            right_start = i

            left_window = copy.deepcopy(docs[left_start])
            left_window.extend(docs[left_start+1])
            right_window = copy.deepcopy(docs[right_start])
            right_window.extend(docs[right_start+1])

            print(left_window)
            print(self.model.to_vector(left_window))

            # GensimのTFIDFモデルを用いた文のベクトル化
            sim_mat.append(cosine_similarity(self.model.to_vector(left_window), self.model.to_vector(right_window)))

