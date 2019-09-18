from lib.ordered_set import OrderedSet


class Sort(object):
    def __init__(self, sentences, sentence_scores, sim_mat, *, sent_limit=10, lambda_=0.7):
        self.sentences = sentences
        self.sentence_scores = sentence_scores
        self.sim_mat = sim_mat
        self.sent_limit = sent_limit
        self.lambda_ = lambda_

    def normal_sort(self):
        print('Normal sort')
        indexes = OrderedSet()
        num_sent = 0
        for i in sorted(self.sentence_scores, key=lambda i: self.sentence_scores[i], reverse=True):
            num_sent += 1
            if num_sent > self.sent_limit:
                break
            indexes.add(i)
        print('Done')
        return indexes

    def mmr_sort(self):
        print('MMR sort')
        indexes = OrderedSet()
        sentence_ids = set(range(len(self.sentences)))
        while len(indexes) < self.sent_limit and set(indexes) != sentence_ids:
            remaining = sentence_ids - set(indexes)
            mmr_score = lambda x: (self.lambda_*self.sentence_scores[x] - (1-self.lambda_)*max([self.sim_mat[x, y] for y in set(indexes)-{x}] or [0]))
            next_selected = max(remaining, key=mmr_score)
            indexes.add(next_selected)
        print('Done')
        return indexes
