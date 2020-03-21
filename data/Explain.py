import jieba


class Explain:
    def __init__(self, segment, label):
        self.segment = segment
        self.label = label
        self.seg_list = jieba.lcut(segment, cut_all=False)  # 精确模式
        self.length = len(self.seg_list)
        self.regularized_seg = []
        self.tokens = []
