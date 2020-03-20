from collections import Counter
from gensim.models import KeyedVectors
import numpy as np


class Vocab(object):
    PAD, UNK = 0, 1

    def __init__(self, word_counter, label, min_occur_count=0):
        self._id2word = ['<pad>', '<unk>']
        self._wordid2freq = [10000, 10000]
        self._id2label = [k for k, v in label.most_common()]
        for word, count in word_counter.most_common():
            if count > min_occur_count:
                self._id2word.append(word)
                self._wordid2freq.append(count)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)
        self._label2id = reverse(self._id2label)
        if len(self._word2id) != len(self._id2word):
            print("serious bug: words dumplicated, please check!")

        print("Vocab info: #words {0}".format(self.vocab_size))

    # def load_pretrained_embs(self, embfile):
    #     embedding_dim = -1
    #     word_count = 0
    #     with open(embfile, encoding='utf-8') as f:
    #         for line in f.readlines():
    #             if word_count < 1:
    #                 values = line.split()
    #                 embedding_dim = len(values) - 1
    #             word_count += 1
    #     print('Total words: ' + str(word_count) + '\n')
    #     print('The dim of pretrained embeddings: ' + str(embedding_dim) + '\n')
    #
    #     index = len(self._id2word)
    #     embeddings = np.zeros((word_count + index, embedding_dim))
    #     with open(embfile, encoding='utf-8') as f:
    #         for line in f.readlines():
    #             values = line.split()
    #             self._id2extword.append(values[0])
    #             vector = np.array(values[1:], dtype='float64')
    #             embeddings[self.UNK] += vector
    #             embeddings[index] = vector
    #             index += 1
    #
    #     embeddings[self.UNK] = embeddings[self.UNK] / word_count
    #     embeddings = embeddings / np.std(embeddings)
    #
    #     reverse = lambda x: dict(zip(x, range(len(x))))
    #     self._extword2id = reverse(self._id2extword)
    #
    #     if len(self._extword2id) != len(self._id2extword):
    #         print("serious bug: extern words dumplicated, please check!")
    #
    #     return embeddings

    def create_pretrained_embs(self, embfile):
        word_vectors = KeyedVectors.load_word2vec_format(embfile, binary=False)
        wv_matrix = list()

        # one for UNK and one for zero padding
        wv_matrix.append(np.zeros(300).astype("float32"))  # zero padding
        wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))  # UNK

        for word in self._id2word[2:]:
            if word in word_vectors.vocab:
                wv_matrix.append(word_vectors.word_vec(word))
            else:
                wv_matrix.append(wv_matrix[1])
        wv_matrix = np.array(wv_matrix)

        assert len(wv_matrix) == len(self._id2word)
        print('embedding size', wv_matrix.shape)
        return wv_matrix

    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.UNK) for x in xs]
        return self._word2id.get(xs, self.UNK)

    def id2word(self, xs):
        if isinstance(xs, list):
            return [self._id2word[x] for x in xs]
        return self._id2word[xs]

    def id2label(self, xs):
        if isinstance(xs, list):
            return [self._id2label[x] for x in xs]
        return self._id2label[xs]

    def label2id(self, xs):
        if isinstance(xs, list):
            return [self._label2id.get(x, 0) for x in xs]
        return self._label2id.get(xs, 0)

    @property
    def vocab_size(self):
        return len(self._id2word)


def creatVocab(train_data, min_occur_count):
    word_counter = Counter()
    label = Counter()
    for instance in train_data:
        for word in instance.seg_list:
            word_counter[word] += 1
        label[instance.label] += 1

    return Vocab(word_counter, label, min_occur_count)
