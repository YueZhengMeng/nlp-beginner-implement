import json
import os

import torch
from tqdm import tqdm


class GloVeTokenizer:
    def __init__(self, glove_path, embedding_dim, use_pretrained=True):
        self.glove_path = glove_path
        self.embedding_dim = embedding_dim
        self.vocab = {}
        self.weights = None
        self.use_pretrained = use_pretrained

        if os.path.exists(self.glove_path.replace('.txt', '_vocab.json')):
            print("开始加载本地已预处理的GloVe词典")
            self.load_vocab()
            print("加载GloVe词典结束，词典大小：", len(self.vocab))
        else:
            print("开始加载并处理原始GloVe文件")
            self.preprocess_glove_vocab()
            self.save_vocab()
            print("加载GloVe词典结束，词典大小：", len(self.vocab))

        if self.use_pretrained:
            if os.path.exists(self.glove_path.replace('.txt', '.pt')):
                print("开始加载本地已预处理的GloVe词向量")
                self.load_weight()
                print("加载GloVe词向量结束，词向量大小：", self.weights.shape)
            else:
                print("开始加载并处理原始GloVe文件")
                self.preprocess_glove_weight()
                self.save_weight()
                print("加载GloVe词向量结束，词向量大小：", self.weights.shape)

    def preprocess_glove_vocab(self):
        with open(self.glove_path, 'r', encoding='utf-8') as f:
            for i, line in tqdm(enumerate(f)):
                word, vec = line.split(' ', 1)
                self.vocab[word] = i
                # 逐行加载后拼接矩阵速度非常慢
                # self.weights = torch.cat([self.weights, vec.view(1, -1)], dim=0)

    def preprocess_glove_weight(self):
        # 先加载字典获得字典大小，根据字典大小一次性初始化权重矩阵，之后再逐行加载向量
        self.weights = torch.zeros(len(self.vocab), self.embedding_dim)
        with open(self.glove_path, 'r', encoding='utf-8') as f:
            for i, line in tqdm(enumerate(f)):
                word, vec = line.split(' ', 1)
                vec = torch.tensor([float(v) for v in vec.split(' ')])
                self.weights[i] = vec

    def save_vocab(self):
        with open(self.glove_path.replace('.txt', '_vocab.json'), 'w') as f:
            json.dump(self.vocab, f)

    def save_weight(self):
        torch.save(self.weights, self.glove_path.replace('.txt', '.pt'))

    def load_vocab(self):
        with open(self.glove_path.replace('.txt', '_vocab.json'), 'r') as f:
            self.vocab = json.load(f)

    def load_weight(self):
        self.weights = torch.load(self.glove_path.replace('.txt', '.pt'))

    def get_vocab_size(self):
        return len(self.vocab)

    def tokenize(self, sent_list):
        # 先计算每个batch中最长的句子长度，然后将所有句子填充到改长度
        # 同时记录每个句子的最后一个有效token的位置，用于TextRNN的forward函数
        max_len = 0
        last_token_pos = torch.zeros(len(sent_list), dtype=torch.int32)
        for idx, sent in enumerate(sent_list):
            words = sent.strip().split(" ")
            if max_len < len(words):
                max_len = len(words)
            last_token_pos[idx] = len(words) - 1
        # 使用vocab中的<unk>填充
        unk_index = self.vocab['<unk>']
        sent_feature = torch.zeros((len(sent_list), max_len), dtype=torch.int32)
        sent_feature.fill_(unk_index)
        # 执行word2index转换
        for idx, sent in enumerate(sent_list):
            words = sent.strip().split(" ")
            for i, word in enumerate(words):
                if word in self.vocab:
                    sent_feature[idx][i] = self.vocab[word]
                else:
                    sent_feature[idx][i] = unk_index
        return sent_feature, last_token_pos


if __name__ == '__main__':
    embedding_dim = 100
    glove_path = './glove.6B/glove.6B.{}d.txt'.format(embedding_dim)
    glove_tokenizer = GloVeTokenizer(glove_path, embedding_dim, True)
    sent_list = ["I love you", "I hate you", "abc"]
    input_ids, last_token_pos = glove_tokenizer.tokenize(sent_list)
    print(input_ids)
    print(input_ids.shape)
    print(last_token_pos)
    print(last_token_pos.shape)
