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
            print("开始加载本地已预处理的GloVe词表")
            self.load_vocab()
            print("加载GloVe词表结束，词表大小：", len(self.vocab))
        else:
            print("开始加载并处理原始GloVe文件")
            self.preprocess_glove_vocab()
            self.save_vocab()
            print("加载GloVe词表结束，词表大小：", len(self.vocab))

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
        # 先加入<PAD>和<UNK>两个特殊token
        self.vocab["<PAD>"] = 0
        self.vocab["<UNK>"] = 1
        with open(self.glove_path, 'r', encoding='utf-8') as f:
            for i, line in tqdm(enumerate(f)):
                word, vec = line.split(' ', 1)
                self.vocab[word] = len(self.vocab)
                # 逐行加载后拼接矩阵速度非常慢
                # self.weights = torch.cat([self.weights, vec.view(1, -1)], dim=0)

    def preprocess_glove_weight(self):
        # 先加载词表获得词表大小，根据词表大小一次性初始化权重矩阵，之后再逐行加载向量
        self.weights = torch.zeros(len(self.vocab), self.embedding_dim)
        with open(self.glove_path, 'r', encoding='utf-8') as f:
            for i, line in tqdm(enumerate(f)):
                word, vec = line.split(' ', 1)
                vec = torch.tensor([float(v) for v in vec.split(' ')])
                # +2是因为前两个token是<PAD>和<UNK>
                self.weights[i + 2] = vec

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
        # 先计算每个batch中最长的句子长度，然后将所有句子填充到该长度
        # 每个句子的实际长度也需要记录，用于对接torch.nn.utils.rnn.pack_padded_sequence与torch.nn.utils.rnn.pad_packed_sequence
        max_len = 0
        input_length = torch.zeros(len(sent_list), dtype=torch.int32)
        for idx, sent in enumerate(sent_list):
            words = sent.strip().split(" ")
            if max_len < len(words):
                max_len = len(words)
            input_length[idx] = len(words)
        # 使用vocab中的<PAD>填充
        pad_index = self.vocab['<PAD>']
        input_ids = torch.zeros((len(sent_list), max_len), dtype=torch.int32)
        input_ids.fill_(pad_index)
        # 使用vocab中的<UNK>表示未知词
        unk_index = self.vocab['<UNK>']
        # 执行word2index转换
        for idx, sent in enumerate(sent_list):
            words = sent.strip().split(" ")
            for i, word in enumerate(words):
                if word in self.vocab:
                    input_ids[idx][i] = self.vocab[word]
                else:
                    input_ids[idx][i] = unk_index
        return input_ids, input_length


if __name__ == '__main__':
    embedding_dim = 50
    glove_path = './glove.6B/glove.6B.{}d.txt'.format(embedding_dim)
    glove_tokenizer = GloVeTokenizer(glove_path, embedding_dim, True)
    sent_list = ["I love you", "I hate you", "<PAD> abc <UNK>", "test"]

    input_ids, input_length = glove_tokenizer.tokenize(sent_list)
    print(input_ids)
    print(input_ids.shape)
    print(input_length)
    print(input_length.shape)
