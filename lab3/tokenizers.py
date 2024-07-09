import json
import os

import torch
from tqdm import tqdm

from read_snli import read_snli


class BagOfWord:
    def __init__(self, do_lower_case=True):
        self.do_lower_case = do_lower_case
        if os.path.exists('vocab_BOW.json'):
            print("开始加载BOW词表")
            with open('vocab_BOW.json', 'r') as f:
                self.vocab = json.load(f)
            print("加载BOW词表成功, 词表大小：", len(self.vocab))
        else:
            print("开始生成BOW词表")
            self.vocab = self.generate_vocab()
            print("生成BOW词表成功")
        self.vocab_size = len(self.vocab)

    def generate_vocab(self):
        # 读取训练集数据，使用前提与假设的句子生成词表
        sent_list = []
        data_dir = "../snli_1.0"
        train_file_name = "snli_1.0_train.txt"
        train_data = read_snli(data_dir, train_file_name)
        sent_list.extend(train_data[0])
        sent_list.extend(train_data[1])
        # 构建词表
        print("开始构建词表")
        vocab = {}
        # 添加[PAD]和[UNK]两个特殊词
        vocab["<PAD>"] = {"index": 0, "count": 99999}
        vocab["<UNK>"] = {"index": 1, "count": 99999}
        for sent in tqdm(sent_list):
            if self.do_lower_case:
                sent = sent.lower()
            words = sent.strip().split(" ")
            for word in words:
                if word not in vocab:
                    vocab[word] = {"index": len(vocab), "count": 1}
                else:
                    vocab[word]["count"] += 1
        print("词表构建完毕")
        print("原始词表大小：", len(vocab))
        print("原始词表词频：", analyze_vocab(vocab))
        print("过滤掉词频小于10的词")
        # 保留出现次数不小于10的词，以及index和count信息
        vocab = {k: v for k, v in vocab.items() if v["count"] >= 10}
        # 重新编号
        for idx, (k, v) in enumerate(vocab.items()):
            vocab[k]["index"] = idx
        print("筛选后词表大小：", len(vocab))
        print("筛选后词表词频：", analyze_vocab(vocab))
        with open('vocab_BOW.json', 'w') as f:
            json.dump(vocab, f)
        return vocab

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
        pad_index = self.vocab['<PAD>']["index"]
        input_ids = torch.zeros((len(sent_list), max_len), dtype=torch.int32)
        input_ids.fill_(pad_index)
        # 使用vocab中的<UNK>表示未知词
        unk_index = self.vocab['<UNK>']["index"]
        # 执行word2index转换
        for idx, sent in enumerate(sent_list):
            words = sent.strip().split(" ")
            for i, word in enumerate(words):
                if word in self.vocab:
                    input_ids[idx][i] = self.vocab[word]["index"]
                else:
                    input_ids[idx][i] = unk_index
        return input_ids, input_length


def analyze_vocab(vocab):
    # 以10为间隔统计词频在0-100之间的词的数量，频率大于100的词统计在最后一个位置
    count_list = [0] * 10
    for k, v in vocab.items():
        if v['count'] // 10 < 10:
            count_list[v['count'] // 10] += 1
        else:
            count_list[-1] += 1
    return count_list


if __name__ == "__main__":
    bow = BagOfWord()
    print(analyze_vocab(bow.vocab))

    sent_list = ["A person on a horse jumps over a broken down airplane .",
                 "A person is training his horse for a competition .",
                 'A person is at a diner , ordering an omelette .',
                 'A person is outdoors , on a horse .']

    input_ids, input_length = bow.tokenize(sent_list)

    print(input_ids)

    print(input_length)
