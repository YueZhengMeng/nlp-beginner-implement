import json
import os

import numpy as np
import pandas as pd


class BagOfWord:
    def __init__(self, do_lower_case=True):
        self.do_lower_case = do_lower_case
        if os.path.exists('vocab_BOW.json'):
            print("开始加载BOW字典")
            with open('vocab_BOW.json', 'r') as f:
                self.vocab = json.load(f)
            print("加载BOW字典成功, 字典大小：", len(self.vocab))
        else:
            print("开始生成BOW字典")
            self.vocab = self.generate_vocab()
            print("生成BOW字典成功")
        self.vocab_size = len(self.vocab)

    def generate_vocab(self, train_path='../Sentiment Analysis on Movie Reviews/train.tsv'):
        df_train = pd.read_csv(train_path, sep='\t', header=0)
        df_train["Phrase"] = df_train["Phrase"].fillna("")
        vocab = {}
        for sent in df_train['Phrase']:
            if self.do_lower_case:
                sent = sent.lower()
            words = sent.strip().split(" ")
            for word in words:
                if word not in vocab:
                    vocab[word] = {"index": len(vocab), "count": 1}
                else:
                    vocab[word]["count"] += 1
        print("原始字典大小：", len(vocab))
        print("原始字典词频：", analyze_vocab(vocab))
        print("过滤掉词频小于10的词")
        # 保留出现次数不小于10的词，以及index和count信息
        vocab = {k: v for k, v in vocab.items() if v["count"] >= 10}
        # 重新编号
        for idx, (k, v) in enumerate(vocab.items()):
            vocab[k]["index"] = idx
        print("筛选后字典大小：", len(vocab))
        print("筛选后字典词频：", analyze_vocab(vocab))
        with open('vocab_BOW.json', 'w') as f:
            json.dump(vocab, f)
        return vocab

    def generate_feature(self, sent_list):
        vocab_size = len(self.vocab)
        sent_feature = np.zeros((len(sent_list), vocab_size))
        for idx, sent in enumerate(sent_list):
            if self.do_lower_case:
                sent = sent.lower()
            words = sent.strip().split(" ")
            for word in words:
                if word in self.vocab:
                    sent_feature[idx][self.vocab[word]["index"]] += 1
        return sent_feature


class NGram:
    def __init__(self, ngram=None, do_lower_case=True):
        if ngram is None:
            ngram = [1, 2]
        self.ngram = ngram
        self.do_lower_case = do_lower_case
        if os.path.exists('vocab_NGram.json'):
            print("开始加载NGram字典")
            with open('vocab_NGram.json', 'r') as f:
                self.vocab = json.load(f)
            print("加载NGram字典结束, 字典大小：", len(self.vocab))
        else:
            print("开始生成NGram字典")
            self.vocab = self.generate_vocab()
            print("生成NGram字典结束")
        self.vocab_size = len(self.vocab)

    def generate_vocab(self, train_path='../Sentiment Analysis on Movie Reviews/train.tsv'):
        df_train = pd.read_csv(train_path, sep='\t', header=0)
        df_train["Phrase"] = df_train["Phrase"].fillna("")
        vocab = {}
        for sent in df_train['Phrase']:
            if self.do_lower_case:
                sent = sent.lower()
            sent = sent.split(" ")
            for gram in self.ngram:
                for i in range(len(sent) - gram + 1):
                    feature = "_".join(sent[i:i + gram])
                    if feature not in vocab:
                        vocab[feature] = {"index": len(vocab), "count": 1}
                    else:
                        vocab[feature]["count"] += 1
        print("原始字典大小：", len(vocab))
        print("原始字典词频：", analyze_vocab(vocab))
        print("过滤掉词频小于10的词")
        # 保留出现次数不小于10的词，以及index和count信息
        vocab = {k: v for k, v in vocab.items() if v["count"] >= 10}
        # 重新编号
        for idx, (k, v) in enumerate(vocab.items()):
            vocab[k]["index"] = idx
        print("筛选后字典大小：", len(vocab))
        print("筛选后字典词频：", analyze_vocab(vocab))
        with open('vocab_NGram.json', 'w') as f:
            json.dump(vocab, f)
        return vocab

    def generate_feature(self, sent_list):
        vocab_size = len(self.vocab)
        sent_feature = np.zeros((len(sent_list), vocab_size))
        for idx, sent in enumerate(sent_list):
            if self.do_lower_case:
                sent = sent.lower()
            sent = sent.split(" ")
            for gram in self.ngram:
                for i in range(len(sent) - gram + 1):
                    feature = "_".join(sent[i:i + gram])
                    if feature in self.vocab:
                        sent_feature[idx][self.vocab[feature]["index"]] += 1
        return sent_feature


def analyze_vocab(vocab):
    # vocab = sorted(vocab.items(), key=lambda x: x[1]["count"], reverse=True)
    # vocab = sorted(vocab.items(), key=lambda x: x[1]["index"], reverse=False)
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
    ngram = NGram()
    print(bow.generate_feature(['this is a good movie', 'this is a bad movie']))
    print(ngram.generate_feature(['this is a good movie', 'this is a bad movie']))
