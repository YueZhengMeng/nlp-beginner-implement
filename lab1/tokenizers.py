import collections
import json
import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm


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

    def generate_vocab(self, train_path='../Sentiment Analysis on Movie Reviews/train.tsv'):
        df_train = pd.read_csv(train_path, sep='\t', header=0)
        df_train["Phrase"] = df_train["Phrase"].fillna("")
        vocab = {}
        for sent in tqdm(df_train['Phrase']):
            if self.do_lower_case:
                sent = sent.lower()
            words = sent.strip().split(" ")
            for word in words:
                if word not in vocab:
                    vocab[word] = {"index": len(vocab), "count": 1}
                else:
                    vocab[word]["count"] += 1
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
        with open('vocab_BOW.json', 'w', encoding='utf-8') as f:
            # indent=4启动格式化写入，否则打开json文件会很卡
            json.dump(vocab, f, indent=4, ensure_ascii=False)
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
            print("开始加载NGram词表")
            with open('vocab_NGram.json', 'r') as f:
                self.vocab = json.load(f)
            print("加载NGram词表结束, 词表大小：", len(self.vocab))
        else:
            print("开始生成NGram词表")
            self.vocab = self.generate_vocab()
            print("生成NGram词表结束")
        self.vocab_size = len(self.vocab)

    def generate_vocab(self, train_path='../Sentiment Analysis on Movie Reviews/train.tsv'):
        df_train = pd.read_csv(train_path, sep='\t', header=0)
        df_train["Phrase"] = df_train["Phrase"].fillna("")
        vocab = {}
        for sent in tqdm(df_train['Phrase']):
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
        with open('vocab_NGram.json', 'w', encoding='utf-8') as f:
            # indent=4启动格式化写入，否则打开json文件会很卡
            json.dump(vocab, f, indent=4, ensure_ascii=False)
        return vocab

    def generate_feature(self, sent_list):
        vocab_size = len(self.vocab)
        sent_feature = np.zeros((len(sent_list), vocab_size))
        for idx, sent in enumerate(sent_list):
            if self.do_lower_case:
                sent = sent.lower()
            sent = sent.split(" ")
            i = 0
            while i < len(sent):
                matched = False
                # Try to match the longest possible n-gram first
                for gram in sorted(self.ngram, reverse=True):
                    if i + gram <= len(sent):
                        feature = "_".join(sent[i:i + gram])
                        if feature in self.vocab:
                            sent_feature[idx][self.vocab[feature]["index"]] += 1
                            i += gram
                            matched = True
                            break
                if not matched:
                    # 直接生成句向量时不需要加入<UNK>
                    # If no n-gram matched, use <UNK>
                    # sent_feature[idx][self.vocab["<UNK>"]["index"]] += 1
                    i += 1
        return sent_feature


def get_max_freq_pair(token_freqs):
    pairs = collections.defaultdict(int)
    for token, freq in token_freqs.items():
        # 将token拆分到字符级
        parts = token.split('#')
        symbols = []
        for i, part in enumerate(parts):
            if part:
                if i == 0:
                    symbols.append(part)
                else:
                    symbols.append('#' + part)
        for i in range(len(symbols) - 1):
            # “pairs”的键是两个连续字符的元组
            # 类似于2-gram
            pairs[symbols[i], symbols[i + 1]] += freq
    if len(pairs) != 0:
        # 返回具有最大freq值的“pairs”键, 以及freq值
        return max(pairs, key=pairs.get), pairs[max(pairs, key=pairs.get)]
    else:
        # 如果已经没有可以合并的符号对，返回None
        return None, None


example = "e#s#c#a#p#a#d#e#s"


def merge_symbols(max_freq_pair, token_freqs, symbols):
    global example
    # 将最频繁的连续符号对合并为一个符号，并添加到词表
    # 合并只消除后缀符号的#，前缀符号保留
    symbols.append(''.join([max_freq_pair[0], max_freq_pair[1].replace("#", "")]))
    new_token_freqs = dict()
    for token, freq in token_freqs.items():

        # 下面这段代码是为了展示某个单词的合并过程，单词定义在全局变量example中。可以注释掉
        if token == example:
            temp = token.replace(''.join(max_freq_pair),
                                 ''.join([max_freq_pair[0], max_freq_pair[1].replace("#", "")]))
            if temp != example:
                print()
                print("merged: ", max_freq_pair[0], max_freq_pair[1], "->", symbols[-1])
                print(example, "->", temp)
                print()
                example = temp
        # 上面这段代码是为了展示某个单词的合并过程，单词定义在全局变量example中。可以注释掉

        # 用最频繁的连续符号对替换所有token中的这对符号
        new_token = token.replace(''.join(max_freq_pair),
                                  ''.join([max_freq_pair[0], max_freq_pair[1].replace("#", "")]))
        # 更新token freq的统计
        new_token_freqs[new_token] = token_freqs[token]
    return new_token_freqs, symbols


def segment_BPE(tokens, vocab):
    outputs = []
    for token in tokens:
        # start与end最初指向token的开头和结尾的字符
        start, end = 0, len(token)
        cur_output = []
        # 具有符号中可能最长子字的词元段
        while start < len(token) and start < end:
            # 找前缀匹配符号
            if token[start: end] in vocab.keys() and len(cur_output) == 0:
                cur_output.append(token[start: end])
                # start指向end，end重新指向token的结尾
                start = end
                end = len(token)
            # 找后缀匹配符号
            elif '#' + token[start: end] in vocab.keys() and len(cur_output) != 0:
                cur_output.append('#' + token[start: end])
                # start指向end，end重新指向token的结尾
                start = end
                end = len(token)
            else:
                if start + 1 == end:
                    # 如果token[start: end]仅为一个字符且不在vocab中
                    # 将<UNK>添加到输出中，并跳过这个字符
                    cur_output.append('<UNK>')
                    start = end
                    end = len(token)
                else:
                    # 否则，end向前移动一个字符，缩短匹配区间
                    end -= 1
        # 如果没有匹配到任何字符，将<UNK>添加到输出中
        if len(cur_output) == 0:
            cur_output.append('<UNK>')
        outputs.append(cur_output)
    return outputs


class BPE:
    def __init__(self, do_lower_case=True, vocab_size=20000):
        self.do_lower_case = do_lower_case
        if os.path.exists('vocab_BPE.json'):
            print("开始加载BPE词表")
            with open('vocab_BPE.json', 'r') as f:
                self.vocab = json.load(f)
                self.vocab_size = len(self.vocab)
            print("加载BPE词表成功, 词表大小：", len(self.vocab))
        else:
            print("开始生成BPE词表")
            print("刚开始生成会显示需要很长时间，后面会越来越快，整体在5分钟左右完成")
            self.vocab_size = vocab_size
            self.vocab = self.generate_vocab()
            print("生成BPE词表成功")

    def generate_vocab(self, train_path='../Sentiment Analysis on Movie Reviews/train.tsv'):
        df_train = pd.read_csv(train_path, sep='\t', header=0)
        df_train["Phrase"] = df_train["Phrase"].fillna("")
        symbols = ['<PAD>', '<UNK>', '<BOS>', '<EOS>', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                   'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '#a', '#b', '#c', '#d', '#e',
                   '#f', '#g', '#h', '#i', '#j', '#k', '#l', '#m', '#n', '#o', '#p', '#q', '#r', '#s', '#t', '#u', '#v',
                   '#w', '#x', '#y', '#z']
        initial_len = len(symbols)
        word_freqs = {}
        for sent in df_train['Phrase']:
            if self.do_lower_case:
                sent = sent.lower()
            words = sent.strip().split(" ")
            for word in words:
                if word not in word_freqs:
                    word_freqs[word] = 1
                else:
                    word_freqs[word] += 1
        token_freqs = {}
        for token, freq in word_freqs.items():
            # token拆分到字符级，然后用插入#分隔
            # 'series' -> 's#e#r#i#e#s'
            token_freqs['#'.join(token)] = word_freqs[token]
        freq_dict = {}
        num_merges = self.vocab_size - len(symbols)
        print("开始BPE合并词表")
        for i in tqdm(range(num_merges)):
            max_freq_pair, max_freq = get_max_freq_pair(token_freqs)
            if not max_freq_pair:
                break
            merge_token = ''.join([max_freq_pair[0], max_freq_pair[1].replace("#", "")])
            freq_dict[merge_token] = max_freq
            token_freqs, symbols = merge_symbols(max_freq_pair, token_freqs, symbols)
        print("BPE合并词表结束")
        # 重新编号
        vocab = {}
        for i, token in enumerate(symbols):
            if i < initial_len:
                vocab[token] = {"index": i, "count": 99999}
            else:
                # 保留出现次数不小于10的词，以及index和count信息
                if freq_dict[token] >= 10:
                    vocab[token] = {"index": i, "count": freq_dict[token]}
        with open('vocab_BPE.json', 'w', encoding='utf-8') as f:
            # indent=4启动格式化写入，否则打开json文件会很卡
            json.dump(vocab, f, indent=4, ensure_ascii=False)
        """
        从软件工程的角度来看，这段代码中的vocab、symbols、freq_dict应该整合在一起
        但我从一个个人学习用项目的角度出发，决定与参考资料《动手学深度学习》中的相关章节的代码保持一致
        """
        return vocab

    def generate_feature(self, sent_list):
        vocab_size = len(self.vocab)
        sent_feature = np.zeros((len(sent_list), vocab_size))
        for idx, sent in enumerate(sent_list):
            if self.do_lower_case:
                sent = sent.lower()
            words = sent.strip().split(" ")
            tokens = segment_BPE(words, self.vocab)
            flatten_tokens = [token for tokens in tokens for token in tokens]
            for token in flatten_tokens:
                sent_feature[idx][self.vocab[token]["index"]] += 1
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
    print("BagOfWord vocab frequency", analyze_vocab(bow.vocab))

    ngram = NGram()
    print("NGram vocab frequency", analyze_vocab(ngram.vocab))

    bpe = BPE()
    print("BPE vocab frequency", analyze_vocab(bpe.vocab))

    time1 = time.time()
    print("BagOfWord result", bow.generate_feature(['this is a good movie', 'this is a bad movie']))
    print("BagOfWord time:", time.time() - time1)

    time2 = time.time()
    print("NGram result", ngram.generate_feature(['this is a good movie', 'this is a bad movie']))
    print("NGram time:", time.time() - time2)

    time3 = time.time()
    print("BPE result", bpe.generate_feature(['this is a good movie', 'this is a bad movie']))
    print("BPE time:", time.time() - time3)

    # 在BPE词表中但不在BOW词表中的词
    diff_vocab_1 = {}
    for k in bpe.vocab.keys():
        if k not in bow.vocab.keys():
            diff_vocab_1[k] = bpe.vocab[k]

    print(len(diff_vocab_1))
    print(analyze_vocab(diff_vocab_1))

    # 在BOW词表中但不在BPE词表中的词
    diff_vocab_2 = {}
    for k in bow.vocab.keys():
        if k not in bpe.vocab.keys():
            diff_vocab_2[k] = bow.vocab[k]

    print(len(diff_vocab_2))
    print(analyze_vocab(diff_vocab_2))

    print(bpe.generate_feature(['considerable', 'considered', 'considering', 'consider',
                                'consider@', '@consider', '@', '']))
