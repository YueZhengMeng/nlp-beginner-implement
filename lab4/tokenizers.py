import json
import os

from tqdm import tqdm

from read_conll import read_conll


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
        data_dir = "../conll2003"
        train_file_name = "train.txt"
        train_data = read_conll(data_dir, train_file_name)
        sent_list = train_data[0]
        # 构建词表
        print("开始构建词表")
        vocab = {}
        # 添加[PAD]和[UNK]两个特殊词
        vocab["<PAD>"] = {"index": 0, "count": 99999}
        vocab["<UNK>"] = {"index": 1, "count": 99999}
        for sent in tqdm(sent_list):
            for word in sent:
                if self.do_lower_case:
                    word = word.lower()
                if word not in vocab:
                    vocab[word] = {"index": len(vocab), "count": 1}
                else:
                    vocab[word]["count"] += 1
        print("词表构建完毕")
        print("原始词表大小：", len(vocab))
        print("原始词表词频：", analyze_vocab(vocab))

        # 重新编号
        for idx, (k, v) in enumerate(vocab.items()):
            vocab[k]["index"] = idx
        print("筛选后词表大小：", len(vocab))
        print("筛选后词表词频：", analyze_vocab(vocab))
        with open('vocab_BOW.json', 'w') as f:
            json.dump(vocab, f)
        return vocab

    def tokenize(self, sentence):
        input_ids = []
        unk_index = self.vocab['<UNK>']["index"]
        for i, word in enumerate(sentence):
            if self.do_lower_case:
                word = word.lower()
            if word in self.vocab:
                input_ids.append(self.vocab[word]["index"])
            else:
                input_ids.append(unk_index)
        return input_ids


def analyze_vocab(vocab):
    # 以10为间隔统计词频在0-100之间的词的数量，频率大于100的词统计在最后一个位置
    count_list = [0] * 100
    for k, v in vocab.items():
        if v['count'] // 1 < 100:
            count_list[v['count'] // 1] += 1
        else:
            count_list[-1] += 1
    return count_list


def count_OOV(vocab, data):
    word_count = 0
    oov_count = 0
    for sent in data:
        for word in sent:
            word = word.lower()
            word_count += 1
            if word not in vocab:
                oov_count += 1
    return oov_count, word_count


if __name__ == "__main__":
    bow = BagOfWord()
    print(analyze_vocab(bow.vocab))

    sent_list = [['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'],
                 ['Peter', 'Blackburn'], ['BRUSSELS', '1996-08-22']]

    for sent in sent_list:
        print(bow.tokenize(sent))

    data_dir = "../conll2003"
    train_file_name = "train.txt"
    train_data = read_conll(data_dir, train_file_name)
    print(count_OOV(bow.vocab, train_data[0]))

    val_file_name = "valid.txt"
    val_data = read_conll(data_dir, val_file_name)
    val_oov_count, val_word_count = count_OOV(bow.vocab, val_data[0])
    print(val_oov_count, val_word_count, val_oov_count / val_word_count)

    test_file_name = "test.txt"
    test_data = read_conll(data_dir, test_file_name)
    test_oov_count, test_word_count = count_OOV(bow.vocab, test_data[0])
    print(test_oov_count, test_word_count, test_oov_count / test_word_count)
