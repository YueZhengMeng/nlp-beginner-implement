import json
import os

from tqdm import tqdm

from read_poetry import read_poetry


class BagOfWord:
    def __init__(self):
        if os.path.exists('vocab_BOW.json'):
            print("开始加载BOW词表")
            with open('vocab_BOW.json', 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)
            print("加载BOW词表成功, 词表大小：", len(self.vocab))
        else:
            print("开始生成BOW词表")
            self.vocab = self.generate_vocab()
            print("生成BOW词表成功")
        self.vocab_size = len(self.vocab)
        self.decode_vocab = {v["index"]: k for k, v in self.vocab.items()}

    def generate_vocab(self):
        # 读取训练集数据，使用前提与假设的句子生成词表
        sent_list = read_poetry()
        # 构建词表
        print("开始构建词表")
        vocab = {}
        # 添加[PAD]和[UNK]两个特殊词
        vocab["<PAD>"] = {"index": 0, "count": 99999}
        vocab["<UNK>"] = {"index": 1, "count": 99999}
        vocab["<BOS>"] = {"index": 2, "count": 99999}
        vocab["<EOS>"] = {"index": 3, "count": 99999}
        vocab["，"] = {"index": 4, "count": 99999}
        vocab["。"] = {"index": 5, "count": 99999}
        for sent in tqdm(sent_list):
            for word in sent:
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
        with open('vocab_BOW.json', 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False)
        return vocab

    def tokenize(self, sentence):
        input_ids = []
        unk_index = self.vocab['<UNK>']["index"]
        for i, word in enumerate(sentence):
            if word in self.vocab:
                input_ids.append(self.vocab[word]["index"])
            else:
                input_ids.append(unk_index)
        return input_ids

    def decode(self, input_ids):
        return [self.decode_vocab[i] for i in input_ids]


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

    sent_list = ["春江潮水连海平，海上明月共潮生。", "滚滚长江东逝水，浪花淘尽英雄。"]

    for sent in sent_list:
        print(bow.tokenize(sent))

    print(bow.decode([751, 752, 451, 129, 136]))
    print(bow.decode([889, 1627,  140, 1639, 1006, 1698,  778]))

    print(bow.vocab["O"])
