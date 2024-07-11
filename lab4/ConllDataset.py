import os
import pickle

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from read_conll import read_conll
from tokenizers import BagOfWord


class ConllDataset(Dataset):
    def __init__(self, data_dir, file_name, data_size=None, device=None):
        if os.path.exists(file_name + '_input_ids.pkl') and os.path.exists(file_name + '_labels.pkl'):
            print("开始加载tokenize后的" + file_name + "数据集")
            with open(file_name + '_input_ids.pkl', 'rb') as f:
                self.input_ids_list = pickle.load(f)
            with open(file_name + '_labels.pkl', 'rb') as f:
                self.labels_list = pickle.load(f)
            print("加载tokenize后的" + file_name + "数据集成功")
        else:
            self.sentences, self.labels_list = read_conll(data_dir, file_name)
            self.tokenizer = BagOfWord()
            self.input_ids_list = []
            print("开始tokenize数据集 " + file_name)
            for i in tqdm(range(len(self.sentences))):
                self.input_ids_list.append(self.tokenizer.tokenize(self.sentences[i]))
            print("数据集 " + file_name + " tokenize完毕")
            print("开始保存tokenize后的" + file_name + "数据集")
            with open(file_name + '_input_ids.pkl', 'wb') as f:
                pickle.dump(self.input_ids_list, f)
            with open(file_name + '_labels.pkl', 'wb') as f:
                pickle.dump(self.labels_list, f)
            print("保存tokenize后的" + file_name + "数据集成功")

        # 由于pickle保存与加载tensor很慢，所以前面的保存与加载都是用list

        # 限制数据集大小
        if data_size is not None:
            self.input_ids_list = self.input_ids_list[:data_size]
            self.labels_list = self.labels_list[:data_size]

        # 计算每个句子的长度，以及最大长度
        self.input_length = [len(input_id) for input_id in self.input_ids_list]
        self.input_length_max = max(self.input_length)
        self.input_length = torch.tensor(self.input_length)
        # 将所有input_ids填充到相同长度，填充值为0
        self.input_ids = torch.zeros(len(self.input_ids_list), self.input_length_max, dtype=torch.int)
        # 将所有labels填充到相同长度，填充值为-1
        # 这里数据格式必须是long，如果是int，会导致后面计算交叉熵损失函数时出错
        self.labels = torch.ones(len(self.labels_list), self.input_length_max, dtype=torch.long) * -1

        # 将input_ids_list和labels_list转换为tensor
        for i, input_id in enumerate(self.input_ids_list):
            self.input_ids[i, :len(input_id)] = torch.tensor(input_id)
        for i, label in enumerate(self.labels_list):
            self.labels[i, :len(label)] = torch.tensor(label)

        o_count = 0
        label_count = 0
        pad_count = (self.input_ids == 0).to(torch.int).sum()
        for i in self.labels_list:
            for j in i:
                label_count += 1
                if j == 0:
                    o_count += 1
        print("O占比：", o_count / label_count)
        print("PAD占比：", pad_count / (self.input_ids.shape[0] * self.input_ids.shape[1]))

        # input_length被要求是在cpu上的tensor，这里不需要转移到CUDA
        if device is not None:
            self.input_ids = self.input_ids.to(device)
            self.labels = self.labels.to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.input_length[idx], self.labels[idx]
