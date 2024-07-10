import os
import pickle

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from read_snli import read_snli
from tokenizers import BagOfWord


class SnliDataset(Dataset):
    def __init__(self, data_dir, file_name, data_size=None):
        if os.path.exists(file_name + '_premises_input_ids.pkl') and os.path.exists(
                file_name + '_hypotheses_input_ids.pkl') and os.path.exists(file_name + '_labels.pkl'):
            print("开始加载tokenize后的" + file_name + "数据集")
            with open(file_name + '_premises_input_ids.pkl', 'rb') as f:
                self.premises_input_ids = pickle.load(f)
            with open(file_name + '_hypotheses_input_ids.pkl', 'rb') as f:
                self.hypotheses_input_ids = pickle.load(f)
            with open(file_name + '_labels.pkl', 'rb') as f:
                self.labels = pickle.load(f)
            print("加载tokenize后的" + file_name + "数据集成功")
        else:
            self.premises, self.hypotheses, self.labels = read_snli(data_dir, file_name)
            self.tokenizer = BagOfWord()
            self.premises_input_ids = []
            self.hypotheses_input_ids = []
            print("开始tokenize数据集 " + file_name)
            for i in tqdm(range(len(self.premises))):
                self.premises_input_ids.append(self.tokenizer.tokenize(self.premises[i]))
                self.hypotheses_input_ids.append(self.tokenizer.tokenize(self.hypotheses[i]))
            print("数据集 " + file_name + " tokenize完毕")
            print("开始保存tokenize后的" + file_name + "数据集")
            with open(file_name + '_premises_input_ids.pkl', 'wb') as f:
                pickle.dump(self.premises_input_ids, f)
            with open(file_name + '_hypotheses_input_ids.pkl', 'wb') as f:
                pickle.dump(self.hypotheses_input_ids, f)
            with open(file_name + '_labels.pkl', 'wb') as f:
                pickle.dump(self.labels, f)
            print("保存tokenize后的" + file_name + "数据集成功")

        if data_size is not None:
            self.premises_input_ids = self.premises_input_ids[:data_size]
            self.hypotheses_input_ids = self.hypotheses_input_ids[:data_size]
            self.labels = self.labels[:data_size]

        # 由于pickle保存与加载tensor很慢，所以前面的保存与加载都是用list，这里转换为tensor
        self.premises_input_ids = [torch.tensor(premise_input_ids) for premise_input_ids in self.premises_input_ids]
        self.hypotheses_input_ids = [torch.tensor(hypothesis_input_ids) for hypothesis_input_ids in
                                     self.hypotheses_input_ids]
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.premises_input_ids[idx], self.hypotheses_input_ids[idx], self.labels[idx]


class BatchPadCollator:
    def __call__(self, batch):
        premises_input_ids_list, hypotheses_input_ids_list, labels = zip(*batch)
        premises_sentence_length = torch.tensor([len(premise) for premise in premises_input_ids_list])
        hypotheses_sentence_length = torch.tensor([len(hypothesis) for hypothesis in hypotheses_input_ids_list])
        premises_sentence_length_max = int(premises_sentence_length.max())
        hypotheses_sentence_length_max = int(hypotheses_sentence_length.max())
        premises_input_ids = torch.zeros(len(premises_input_ids_list), premises_sentence_length_max, dtype=torch.int)
        hypotheses_input_ids = torch.zeros(len(hypotheses_input_ids_list), hypotheses_sentence_length_max,
                                           dtype=torch.int)
        for i, premise_input_ids in enumerate(premises_input_ids_list):
            premises_input_ids[i, :len(premise_input_ids)] = premise_input_ids.clone().detach()
        for i, hypothesis_input_ids in enumerate(hypotheses_input_ids_list):
            hypotheses_input_ids[i, :len(hypothesis_input_ids)] = hypothesis_input_ids.clone().detach()
        # zip(*batch)返回的labels是tuple，需要转换为tensor
        labels = torch.tensor(labels)
        return premises_input_ids, premises_sentence_length, hypotheses_input_ids, hypotheses_sentence_length, labels
