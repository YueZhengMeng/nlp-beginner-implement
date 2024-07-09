import torch
from torch.utils.data import Dataset

from read_snli import read_snli


class SnliDataset(Dataset):
    def __init__(self, data_dir, file_name, data_size=None):
        self.premises, self.hypotheses, self.labels = read_snli(data_dir, file_name)
        if data_size is not None:
            self.premises = self.premises[:data_size]
            self.hypotheses = self.hypotheses[:data_size]
            self.labels = self.labels[:data_size]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.premises[idx], self.hypotheses[idx], self.labels[idx]


class TokenizeCollator:
    def __init__(self, tokenizer):
        # 分离为Collator类，目的是为了进行batch_tokenization
        self.tokenizer = tokenizer

    def __call__(self, batch):
        premises, hypotheses, labels = zip(*batch)
        premises_input_ids, premises_sentence_length = self.tokenizer.tokenize(premises)
        hypotheses_input_ids, hypotheses_sentence_length = self.tokenizer.tokenize(hypotheses)
        # zip(*batch)返回的labels是tuple，需要转换为tensor
        labels = torch.tensor(labels)
        return premises_input_ids, premises_sentence_length, hypotheses_input_ids, hypotheses_sentence_length, labels
