import torch
from torch.utils.data import Dataset


class SentenceDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['Phrase']
        label = None
        # 这里的判断是为了适配测试集
        if 'Sentiment' in self.df.columns:
            label = self.df.iloc[idx]['Sentiment']
        return text, label


class TokenizeCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        texts, labels = zip(*batch)
        input_ids, last_token_pos = self.tokenizer.tokenize(texts)
        # 这里的判断是为了适配测试集
        if labels[0] is not None:
            labels = torch.tensor(labels)
        return input_ids, last_token_pos, labels
