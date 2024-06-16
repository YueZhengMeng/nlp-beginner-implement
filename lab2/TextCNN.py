import torch
import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, kernel_sizes=(2, 3, 4),
                 kernel_num=100, glove_weight=None, num_classes=5, dropout_prob=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if glove_weight is not None:
            self.embedding.weight.data.copy_(glove_weight)
            self.embedding.weight.requires_grad = False
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, kernel_num, (k, embedding_size)) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(kernel_num * len(kernel_sizes), num_classes)

    def forward(self, x, last_token_pos):
        # 这里的last_token_pos是为了与TextRNN的forward函数统一接口
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = TextCNN(100, 32, None, 5)
    x = torch.randint(0, 10, (2, 10))
    print(model(x))
