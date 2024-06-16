import torch
import torch.nn as nn


class TextRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, rnn_type, num_layers=3, hidden_size=100, glove_weight=None,
                 num_classes=5, dropout_prob=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if glove_weight is not None:
            self.embedding.weight.data.copy_(glove_weight)
            self.embedding.weight.requires_grad = False
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, last_token_pos):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        # 根据last_token_pos获取最后一个有效token的hidden state
        # 避免输出受到padding的影响
        # 这种实现方法是我拍脑门想出来的，不一定是最优解
        # 最合适的方法应该是调用torch.nn.utils.rnn.pack_padded_sequence和torch.nn.utils.rnn.pad_packed_sequence进行处理
        x = x[torch.arange(x.size(0)), last_token_pos, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = TextRNN(100, 32, 'lstm', 3, 64, None, 5)
    x = torch.randint(0, 10, (2, 10))
    last_token_pos = torch.tensor([5, 3])
    print(model(x, last_token_pos))
