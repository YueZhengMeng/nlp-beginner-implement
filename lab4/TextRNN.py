import torch
import torch.nn as nn


class TextRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, rnn_type, num_layers=3, hidden_size=100, num_classes=9,
                 dropout_prob=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, input_length):
        x = self.embedding(x)
        # 压缩padding部分
        x = nn.utils.rnn.pack_padded_sequence(x, input_length, batch_first=True, enforce_sorted=False)
        # hidden :
        # gru : (D * num_layers, batch_size, hidden_size)
        # lstm : ((D * num_layers, batch_size, hidden_size), (D * num_layers, batch_size, hidden_cell_size))
        x, hidden = self.rnn(x)
        # 还原x的形状
        x, _input_length = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.dropout(x)
        # 映射到实体类别
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = TextRNN(100, 32, 'gru', 3, 64, 9, 0.1)
    x = torch.randint(0, 10, (2, 10))
    input_length = torch.tensor([8, 9])
    print(model(x, input_length))
