import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_size, rnn_type, num_layers=3, hidden_size=100, dropout_prob=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.rnn_type = rnn_type
        if rnn_type == 'lstm':
            self.encoder = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
            self.decoder = nn.LSTM(embedding_size + hidden_size, hidden_size, num_layers, batch_first=True)
        else:
            self.encoder = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True)
            self.decoder = nn.GRU(embedding_size + hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids, input_length, output_ids, output_length):
        # encoder
        input_x = self.embedding(input_ids)
        input_x = nn.utils.rnn.pack_padded_sequence(input_x, input_length, batch_first=True, enforce_sorted=False)
        _, encoder_hidden = self.encoder(input_x)

        # decoder
        output_x = self.embedding(output_ids)
        if self.rnn_type == 'lstm':
            context = encoder_hidden[0][-1].unsqueeze(1).expand(-1, output_x.size(1), -1)
        else:
            context = encoder_hidden[-1].unsqueeze(1).expand(-1, output_x.size(1), -1)
        # 叠加context信息
        output_x = torch.cat([output_x, context], dim=-1)
        output_x = nn.utils.rnn.pack_padded_sequence(output_x, output_length, batch_first=True, enforce_sorted=False)
        output_x, decoder_hidden = self.decoder(output_x, encoder_hidden)
        output_x, _ = nn.utils.rnn.pad_packed_sequence(output_x, batch_first=True)

        # 输出层
        output_x = self.dropout(output_x)
        output_x = self.fc(output_x)
        return output_x, decoder_hidden


if __name__ == '__main__':
    model = Seq2Seq(100, 32, 'gru', 3, 64, 0.1)
    input_ids = torch.randint(0, 10, (2, 10))
    input_length = torch.tensor([8, 9])
    output_ids = torch.randint(0, 10, (2, 10))
    output_length = torch.tensor([8, 9])
    print(model(input_ids, input_length, output_ids, output_length))
