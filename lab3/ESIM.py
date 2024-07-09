import torch
import torch.nn as nn
from torch import tensor
from torch.nn import functional as F


class ESIM(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_layers=3, hidden_size=100, dropout_prob=0.1, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        # 第一次RNN
        self.rnn_encoder_1 = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # 用于将组合增强的向量映射回原始维度。 *2是因为双向LSTM，*4是因为拼接了4个特征
        self.projection = nn.Sequential(nn.Linear(hidden_size * 2 * 4, hidden_size), nn.ReLU())
        # 第二次RNN
        self.rnn_encoder_2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # 分类层。 *2是因为双向LSTM，*4是因为拼接了4个特征
        self.fc = nn.Linear(hidden_size * 2 * 4, num_classes)
        self.dropout = nn.Dropout(dropout_prob)

    def generate_attention_mask(self, premises_sentence_length, hypotheses_sentence_length):
        batch_size = premises_sentence_length.size(0)
        premises_max_len = premises_sentence_length.max()
        hypotheses_max_len = hypotheses_sentence_length.max()
        # 创建一个全1的mask
        mask = torch.ones(batch_size, premises_max_len, hypotheses_max_len)
        # 要保留的部分设置为0
        for i in range(batch_size):
            mask[i, :premises_sentence_length[i], :hypotheses_sentence_length[i]] = 0
        # 将1变为大负数
        mask = mask * torch.finfo(torch.float32).min
        return mask

    def generate_padding_mask(self, sentence_length):
        batch_size = sentence_length.size(0)
        max_len = sentence_length.max()
        # 创建一个全1的mask
        mask = torch.ones(batch_size, max_len)
        # 要保留的部分设置为0
        for i in range(batch_size):
            mask[i, :sentence_length[i]] = 0
        return mask

    def forward(self, premises_input_ids, premises_sentence_length, hypotheses_input_ids, hypotheses_sentence_length):
        # embedding
        x_a = self.embedding(premises_input_ids)
        x_b = self.embedding(hypotheses_input_ids)
        # 压缩padding部分
        x_a = nn.utils.rnn.pack_padded_sequence(x_a, premises_sentence_length, batch_first=True, enforce_sorted=False)
        x_b = nn.utils.rnn.pack_padded_sequence(x_b, hypotheses_sentence_length, batch_first=True, enforce_sorted=False)
        # 第一次RNN
        # hidden :
        # lstm : ((D * num_layers, batch_size, hidden_size), (D * num_layers, batch_size, hidden_cell_size))
        x_a, hidden = self.rnn_encoder_1(x_a)
        x_b, hidden = self.rnn_encoder_1(x_b)
        # 还原x的形状
        x_a, _premises_sentence_length = nn.utils.rnn.pad_packed_sequence(x_a, batch_first=True)
        x_b, _hypotheses_sentence_length = nn.utils.rnn.pad_packed_sequence(x_b, batch_first=True)
        # 计算注意力分数
        e = torch.bmm(x_a, x_b.permute(0, 2, 1))
        # 计算attention mask，将padding部分的attention score设置为负无穷
        # 负无穷在softmax后会变为0，即PAD的token的在加权求和时权重为0
        attention_mask = self.generate_attention_mask(premises_sentence_length, hypotheses_sentence_length)
        e = e + attention_mask.to(e.device)
        # 计算对齐分布
        x_a_align = torch.bmm(F.softmax(e, dim=-1), x_b)
        x_b_align = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), x_a)
        # 聚合 [x_a; x_a_align; x_a - x_a_align; x_a * x_a_align]
        x_a = torch.cat([x_a, x_a_align, x_a - x_a_align, x_a * x_a_align], dim=-1)
        x_b = torch.cat([x_b, x_b_align, x_b - x_b_align, x_b * x_b_align], dim=-1)
        # 映射回原始维度
        x_a = self.projection(x_a)
        x_b = self.projection(x_b)
        # 第二次RNN
        x_a = nn.utils.rnn.pack_padded_sequence(x_a, premises_sentence_length, batch_first=True, enforce_sorted=False)
        x_b = nn.utils.rnn.pack_padded_sequence(x_b, hypotheses_sentence_length, batch_first=True, enforce_sorted=False)
        x_a, hidden = self.rnn_encoder_2(x_a)
        x_b, hidden = self.rnn_encoder_2(x_b)
        x_a, _premises_sentence_length = nn.utils.rnn.pad_packed_sequence(x_a, batch_first=True)
        x_b, _hypotheses_sentence_length = nn.utils.rnn.pad_packed_sequence(x_b, batch_first=True)
        # 池化特征提取
        # 每个句子的padding mask
        mask_a = self.generate_padding_mask(premises_sentence_length).to(x_a.device)
        mask_b = self.generate_padding_mask(hypotheses_sentence_length).to(x_b.device)
        # 平均池化，将padding部分的token的词向量设置为0
        # mask_a中1是被mask的padding。现在将其中0替换为1，1替换为0，然后乘以x_a即可实现padding部分的token的词向量设置为0
        # unsqueeze(-1)后mask_a的形状为(batch_size, max_len, 1)，与x_a的形状(batch_size, max_len, hidden_size)广播后相乘
        # 实际上，nn.utils.rnn.pad_packed_sequence的返回值中，padding部分的token的词向量已经被设置为0了，这里的操作只是为了演示
        # 处理后的x_a再通过permute变换为(batch_size, hidden_size, max_len)的形状，然后再通过avg_pool1d进行平均池化
        # 最后通过squeeze(-1)将维度为1的维度去掉，avg_a的形状为(batch_size, hidden_size)
        avg_a = F.avg_pool1d((x_a * (1 - mask_a).unsqueeze(-1)).permute(0, 2, 1), kernel_size=x_a.size(1)).squeeze(-1)
        avg_b = F.avg_pool1d((x_b * (1 - mask_b).unsqueeze(-1)).permute(0, 2, 1), kernel_size=x_b.size(1)).squeeze(-1)
        # 最大池化
        # mask_a中1是被mask的padding。现在将其中1替换为大负数，然后加到x_a上，这样padding部分的token的词向量就不会参与到max_pool1d中
        max_a = F.max_pool1d((x_a + mask_a.unsqueeze(-1) * torch.finfo(torch.float32).min).permute(0, 2, 1), kernel_size=x_a.size(1)).squeeze(-1)
        max_b = F.max_pool1d((x_b + mask_b.unsqueeze(-1) * torch.finfo(torch.float32).min).permute(0, 2, 1), kernel_size=x_b.size(1)).squeeze(-1)
        # 拼接
        x = torch.cat([avg_a, max_a, avg_b, max_b], dim=-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    hypotheses_input_ids = tensor([[1, 30, 19, 145, 11, 0, 0, 0, 0],
                                   [1, 279, 19, 471, 244, 205, 11, 0, 0],
                                   [1, 198, 191, 6, 86, 23, 196, 11, 0],
                                   [1, 30, 14, 33, 5094, 50, 22, 237, 11]], dtype=torch.int32)
    hypotheses_sentence_length = tensor([5, 7, 8, 9], dtype=torch.int32)

    premises_input_ids = tensor([[1, 30, 22, 2, 1895, 199, 19, 145, 22, 130, 25, 2,
                                  255, 11, 0, 0, 0],
                                 [1, 78, 279, 19, 430, 4, 2, 1083, 118, 1268, 32, 136,
                                  137, 4, 23, 456, 11],
                                 [1, 198, 191, 20, 86, 2, 553, 283, 11, 0, 0, 0,
                                  0, 0, 0, 0, 0],
                                 [1, 30, 14, 33, 5094, 1660, 2, 1826, 340, 22, 776, 16,
                                  2, 9059, 1301, 11, 0]], dtype=torch.int32)
    premises_sentence_length = tensor([14, 17, 9, 16], dtype=torch.int32)

    model = ESIM(12527, 50, 3, 100, 0.1, 3)

    print(model(premises_input_ids, premises_sentence_length, hypotheses_input_ids, hypotheses_sentence_length))
