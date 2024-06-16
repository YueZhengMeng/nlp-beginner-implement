import random

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from SentenceDataset import SentenceDataset, TokenizeCollator
from TextCNN import TextCNN
from TextRNN import TextRNN
from tokenizers import GloVeTokenizer

seed = 42
batch_size = 32
start_learning_rate = 1e-3
epochs = 10
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
embedding_dim = 50
glove_path = './glove.6B/glove.6B.{}d.txt'.format(embedding_dim)
use_pretrained = True
# model_type = 'cnn' 或 'lstm' 或 'gru'
model_type = 'cnn'
rnn_num_layers = 3
hidden_size = 2 * embedding_dim
num_class = 5
data_size = 150000
dropout_prob = 0.1
img_save_path = './exp/accuracy-%s-ds=%d-bs=%d-lr=%.4f-hs=%d.png' % (
    model_type, data_size, batch_size, start_learning_rate, hidden_size)
model_save_path = './exp/model-%s-ds=%d-bs=%d-lr=%.4f-hs=%d.pt' % (
    model_type, data_size, batch_size, start_learning_rate, hidden_size)
submission_save_path = './exp/submission-%s-ds=%d-bs=%d-lr=%.4f-hs=%d.csv' % (
    model_type, data_size, batch_size, start_learning_rate, hidden_size)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    return seed


# 设置随机种子
seed_everything(seed)

# 实例化tokenizer
glove_tokenizer = GloVeTokenizer(glove_path, embedding_dim, use_pretrained)

# 读取训练集数据
train_df = pd.read_csv('../Sentiment Analysis on Movie Reviews/train.tsv', sep='\t', header=0)
train_df["Phrase"] = train_df["Phrase"].fillna("")
train_dataset = SentenceDataset(train_df[:data_size])
collete_fn = TokenizeCollator(glove_tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collete_fn)
steps_every_epoch = len(train_dataloader)
total_train_steps = steps_every_epoch * int(epochs)

# 划分验证集
val_dataset = SentenceDataset(train_df[data_size:])
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collete_fn)

# 初始化模型
if model_type == 'lstm' or model_type == 'gru':
    model = TextRNN(glove_tokenizer.get_vocab_size(), embedding_dim, model_type, rnn_num_layers, hidden_size,
                    glove_tokenizer.weights, num_class, dropout_prob).to(device)
elif model_type == 'cnn':
    model = TextCNN(glove_tokenizer.get_vocab_size(), embedding_dim, (2, 3, 4), hidden_size, glove_tokenizer.weights,
                    num_class, dropout_prob).to(device)

# 定义优化器
optimizer = AdamW(model.parameters(), lr=start_learning_rate)

# 定义带预热的线性递减学习率调度器
lr_scheduler = OneCycleLR(optimizer, max_lr=start_learning_rate, total_steps=total_train_steps,
                          anneal_strategy='linear', pct_start=0.1, div_factor=10.0, final_div_factor=10.0)

# 定义损失函数
criterion = torch.nn.CrossEntropyLoss()


def train_per_epoch(train_dataloader):
    model.train()
    correct = 0
    total = 0
    for step, (input_ids, last_token_pos, labels) in enumerate(tqdm(train_dataloader)):
        input_ids, last_token_pos, labels = input_ids.to(device), last_token_pos.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, last_token_pos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_accuracy = correct / total
    print("Train Accuracy:", train_accuracy)
    return train_accuracy


def eval_per_epoch(val_dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for step, (input_ids, last_token_pos, labels) in enumerate(tqdm(val_dataloader)):
            input_ids, last_token_pos, labels = input_ids.to(device), last_token_pos.to(device), labels.to(device)
            outputs = model(input_ids, last_token_pos)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracy = correct / total
    print("Val Accuracy:", val_accuracy)
    return val_accuracy


train_accuracy_list = []
val_accuracy_list = []


# 训练模型
def train():
    best_accuracy = 0
    for i in range(epochs):
        print("Epoch:", i)
        train_accuracy = train_per_epoch(train_dataloader)
        val_accuracy = eval_per_epoch(val_dataloader)
        train_accuracy_list.append(train_accuracy)
        val_accuracy_list.append(val_accuracy)
        # 保存验证集上表现最好的模型
        if best_accuracy < val_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            print("best model saved with train accuracy:", best_accuracy)

    plt.plot(train_accuracy_list, label='train')
    plt.plot(val_accuracy_list, label='val')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig(img_save_path)
    plt.show()


def generate_submission():
    test_df = pd.read_csv('../Sentiment Analysis on Movie Reviews/test.tsv', sep='\t', header=0)
    test_df["Phrase"] = test_df["Phrase"].fillna("")
    test_dataset = SentenceDataset(test_df)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                 collate_fn=collete_fn)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    with torch.no_grad():
        results = []
        for step, (input_ids, last_token_pos, _) in enumerate(tqdm(test_dataloader)):
            input_ids, last_token_pos = input_ids.to(device), last_token_pos.to(device)
            outputs = model(input_ids, last_token_pos)
            _, predicted = torch.max(outputs, 1)
            results.extend(predicted.cpu().numpy().tolist())
        submission_df = pd.DataFrame({'PhraseId': test_df['PhraseId'], 'Sentiment': results})
        submission_df.to_csv(submission_save_path, index=False)


if __name__ == '__main__':
    # 开始训练之前先评估一下模型的初始性能
    train_accuracy_before = eval_per_epoch(train_dataloader)
    val_accuracy_before = eval_per_epoch(val_dataloader)
    train_accuracy_list.append(train_accuracy_before)
    val_accuracy_list.append(val_accuracy_before)
    # 开始训练
    train()
    # 评估训练后的模型性能
    model.load_state_dict(torch.load(model_save_path))
    eval_per_epoch(val_dataloader)
    # 生成提交文件
    generate_submission()
