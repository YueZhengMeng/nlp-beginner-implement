import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ESIM import ESIM
from SnliDataset import SnliDataset, BatchPadCollator
from tokenizers import BagOfWord

seed = 42
batch_size = 1024
start_learning_rate = 1e-3
epochs = 10
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_data_size = None
val_data_size = None
test_data_size = None
vocab_size = 12527
rnn_num_layers = 3
embedding_size = 50
hidden_size = 2 * embedding_size
num_class = 3
dropout_prob = 0.1
img_save_path = './exp/accuracy--bs=%d-lr=%.4f-hs=%d.png' % (batch_size, start_learning_rate, hidden_size)
model_save_path = './exp/model-bs=%d-lr=%.4f-hs=%d.pt' % (batch_size, start_learning_rate, hidden_size)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    return seed


# 设置随机种子
seed_everything(seed)

collete_fn = BatchPadCollator()

# 读取训练集
data_dir = "../snli_1.0"
train_file_name = "snli_1.0_train.txt"
dev_file_name = "snli_1.0_dev.txt"
test_file_name = "snli_1.0_test.txt"

train_dataset = SnliDataset(data_dir, train_file_name, data_size=train_data_size)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collete_fn)

steps_every_epoch = len(train_dataloader)
total_train_steps = steps_every_epoch * int(epochs)

# 读取验证集
val_dataset = SnliDataset(data_dir, dev_file_name, data_size=val_data_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collete_fn)

# 读取测试集
test_dataset = SnliDataset(data_dir, test_file_name, data_size=test_data_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collete_fn)

# 初始化模型
model = ESIM(vocab_size, embedding_size, rnn_num_layers, hidden_size, dropout_prob, num_class).to(device)

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
    for step, (premises_input_ids, premises_sentence_length, hypotheses_input_ids, hypotheses_sentence_length,
               labels) in enumerate(tqdm(train_dataloader)):
        # input_length被要求是在cpu上的tensor，这里不需要转移到CUDA
        premises_input_ids, hypotheses_input_ids, labels = premises_input_ids.to(device), hypotheses_input_ids.to(
            device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(premises_input_ids, premises_sentence_length, hypotheses_input_ids, hypotheses_sentence_length)
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
        for step, (premises_input_ids, premises_sentence_length, hypotheses_input_ids, hypotheses_sentence_length,
                   labels) in enumerate(tqdm(val_dataloader)):
            # input_length被要求是在cpu上的tensor，这里不需要转移到CUDA
            premises_input_ids, hypotheses_input_ids, labels = premises_input_ids.to(device), hypotheses_input_ids.to(
                device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(premises_input_ids, premises_sentence_length, hypotheses_input_ids,
                            hypotheses_sentence_length)
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
            print("best model saved with val accuracy:", best_accuracy)

    plt.plot(train_accuracy_list, label='train')
    plt.plot(val_accuracy_list, label='val')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig(img_save_path)
    plt.show()


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
    print("验证集上的最佳模型在验证集上的性能")
    eval_per_epoch(val_dataloader)
    print("验证集上的最佳模型在测试集上的性能")
    eval_per_epoch(test_dataloader)
