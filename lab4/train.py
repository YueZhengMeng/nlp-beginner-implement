import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ConllDataset import ConllDataset
from TextRNN import TextRNN
from tokenizers import BagOfWord
from torchcrf import CRF

seed = 42
batch_size = 1024
start_learning_rate = 1e-3
epochs = 100
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_data_size = None
val_data_size = None
test_data_size = None
vocab_size = BagOfWord().vocab_size
# rnn_type = 'lstm' 或 'gru'
rnn_type = 'lstm'
loss_type = 'crf+ce'
rnn_num_layers = 3
embedding_size = 50
hidden_size = 2 * embedding_size
num_classes = 9
dropout_prob = 0.1

# 由于crf_loss的数值较大(大约是ce_loss的10倍以上)，所以需要乘以一个较小的系数
crf_loss_weight = 0.0
ce_loss_weight = 1

if crf_loss_weight > 0 and ce_loss_weight > 0:
    loss_type = 'crf+ce'
    start_learning_rate = 5e-4
elif crf_loss_weight > 0:
    loss_type = 'crf'
elif ce_loss_weight > 0:
    loss_type = 'ce'

img_save_path = './exp/f1-%s-%s-bs=%d-lr=%.4f-hs=%d.png' % (
    rnn_type, loss_type, batch_size, start_learning_rate, hidden_size)
model_save_path = './exp/model-%s-%s-bs=%d-lr=%.4f-hs=%d.pth' % (
    rnn_type, loss_type, batch_size, start_learning_rate, hidden_size)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    return seed


# 设置随机种子
seed_everything(seed)

# 读取训练集
data_dir = "../conll2003"
train_file_name = "train.txt"
val_file_name = "valid.txt"
test_file_name = "test.txt"

train_dataset = ConllDataset(data_dir, train_file_name, data_size=train_data_size, device=device)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

steps_every_epoch = len(train_dataloader)
total_train_steps = steps_every_epoch * int(epochs)

# 读取验证集
val_dataset = ConllDataset(data_dir, val_file_name, data_size=val_data_size, device=device)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# 读取测试集
test_dataset = ConllDataset(data_dir, test_file_name, data_size=test_data_size, device=device)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# 初始化模型
model = TextRNN(vocab_size, embedding_size, rnn_type, rnn_num_layers, hidden_size, num_classes, dropout_prob).to(device)

# 定义优化器
optimizer = AdamW(model.parameters(), lr=start_learning_rate)

# 定义带预热的线性递减学习率调度器
lr_scheduler = OneCycleLR(optimizer, max_lr=start_learning_rate, total_steps=total_train_steps,
                          anneal_strategy='linear', pct_start=0.1, div_factor=10.0, final_div_factor=10.0)

# 定义损失函数
criterion = torch.nn.CrossEntropyLoss()
crf = CRF(num_classes, batch_first=True).to(device)


def train_per_epoch(train_dataloader):
    model.train()
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long).to(device)
    for step, (input_ids, input_length, labels) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        outputs = model(input_ids, input_length)
        batch_max_length = outputs.shape[1]
        labels = labels[:, :batch_max_length].contiguous()
        mask = (torch.arange(batch_max_length).unsqueeze(0) < input_length.unsqueeze(1)).to(device)
        # 计算CRF Loss，注意要取负数
        crf_loss = -crf(outputs, labels, mask=mask, reduction='mean')
        # 根据input_length取出有效token
        outputs = outputs.view(-1, num_classes)[mask.view(-1)]
        labels = labels.view(-1)[mask.view(-1)]
        # 计算CrossEntropyLoss
        ce_loss = criterion(outputs, labels)
        # 总的loss，两者加权求和
        loss = crf_loss * crf_loss_weight + ce_loss * ce_loss_weight
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        _, predicted = torch.max(outputs, 1)
        # 二重循环计算混淆矩阵效率很低，会导致GPU等待CPU
        # for t, p in zip(labels, predicted):
        #    confusion_matrix[t.long(), p.long()] += 1
        # 基于向量化的计算混淆矩阵，效率更高
        # for c in range(num_classes):
        #     target_mask = (labels == c)
        #     pred_labels = predicted[target_mask]
        #     for c_pred in range(num_classes):
        #         confusion_matrix[c, c_pred] = (pred_labels == c_pred).sum()
        # 使用散点加法操作填充混淆矩阵
        with torch.no_grad():
            idx = num_classes * labels + predicted
            conf_mat = torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
            confusion_matrix += conf_mat
    (train_macro_precision, train_macro_recall, train_macro_f1, train_micro_precision, train_micro_recall,
     train_micro_f1, train_accuracy) = calculate_metrics(confusion_matrix)
    print("train_macro_precision:", train_macro_precision, "train_macro_recall:", train_macro_recall, "train_macro_f1:",
          train_macro_f1)
    print("train_micro_precision:", train_micro_precision, "train_micro_recall:", train_micro_recall, "train_micro_f1:",
          train_micro_f1)
    print("train_accuracy:", train_accuracy)
    return train_macro_f1, train_micro_f1, train_accuracy


def eval_per_epoch(val_dataloader):
    model.eval()
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long).to(device)
    with torch.no_grad():
        for step, (input_ids, input_length, labels) in enumerate(tqdm(val_dataloader)):
            outputs = model(input_ids, input_length)
            batch_max_length = outputs.shape[1]
            mask = (torch.arange(batch_max_length).unsqueeze(0) < input_length.unsqueeze(1)).view(-1).to(device)
            outputs = outputs.view(-1, num_classes)[mask]
            labels = labels[:, :batch_max_length].contiguous().view(-1)[mask]
            _, predicted = torch.max(outputs, 1)
            # 二重循环计算混淆矩阵效率很低，会导致GPU等待CPU
            # for t, p in zip(labels, predicted):
            #    confusion_matrix[t.long(), p.long()] += 1
            # 基于向量化的计算混淆矩阵，效率更高
            # for c in range(num_classes):
            #     target_mask = (labels == c)
            #     pred_labels = predicted[target_mask]
            #     for c_pred in range(num_classes):
            #         confusion_matrix[c, c_pred] = (pred_labels == c_pred).sum()
            # 使用散点加法操作填充混淆矩阵
            with torch.no_grad():
                idx = num_classes * labels + predicted
                conf_mat = torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
                confusion_matrix += conf_mat
    (val_macro_precision, val_macro_recall, val_macro_f1, val_micro_precision, val_micro_recall, val_micro_f1,
     val_accuracy) = calculate_metrics(confusion_matrix)
    print("val_macro_precision:", val_macro_precision, "val_macro_recall:", val_macro_recall, "val_macro_f1:",
          val_macro_f1)
    print("val_micro_precision:", val_micro_precision, "val_micro_recall:", val_micro_recall, "val_micro_f1:",
          val_micro_f1)
    print("val_accuracy:", val_accuracy)
    return val_macro_f1, val_micro_f1, val_accuracy


def calculate_metrics(confusion_matrix):
    num_classes = confusion_matrix.size(0)
    tp = torch.diag(confusion_matrix)
    fp = confusion_matrix.sum(dim=0) - tp
    fn = confusion_matrix.sum(dim=1) - tp
    tn = confusion_matrix.sum() - (tp + fp + fn)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    # Handle cases where the denominator is zero
    precision[torch.isnan(precision)] = 0
    recall[torch.isnan(recall)] = 0
    f1[torch.isnan(f1)] = 0

    macro_precision = precision.mean().item()
    macro_recall = recall.mean().item()
    macro_f1 = f1.mean().item()

    micro_precision = (tp.sum() / (tp.sum() + fp.sum())).item()
    micro_recall = (tp.sum() / (tp.sum() + fn.sum())).item()
    micro_f1 = (2 * (micro_precision * micro_recall) / (micro_precision + micro_recall))

    accuracy = ((tp.sum() + tn.sum()) / (tp.sum() + tn.sum() + fp.sum() + fn.sum())).item()

    return macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1, accuracy


train_macro_f1_list = []
train_micro_f1_list = []
train_accuracy_list = []
val_macro_f1_list = []
val_micro_f1_list = []
val_accuracy_list = []


# 训练模型
def train():
    best_macro_f1 = 0
    for i in range(epochs):
        print("Epoch:", i)
        train_macro_f1, train_micro_f1,train_accuracy = train_per_epoch(train_dataloader)
        val_macro_f1, val_micro_f1,val_accuracy = eval_per_epoch(val_dataloader)
        train_macro_f1_list.append(train_macro_f1)
        train_micro_f1_list.append(train_micro_f1)
        train_accuracy_list.append(train_accuracy)
        val_macro_f1_list.append(val_macro_f1)
        val_micro_f1_list.append(val_micro_f1)
        val_accuracy_list.append(val_accuracy)
        # 保存验证集上表现最好的模型
        if best_macro_f1 < val_macro_f1:
            best_macro_f1 = val_macro_f1
            torch.save(model.state_dict(), model_save_path)
            print("best model saved with val macro f1 list:", val_macro_f1)

    plt.plot(train_macro_f1_list, label='train macro f1')
    plt.plot(val_macro_f1_list, label='val macro f1')
    plt.plot(train_micro_f1_list, label='train micro f1')
    plt.plot(val_micro_f1_list, label='val micro f1')
    plt.plot(train_accuracy_list, label='train accuracy')
    plt.plot(val_accuracy_list, label='val accuracy')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('f1')
    plt.savefig(img_save_path)
    plt.show()


if __name__ == '__main__':
    # 开始训练之前先评估一下模型的初始性能
    train_macro_f1_before, train_micro_f1_before,train_accuracy_before = eval_per_epoch(train_dataloader)
    val_macro_f1_before, val_micro_f1_before,val_accuracy_before = eval_per_epoch(val_dataloader)
    train_macro_f1_list.append(train_macro_f1_before)
    train_micro_f1_list.append(train_micro_f1_before)
    train_accuracy_list.append(train_accuracy_before)
    val_macro_f1_list.append(val_macro_f1_before)
    val_micro_f1_list.append(val_micro_f1_before)
    val_accuracy_list.append(val_accuracy_before)
    # 开始训练
    train()
    # 评估训练后的模型性能
    model.load_state_dict(torch.load(model_save_path))
    print("验证集上的最佳模型在验证集上的性能")
    eval_per_epoch(val_dataloader)
    print("验证集上的最佳模型在测试集上的性能")
    eval_per_epoch(test_dataloader)
