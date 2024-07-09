import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from SentenceDataloader import SentenceDataloader
from model_numpy import SentenceClassificationModel, softmax
from tokenizers import BagOfWord, NGram, BPE

seed = 42
batch_size = 32
# 手搓模型的梯度值很小，所以学习率设置的比较大
# 整个训练过程中学习率从0.1线性下降到0.01
start_learning_rate = 0.1
end_learning_rate = 0.01
epochs = 10

input_size = 128
hidden_size = 4 * input_size
num_class = 5
data_size = 150000
# tokenizer_type = 'bow' 或 'ngram' 或 ‘bpe’
tokenizer_type = 'bpe'
img_save_path = './exp/accuracy-%s-ds=%d-bs=%d-lr=%.2f-%.2f-hs=%d.png' % (
    tokenizer_type, data_size, batch_size, start_learning_rate, end_learning_rate, hidden_size)
model_save_path = './exp/model-%s-ds=%d-bs=%d-lr=%.2f-%.2f-hs=%d.npz' % (
    tokenizer_type, data_size, batch_size, start_learning_rate, end_learning_rate, hidden_size)
submission_save_path = './exp/submission-%s-ds=%d-bs=%d-lr=%.2f-%.2f-hs=%d.csv' % (
    tokenizer_type, data_size, batch_size, start_learning_rate, end_learning_rate, hidden_size)

# 设置随机种子
np.random.seed(seed)
random.seed(seed)

# 实例化tokenizer
if tokenizer_type == 'bow':
    tokenizer = BagOfWord()
elif tokenizer_type == 'ngram':
    tokenizer = NGram()
else:
    tokenizer = BPE()

# 读取训练集数据
train_df = pd.read_csv('../Sentiment Analysis on Movie Reviews/train.tsv', sep='\t', header=0)
train_df["Phrase"] = train_df["Phrase"].fillna("")
train_dataloader = SentenceDataloader(train_df, tokenizer, batch_size, data_size)
print("train data size:", train_dataloader.data_size)
total_train_step = epochs * train_dataloader.train_step

# 划分验证集
val_df = train_df[data_size:]
val_dataloader = SentenceDataloader(val_df, tokenizer, batch_size)
print("val data size:", val_dataloader.data_size)

# 初始化模型
model = SentenceClassificationModel(tokenizer.vocab_size, input_size, hidden_size, num_class)


def train_per_epoch(train_dataloader):
    correct = 0
    for step in tqdm(range(train_dataloader.train_step)):
        x, label = train_dataloader.get_batch()
        output = model.forward(x)
        prob = softmax(output)
        loss = model.compute_loss(output, label)
        pred = np.argmax(prob, axis=1)
        correct += np.sum(pred == label)
        model.backward()
        learning_rate = start_learning_rate - (start_learning_rate - end_learning_rate) * step / total_train_step
        model.update(learning_rate)
    train_accuracy = correct / train_dataloader.data_size
    print("Train Accuracy:", train_accuracy)
    return train_accuracy


def eval_per_epoch(val_dataloader):
    correct = 0
    for step in tqdm(range(val_dataloader.train_step)):
        x, label = val_dataloader.get_batch()
        output = model.forward(x)
        prob = softmax(output)
        pred = np.argmax(prob, axis=1)
        correct += np.sum(pred == label)
    val_accuracy = correct / val_dataloader.data_size
    print("val Accuracy:", val_accuracy)
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

        if best_accuracy < val_accuracy:
            best_accuracy = val_accuracy
            model.save_model(model_save_path)
            print("best model saved with val accuracy:", best_accuracy)

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
    test_df.insert(3, 'Sentiment', 0)
    test_dataloader = SentenceDataloader(test_df, tokenizer, 1)
    model.load_model(model_save_path)
    pred_list = []
    for step in tqdm(range(test_dataloader.train_step)):
        x, _ = test_dataloader.get_batch()
        output = model.forward(x)
        prob = softmax(output)
        pred = np.argmax(prob, axis=1)
        pred_list.extend(pred)
    test_df['Sentiment'] = pred_list
    test_df = test_df[['PhraseId', 'Sentiment']]
    test_df.to_csv(submission_save_path, index=False)


if __name__ == '__main__':
    # 开始训练之前先评估一下模型的初始性能
    train_accuracy_before = eval_per_epoch(train_dataloader)
    val_accuracy_before = eval_per_epoch(val_dataloader)
    train_accuracy_list.append(train_accuracy_before)
    val_accuracy_list.append(val_accuracy_before)
    # 开始训练
    train()
    # 评估训练后的模型性能
    model.load_model(model_save_path)
    eval_per_epoch(val_dataloader)
    # 生成提交文件
    generate_submission()
