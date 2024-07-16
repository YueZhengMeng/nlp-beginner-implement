import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from PoetryDataset import PoetryDataset, BatchPadCollator
from Seq2Seq import Seq2Seq
from read_poetry import read_poetry, read_poetry_sentences
from tokenizers import BagOfWord

seed = 42
batch_size = 32
learning_rate = 1e-3
epochs = 100
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# rnn_type = 'lstm' 或 'gru'
rnn_type = 'lstm'
rnn_num_layers = 3
embedding_size = 256
hidden_size = embedding_size
dropout_prob = 0.1
# 任务类型，'short'或'long'
task = 'long'

img_save_path = './exp/performance-%s-%s-bs=%d-lr=%.4f-hs=%d.png' % (
    rnn_type, task, batch_size, learning_rate, hidden_size)
model_save_path = './exp/model-%s-%s-bs=%d-lr=%.4f-hs=%d-test.pth' % (rnn_type, task, batch_size, learning_rate, hidden_size)

tokenizer = BagOfWord()
vocab_size = tokenizer.vocab_size


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    return seed


# 设置随机种子
seed_everything(seed)

collator = BatchPadCollator()

if task == 'short':
    poetry_list = read_poetry_sentences()
else:
    poetry_list = read_poetry()

train_count = int(len(poetry_list) * 0.8)
train_dataset = PoetryDataset(poetry_list[:train_count], tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collator)

test_dataset = PoetryDataset(poetry_list[train_count:], tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collator)

steps_every_epoch = len(train_dataloader)
total_train_steps = steps_every_epoch * (epochs + 1)

# 初始化模型
model = Seq2Seq(vocab_size, embedding_size, rnn_type, rnn_num_layers, hidden_size, dropout_prob).to(device)

# 定义优化器
optimizer = AdamW(model.parameters(), lr=learning_rate)

# 定义带预热的线性递减学习率调度器
lr_scheduler = OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_train_steps,
                          anneal_strategy='linear', pct_start=0.1, div_factor=10.0, final_div_factor=10.0,
                          cycle_momentum=False)

# 定义损失函数
criterion = torch.nn.CrossEntropyLoss()


def train_per_epoch(train_dataloader):
    model.train()
    total_train_loss = 0
    correct = 0
    total = 0
    for step, (input_ids, input_length, output_ids, output_length, labels) in enumerate(tqdm(train_dataloader)):
        # length被要求是CPU上的tensor，这里不需要to(device)
        input_ids, output_ids, labels = input_ids.to(device), output_ids.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, decoder_hidden = model(input_ids, input_length, output_ids, output_length)
        batch_max_length = outputs.shape[1]
        mask = (torch.arange(batch_max_length).unsqueeze(0) < output_length.unsqueeze(1)).to(device)
        # 根据input_length取出有效token与label
        outputs = outputs.view(-1, vocab_size)[mask.view(-1)]
        labels = labels.view(-1)[mask.view(-1)]
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        loss = criterion(outputs, labels)
        loss.backward()
        total_train_loss += loss.item()
        optimizer.step()
        lr_scheduler.step()
    average_train_loss = total_train_loss / len(train_dataloader)
    train_perplexity = float(torch.exp(torch.tensor(average_train_loss)))
    train_accuracy = correct / total
    print("average loss:", average_train_loss, "perplexity:", train_perplexity, "accuracy:", train_accuracy)
    return average_train_loss, train_perplexity, train_accuracy


def eval_per_epoch(test_dataloader):
    model.eval()
    total_test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for step, (input_ids, input_length, output_ids, output_length, labels) in enumerate(tqdm(test_dataloader)):
            input_ids, output_ids, labels = input_ids.to(device), output_ids.to(device), labels.to(device)
            outputs, decoder_hidden = model(input_ids, input_length, output_ids, output_length)
            batch_max_length = outputs.shape[1]
            mask = (torch.arange(batch_max_length).unsqueeze(0) < output_length.unsqueeze(1)).to(device)
            outputs = outputs.view(-1, vocab_size)[mask.view(-1)]
            labels = labels.view(-1)[mask.view(-1)]
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()
    average_test_loss = total_test_loss / len(test_dataloader)
    test_perplexity = float(torch.exp(torch.tensor(average_test_loss)))
    test_accuracy = correct / total
    print("average loss:", average_test_loss, "perplexity:", test_perplexity, "accuracy:", test_accuracy)
    return average_test_loss, test_perplexity, test_accuracy


average_train_loss_list = []
train_perplexity_list = []
train_accuracy_list = []
average_test_loss_list = []
test_perplexity_list = []
test_accuracy_list = []


# 训练模型
def train():
    # 开始训练之前先评估一下模型的初始性能
    average_train_loss, train_perplexity, train_accuracy = eval_per_epoch(train_dataloader)
    average_train_loss_list.append(average_train_loss)
    train_perplexity_list.append(train_perplexity)
    train_accuracy_list.append(train_accuracy)
    average_test_loss, test_perplexity, test_accuracy = eval_per_epoch(test_dataloader)
    average_test_loss_list.append(average_test_loss)
    test_perplexity_list.append(test_perplexity)
    test_accuracy_list.append(test_accuracy)

    best_test_perplexity = 99999
    for i in range(epochs):
        print("Epoch:", i)
        average_train_loss, train_perplexity, train_accuracy = train_per_epoch(train_dataloader)
        average_train_loss_list.append(average_train_loss)
        train_perplexity_list.append(train_perplexity)
        train_accuracy_list.append(train_accuracy)
        average_test_loss, test_perplexity, test_accuracy = eval_per_epoch(test_dataloader)
        average_test_loss_list.append(average_test_loss)
        test_perplexity_list.append(test_perplexity)
        test_accuracy_list.append(test_accuracy)

        # 保存验证集上表现最好的模型
        if best_test_perplexity > test_perplexity:
            best_test_perplexity = test_perplexity
            torch.save(model.state_dict(), model_save_path)
            print("best model saved with perplexity:", test_perplexity)

    # 保存充分拟合训练集的模型
    torch.save(model.state_dict(), model_save_path.replace('test', 'train'))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(average_train_loss_list, label='average train loss')
    ax1.plot(average_test_loss_list, label='average test loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.legend()
    ax2.plot(train_perplexity_list, label='train perplexity')
    ax2.plot(test_perplexity_list, label='test perplexity')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('perplexity')
    ax2.legend()
    ax3.plot(train_accuracy_list, label='train accuracy')
    ax3.plot(test_accuracy_list, label='test accuracy')
    ax3.set_xlabel('epoch')
    ax3.set_ylabel('accuracy')
    ax3.legend()
    fig.tight_layout()
    plt.savefig(img_save_path)
    plt.show()


def inference(model, tokenizer, input_sentence, max_length=50):
    model.eval()
    if input_sentence[-1] == '，':
        input_sentence = input_sentence[:-1]
    input_ids = torch.tensor(tokenizer.tokenize(input_sentence)).unsqueeze(0)
    input_length = torch.tensor([input_ids.shape[-1]])
    output_ids = torch.tensor([tokenizer.vocab['<BOS>']["index"]]).unsqueeze(0)
    output_length = torch.tensor([1])
    input_ids, output_ids = input_ids.to(device), output_ids.to(device)
    prob_list = []
    # 这里其实可以优化一下，不需要每次都从头开始推理next_token，可以直接使用上一次的decoder_hidden
    # 但是优化写法需要大改模型代码，这里就不改了
    with torch.no_grad():
        for i in range(max_length):
            outputs, decoder_hidden = model(input_ids, input_length, output_ids, output_length)
            outputs = outputs.squeeze(0)
            prob, next_token = torch.max(torch.softmax(outputs[-1], dim=0), dim=0)
            prob_list.append(prob.item())
            if next_token == tokenizer.vocab['<EOS>']['index']:
                break
            output_ids = torch.cat([output_ids, torch.tensor([[next_token]]).to(device)], dim=1)
            output_length += 1
    output_sentence = tokenizer.decode(output_ids.squeeze(0).cpu().tolist())
    # 计算困惑度
    probs = torch.tensor(prob_list)
    perplexity = torch.exp(-torch.mean(torch.log(probs)))
    return output_sentence, perplexity.item()


if __name__ == '__main__':
    # 开始训练
    train()

    # 加载充分拟合训练集的模型
    model.load_state_dict(torch.load(model_save_path.replace('test', 'train')))

    # 生成诗句，评估过拟合训练集的模型的性能
    input_sentence = "赤城映朝日"
    output_sentence, perplexity = inference(model, tokenizer, input_sentence)
    print("input:", input_sentence)
    print("output:", output_sentence)
    print("perplexity:", perplexity)

    input_sentence = "遥想紫泥封诏罢"
    output_sentence, perplexity = inference(model, tokenizer, input_sentence)
    print("input:", input_sentence)
    print("output:", output_sentence)
    print("perplexity:", perplexity)

    input_sentence = "细软青丝履"
    output_sentence, perplexity = inference(model, tokenizer, input_sentence)
    print("input:", input_sentence)
    print("output:", output_sentence)
    print("perplexity:", perplexity)

    input_sentence = "少陵野老吞声哭"
    output_sentence, perplexity = inference(model, tokenizer, input_sentence)
    print("input:", input_sentence)
    print("output:", output_sentence)
    print("perplexity:", perplexity)

    # 评估测试集上最佳模型性能
    model.load_state_dict(torch.load(model_save_path))

    # 生成诗句，评估测试集上最佳模型性能
    input_sentence = "赤城映朝日"
    output_sentence, perplexity = inference(model, tokenizer, input_sentence)
    print("input:", input_sentence)
    print("output:", output_sentence)
    print("perplexity:", perplexity)

    input_sentence = "遥想紫泥封诏罢"
    output_sentence, perplexity = inference(model, tokenizer, input_sentence)
    print("input:", input_sentence)
    print("output:", output_sentence)
    print("perplexity:", perplexity)

    input_sentence = "细软青丝履"
    output_sentence, perplexity = inference(model, tokenizer, input_sentence)
    print("input:", input_sentence)
    print("output:", output_sentence)
    print("perplexity:", perplexity)

    input_sentence = "少陵野老吞声哭"
    output_sentence, perplexity = inference(model, tokenizer, input_sentence)
    print("input:", input_sentence)
    print("output:", output_sentence)
    print("perplexity:", perplexity)
