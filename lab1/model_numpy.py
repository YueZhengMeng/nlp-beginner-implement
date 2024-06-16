import numpy as np


class LinearLayer:
    def __init__(self, input_size, output_size):
        # 初始化权重矩阵和偏置向量
        self.weights = np.random.normal(loc=0.0, scale=0.1, size=(input_size, output_size))
        self.bias = np.zeros((1, output_size))

        # 存储输入输出用于反向传播
        self.x = None

        # 存储梯度用于反向传播
        self.d_weights = None
        self.d_bias = None

    def forward(self, x):
        # 备份输入x，反向传播时会用到
        self.x = x
        # 前向传播计算 y = xA + b
        # bias会自动broadcasting成(batch_size, output_size)
        output = np.matmul(self.x, self.weights) + self.bias
        return output

    def backward(self, d_output):
        # 反向传播计算权重的梯度
        self.d_weights = np.matmul(self.x.T, d_output)
        # 反向传播计算偏置的梯度
        self.d_bias = np.sum(d_output, axis=0)
        # 反向传播计算输入的梯度
        d_input = np.matmul(d_output, self.weights.T)
        return d_input

    def update(self, learning_rate):
        # 使用梯度下降法更新权重和偏置
        self.weights -= learning_rate * self.d_weights
        self.bias -= learning_rate * self.d_bias


class ReLULayer(object):
    def __init__(self):
        self.x = None

    def forward(self, x):
        # 备份输入x，反向传播时会用到
        self.x = x
        # 前向传播计算 y = max(0, x)
        output = np.maximum(0, x)
        return output

    def backward(self, d_output):
        # 反向传播的计算
        d_input = d_output
        d_input[self.x < 0] = 0
        return d_input


def softmax(input):
    # 通过softmax函数计算概率
    # 减去输入的最大值，防止指数爆炸
    input_max = np.max(input, axis=1, keepdims=True)
    input_exp = np.exp(input - input_max)
    # 计算概率
    prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)
    return prob


class CrossEntropyLossLayer:
    def __init__(self):
        self.prob = None
        self.label_onehot = None

    def forward(self, prob, label):
        # 备份概率值，反向传播时会用到
        self.prob = prob
        # 将标签转换为one-hot编码并备份
        batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(batch_size), label] = 1.0
        # 计算交叉熵损失
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / batch_size
        return loss

    def backward(self):
        # 反向传播计算输入的梯度
        batch_size = self.prob.shape[0]
        d_input = (self.prob - self.label_onehot) / batch_size
        return d_input


class SentenceClassificationModel:
    def __init__(self, vocab_size, input_size, hidden_size, output_size):
        # 初始化模型的各个层
        self.embedding = LinearLayer(vocab_size, input_size)
        self.linear1 = LinearLayer(input_size, hidden_size)
        self.relu = ReLULayer()
        self.linear2 = LinearLayer(hidden_size, output_size)
        self.softmax = softmax
        self.loss_layer = CrossEntropyLossLayer()

    def forward(self, x):
        # 前向传播计算
        x = self.embedding.forward(x)
        x = self.linear1.forward(x)
        x = self.relu.forward(x)
        x = self.linear2.forward(x)
        x = self.softmax(x)
        return x

    def compute_loss(self, prob, label):
        return self.loss_layer.forward(prob, label)

    def backward(self):
        # 反向传播计算
        d_output = self.loss_layer.backward()
        d_output = self.linear2.backward(d_output)
        d_output = self.relu.backward(d_output)
        d_output = self.linear1.backward(d_output)
        d_output = self.embedding.backward(d_output)
        return d_output

    def update(self, learning_rate):
        # 更新模型参数
        self.embedding.update(learning_rate)
        self.linear1.update(learning_rate)
        self.linear2.update(learning_rate)

    def save_model(self, path):
        # 保存模型参数
        np.savez(path, embedding=self.embedding.weights, linear1=self.linear1.weights, linear2=self.linear2.weights)

    def load_model(self, path):
        # 加载模型参数
        data = np.load(path)
        self.embedding.weights = data['embedding']
        self.linear1.weights = data['linear1']
        self.linear2.weights = data['linear2']


if __name__ == '__main__':
    # 测试模型
    model = SentenceClassificationModel(100, 50, 30, 5)
    x = np.random.randint(0, 100, (32, 100))
    label = np.random.randint(0, 5, 32)
    prob = model.forward(x)
    loss = model.compute_loss(prob, label)
    print('loss:', loss)
    model.backward()
    model.update(0.01)
