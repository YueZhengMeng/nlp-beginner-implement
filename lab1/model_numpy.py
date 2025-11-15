import numpy as np


class LinearLayer:
    def __init__(self, input_size, output_size, optimizer='sgd',
                 beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        """
        初始化线性层
        Args:
            input_size (int): 输入维度
            output_size (int): 输出维度
            optimizer (str): 优化器类型, 'sgd' 或 'adamw'
            beta1 (float): AdamW超参数
            beta2 (float): AdamW超参数
            eps (float): AdamW超参数
            weight_decay (float): AdamW超参数
        """
        # 初始化权重矩阵和偏置向量
        self.weights = np.random.normal(loc=0.0, scale=0.1, size=(input_size, output_size))
        self.bias = np.zeros((1, output_size))

        # 存储输入输出用于反向传播
        self.x = None

        # 存储梯度用于反向传播
        self.d_weights = None
        self.d_bias = None

        # -------- 优化器相关 --------
        self.optimizer = optimizer
        if self.optimizer == 'adamw':
            # 初始化 AdamW 的状态变量
            self.v_w = np.zeros_like(self.weights)
            self.s_w = np.zeros_like(self.weights)
            self.v_b = np.zeros_like(self.bias)
            self.s_b = np.zeros_like(self.bias)
            self.t = 0  # 时间步

            # 将超参数保存为实例属性
            self.beta1 = beta1
            self.beta2 = beta2
            self.eps = eps
            self.weight_decay = weight_decay
        # ---------------------------------

    def forward(self, x):
        # 备份输入x，反向传播时会用到
        self.x = x
        # 前向传播计算输出，对应公式(1.1)
        # bias会自动broadcasting到(batch_size, output_size)维
        output = np.matmul(self.x, self.weights) + self.bias
        return output

    def backward(self, d_output):
        # 反向传播计算权重的梯度
        self.d_weights = np.matmul(self.x.T, d_output)
        # 反向传播计算偏置的梯度
        self.d_bias = np.sum(d_output, axis=0, keepdims=True)
        # 反向传播计算输入的梯度
        d_input = np.matmul(d_output, self.weights.T)
        return d_input

    def update(self, learning_rate):
        if self.optimizer == 'sgd':
            # 使用梯度下降法更新权重和偏置
            self.weights -= learning_rate * self.d_weights
            self.bias -= learning_rate * self.d_bias

        elif self.optimizer == 'adamw':
            # 时间步加 1
            self.t += 1

            # --- 步骤 1: 对所有参数应用解耦的权重衰减 ---
            self.weights -= learning_rate * self.weight_decay * self.weights
            self.bias -= learning_rate * self.weight_decay * self.bias

            # --- 步骤 2: 更新所有参数的一阶矩估计和二阶矩估计 ---
            # 更新权重的动量
            self.v_w = self.beta1 * self.v_w + (1 - self.beta1) * self.d_weights
            self.s_w = self.beta2 * self.s_w + (1 - self.beta2) * np.square(self.d_weights)
            # 更新偏置的动量
            self.v_b = self.beta1 * self.v_b + (1 - self.beta1) * self.d_bias
            self.s_b = self.beta2 * self.s_b + (1 - self.beta2) * np.square(self.d_bias)

            # --- 步骤 3: 计算偏差校正因子 (只需计算一次) ---
            bias_correction1 = 1 - self.beta1 ** self.t
            bias_correction2 = 1 - self.beta2 ** self.t

            # --- 步骤 4: 对所有参数执行最终的 Adam 更新 ---
            # 更新权重
            v_w_corr = self.v_w / bias_correction1
            s_w_corr = self.s_w / bias_correction2
            self.weights -= learning_rate * v_w_corr / (np.sqrt(s_w_corr) + self.eps)

            # 更新偏置
            v_b_corr = self.v_b / bias_correction1
            s_b_corr = self.s_b / bias_correction2
            self.bias -= learning_rate * v_b_corr / (np.sqrt(s_b_corr) + self.eps)


class ReLULayer(object):
    def __init__(self):
        self.x = None

    def forward(self, x):
        # 备份输入x，反向传播时会用到
        self.x = x
        # 前向传播计算，对应公式(2.1)
        output = np.maximum(0, x)
        return output

    def backward(self, d_output):
        # 反向传播的计算，对应公式(2.2)
        d_input = d_output
        d_input[self.x < 0] = 0
        return d_input


def softmax(input):
    # 通过softmax函数计算概率
    # 减去输入的最大值，防止指数爆炸
    # 对应公式(3.2)
    input_max = np.max(input, axis=1, keepdims=True)
    input_exp = np.exp(input - input_max)
    # 计算概率
    prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)
    return prob
    # 反向传播的代码，与后面的交叉熵损失函数结合实现


class CrossEntropyLossLayer:
    def __init__(self):
        self.prob = None
        self.label_onehot = None

    def forward(self, output, label):
        # 备份概率值，反向传播时会用到
        self.prob = softmax(output)
        # 将标签转换为one-hot编码并备份
        batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(batch_size), label] = 1.0
        # 计算交叉熵损失，对应公式(4.1)
        # loss = -np.sum(np.log(self.prob) * self.label_onehot) / batch_size
        # 为了防止溢出，使用下面的计算方式，对应公式(4.2)
        output_max = np.max(output, axis=1, keepdims=True)
        log_prob = output - output_max - np.log(np.sum(np.exp(output - output_max), axis=1, keepdims=True))
        loss = -np.sum(log_prob * self.label_onehot) / batch_size
        return loss

    def backward(self):
        # 反向传播计算输入的梯度，对应公式(4.3)
        # 虽然我们在前向计算时使用了优化过的公式，但这些修改不影响梯度的计算
        batch_size = self.prob.shape[0]
        d_input = (self.prob - self.label_onehot) / batch_size
        return d_input


class SentenceClassificationModel:
    def __init__(self, vocab_size, input_size, hidden_size, output_size, optimizer='sgd', **optimizer_kwargs):
        """
        初始化模型
        Args:
            ...
            optimizer (str): 优化器名称
            **optimizer_kwargs: 优化器的特定参数, 如 beta1, weight_decay 等
        """
        # 初始化模型的各个层，并使用 **optimizer_kwargs 传递优化器参数
        self.embedding = LinearLayer(vocab_size, input_size, optimizer=optimizer, **optimizer_kwargs)
        self.linear1 = LinearLayer(input_size, hidden_size, optimizer=optimizer, **optimizer_kwargs)
        self.relu = ReLULayer()
        self.linear2 = LinearLayer(hidden_size, output_size, optimizer=optimizer, **optimizer_kwargs)
        self.loss_layer = CrossEntropyLossLayer()

    def forward(self, x):
        # 前向传播计算
        x = self.embedding.forward(x)
        x = self.linear1.forward(x)
        x = self.relu.forward(x)
        x = self.linear2.forward(x)
        # 直接返回模型的output，不用计算概率
        # 用于防溢出交叉熵损失的计算
        return x

    def compute_loss(self, output, label):
        return self.loss_layer.forward(output, label)

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
    # -------- 使用 AdamW 优化器进行测试 --------
    print("Testing with AdamW optimizer...")

    # 1. 创建模型实例，直接传入 AdamW 的特定参数
    #    使用 **optimizer_kwargs 的方式可以非常灵活地将优化器参数传递给底层的 LinearLayer
    model = SentenceClassificationModel(
        vocab_size=100,
        input_size=50,
        hidden_size=30,
        output_size=5,
        optimizer='adamw',
        weight_decay=0.01  # 可以在这里覆盖默认值
    )

    # 2. 定义学习率
    lr = 1e-3

    # 3. 模拟一次训练迭代
    x = np.random.rand(32, 100)
    label = np.random.randint(0, 5, 32)

    # 前向传播
    output = model.forward(x)
    # 计算损失
    loss = model.compute_loss(output, label)
    print(f'Initial loss with AdamW: {loss:.4f}')

    # 反向传播
    model.backward()

    # 4. 更新参数时传入学习率
    model.update(learning_rate=lr)

    # 再次计算损失，检查是否下降
    output_after_update = model.forward(x)
    loss_after_update = model.compute_loss(output_after_update, label)
    print(f'Loss after one AdamW update: {loss_after_update:.4f}')

    print("\n" + "=" * 30 + "\n")

    # -------- 使用 SGD 优化器进行对比测试 --------
    print("Testing with SGD optimizer...")
    model_sgd = SentenceClassificationModel(
        vocab_size=100,
        input_size=50,
        hidden_size=30,
        output_size=5,
        optimizer='sgd'
        # SGD 不需要额外参数，所以不用传
    )

    lr_sgd = 0.1

    output_sgd = model_sgd.forward(x)
    loss_sgd = model_sgd.compute_loss(output_sgd, label)
    print(f'Initial loss with SGD: {loss_sgd:.4f}')

    model_sgd.backward()
    model_sgd.update(learning_rate=lr_sgd)

    output_sgd_after_update = model_sgd.forward(x)
    loss_sgd_after_update = model_sgd.compute_loss(output_sgd_after_update, label)
    print(f'Loss after one SGD update: {loss_sgd_after_update:.4f}')
