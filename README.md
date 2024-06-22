# nlp-beginner-implement
经典NLP入门实验项目 https://github.com/FudanNLP/nlp-beginner 实现

# 任务一：基于机器学习的文本分类

实现基于logistic/softmax regression的文本分类

1. 参考
   1. [文本分类](文本分类.md)
   2. 《[神经网络与深度学习](https://nndl.github.io/)》 第2/3章
2. 数据集：[Classify the sentiment of sentences from the Rotten Tomatoes dataset](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)
3. 实现要求：NumPy
4. 需要了解的知识点：
   1. 文本特征表示：Bag-of-Word，N-gram
   2. 分类器：logistic/softmax  regression，损失函数、（随机）梯度下降、特征选择
   3. 数据集：训练集/验证集/测试集的划分
5. 实验：
   1. 分析不同的特征、损失函数、学习率对最终分类性能的影响
   2. shuffle 、batch、mini-batch 
6. 时间：两周

## 实现方法与结果
1. 根据数学公式，基于numpy手搓线性层、ReLU、Softmax、交叉熵等模块，实现前向计算、反向传播、参数更新等过程，搭建一个简单的神经网络  
    详细公式推导与代码解析见 ./lab1/math_formula.ipynb，模型代码见 ./lab1/model_numpy.py  
    参考资料：  
    https://zhuanlan.zhihu.com/p/380036598   
    https://zh.d2l.ai/chapter_multilayer-perceptrons/backprop.html  
    https://zh.d2l.ai/chapter_linear-networks/softmax-regression-concise.html  
    https://www.cnblogs.com/gczr/p/16345902.html   
    https://blog.csdn.net/chaipp0607/article/details/101946040  
2. 手搓BOW与NGram两种tokenizer，在训练集上训练词表  
    为了降低手搓模型的复杂度与工程量，我选择将句子tokenize为Multi-hot向量，经过线性层映射后直接得到句向量，而不是使用词向量。代码见 ./lab1/tokenizers.py
3. 拟合实验：
    使用训练集的前3000条数据训练模型，使用训练集的3000-3999条数据验证模型，进行100个epoch的训练。  
    基于BOW与NGram的模型在训练集上最终都可以达到97%以上的准确率，在验证集上只需要1个epoch就可以达到56%左右的准确率，之后不会继续上升，10个epoch之后还会下降。  
    实验结果见 ./lab1/exp 文件夹下文件名包含ds=3000的图片和csv文件。使用ngram的模型参数量太大，不便上传到github。
4. 正式训练：
    使用训练集的前150000条数据训练模型，使用训练集的后6060条数据验证模型。基于拟合实验的结论，只进行10个epoch的训练。保存验证集精度最高的模型，对测试集进行预测。
    实验结果见 ./lab1/exp 文件夹下文件名包含ds=150000的图片和csv文件。使用ngram的模型参数量太大，不便上传到github。  
5. 上传kaggle：
    将测试集的预测结果上传到kaggle评测。  
    基于BOW的模型精度为59.613%，基于NGram的模型精度为60.108%  
    截图见 ./lab1/exp/kaggle_result_lab1.png

## Tips
1. 手搓模型的梯度值很小，要设置较大的学习率，但也不能太大，否则会发生溢出。实测0.1左右的学习率比较合适。
2. 增加batch_size可以加快训练速度，但是精度会下降。实测batch_size=32比较合适。
3. 现存的实验数据是在"lab1加入了防溢出CrossEntropyLoss"这个commit之前完成的，使用最新的代码进行复现可能会有一些不同。  

# 任务二：基于深度学习的文本分类

熟悉Pytorch，用Pytorch重写《任务一》，实现CNN、RNN的文本分类；

1. 参考
   1. https://pytorch.org/
   2. Convolutional Neural Networks for Sentence Classification <https://arxiv.org/abs/1408.5882>
   3. <https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/>
2. word embedding 的方式初始化
   1. 随机embedding的初始化方式
   2. 用glove 预训练的embedding进行初始化 https://nlp.stanford.edu/projects/glove/
3. 知识点：
   1. CNN/RNN的特征抽取
   2. 词嵌入
   3. Dropout
4. 时间：两周

## 实现方法与结果
1. 下载glove.6B预训练词向量，手搓代码进行解析，获取vocab与词向量矩阵，并保存到文件  
    由于glove.6B的vocab太大(400K)，embedding层计算量相当可观，我选择仅使用最小的50维词向量。  
    代码见 ./lab2/tokenizers.py  
2. 搭建TextRNN模型，使用3层BiLSTM或BiGRU，对句子进行特征抽取  
    代码见 ./lab2/TextRNN.py  
3. 搭建TextCNN模型，使用大小为2,3,4的卷积核，每个大小的卷积核各100个，以及maxpooling层，对句子进行特征抽取  
    代码见 ./lab2/TextCNN.py  
4. 拟合实验：  
    三款模型(lstm, gru, cnn)，两种embedding初始化方式(random, glove)。  
    1. 在使用glove初始化的情况下，embedding层不参与训练。使用训练集的前3000条数据训练模型，使用训练集的3000-3999条数据验证模型，进行100个epoch的训练。  
        三款模型在训练集上最终都可以达到98%以上的准确率，在验证集上都只需要1个epoch就可以达到56%左右的准确率。基于RNN的模型的验证集精度之后不会继续上升。基于CNN的模型的验证集精度在20个epoch时达到最高，之后不会继续上升。  
    2. 在随机初始化embedding层的情况下。使用训练集的前2999条数据训练模型，使用训练集的2999-3998条数据验证模型，进行100个epoch的训练。  
        三款模型在训练集上最终都可以达到97%以上的准确率，在验证集上都只需要1个epoch就可以达到56%左右的准确率，之后会下降。  
    实验结果见 ./lab2/exp 文件夹下文件名包含ds=3000或ds=2999的图片和csv文件。由于embedding层参数量太大，模型权重不便上传到github。  
5. 正式训练：  
    三款模型(lstm, gru, cnn)，两种embedding初始化方式(random, glove)。  
    1. 在使用glove初始化的情况下，embedding层不参与训练。使用训练集的前150000条数据训练模型，使用训练集的后6060条数据验证模型。TextCNN模型训练20个epoch，TextRNN模型训练10个epoch。保存验证集精度最高的模型，对测试集进行预测。  
    2. 在随机初始化embedding层的情况下，使用训练集的前149999条数据训练模型，使用训练集的后6061条数据验证模型，进行10个epoch的训练。保存验证集精度最高的模型，对测试集进行预测。  
    实验结果见 ./lab2/exp 文件夹下文件名包含ds=150000或ds=149999的图片和csv文件。由于embedding层参数量太大，模型权重不便上传到github。  
6. 上传kaggle：
    将测试集的预测结果上传到kaggle评测。  
    评测结果：  
    cnn  + random ： 62.275  
    cnn  + glove  ： 61.533  
    gru  + random ： 61.745  
    gru  + glove  ： 63.169  
    lstm + random ： 62.298  
    lstm + glove  ： 63.579  
    截图见 ./lab2/exp/kaggle_result_lab2.png


## Tips
1. 我实现的tokenizer在会返回根据last_token_pos，给出最后一个有效token的位置。在TextRNN中的rnn前向计算结束后，取出对应的hidden state，避免输出受到padding的影响。  
    这种实现方法是我拍脑门想出来的，不一定是最优解。最合适的方法应该是调用torch.nn.utils.rnn.pack_padded_sequence和torch.nn.utils.rnn.pad_packed_sequence进行处理
2. 即使Glove词典规模已经很大(400K), 但是在实际应用中，未登录词依然较多。
