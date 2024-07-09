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
    为了降低手搓模型的复杂度与工程量，我选择将句子tokenize为multi-hot向量，经过线性层映射后直接得到句向量，而不是使用词向量。没有使用PAD与UNK。代码见 ./lab1/tokenizers.py
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
2. 增加batch_size并等比例增加学习率可以加快训练速度，但是精度会下降。实测batch_size=32比较合适。
3. 以上实验数据是在"lab1加入了防溢出CrossEntropyLoss"这个commit之前完成的，之后重复实验结果提交kaggle评测精度完全一致(小数点后三位)。  

## 更新
&emsp;参考 https://zh.d2l.ai/chapter_natural-language-processing-pretraining/subword-embedding.html 实现了BPE分词  
&emsp;并且实现了类似BERT的WordPiece的前缀后缀分词。参考资料中的代码仅基于字母分词。代码见 ./lab1/tokenizers.py  
&emsp;相比于之前实验的BOW与NGram，词表加入了PAD与UNK，以及26个字母。由于是tokenizer到multi-hot向量，PAD并没有用上。  
&emsp;原计划生成一个大小为20000的词表，但是在生成过程中，合并次数达到11882时，学习到的的合并规则，已经将训练集的所有被拆分为字符级的词汇，重新合并为完整的词，因此停止继续生成。  
&emsp;在此基础上去掉低频词，生成了一个大小为9069的词表。相比于BOW(11452)与NGram(39708)的词表，要更小。  
&emsp;拟合实验确认训练集上精度可以达到99%以上，验证集上精度在epoch 3达到56%左右，之后下降。
&emsp;正式训练实验方法同上，结果提交kaggle评测精度为59.791%。截图见 ./lab1/exp/kaggle_result_lab1_BPE.png  
&emsp;与BOW的词表相比，有2325个词在BPE词表中但不在BOW词表中，这些词主要是前缀后缀子词，符合BPE的分词规则。  
&emsp;有4708个词在BOW词表中但不在BPE词表中，主要是完整的词。最初我对此感到困惑，直觉上认为在BPE的训练过程中，如果某个词被完整复原，那么应该有一条合并规则是将这个词的前半部分与后半部分合并，合并的结果，也就是这个词，会被加入到词表中。之前的训练过程中，训练集的所有的词都被完整复原了，那么这些词应该都会被加入到BPE词表中。换句话说，BOW词表应该是BPE词表的一个子集才对。  
&emsp;通过分析BPE的词表生成过程，以escapades这个单词的分词与合并过程为例：  
&emsp;&emsp;merged:  #e #s -> #es &emsp;&emsp;&emsp;       e#s#c#a#p#a#d#e#s -> e#s#c#a#p#a#d#es  
&emsp;&emsp;merged:  #a #d -> #ad &emsp;&emsp;&emsp;       e#s#c#a#p#a#d#es -> e#s#c#a#p#ad#es  
&emsp;&emsp;merged:  d #e -> de  &emsp;&emsp;&emsp;&emsp;  e#s#c#a#p#ad#es -> e#s#c#a#p#ades  
&emsp;&emsp;merged:  #a #p -> #ap &emsp;&emsp;&emsp;       e#s#c#a#p#ades -> e#s#c#ap#ades  
&emsp;&emsp;merged:  s #c -> sc &emsp;&emsp;&emsp;&emsp;        e#s#c#ap#ades -> e#sc#ap#ades  
&emsp;&emsp;merged:  c #ap -> cap &emsp;&emsp;&emsp;                  e#sc#ap#ades -> e#scap#ades  
&emsp;&emsp;merged:  p #a -> pa &emsp;&emsp;&emsp;&emsp;              e#scap#ades -> e#scapades  
&emsp;&emsp;merged:  e #scap -> escap &emsp;                          e#scapades -> escapades  
&emsp;发现每次合并的两个子词可以在原单词的任意位置，最后一步也不一定正好是合并单词的前半部分与后半部分。因此，BPE的词表中不一定会包含每一个完整的词。  
&emsp;至此疑惑得到解答。

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
    由于glove.6B的词表很大(400K)，embedding层参数量相当可观，我选择仅使用最小的50维词向量。  
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
1. 我实现的tokenizer会返回last_token_pos，给出最后一个有效token的位置。在TextRNN中的rnn前向计算结束后，取出对应的hidden state，避免输出受到padding的影响。  
    这种实现方法是我拍脑门想出来的，不一定是最优解。最常用的方法应该是调用torch.nn.utils.rnn.pack_padded_sequence和torch.nn.utils.rnn.pad_packed_sequence进行处理。  
    查阅词表发现，由于没有手动额外加入PAD与UNK，使得 'the' 的id是0，padding用的也是0，我并不能确定这种重叠在数学上是否有影响。以后有时间会重新实验。  
    至于TextCNN中如何处理padding，我暂时没有找到参考资料。  
2. 即使Glove词表规模已经很大(400K), 但是在实际应用中，未登录词依然较多。  

## 更新
改进tokenizer与TextRNN模型的实现，手动在glove词表中加入PAD与UNK，并初始化为全0向量。  
接入torch.nn.utils.rnn.pack_padded_sequence和torch.nn.utils.rnn.pad_packed_sequence，用于处理padding。  
在TextCNN中将PAD embedding为全0词向量，使其几乎不影响卷积和maxpooling结果。  
此外，之前实验因疏忽导致验证集dataloader的shuffle参数错误设置为True，本组实验中已修正。由于之前的验证都是基于完整的训练集进行的，因此这个错误影响不大。  
重新实验结果上传到kaggle评测，评测结果：  
cnn  + random ： 62.022  
cnn  + glove  ： 61.859  
gru  + random ： 61.859  
gru  + glove  ： 63.399  
lstm + random ： 62.497  
lstm + glove  ： 63.193  
截图见 ./lab2/exp/kaggle_result_lab2_update.png  
注意：重新试验后 ./lab2/exp 文件夹下的图片和csv文件已被更新覆盖。  

# 任务三：基于注意力机制的文本匹配

输入两个句子判断，判断它们之间的关系。参考[ESIM]( https://arxiv.org/pdf/1609.06038v3.pdf)（可以只用LSTM，忽略Tree-LSTM），用双向的注意力机制实现。

1. 参考
   1. 《[神经网络与深度学习](https://nndl.github.io/)》 第8章
   2. Reasoning about Entailment with Neural Attention <https://arxiv.org/pdf/1509.06664v1.pdf>
   3. Enhanced LSTM for Natural Language Inference <https://arxiv.org/pdf/1609.06038v3.pdf>
2. 数据集：https://nlp.stanford.edu/projects/snli/
3. 实现要求：Pytorch
4. 知识点：
   1. 注意力机制
   2. token2token attention
5. 时间：两周

## 实现方法与结果
1. 下载snli数据集，参考 https://zh.d2l.ai/chapter_natural-language-processing-applications/natural-language-inference-and-dataset.html 实现数据读取函数  
    代码见 ./lab3/read_snli.py  &emsp;&emsp;  解压后数据集文件太大，不便上传到github。
2. 为了使实现尽可能简单，使用BagOfWord tokenizer，基于训练集的premises与hypotheses的句子生成词表。  
   最终词表包括PAD与UNK在内，大小为12527  
    代码见 ./lab3/tokenizers.py
3. 搭建ESIM模型，参考资料：  
    https://zh.d2l.ai/chapter_natural-language-processing-applications/natural-language-inference-attention.html
    https://zhuanlan.zhihu.com/p/509287055  
    https://zhuanlan.zhihu.com/p/647038302  
    代码见 ./lab3/ESIM.py
4. 拟合实验：  
    使用训练集的前3000条数据训练模型，使用验证集的前1000条数据验证模型，进行50个epoch的训练。  
    batch_size=32，每个epoch迭代93步，使用Adam优化器，学习率0.001，损失函数为交叉熵。
    模型在训练集上最终可以达到99%以上的准确率，在验证集上的精度在15个epoch时达到最高，迭代约1395步，之后不会继续上升。  
    实验结果见 ./lab3/exp 文件夹下文件名包含bs=32的文件。  
5. 正式训练：  
    使用完整训练集和验证集训练模型，使用测试集进行预测。  
    由于训练集数据量很大，扩大batch_size=1024，每个epoch迭代536步，进行10个epoch的训练。同样使用Adam优化器，学习率0.001，损失函数为交叉熵。每个epoch结束时在验证集上进行一次评测，保存到目前为止验证精度最高的权重。  
    实验结果见 ./lab3/exp 文件夹下文件名包含bs=1024的文件。  
6. 实验结果：  
    验证集上的最佳模型在验证集上的精度为79.785%  
    最佳模型在测试集上的精度为79.4686%  
    复现运行 ./lab3/train.py 最后几行的加载权重与预测的代码即可。  

## Tips
1. 我将模型的参数量设置的很小，权重保存为文件后仅7.47MB。  
   在使用AutoDL平台租用的3080进行正式实验时，batch_size=1024，显存的占用量约为2840MB。  
   GPU使用率在5%与40%直接快速波动，CPU占用率一直是最高。  
   猜测原因是我实现的tokenizer不够高效，导致CPU计算量过大。  
