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
    基于BOW的模型精度为59.613%，基于NGram的模型精度为57.804%  
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

## 更新
&emsp;在第一版的实现中，NGram tokenizer的generate_feature函数在分词时，没有优先匹配最长的token并跳过已匹配的字符，导致了tokenize结果中出现了重复。  
&emsp;该错误目前已经纠正。重新提交kaggle评测，精度为57.804%。  
&emsp;纠错前NGram的精度60.108%比纠错后更高，猜测原因是之前的实现在句向量中加入了更多细粒度token。  
&emsp;NGram的词表中词频介于10-20之间的低频词很多（26656/39708），有重复的NGram tokenizer使得词向量可以被更充分地训练，句向量汇总的信息也更多，因此经过本lab的简单MLP模型映射后，取得了更好的效果。  
&emsp;此外，在保存vocab时，设置了json文件的格式化，避免超长单行导致的IDE卡顿问题。  

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
    batch_size=128，每个epoch迭代23步，使用AdamW优化器，学习率0.001，损失函数为交叉熵。
    模型在训练集上最终可以达到99%以上的准确率，在验证集上的精度在40个epoch时达到最高，迭代约920步，之后不会继续上升。  
    实验结果见 ./lab3/exp 文件夹下文件名包含bs=32的文件。  
5. 正式训练：  
    使用完整训练集和验证集训练模型，使用测试集进行预测。  
    由于训练集数据量很大，扩大batch_size=1024，每个epoch迭代536步，进行10个epoch的训练。同样使用AdamW优化器，学习率0.001，损失函数为交叉熵。每个epoch结束时在验证集上进行一次评测，保存到目前为止验证精度最高的权重。  
    实验结果见 ./lab3/exp 文件夹下文件名包含bs=1024的文件。  
6. 实验结果：  
    验证集上的最佳模型在验证集上的精度为79.785%  
    最佳模型在测试集上的精度为79.4686%  
    复现运行 ./lab3/train.py 最后几行的加载权重与预测的代码即可。  

## Tips
1. 我将模型的参数量设置的很小，权重保存为文件后仅7.47MB。  
   在使用AutoDL平台租用的3080进行正式实验时，batch_size=1024，显存的占用量约为3256MB。  
   GPU使用率在5%与40%直接快速波动，CPU占用率一直是最高。  
   猜测原因是我实现的tokenizer不够高效，导致CPU计算量过大，GPU在等待数据时无法充分利用。   

## 更新
   改进tokenizer、Dataset类与collate_fn方法。  
   在加载数据集时，先完成token2id的映射。该结果可以保存与加载。  
   由于pickle保存与加载tensor很慢，所以这里的数据类型都是list，加载完成后转换为tensor。由于转换为tensor也有一定的时间开销，所以在dataset的init方法中一次性完成，避免在collate_fn中反复转换。  
   在dataloader的collate_fn中只执行batch pad，将同batch内的所有input_ids对齐到最长的长度。这一步不能预先完成，是因为训练时dataloader shuffle=True，每个batch需要pad对齐的长度不一定一样。
   在使用AutoDL平台租用的3080上重新实验，优化前一个batch的时间约165s，优化后约96s。GPU占用率稳定在46%左右，显存占用量无明显变化，CPU占用率还一直是最高。  
   此外，在改进过程中发现之前的tokenizer在生成词表时将所有句子转换为小写，但是在tokenize时没有做同样的处理，导致含义大写字母的词被标记为UNK。该错误已修正。  
   进行15个epoch的重新实验，模型在第8个epoch时在验证集上的精度达到最高，为81.7057%，此时在测试集上的精度为81.331%  

## Tips
   lab2的tokenizer、Dataset类与collate_fn方法也可以进行类似的优化。因为数据集较小，以及没有其他重新实验的必要，就不浪费时间了。  
   后来查阅文档发现torch.nn.utils.rnn.pack_padded_sequence和torch.nn.utils.rnn.pad_packed_sequence已经包含了batch pad功能，不需要像我这样手动实现。  
   最简单的实现，只需要将整个数据集的所有input_ids对齐到最长的长度，同时记录真实长度，然后直接取batch即可。但这种实现方法在数据集样本数巨大且其中最长句子很长时，会产生严重的空间浪费与访存压力。  
   本实验数据集规模较大(超过50万条数据)，因为我的实现方法是也算是合理的。  

# 任务四：基于LSTM+CRF的序列标注

用LSTM+CRF来训练序列标注模型：以Named Entity Recognition为例。

1. 参考
   1. 《[神经网络与深度学习](https://nndl.github.io/)》 第6、11章
   2. https://arxiv.org/pdf/1603.01354.pdf
   3. https://arxiv.org/pdf/1603.01360.pdf
2. 数据集：CONLL 2003，https://www.clips.uantwerpen.be/conll2003/ner/
3. 实现要求：Pytorch
4. 知识点：
   1. 评价指标：precision、recall、F1
   2. 无向图模型、CRF
5. 时间：两周

## 实现方法与结果
1. 下载conll2003数据集，https://data.deepai.org/conll2003.zip 该网址下载更方便。  
2. 由于conll2003数据集本身分好了词，每个词对应一个标签，所以直接采用BagOfWord tokenizer。  
   分析词表发现16379个词的词频小于5,10060个词词频仅为1。考虑到每个词都要进行NER标注，所以没有去掉低频词。  
    代码见 ./lab4/tokenizers.py
3. 搭建TextRNN结构的NER模型，采用BiLSTM+CRF。
   近期时间有限，没有实现手搓CRF的前向计算与反向传播，直接使用torchcrf库。https://github.com/kmkurn/pytorch-crf/blob/master/torchcrf/__init__.py  
4. 由于数据集规模较小，基于lab3的经验，在dataloader的init方法中完成所有数据的token2id的映射，得到一个(example_num, max_len)维的input_ids矩阵，以及一个(example_num)维的seq_len向量。并且全部转移到GPU上，节约访存时间。之后也不再在collate_fn中进行batch pad。  
   实测训练集input_ids矩阵维度为(14986, 113)，PAD占比为87.92%。从这一结果来看，该实现方法造成的空间浪费很严重。  
    代码见 ./lab4/ConllDataset.py
5. 实现混淆矩阵计算，以及precision、recall、F1的计算。  
   最开始我使用二重循环计算混淆矩阵，但是这种方法非常慢。在笔记本1660ti上实验，GPU占用率仅5%。后改用向量化操作或散点加法操作实现，速度大幅提升，GPU占用率达到100%。  
   由于数据集中大部分词都不是实体，类别不平衡问题严重。以训练集为例，'O'占比为83.35%。因此实现了宏平均与微平均两种F1计算方法。macro F1可以平均地评估每个类别的性能，micro F1可以评估整体性能。
6. 正式实验：
   由于数据集较小，不再进行拟合实验。直接使用全部数据进行训练。  
   batch_size=1024，每个epoch迭代14步，使用AdamW优化器。损失函数为CRF与交叉熵的结合。调试发现CRF loss的数值较大，即使根据batch_size取平均值，其数值仍是CE loss的十几倍。因此乘以0.1的权重。  
   仅使用CRF loss或CE loss时，学习率为0.001时。使用CRF loss + CE loss作为损失函数时，学习率为0.0005。  
   每个训练epoch结束后在验证集上进行一次评测，取val_macro_f1最高的检查点保存，用于之后在测试集评测。
7. 实验结果：  

| 损失函数     | val_macro_f1 | val_micro_f1 | val_accuracy | test_macro_f1 | test_micro_f1 | test_accuracy |  
|----------|--------------|--------------|--------------|---------------|---------------|---------------|  
| CE       | 0.7317       | 0.9327       | 0.9850       | 0.6408        | 0.9087        | 0.9797        |  
| CRF      | 0.7320       | 0.9334       | 0.9852       | 0.6445        | 0.9098        | 0.9800        |  
| CE + CRF | 0.6633       | 0.9190       | 0.9820       | 0.5926        | 0.8987        | 0.9775        |
   F1变化曲线见 ./lab4/exp 文件夹下的图片。

## Tips
1. 分析训练过程中的F1变化曲线，可以发现仅1个epoch之后，micro f1就达到了一个比较高的值。但是macro f1一直很低。  
   分析混淆矩阵发现模型在第一个epoch结束后陷入局部最优解，所有的样本都被预测为标签数量最多的‘O’，导致了这一现象。  
   之后训练进入鞍点，在大约17个epoch后走出局部最优解，micro f1与macro f1都开始上升。60个epoch后模型完全收敛。    
   至于为什么能从局部最优解走出来，我还没想明白背后的数学原理，之后找人请教一下。  
2. 分析训练过程中的F1变化曲线，可以发现模型在验证集与训练集上泛化能力很差。除了类别不平衡这一因素之外，还可能是因为OOV的词过多。  
   最初在分析tokenizer的词表时就发现，数据集中近一半的词都只出现了一次，绝大部分词是低频词。  
   如果是OOV导致的泛化能力差，可以考虑换用细粒度的tokenizer，比如BPE。同时注意label的复制与对齐。  
   但是逐个检查验证集与测试集的词后发现，验证集OOV率为7.51%，测试集OOV率为10.81%，并不是很高。  
   从这一角度来看，OOV并不是导致泛化能力差的主要原因。改进模型可能需要从类别不平衡问题入手。  

## 更新
   猜测模型能走出局部最优解的原因可能是由于AdamW优化器的特性。  
   更换SGD + momentum优化器、Adagrad优化器、Adadelta优化器、RMSProp优化器、Adam优化器，采用CRF loss，进行一轮控制变量实验。  
   SGD + momentum优化器彻底陷入局部最优解，无法走出。  
   Adagrad优化器彻底陷入局部最优解，无法走出。  
   Adadelta优化器前40个epoch完全不收敛，之后到80个epoch陷入局部最优解，且该局部最优解不如其他优化器的局部最优解。分析混淆矩阵发现，大部分样本被预测为‘O’，小部分样本被预测为'B-PER'。  
   RMSProp优化器与AdamW相比，大约3个epoch就走出了局部最优解，而且30个epoch后模型就完全收敛，最终性能表现最好。  
   Adam优化器的曲线与AdamW的几乎完全相同。  

| 优化器      | val_macro_f1 | val_micro_f1 | val_accuracy | test_macro_f1 | test_micro_f1 | test_accuracy |  
|----------|--------------|--------------|--------------|---------------|---------------|---------------|  
| AdamW    | 0.7317       | 0.9327       | 0.9850       | 0.6408        | 0.9087        | 0.9797        |  
| SGD+M    | 0.1006       | 0.8278       | 0.9617       | 0.1005        | 0.8262        | 0.9614        |  
| Adagrad  | 0.1051       | 0.8088       | 0.9575       | 0.1056        | 0.7985        | 0.9552        |
| Adadelta | 0.1129       | 0.7645       | 0.9477       | 0.1107        | 0.7533        | 0.9452        |
| RMSProp  | 0.7704       | 0.9427       | 0.9873       | 0.6374        | 0.9104        | 0.9801        |
| Adam     | 0.7284       | 0.9325       | 0.9850       | 0.6451        | 0.9100        | 0.9800        |

  综合对比分析以上实验结果，可以给出这样一个解释：  
  类别的不平衡，导致模型的的参数中和多数类有关的参数的梯度较大，而与少数类有关的参数的梯度较小。  
  RMSprop、Adam、AdamW三款优化器都有自适应学习率的特性，可以根据参数的梯度大小自动调整学习率。对于大的梯度值，可以减小学习率，从而减少震荡，而对于小的梯度值，可以增加学习率，有助于逃离局部最优。  
  Adam与AdamW在调整学习率时，会考虑梯度的一阶矩估计和二阶矩的指数加权滑动平均，而RMSprop只考虑二阶矩的。如果局部最优解主要是由梯度方差引起的，RMSprop可以更高效地调整学习率，从而更快地走出局部最优解。  
  此外，Adam与AdamW的一阶矩二阶矩0初始化与偏差纠正机制，AdamW的权重衰减机制，Adam的正则化机制，也可能与该现象相关，但没法定性解释。  
  Adadelta优化器虽然也是基于梯度二阶矩的自适应学习率优化器，但是其学习率的更新并不基于全局学习率，而是历史参数变化量的二阶矩。这使得其在梯度平坦的局部最优解区域的学习率较小，不足以推动权重远离局部最优解。  
  此外，RMSprop认的二阶矩估计的指数加权滑动平均系数为0.99，而Adam与AdamW默认的二阶矩系数为0.999，Adadelta默认的系数为0.9。这个系数会影响在梯度平坦区的学习率调整系数的分母减小的速度。  
  之前的实验都使用默认系数，调整AdamW与Adadelata的二阶矩系数为0.99补充实验，发现AdamW的曲线几乎没有变化，Adadelta的曲线提前数个epoch开始向局部最优解收敛，但最终性能几乎没有变化。说明各优化器的表现的区别源于算法本身的设计，而不是超参数的选择。  

## 更新
  通过调整损失函数对类别不平衡问题进行处理。  
  采用Weighted Cross Entropy Loss，对每个类别的损失进行加权，权重为每个类别的逆频率。  
  采用Focal Loss，降低易分类的样本的权重，增加难分类的样本的权重。  
  实验结果： 

| 损失函数     | val_macro_f1 | val_micro_f1 | val_accuracy | test_macro_f1 | test_micro_f1 | test_accuracy |  
|----------|--------------|--------------|--------------|---------------|---------------|---------------|  
| CE       | 0.7317       | 0.9327       | 0.9850       | 0.6408        | 0.9087        | 0.9797        |  
| CRF      | 0.7320       | 0.9334       | 0.9852       | 0.6445        | 0.9098        | 0.9800        |  
| CE + CRF | 0.6633       | 0.9190       | 0.9820       | 0.5926        | 0.8987        | 0.9775        |
| W_CE     | 0.7147       | 0.9177       | 0.9817       | 0.6294        | 0.8940        | 0.9764        |
| Focal    | 0.7299       | 0.9302       | 0.9844       | 0.6240        | 0.9040        | 0.9787        |

  采用Weighted Cross Entropy Loss训练，模型在向局部最优解收敛的过程中就开始走出，仅表现为初期突刺。  
  采用Focal Loss训练，过程曲线与CE几乎没有变化。  
  两种方法都没有取得性能上的提升。

# 任务五：基于神经网络的语言模型

用LSTM、GRU来训练字符级的语言模型，计算困惑度

1. 参考
   1. 《[神经网络与深度学习](https://nndl.github.io/)》 第6、15章
2. 数据集：poetryFromTang.txt
3. 实现要求：Pytorch
4. 知识点：
   1. 语言模型：困惑度等
   2. 文本生成
5. 时间：两周

## 实现方法与结果

1. 下载poetryFromTang.txt数据集，进行预处理。  
   去除其中多余的空格，分为整首诗和诗句两种加载方式。  
   代码见 ./lab5/read_poetry.py  
2. 采用BagOfWord tokenizer，中文一字一token，生成词表。  
   包括PAD、UNK、BOS、EOS、逗号、句号在内，词表大小为2508。  
   代码见 ./lab5/tokenizers.py  
3. 以诗句的第一个逗号前的短句作为encoder的输入。  
   这个逗号替换为BOS，与诗句后面的部分拼接后作为decoder的输入。  
   诗句最后一个句号替换为EOS，拼接在诗句后半段的末尾，作为预测的target。  
   忽略EOS与BOS，target相当于decoder input右移一位，即预测下一个词。  
   代码见 ./lab5/PoetryDataset.py  
4. 参考 https://zh.d2l.ai/chapter_recurrent-modern/seq2seq.html 搭建基于LSTM的seq2seq模型。    
   使用teacher forcing的方式训练模型，即训练时decoder的输入为真实目标序列，而不是上一个时间步的预测。这一机制可以加速训练，帮助模型收敛，但是会导致模型在测试时产生累积误差。    
   代码见 ./lab5/Seq2Seq.py  
5. 以9:1的比例划分训练集和测试集，进行训练。  
   训练过程中，每个epoch结束后在测试集上进行一次评测，计算困惑度。  
   困惑度的计算公式为交叉熵损失的指数。  
   保存困惑度最低的模型，用于之后的生成。  
   代码见 ./lab5/train.py
6. 实验结果：  
   使用诗句进行训练，模型在训练集上可以充分拟合，困惑度最低1.1834，next token accuracy最高为99.38%  
   测试集上的困惑度在第3个epoch后达到最低，为871.0938，next token accuracy为15.38%，几乎没有任何泛化能力。此时训练集上的困惑度为550.1983，next token accuracy为14.91%，还没有收敛。    
   之后随着训练集上的困惑度下降，测试集上的困惑度反而快速上升，最终超过15000。  
   使用充分拟合训练集的模型进行推理测试，取训练集中五言诗与七言诗各一句作为input，模型可以准确生成后半句。生成过程的困惑度在1.1左右。  
   取测试集中五言诗与七言诗各一句作为input，模型生成的诗句无意义，长度也不一定准确。生成过程的困惑度在4.8左右。  
   使用测试集上困惑度最低的模型进行推理测试，以上四句诗任一作为input，模型在生成一两个“不”字后就会输出EOS。生成过程的困惑度在102左右。    
&emsp;  
   使用整首诗进行训练，模型在训练集上都无法收敛，困惑度最低381.6075，next token accuracy最高为11.53%  
   测试集上的困惑度在第6个epoch后达到最低，为1039.4816，next token accuracy为7.16%，几乎没有任何泛化能力。此时训练集上的困惑度为892.3888，next token accuracy为7.19%，还没有收敛。    
   之后随着训练集上的困惑度下降，测试集上的困惑度反而快速上升，最终超过2000。  
   使用充分拟合训练集的模型进行推理测试，取训练集中五言诗与七言诗各一句作为input，模型在输出三到四个无意的字后会持续输出一串句号与逗号。生成过程的困惑度在12.4左右。  
   取测试集中五言诗与七言诗各一句作为input，模型在输出三到四个随机的字后会持续输出一串句号和逗号。生成过程的困惑度在12.37左右。  
   使用测试集上困惑度最低的模型进行推理测试，以上四句诗任一作为input，模型会持续输出一串逗号。生成过程的困惑度在19.9左右。    

## Tips
   我之前做过其他基于RNN的seq2seq任务，也是以失败而告终。该类任务对于模型而言，学习难度很大，简单的模型很难学到有意义的规律。  
   要改进基于RNN的seq2seq模型，可以从数据增广、增大模型参数量、注意力机制等方面入手。但是在这个Attention is All You Need的时代，我不打算花费大量时间与精力去在这个问题上考古。  
   关于本实验的尝试到此为止，要是以后有机会碰巧学习了更先进的基于lstm的seq2seq模型，再回来尝试一下吧。  

## 更新
  第一版实验时，由于我的疏忽，导致计算交叉熵损失与困惑度时，错误地使用input_length进行有效token的提取，而不是使用output_length。   
  该问题已修正，实验结果章节已经重写。  
