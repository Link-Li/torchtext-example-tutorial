# torchtext-example-tutorial

## 1.运行环境

```
Python3
pytorch1.2+cuda
```

## 2.目录结构

CNN_torchtext和LSTM_torchtext目录一致

/checkpoint 保存训练好的模型
/data 存放训练数据和词向量

## 3.实验结果

一般来说CNN预测的结果F1值大概在0.72左右，LSTM预测结果的F1值在0.75左右。

## 4.torchtext讲解

# torchtext的使用

目录
=================

   * [torchtext的使用](#torchtext的使用)
      * [1.引言](#1引言)
      * [2.torchtext简介](#2torchtext简介)
      * [3.代码讲解](#3代码讲解)
         * [3.1 Field](#31-field)
         * [3.2 Dataset](#32-dataset)
         * [3.4 使用Field构建词向量表](#34-使用field构建词向量表)
         * [3.3 Iteration](#33-iteration)
      * [4. 总结](#4-总结)


## 1.引言

&emsp;&emsp;这两天看了一些torchtext的东西， 其实torchtext的教程并不是很多，当时想着使用torchtext的原因就是， 其中提供了一个BucketIterator的桶排序迭代器，通过这个输出的批数据中，每批文本长度基本都是一致的，当时就感觉这个似乎可以提升模型的性能，毕竟每次训练的数据的长度都差不多，不会像以前一样像狗牙一样参差不齐，看着揪心了。 

&emsp;&emsp;但是实际使用起来， 其实发现torchtext并不是那么好用，而且实际实验结果表明，随机抽取文本和按文本长度排序之后再去抽取文本，模型的性能似乎都是一样的，我在CNN和LSTM上面都做了实验，发现没啥提升。至于为啥没用预训练模型做实验，主要是发现torchtext的限制太多了，而预训练模型都有自己的tokenizer方式，灵活性比较高，导致这个模块使用起来特别的别扭。最主要的还是因为torchtext封装的太厉害了，而它的官方文档说实话，写的也不是特别清楚，有些地方用的就有点糊里糊涂，还得靠做实验来看看到底是啥情况，有些操作感觉还是没有自己动手写感觉踏实。

*参考链接*
<a href='https://blog.csdn.net/nlpuser/article/details/88067167' target='_blank'>CSDN上面的一个torchtext的简单介绍</a>

<a href='https://www.kaggle.com/swarnabha/pytorch-text-classification-torchtext-lstm' target='_blank'>Kaggle上面的一个torchtext+LSTM的示例</a>

<a hef='https://github.com/pytorch/text/issues/609' target='_blank'>github上面关于torchtext+HuggingFace 的使用讨论</a>

<a href='' target='_blank'>本文代码的github地址</a>

## 2.torchtext简介

&emsp;&emsp;因为也没研究的特别深，所以这里介绍的就是平时用的一些方法，而torchtext本身是有很多其他的用途的，例如它里面提供了很多nlp方面的数据集，可以直接加载使用，也提供了不少训练好的词向量之类的，这一点和torchvisio是一样的（但是限于国内的一些网络，这些功能一般好像都是处于荒废的状态）。

&emsp;&emsp;一般我们常用的torchtext主要是3大部分，分别是**Field**，**Dataset**和**Iteration**三大部分。其中`Dataset`是对数据进行一些处理操作等，这点和torchvisio还是比较像的，但是这里的`Dataset`其实能做的操作并不是很多，因为它的很多任务都被`Field`所承担了；至于`Iteration`，这个和torchvisio模块中的`DataLoader`很类似，但是`Iteration`提供了很多NLP里面需要的功能，例如对每个batch的数据进行batch内排序，设置排序的关键字等。

## 3.代码讲解

&emsp;&emsp;这里使用一个基于LSTM的情感分析模型进行讲解torchtext的简单使用

### 3.1 Field

&emsp;&emsp;一般来说，第一步是首先设定好Field，Field是对数据格式的一种定义，可以看到官方提供的Field参数如下所示：

```
~Field.sequential – 输入的数据是否是序列型的，如果不是，将不使用tokenzie对数据进行处理

~Field.use_vocab – 是否使用Vocab对象，也就是使用输入的词向量，这里以后会讲到，如果不使用，那么输入Field的对象一定是数字类型的。

~Field.init_token – 给example数据的开头加一个token，感觉类似<CLS>标签，example之后会将

~Field.eos_token – 给example数据加一个结束token

~Field.fix_length – 设定序列的长度，不够的进行填充

~Field.dtype – 表示输入的example数据的类型

~Field.preprocessing – 将example数据在tokenize之后，但在转换成数值之前的管道设置，这个我没有用过，所以不确定具体怎么用

~Field.postprocessing – 将example数据在转换成数值之后，但在变成tensor数据之前的管道设置. 管道将每个batch的数据当成一个list进行处理

~Field.lower – 是否将输入的文本变成小写

~Field.tokenize – 设置一个tokenize分词器给Field用，这里也有内置的一些分词器可以用

~Field.tokenizer_language – 分词器tokenize的语言，这里是针对SpaCy的

~Field.include_lengths – 是否在返回文本序列的时候返回文本的长度，这里是对LSTM的变长输入设置非常好用

~Field.batch_first – 输出的数据的维度中batch的大小放到前面

~Field.pad_token – 用于填充文本的关键字，默认是<pad>

~Field.unk_token – 用于填充不在词汇表中的关键字，默认是<unk>

~Field.pad_first – 是否将填充放到文本最前面

~Field.truncate_first – 是否从文本开始的地方将文本截断

~Field.stop_words – 停止词的设置

~Field.is_target – 没看明白干啥用的
```

&emsp;&emsp;可以看到，Field的功能还是非常多的，毕竟这个是用来对输入的文本进行一些数据的预处理，首先进行初始化Field，如下所示：

```
def tokenize(x): return jieba.lcut(x)
sentence_field = Field(sequential=True, tokenize=tokenize,
                        lower=False, batch_first=True, include_lengths=True)
label_field = Field(sequential=False, use_vocab=False)
```

&emsp;&emsp;然后Field的代码就这么多。

### 3.2 Dataset

&emsp;&emsp;这里是Dataset的代码介绍，这里我们需要做的一般是继承`torchtext.data.Dataset`类，然后重写自己的Dataset，不过torchtext提供了一些内置的Dataset，如果处理的数据不是特别复杂，直接使用官方内置的一些Dataset可以满足要求，那么直接使用官方的就行了。不过一般都要自己定制一下吧，毕竟很多时候数据的输入都要进行一些修改，官方的不一定能满足要求。

&emsp;&emsp;写Dataset的时候，最主要的其实是一个Example和Field的结合，可以看下面的代码：

```
from torchtext.data import Dataset, Example

class SentenceDataset(Dataset):

    def __init__(self, data_path, sentence_field, label_field):

        fields = [('sentence', sentence_field), ('label', label_field)]
        examples = []

        with open(data_path, 'r') as f_json:
            file_content = json.load(f_json)
            self.sentence_list = []
            self.label_list = []
            for data in file_content:
                self.sentence_list.append(data['sentence'])
                self.label_list.append(data['label'])
        
        for index in trange(len(self.sentence_list)):
            examples.append(Example.fromlist([self.sentence_list[index], self.label_list[index]], fields))

        super().__init__(examples, fields)

    @staticmethod
    def sort_key(input):
        return len(input.sentence)
```

&emsp;&emsp;可以看到，这里将输入的文本，标签和Field进行绑定，也就是告诉Field它要具体处理哪些东西，然后最后还要使用`super().__init__(examples, fields)`来调用一下父类的初始化方法。这里还有一个`def sort_key(input):`方法，这个方法是帮助后面的Iteration进行数据排序用的关键字，其实在Iteration中可以直接设置用于排序的关键字，但是因为在前面的Field里面使用了`include_lengths`关键字，好像导致后面的 Iteration直接指定关键字无法进行正常的排序，然后在Dataset里面直接指定关键字，Iteration就可以直接进行正常的排序。这里不排除是我代码写的有问题，但是使用上面代码的那种方法可以正常排序是经过实验验证的。

&emsp;&emsp;对于上面的example和Field进行绑定的时候，因为我这里使用的训练数据和测试数据都是有标签的，所以标签那个位置直接就写上了，但是一般测试数据都是没有标签，如果是没有标签的，将上面的代码改成下面这样就行了：

```
examples.append(Example.fromlist([self.sentence_list[index], None], fields)) # 没有标签就使用None来代替
```

&emsp;&emsp;当然这里的Dataset还有其他的一些功能，例如`split`方法等，这些大家可以去看官方的API文档。

### 3.4 使用Field构建词向量表

&emsp;&emsp;在使用LSTM等一些网络的时候，我们喜欢使用词向量对网络的Embedding层进行初始化，而Field中的`build_vocab`提供了这些处理操作。首先我们需要将词向量读取进来，在一个txt文本中保存如下格式的词向量：

```
公司 0.3919137716293335 0.4011327922344208 ...
公园 0.17394110560417175 0.10003302991390228 ...
公布 0.24726712703704834 0.06743448227643967 ...
公正 0.1161544919013977 0.07093961536884308 ...
公道 0.44119203090667725 0.21420961618423462 ...
```

&emsp;&emsp;首先需要注意的是，词向量表中只包含词语的词向量，不包含<pad><unk>等关键字的词向量，这部分的词向量可以在`build_vocab`进行一定的设置，首先看`build_vocab`的参数（实际是`torchtext.vocab.Vocab`的参数，但是这里是经过`build_vocab`处理之后将参数传入到`torchtext.vocab.Vocab`）：

```
counter – 这里用来计算输入的数据的频率的，其实没太看明白英文翻译，不过这里对应build_vocab输入的是Dataset，经过build_vocab处理之后传递给torchtext.vocab.Vocab

max_size – 词向量表的最大大小

min_freq – 参与转换成词向量的最小词频率，不满足这个词频率的直接就是<unk>了

specials – 需要添加到词向量表中的一些特殊字符，默认的包含['<unk'>，'<pad>']两种，也是因为这个参数，所以我们的txt文件中的词向量不需要包含这两个特殊字符。

vectors – 用于加载的预训练好的词向量

unk_init (callback) – 用于初始化未知词汇的词向量，默认是0

vectors_cache – 存放缓存的目录，这个不一定在这里设定，在Vectors类中设置也行

specials_first – 改变特征字符在词汇表中的位置，是放在最前面还是放在最后面
```

&emsp;&emsp;不过在`build_vocab`词向量之前，我们需要先将词向量加载进来，这里的操作就是使用`torchtext.vocab.Vectors`，主要包含以下参数：

```
name – 这里实际应该是保存词向量文件的位置

cache – 用于存放缓存的目录

url – url for download if vectors not found in cache

unk_init (callback) – 初始化未知词的词向量

max_vectors (int) – 用于设置词向量的大小，API文档中说，一般保存词向量的文件中，是按照词向量的频率大小从上大小进行排序，然后存储到文件中，所以放弃一些低频率的词向量，对性能可能没影响，但是还可以节省内存
```

&emsp;&emsp;上面的参数介绍完了，就可以来看代码了，代码其实并不怎么复杂：

```
cache = 'data/vector_cache'
if not os.path.exists(cache):
    os.mkdir(cache)
vectors = Vectors(name=vector_path, cache=cache)
sentence_field.build_vocab(train_dataset, min_freq=min_freq, vectors=vectors)
```

### 3.3 Iteration

&emsp;&emsp;最后是Iteration方面的介绍，这部分官方提供了三个Iteration，当然也可以自定义，但是目前看官方提供的Iteration就可以满足大部分情况，所以这里就没有进行自定义的Iteration。

```
train_iterator = BucketIterator(train_dataset, batch_size=batch_size,
                                device='cuda',
                                sort_within_batch=True, shuffle=True)
test_iterator = Iterator(test_dataset, batch_size=batch_size,
                        device='cuda', train=False,
                        shuffle=False, sort=False, sort_within_batch=True)
```

&emsp;&emsp;这里使用了`BucketIterator`和`Iterator`，因为`BucketIterator`可以自动的选择长度类似的文本组成一个batch，所以用于训练数据，而测试数据一般而言不想进行排序或者其他的操作，就使用了`Iterator`，这里就不对`Iterator`的一些参数进行介绍了，一些重要的常用的基本就是上面列出来的那些了。

## 4. 总结

&emsp;&emsp;torchtext模块对于传统的一些模型，例如CNN，LSTM等，使用起来还是比较方便的，特别是一些常用的操作，一些常用的数据集等等，torchtext都是包含的，但是对于目前的预训练模型，大家可以去网上找一下资料，比如<a hef='https://github.com/pytorch/text/issues/609' target='_blank'>github上面关于torchtext+HuggingFace 的使用讨论</a>，其实受限于torchtext本身的一些规则太多，导致很多操作都被隐层了起来，一些想要自定义的功能却不怎么方便去自定义，所以感觉对于预训练模型，或者一些其他的，例如多模态方面的模型，使用起来并不怎么方便，估计也是因为这个原因，导致一些相关教程和讨论比较少。然后我基于CNN和LSTM，写了一个torchtext的代码，所以这里贴出来github地址<a href='' target='_blank'>本文代码的github地址</a>
