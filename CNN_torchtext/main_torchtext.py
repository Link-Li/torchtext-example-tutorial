# -*- encoding: utf-8 -*-
'''
@File    :   main_torchtext.py
@Time    :   2020/06/07 18:19:04
@Version :   1.0
TODO     :   这个是将torchvisio替换成了torchtext, 按道理来说, 输入的数据基本都是等长的话, 那么应该可以在一定程度上提升模型的效果.
             事实上是, 在这个模型里面, 似乎torchvisio的效果更好一点, 而torchtext的效果略差一点, 如果将BucketIterator换成
             Iterator, 好像和torchvisio的效果就一样了, 怎么说呢, 或许有点玄学. 如果使用BucketIterator, 随机参数设为16好像效果好点.

             后来又进行了实验, 发现可能是词汇频率那个设置的问题, 将min_freq设置为5就好了, 当时设置的是3, 这里改为5好像就正常了呢
'''

# here put the code

import torch
import numpy as np
import torch.nn as nn
from datetime import datetime
import get_data_loader_torchtext
import model
import sklearn as sk
import sklearn.metrics
import os
from test_model import test_model
import argparse
import jieba
from torchtext.data import Field
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-min_sentence_length', type=int, default=4, help='min sentence length, which is decided by CNN cell length')
    parse.add_argument('-no_train', action='store_true', default=False, help='train or test')
    parse.add_argument('-batch_size', type=int, default=8, help='batch size number')
    parse.add_argument('-lr', type=float, default=1e-5, help='learning rate')
    parse.add_argument('-gpu_num', type=str, default='0', help='gpu index')
    parse.add_argument('-cuda', action='store_true', default=False, help='if True: use cuda. if False: use cpu')
    parse.add_argument('-seed_num', type=int, default=13, help='random seed')  # 13 16的时候都还差不多
    opt = parse.parse_args()
    print(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_num)
    set_seed(seed=int(opt.seed_num))

    tokenize = lambda x: jieba.lcut(x)
    sentence_field = Field(
        sequential=True, tokenize=tokenize, lower=False, batch_first=True)
    label_field = Field(sequential=False, use_vocab=False)

    train_iterator, test_iterator, embeding_vector = \
        get_data_loader_torchtext.get_iterator(opt, train_data_path='data/train.json',
                                               test_data_path='data/test.json', sentence_field=sentence_field,
                                               label_field=label_field, vectors_path='data/train_word2vec_200.txt')

    # 加载预训练好的词向量
    emotion_net = model.EmotionNet(vocab_size=embeding_vector.shape[0], embeding_dim=embeding_vector.shape[1],
                                   conv_channel=opt.min_sentence_length, embeding_vector=embeding_vector, droput_rate=0.1)
    emotion_net = emotion_net.cuda()

    # 定义损失函数
    critertion = nn.CrossEntropyLoss()
    # 定义优化函数，同时设定模型中只有被允许反向传播的参数才会被优化函数更新，并设定学习率
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, emotion_net.parameters()), lr=1e-5,
                                 betas=(0.9, 0.99))

    if opt.no_train is False:
        print('Train Begin:')
        last_F1 = 0
        dt = datetime.now()
        save_model_path = 'checkpoint/' + dt.strftime('%Y-%m-%d-%H-%M-%S-%f')
        os.mkdir(save_model_path)
        for epoch in range(100):
            print('Epoch: ', str(epoch))
            emotion_net.train()
            total_labels = 0
            run_loss = 0.0
            y_true = []
            y_pre = []
            for i, data in enumerate(train_iterator):
                output = emotion_net(data.sentence)

                loss = critertion(output, data.label)
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                run_loss += loss.item()
                _, predicted = torch.max(output, 1)
                y_pre.extend(predicted.cpu())
                y_true.extend(data.label.cpu())
                total_labels += data.label.size(0)

            # 计算F1值等一些评估参数
            run_loss /= total_labels
            train_correct = sk.metrics.accuracy_score(y_true, y_pre)
            train_F1 = sk.metrics.f1_score(y_true, y_pre, average='macro')
            train_R = sk.metrics.recall_score(y_true, y_pre, average='macro')
            train_precision = sk.metrics.precision_score(y_true, y_pre, average='macro')

            save_content = 'Train: Correct: %.5f, Precision: %.5f, R: %.5f, F1(macro): %.5f, test_loss: %f' % \
                           (train_correct, train_precision, train_R, train_F1, run_loss)
            print(save_content)

            last_F1 = test_model(emotion_net, test_iterator, critertion, save_model_path, last_F1)
            print()

    else:
        print('Test Begin:')
        # 加载模型进行训练
        emotion_net.load_state_dict(torch.load('checkpoint/2020-02-17-18-04-55-003906/0.7363878105723609.pth'))
        test_model(emotion_net, test_iterator, critertion)
