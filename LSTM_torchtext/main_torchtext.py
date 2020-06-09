import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import argparse
import get_data_loader_torchtext
import model
from tqdm import tqdm, trange
import sklearn as sk
import sklearn.metrics
import os
from test_model import test_model
import jieba
from torchtext.data import Field
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-no_train', action='store_true', default=False, help='test or train')
    parse.add_argument('-cuda', action='store_true', default=False, help='if True: use cuda. if False: use cpu')
    parse.add_argument('-gpu_num', type=str, default='0', help='gpu index')
    parse.add_argument('-lr', type=float, default=1e-3, help='learning rate')
    parse.add_argument('-seed_num', type=int, default=13, help='random seed') 
    parse.add_argument('-batch_size', type=int, default=16, help='batch size number')
    parse.add_argument('-iterator_type', type=str, default='bucket', help='bucket, iterator')
    parse.add_argument('-min_freq', type=int, default=5, help='build vocab\'s min word freq')
    opt = parse.parse_args()
    print(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_num)
    set_seed(opt.seed_num)

    def tokenize(x): return jieba.lcut(x)
    sentence_field = Field(sequential=True, tokenize=tokenize,
                           lower=False, batch_first=True, include_lengths=True)
    label_field = Field(sequential=False, use_vocab=False)
    train_iterator, test_iterator, embeding_vector = \
        get_data_loader_torchtext.get_iterator(
            opt, 'data/train.json', 'data/test.json', sentence_field, label_field, 'data/train_word2vec_200.txt')

    # 加载预训练好的词向量
    emotion_net = model.EmotionNet(vocab_size=embeding_vector.shape[0], embeding_dim=embeding_vector.shape[1],
                                   hidden_dim1=64, hidden_dim2=32, embeding_vector=embeding_vector)

    if opt.cuda is True:
        assert torch.cuda.is_available()
        emotion_net = emotion_net.cuda()

    # 定义损失函数
    critertion = nn.CrossEntropyLoss()
    # 定义优化函数，同时设定模型中只有被允许反向传播的参数才会被优化函数更新，并设定学习率
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, emotion_net.parameters()), lr=opt.lr,
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
            for i, data in enumerate(tqdm(train_iterator)):
                input_texts, seq_lengths = data.sentence
                output = emotion_net(input_texts, seq_lengths)

                optimizer.zero_grad()
                loss = critertion(output, data.label)
                # 反向传播
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

            last_F1 = test_model(opt, emotion_net, test_iterator, critertion, save_model_path, last_F1)
            print()

    else:
        print('Test Begin:')
        # 加载模型进行训练
        if opt.cuda is True:
            emotion_net.load_state_dict(torch.load('checkpoint/2020-02-19-11-24-13-270048/0.7587675397601501.pth'))
        else:
            emotion_net.load_state_dict(torch.load('checkpoint/2020-02-19-11-24-13-270048/0.7587675397601501.pth', map_location=torch.device('cpu')))
        test_model(opt, emotion_net, test_iterator, critertion)
