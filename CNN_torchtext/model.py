import torch.nn as nn
import torch


class EmotionNet(nn.Module):
    def __init__(self, vocab_size, embeding_dim, conv_channel, embeding_vector, droput_rate=0.1):
        super(EmotionNet, self).__init__()

        # 词嵌入及预训练词向量的导入
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embeding_dim)
        # self.embedding.weight.data.copy_(torch.from_numpy(embeding_vector))
        self.embedding.weight.data.copy_(embeding_vector)
        self.embedding.weight.requires_grad = False

        # 卷积神经网络的搭建
        self.conv1 = nn.Conv2d(1, conv_channel, (1, embeding_dim))
        self.conv2 = nn.Conv2d(1, conv_channel, (2, embeding_dim))
        self.conv3 = nn.Conv2d(1, conv_channel, (3, embeding_dim))
        self.conv4 = nn.Conv2d(1, conv_channel, (4, embeding_dim))

        # 最后全连接层进行分类
        self.classifer = nn.Sequential(
            nn.Dropout(p=droput_rate),
            nn.Linear(conv_channel*4, 2),
            nn.Tanh()
        )

    def forward(self, input):

        # 词嵌入，输出大小(batch size, word_number, embedding_dim)
        embeds = self.embedding(input)
        # 添加一个通道，因为词嵌入输出的是没有通道的, 输出大小(batch size, input_channel, word_number, embedding_dim)
        embeds = embeds.unsqueeze(1)

        # 让词向量通过每一个卷积神经网络，这里采用最大池化，提取每次输出的最大值，同时剔除多余的维度
        # 卷积输出输出大小均为(batch size, conv_channel, word_number, 1)
        # 池化输出输出大小均为(batch size, conv_channel, 1, 1)
        # 剔除多余维度之后的输出输出大小均为(batch size, conv_channel)
        conv1_out = torch.max(self.conv1(embeds), dim=2)[0].squeeze(-1)
        conv2_out = torch.max(self.conv2(embeds), dim=2)[0].squeeze(-1)
        conv3_out = torch.max(self.conv3(embeds), dim=2)[0].squeeze(-1)
        conv4_out = torch.max(self.conv4(embeds), dim=2)[0].squeeze(-1)

        # 对于上面的输出进行拼接， (batch size, conv_channel*4)
        conv_out = torch.cat((conv1_out, conv2_out, conv3_out, conv4_out), dim=1)

        # 因为只有两个类别，所以输出为(batch size, 2)
        output = self.classifer(conv_out)
        return output
