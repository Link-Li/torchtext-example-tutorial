import torch.nn as nn
import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, lstm_output):
        output = self.projection(lstm_output) / 8
        weights = F.softmax(output.squeeze(-1), dim=1)
        output = (lstm_output * weights.unsqueeze(-1)).sum(dim=1) * 0.001
        return output


class EmotionNet(nn.Module):
    def __init__(self, vocab_size, embeding_dim, hidden_dim1, hidden_dim2, embeding_vector):
        super(EmotionNet, self).__init__()

        self.hidden_dim1 = hidden_dim1
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embeding_dim)
        self.embedding.weight.data.copy_(embeding_vector)
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(embeding_dim, hidden_dim1, batch_first=True, num_layers=2, bidirectional=True, dropout=0.5)
        self.attention = SelfAttention(hidden_dim1)

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.Tanh(),
            nn.Dropout(p=0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim2, 2),
            nn.Tanh()
        )

    def forward(self, input, seq_lengths):
        embeds = self.embedding(input)

        input_pack = rnn_utils.pack_padded_sequence(embeds, seq_lengths, batch_first=True)

        output, hidden = self.lstm(input_pack)
        out_pad, out_len = rnn_utils.pad_packed_sequence(output, batch_first=True)
        out_pad = out_pad[:, :, : self.hidden_dim1] + out_pad[:, :, self.hidden_dim1: ]
        output_attention = self.attention(out_pad)

        output_linear = self.fc1(output_attention)
        output = self.fc2(output_linear)
        return output
