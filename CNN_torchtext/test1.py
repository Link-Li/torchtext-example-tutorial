import torch
import torch.nn as nn


cnn = nn.Conv2d(1, 2, kernel_size=(5, 10))
text = torch.FloatTensor(5, 10).view(1, 1, 5, 10)
output = cnn(text)

a = 1
