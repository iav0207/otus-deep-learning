import torch.nn as nn
import torch.nn.functional as F

import utils

train_loader, test_loader = utils.mnist()
batch_x, batch_y = next(iter(train_loader))
print(batch_x.shape, batch_y.shape)

flatten_x = batch_x.view(-1, 784)
print(flatten_x.shape)

layer = nn.Linear(784, 10)
print([p.shape for p in layer.parameters()])

params = [p for p in layer.parameters()]
print(params[1])

hidden_x = layer(flatten_x)
print(hidden_x.shape, hidden_x[0][:10])

rectified = F.relu(hidden_x)
print(hidden_x[0])
print(rectified[0])
