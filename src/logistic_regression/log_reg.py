#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# get_ipython().run_line_magic('matplotlib', 'inline')

# Given distributions of two classes, build a classifier

# distribution params
mu0, sigma0 = -2., 1.
mu1, sigma1 = 3., 2.


def sample(d0, d1, n=32):
    x0 = d0.sample((n,))
    x1 = d1.sample((n,))
    y0 = torch.zeros((n, 1))
    y1 = torch.ones((n, 1))
    return torch.cat([x0, x1], 0), torch.cat([y0, y1], 0)


d0 = torch.distributions.Normal(torch.tensor([mu0]), torch.tensor([sigma0]))
d1 = torch.distributions.Normal(torch.tensor([mu1]), torch.tensor([sigma1]))

layer = nn.Linear(1, 1)  # 1 in 1 out
print([p.data[0] for p in layer.parameters()])
layer_opt = optim.SGD(lr=1e-3, params=list(layer.parameters()))

log_freq = 500
for i in range(10000):
    if i % log_freq == 0:
        with torch.no_grad():   # disabling gradient calculation
            x, y = sample(d0, d1, 100000)
            out = torch.sigmoid(layer(x))
            loss = F.binary_cross_entropy(out, y)
        print('Loss after %d iterations: %f' % (i / log_freq, loss))
    layer_opt.zero_grad()
    x, y = sample(d0, d1, 1024)
    out = torch.sigmoid(layer(x))
    loss = F.binary_cross_entropy(out, y)
    loss.backward()
    layer_opt.step()

x_scale = np.linspace(-10, 10, 5000)
d0_pdf = stats.norm.pdf(x_scale, mu0, sigma0)
d1_pdf = stats.norm.pdf(x_scale, mu1, sigma1)
x_tensor = torch.tensor(x_scale.reshape(-1, 1), dtype=torch.float)
with torch.no_grad():
    dist = torch.sigmoid(layer(x_tensor)).numpy()
ratio = d1_pdf / (d1_pdf + d0_pdf)

plt.plot(x_scale, d0_pdf * 2, label='d0')  # multiplying by 2 just for better visualization
plt.plot(x_scale, d1_pdf * 2, label='d1')
plt.plot(x_scale, dist.flatten(), label='pred')
plt.plot(x_scale, ratio, label='ratio')
plt.legend()

print([p.data[0] for p in layer.parameters()])

# torch.log(F.sigmoid(torch.tensor(-100.))) # fails due to -infinity

F.logsigmoid(torch.tensor(-100.))   # correct way
