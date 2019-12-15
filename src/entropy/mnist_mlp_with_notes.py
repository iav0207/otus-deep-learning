#!/usr/bin/env python
# coding: utf-8

# In[13]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .utils import mnist

# In[14]:


train_loader, test_loader = mnist()


# In[15]:


class Net(nn.Module):  # nn.Module is a class with some predefined methods and self.parameters()
    def __init__(self, log_softmax=False):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # 128 is chosen arbitrarily. Can be adjusted for better performance
        self.fc2 = nn.Linear(128, 10)
        self.log_softmax = log_softmax
        self.optim = optim.SGD(self.parameters(), lr=1.0)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        if self.log_softmax:  # for demo purposes, to compare
            # dim=1: softmax is an operation on vector, x here is a matrix [pixel_i, picture_j]
            # we calculate softmax on a vector of each picture
            x = F.log_softmax(x, dim=1)  # this function uses the two tricks (see slides)
        else:
            x = torch.log(F.softmax(x, dim=1))
        return x

    def loss(self, output, target, **kwargs):
        self._loss = F.nll_loss(output, target, **kwargs)  # negative log likelihood
        return self._loss


# In[16]:


def train(epoch, models):
    for batch_idx, (data, target) in enumerate(train_loader):
        for model in models:
            model.optim.zero_grad()  # optimizer accumulates gradients. it's crucial to reset them in each iteration
            output = model(data)  # calls forward(x) function, gets logarithmic probabilities
            loss = model.loss(output, target)  # scalar value
            loss.backward()  # backprop step. in each vertex of the evaluation graph (see slides) sets gradient
            # of loss function by the vertex's parameter
            model.optim.step()  # optimization step. gradients are added to parameter values according to opt.function
            # which in our case is SGD with lambda=1.0

        if batch_idx % 200 == 0:
            line = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLosses '.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader))
            losses = ' '.join(['{}: {:.6f}'.format(i, m._loss.item()) for i, m in enumerate(models)])
            print(line + losses)

    else:
        batch_idx += 1
        line = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLosses '.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader))
        losses = ' '.join(['{}: {:.6f}'.format(i, m._loss.item()) for i, m in enumerate(models)])
        print(line + losses)


# In[17]:


models = [Net(), Net(True)]

# In[18]:


avg_lambda = lambda l: 'Loss: {:.4f}'.format(l)
acc_lambda = lambda c, p: 'Accuracy: {}/{} ({:.0f}%)'.format(c, len(test_loader.dataset), p)
line = lambda i, l, c, p: '{}: '.format(i) + avg_lambda(l) + '\t' + acc_lambda(c, p)


def test(models):
    test_loss = [0] * len(models)
    correct = [0] * len(models)
    with torch.no_grad():  # no need to calculate gradients during test run. makes impossible to perform SDG, but it's not needed here, right?
        for data, target in test_loader:
            output = [m(data) for m in models]
            for i, m in enumerate(models):
                test_loss[i] += m.loss(output[i], target, reduction='sum').item()  # sum up batch loss
                pred = output[i].data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct[i] += pred.eq(target.data.view_as(pred)).cpu().sum()

    for i in range(len(models)):
        test_loss[i] /= len(test_loader.dataset)
    correct_pct = [100. * c / len(test_loader.dataset) for c in correct]
    lines = '\n'.join([line(i, test_loss[i], correct[i], correct_pct[i]) for i in range(len(models))]) + '\n'
    report = 'Test set:\n' + lines

    print(report)


# In[19]:


for epoch in range(1, 3):
    train(epoch, models)
    test(models)

# In[ ]:


# here the results of models 1 and 2 are somewhat similar, the training process is similar as well, no problems
# with manual log(softmax) calculation
# BUT if we replace activation function (in forward(x) method) from sigmoid to relu (which does not limit
# values in the middle-layer neurons),
# the first model will seize giving meaningful results due to the numeric overflow problem
