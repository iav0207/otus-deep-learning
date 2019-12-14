import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torch import utils
from torchvision import datasets, transforms


def plot_mnist(images, shape):
    fig = plt.figure(figsize=shape[::-1], dpi=80)
    for j in range(1, len(images) + 1):
        ax = fig.add_subplot(shape[0], shape[1], j)
        ax.matshow(images[j - 1][0], cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.show()


path = './MNIST_data'

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,)),
])

train_data = datasets.MNIST(path, train=True, download=True, transform=mnist_transform)
test_data = datasets.MNIST(path, train=False, download=True, transform=mnist_transform)

print(train_data[0][0])

images = [train_data[i][0] for i in range(50)]

plot_mnist(images, (5, 10))

print(images[0][0].shape)

train_loader = utils.data.DataLoader(train_data, batch_size=50, shuffle=True)

batch_x, batch_y = next(iter(train_loader))

print(batch_x.shape)

print(batch_y)
