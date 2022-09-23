import os
import torch
import numpy as np
from torch.nn import functional as F

import config_bayesian as cfg

# cifar10 classes
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


def logmeanexp(x, dim=None, keepdim=False):
    """Stable computation of log(mean(exp(x))"""

    if dim is None:
        x, dim = x.view(-1), 0
    x_max, _ = torch.max(x, dim, keepdim=True)
    x = x_max + torch.log(torch.mean(torch.exp(x - x_max), dim, keepdim=True))
    return x if keepdim else x.squeeze(dim)


def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_array_to_file(numpy_array, filename):
    file = open(filename, 'a')
    shape = " ".join(map(str, numpy_array.shape))
    np.savetxt(file, numpy_array.flatten(), newline=" ", fmt="%.3f")
    file.write("\n")
    file.close()


# 数据下采样
def lower_sample_data(data, label, percent=1):
    if not (type(label) is np.ndarray):
        label = np.array(label)

    number = min(np.count_nonzero(label), len(label)-np.count_nonzero(label))
    loc = []
    c0 = (50 * number) // 50
    c1 = number
    for i in range(len(label)):
        if label[i] == 0 and c0 >0:
            c0 -= 1
            loc.append(i)
        elif label[i] == 1 and c1 >0:
            c1 -= 1
            loc.append(i)
    return data[loc], label[loc]

# 使用SMOTE算法进行过采样
from imblearn.over_sampling import SMOTE
def oversample_data(data, label):
    oversample = SMOTE()
    matrix_data = data.reshape([-1, 12*32])
    os_X, os_labels = oversample.fit_resample(matrix_data, label)
    matrix_data = os_X.reshape([-1, 1, 12, 32])
    labels = os_labels
    return matrix_data, labels

# 添加白噪音
def get_Normal_noise(shape, k=10, scale = 0.1, loc = 0):
    np.random.seed(0)
    return np.random.normal(loc, scale, shape)*k

def get_Poisson_noise(shape, k=1, lam=1):
    np.random.seed(0)
    return np.random.poisson(lam, shape)*1


def normalize(data):
    if data is torch.Tensor:
        pass
    return F.softplus(torch.from_numpy(np.float32(data))).data.numpy()

if __name__ == '__main__':
    shape = np.array([4, 4])
    print(get_Normal_noise(shape))
    print(get_Poisson_noise(shape))
