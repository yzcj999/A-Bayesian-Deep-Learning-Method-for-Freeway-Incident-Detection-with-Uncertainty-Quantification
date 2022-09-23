import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

import utils


def getDataloader(trainset, testset, valid_size, batch_size, num_workers):
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.seed(7)
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              num_workers=num_workers)

    return train_loader, valid_loader, test_loader


def getMyDataset(data, type="csv", size=(12, 32)):
    # 设置超参数
    size_h = 12
    size_w = 32
    a = pd.read_csv("data/TRAFFIC/occupancy.csv").to_numpy()[:, 1:size_w + 1]
    b = pd.read_csv("data/TRAFFIC/speed.csv").to_numpy()[:, 1:size_w + 1]
    c = pd.read_csv("data/TRAFFIC/volume.csv").to_numpy()[:, 1:size_w + 1]
    d = pd.read_csv("data/TRAFFIC/label.csv").to_numpy()

    # 处理成（-1，size_h，size_w）数据
    matrix_data = []
    labels = []
    for begin_index in range(len(a) - size_h):
        matrix_data.append([a[begin_index:begin_index + size_h, :],
                            b[begin_index:begin_index + size_h, :],
                            c[begin_index:begin_index + size_h, :]])
        temp = d[(begin_index + size_h - 2), :]
        labels.append([1] if (1 in temp) else [0])

    X = np.array(matrix_data).astype(float)
    matrix_data = X.reshape(-1, 3, 12, 32)
    labels = np.array(labels).flatten().astype(int)

    # matrix_data, labels = utils.lower_sample_data(matrix_data, labels)

    # 使用SMOTE算法进行过采样
    from imblearn.over_sampling import SMOTE
    oversample = SMOTE()
    matrix_data = matrix_data.reshape([-1, 3 * 12 * 32])
    os_X, os_labels = oversample.fit_resample(matrix_data, labels)
    matrix_data = os_X.reshape([-1, 3, 12, 32])
    labels = os_labels

    # 转tensor格式
    X_train_tensor = torch.from_numpy(np.array(matrix_data).astype(float)).float()
    y_train_tensor = torch.from_numpy(np.array(labels).astype(float)).float()

    X_test_tensor = torch.from_numpy(np.array(matrix_data).astype(float)).float()
    y_test_tensor = torch.from_numpy(np.array(labels).astype(float)).float()

    trainset = TensorDataset(X_train_tensor, y_train_tensor)
    testset = TensorDataset(X_test_tensor, y_test_tensor)
    num_classes = 2
    inputs = 3
    return trainset, testset, inputs, num_classes


# 增大时间间隔
def getMyDataset_2(data, type="csv", size=(12, 32)):
    # 设置超参数
    size_h = 12
    size_w = 32
    a = pd.read_csv("data/TRAFFIC/occupancy.csv").to_numpy()[:, 1:size_w + 1]
    # b = pd.read_csv("data/TRAFFIC/speed.csv").to_numpy()[:, 1:size_w + 1]
    # c = pd.read_csv("data/TRAFFIC/volume.csv").to_numpy()[:, 1:size_w + 1]
    d = pd.read_csv("data/TRAFFIC/label.csv").to_numpy()

    # 处理成（-1，size_h，size_w）数据
    matrix_data = []
    labels = []
    for begin_index in range(len(a) - size_h):
        a1 = np.mean(a[begin_index:begin_index + 3, :], axis=0)
        a3 = np.mean(a[begin_index + 3:begin_index + 6, :], axis=0)
        a5 = np.mean(a[begin_index + 6:begin_index + 9, :], axis=0)
        a7 = np.mean(a[begin_index + 9:begin_index + 12, :], axis=0)
        a2 = (a1 + a3) / 2
        a4 = (a3 + a5) / 2
        a6 = (a5 + a7) / 2
        matrix_data.append([np.array([a1, a2, a3, a4, a5, a6, a7])
                            ])
        temp = d[(begin_index + size_h - 2), :]
        labels.append([1] if (1 in temp) else [0])

    X = np.array(matrix_data).astype(float)
    matrix_data = X.reshape(-1, 1, 7, 32)
    labels = np.array(labels).flatten().astype(int)

    matrix_data, labels = utils.lower_sample_data(matrix_data, labels)

    # # 使用SMOTE算法进行过采样
    # from imblearn.over_sampling import SMOTE
    # oversample = SMOTE()
    # matrix_data = matrix_data.reshape([-1, 1*7 * 32])
    # os_X, os_labels = oversample.fit_resample(matrix_data, labels)
    # matrix_data = os_X.reshape([-1, 1, 7, 32])
    # labels = os_labels

    # 转tensor格式
    X_train_tensor = torch.from_numpy(np.array(matrix_data).astype(float)).float()
    y_train_tensor = torch.from_numpy(np.array(labels).astype(float)).float()

    X_test_tensor = torch.from_numpy(np.array(matrix_data).astype(float)).float()
    y_test_tensor = torch.from_numpy(np.array(labels).astype(float)).float()

    trainset = TensorDataset(X_train_tensor, y_train_tensor)
    testset = TensorDataset(X_test_tensor, y_test_tensor)
    num_classes = 2
    inputs = 1
    return trainset, testset, inputs, num_classes


# 增大时间间隔
def getMyDataset_3(data, type="csv", size=(12, 32)):
    # 设置超参数
    size_h = 12
    size_w = 32
    a = pd.read_csv("data/TRAFFIC/occupancy.csv").to_numpy()[:, 1:size_w + 1]
    # b = pd.read_csv("data/TRAFFIC/speed.csv").to_numpy()[:, 1:size_w + 1]
    # c = pd.read_csv("data/TRAFFIC/volume.csv").to_numpy()[:, 1:size_w + 1]
    d = pd.read_csv("data/TRAFFIC/label.csv").to_numpy()

    # 处理成（-1，size_h，size_w）数据
    matrix_data = []
    labels = []
    for begin_index in range(len(a) - size_h):
        a1 = np.mean(a[begin_index:begin_index + 2, :], axis=0)
        a3 = np.mean(a[begin_index + 2:begin_index + 4, :], axis=0)
        a5 = np.mean(a[begin_index + 4:begin_index + 6, :], axis=0)
        a7 = np.mean(a[begin_index + 6:begin_index + 8, :], axis=0)
        a9 = np.mean(a[begin_index + 8:begin_index + 10, :], axis=0)
        a11 = np.mean(a[begin_index + 10:begin_index + 12, :], axis=0)
        a2 = (a1 + a3) / 2
        a4 = (a3 + a5) / 2
        a6 = (a5 + a7) / 2
        a8 = (a7 + a9) / 2
        a10 = (a9 + a11) / 2
        matrix_data.append([np.array([a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11])
                            ])
        # matrix_data.append([np.array([a1, a3, a5, a7, a9, a11])])
        temp = d[(begin_index + size_h - 2), :]
        labels.append([1] if (1 in temp) else [0])

    X = np.array(matrix_data).astype(float)
    matrix_data = X.reshape(-1, 1, 11, 32)
    labels = np.array(labels).flatten().astype(int)

    matrix_data, labels = utils.lower_sample_data(matrix_data, labels)

    # # 使用SMOTE算法进行过采样
    # from imblearn.over_sampling import SMOTE
    # oversample = SMOTE()
    # matrix_data = matrix_data.reshape([-1, 1*7 * 32])
    # os_X, os_labels = oversample.fit_resample(matrix_data, labels)
    # matrix_data = os_X.reshape([-1, 1, 7, 32])
    # labels = os_labels

    # 转tensor格式
    X_train_tensor = torch.from_numpy(np.array(matrix_data).astype(float)).float()
    y_train_tensor = torch.from_numpy(np.array(labels).astype(float)).float()

    X_test_tensor = torch.from_numpy(np.array(matrix_data).astype(float)).float()
    y_test_tensor = torch.from_numpy(np.array(labels).astype(float)).float()

    trainset = TensorDataset(X_train_tensor, y_train_tensor)
    testset = TensorDataset(X_test_tensor, y_test_tensor)
    num_classes = 2
    inputs = 1
    return trainset, testset, inputs, num_classes


def getMyDataset_4(data, type="csv", size=(12, 32)):
    # 设置超参数
    size_h = 12
    size_w = 32
    a = pd.read_csv("data/TRAFFIC/occupancy.csv").to_numpy()[:, 1:size_w + 1]
    # b = pd.read_csv("data/TRAFFIC/speed.csv").to_numpy()[:, 1:size_w + 1]
    # c = pd.read_csv("data/TRAFFIC/volume.csv").to_numpy()[:, 1:size_w + 1]
    d = pd.read_csv("data/TRAFFIC/label.csv").to_numpy()

    # 处理成（-1，size_h，size_w）数据
    matrix_data = []
    labels = []
    for begin_index in range(len(a) - size_h):
        a1 = np.mean(a[begin_index:begin_index + 4, :], axis=0)
        a3 = np.mean(a[begin_index + 4:begin_index + 8, :], axis=0)
        a5 = np.mean(a[begin_index + 8:begin_index + 12, :], axis=0)
        a2 = (a1 + a3) / 2
        a4 = (a3 + a5) / 2

        matrix_data.append([np.array([a1, a2, a3, a4, a5])
                            ])
        # matrix_data.append([np.array([a1, a3, a5, a7, a9, a11])])
        temp = d[(begin_index + size_h - 2), :]
        labels.append([1] if (1 in temp) else [0])

    X = np.array(matrix_data).astype(float)
    matrix_data = X.reshape(-1, 1, 5, 32)
    labels = np.array(labels).flatten().astype(int)

    matrix_data, labels = utils.lower_sample_data(matrix_data, labels)

    # # 使用SMOTE算法进行过采样
    # from imblearn.over_sampling import SMOTE
    # oversample = SMOTE()
    # matrix_data = matrix_data.reshape([-1, 1*7 * 32])
    # os_X, os_labels = oversample.fit_resample(matrix_data, labels)
    # matrix_data = os_X.reshape([-1, 1, 7, 32])
    # labels = os_labels

    # 转tensor格式
    X_train_tensor = torch.from_numpy(np.array(matrix_data).astype(float)).float()
    y_train_tensor = torch.from_numpy(np.array(labels).astype(float)).float()

    X_test_tensor = torch.from_numpy(np.array(matrix_data).astype(float)).float()
    y_test_tensor = torch.from_numpy(np.array(labels).astype(float)).float()

    trainset = TensorDataset(X_train_tensor, y_train_tensor)
    testset = TensorDataset(X_test_tensor, y_test_tensor)
    num_classes = 2
    inputs = 1
    return trainset, testset, inputs, num_classes