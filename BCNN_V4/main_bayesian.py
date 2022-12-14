from __future__ import print_function

import os
import numpy as np
import torch
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset, TensorDataset
from torch.autograd import Variable
from torch.nn import functional as F

import data
import utils
import metrics
import config_bayesian as cfg
# from models.BayesianLeNet_0710 import BBBLeNet
# from models.BayesianLeNet_40s import BBBLeNet
from models.BayesianLeNet_80s import BBBLeNet
from test.test import Test


# CUDA settings
device = cfg.device

def getModel(net_type, inputs, outputs, priors, layer_type, activation_type):
    if (net_type == 'lenet'):
        return BBBLeNet(outputs, inputs, priors, layer_type, activation_type)
    else:
        raise ValueError('Network should be either [LeNet / AlexNet / 3Conv3FC')


def train_model(net, optimizer, criterion, trainloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    """训练模型"""
    net.train()
    training_loss = 0.0
    accs = []
    kl_list = []
    for i, (inputs, labels) in enumerate(trainloader, 1):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = Variable(inputs, requires_grad=True)
        labels = Variable(labels, requires_grad=True)
        optimizer.zero_grad()

        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)
        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1)
        
        kl = kl / num_ens
        kl_list.append(kl.item())
        log_outputs = utils.logmeanexp(outputs, dim=2)

        beta = metrics.get_beta(i-1, len(trainloader), beta_type, epoch, num_epochs)
        loss = criterion(log_outputs, labels.long().flatten(), kl, beta)
        loss.backward()
        optimizer.step()
        accs.append(metrics.acc(log_outputs.data, labels.long().flatten()))
        training_loss += loss.cpu().data.numpy()
    return training_loss/len(trainloader), np.mean(accs), np.mean(kl_list)


def validate_model(net, criterion, validloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    """计算准确度"""
    net.train()
    valid_loss = 0.0
    accs = []

    for i, (inputs, labels) in enumerate(validloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)
        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1).data

        log_outputs = utils.logmeanexp(outputs, dim=2)
        # x_max, _ = torch.max(x, dim, keepdim=True)
        beta = metrics.get_beta(i-1, len(validloader), beta_type, epoch, num_epochs)
        valid_loss += criterion(log_outputs, labels.long().flatten(), kl, beta).item()

        accs.append(metrics.acc(log_outputs, labels.long().flatten()))
        # y = labels.long().flatten().reshape(-1,1)
        # y_ = log_outputs.cpu().numpy().argmax(axis=1).reshape(-1,1)
        #
        # print(y.shape, y_.shape)
        # print(test.pre(y, y_))
    return valid_loss/len(validloader), np.mean(accs)



def run(dataset, net_type):
    # 超参数设置
    layer_type = cfg.layer_type
    activation_type = cfg.activation_type
    priors = cfg.priors

    train_ens = cfg.train_ens
    valid_ens = cfg.valid_ens
    n_epochs = cfg.n_epochs
    lr_start = cfg.lr_start
    num_workers = cfg.num_workers
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size
    beta_type = cfg.beta_type

    trainset, testset, inputs, outputs = data.getMyDataset_4(dataset)

    train_loader, valid_loader, test_loader = data.getDataloader(
        trainset, testset, valid_size, batch_size, num_workers)

    net = getModel(net_type, inputs, outputs, priors, layer_type, activation_type).to(device)
    # state_dict = torch.load("checkpoints/TRAFFIC/bayesian/model_lenet_lrt_softplus.pt")
    # net.load_state_dict(state_dict)


    ckpt_dir = f'checkpoints/{dataset}/bayesian'
    ckpt_name = f'checkpoints/{dataset}/bayesian/model_{net_type}_{layer_type}_{activation_type}_80s.pt'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    criterion = metrics.ELBO(len(trainset)).to(device)
    optimizer = Adam(net.parameters(), lr=lr_start)
    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, verbose=True)
    valid_loss_max = np.Inf
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        train_loss, train_acc, train_kl = train_model(net, optimizer, criterion, train_loader, num_ens=train_ens, beta_type=beta_type, epoch=epoch, num_epochs=n_epochs)

        valid_loss, valid_acc = validate_model(net, criterion, valid_loader, num_ens=valid_ens, beta_type=beta_type, epoch=epoch, num_epochs=n_epochs)
        lr_sched.step(valid_loss)


        # print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f} \ttrain_kl_div: {:.4f}'.format(
        #     epoch, train_loss, train_acc, valid_loss, valid_acc, train_kl))
        if valid_loss_max != 0:
            with open("res.txt", "a+", encoding="utf-8") as f:
                f.write('{},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(epoch, train_loss, train_acc, valid_loss, valid_acc))
            
        print('{},{:.4f},{:.4f},{:.4f},{:.4f}'.format(
              epoch, train_loss, train_acc, valid_loss, valid_acc))
        # save model if validation accuracy has increased
        if valid_loss <= valid_loss_max:
            # print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            #     valid_loss_max, valid_loss))
            torch.save(net.state_dict(), ckpt_name)
            valid_loss_max = valid_loss

import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # 设置随机数种子
    setup_seed(20)
    # 训练
    run(dataset="TRAFFIC", net_type="lenet")

    # 评价指标
    cfg.device = "cpu"
    a = Test()
    a.predict("TRAFFIC")

    # for i in range(1, 7):
    #     a.predict("TRAFFIC",g_p=i/10, no="G")
    # print("_____________________________________")
    # for i in range(1, 7):
    #     a.predict("TRAFFIC",p_p=i, no="P")

