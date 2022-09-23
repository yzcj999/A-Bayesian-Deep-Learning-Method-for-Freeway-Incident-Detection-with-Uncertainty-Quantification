import torch
import utils
import pandas as pd
import numpy as np
from tqdm import tqdm
import config_bayesian as cfg
from models.BayesianLeNet import BBBLeNet
# from models.BayesianLeNet_0710 import BBBLeNet
# from models.BayesianLeNet_80s import BBBLeNet
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, TensorDataset


class Test():
    DR = 0
    FAR = 0
    MTTD = 0
    PI = 0
    score = 0
    def __init__(self):
        # 默认测试数据量是 n
        # y:[n, 2] --time, real_label
        # y:[n] --predict_label
        pass

    def score_DR(self, y, y_):
        right = 0.0
        sum = 0.0
        for i in range(len(y)):
            if y[i][1] == 1:
                sum += 1
                if y_[i] == 1:
                    right += 1
        self.DR = right / sum
        return self.DR

    def score_FAR(self, y, y_):
        wrong = 0.0
        sum = 0.0
        for i in range(len(y)):
            if y[i][1] == 0:
                sum+=1
                if y_[i] == 1:
                    wrong += 1
        self.FAR = wrong / sum
        return self.FAR

    def score_MTTD(self, y, y_):
        start = False
        sum = 0.0
        count = 0.0
        for i in range(1, len(y)):
            # a = datetime.strptime(y[i-1][0], '%Y-%m-%d %H:%M:%S')
            # b = datetime.strptime(y[i][0], '%Y-%m-%d %H:%M:%S')
            # if (b - a).seconds > 60:
            #     start = False
            #     continue
            if y[i-1][1] == 0 and y[i][1]==1:
                start = True
            if (y[i][1] == 0) or (y_[i] == 1):
            # if (y[i][1] == 1):
                start = False
            if start:
                sum += 1

            if y_[i] == 1 and y_[i-1]==0:
                count += 1
        self.MTTD = sum/count
        return self.MTTD

    def score_PI(self, y, y_):
        self.score_DR(y, y_)
        self.score_FAR(y, y_)
        self.score_MTTD(y, y_)
        self.PI = (1.01 - self.DR)*(self.FAR + 0.001)*self.MTTD
        return self.PI

    def score_S(self, y, y_):
        count = 0
        for i in range(len(y_)):
            if y[i][1] == y_[i]:
                count+=1
        self.score = count /len(y_)

    def getScore(self, y, y_):
        self.score_PI(y, y_)
        self.score_S(y, y_)
        return {
            "score": self.score,
            "DR":self.DR,
            "FAR":self.FAR,
            "MTTD":self.MTTD,
            "PI":self.PI
        }

    def predict(self, data_path, g_p=0, p_p=0, no="G"):
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

        # 添加噪音
        if no == "G" and g_p!=0:
            noise = utils.get_Normal_noise(matrix_data.shape, scale=g_p)
            matrix_data = matrix_data + noise
        elif no == "P" and p_p != 0:
            noise = utils.get_Poisson_noise(matrix_data.shape, lam=p_p)
            matrix_data = matrix_data + noise
        data = torch.from_numpy(matrix_data.astype(float))
        data = data.to(torch.float32)

        # 加载训练过的模型
        model = BBBLeNet(2, 3, cfg.priors)
        # state_dict = torch.load("checkpoints/"+data_path+"/bayesian/model_lenet_lrt_softplus.pt")
        state_dict = torch.load(r"D:\Papers\交通事件检测\TRAFFIC(第三版)\BCNN_V4\checkpoints\TRAFFIC\best_models\model_lenet_lrt_softplus.pt")
        model.load_state_dict(state_dict)
        model.eval()
        from uncertainty_estimation import get_uncertainty_per_batch as elv

        a, b, c = elv(model, data)
        print(f"aleatoric: {np.mean(a, axis=0)[0]}, epistemic: {np.mean(b, axis=0)[0]}")

        pre_y = []
        real_y = []
        #
        # li = np.linspace(0, 1013304, 100).astype(int)
        # torch.chunk(data, chunks=100, dim=1)
        list_of_tensor = torch.chunk(data, chunks=200, dim=0)
        res = []
        for i in list_of_tensor:
            tu, a = model(i)
            res.extend(tu.tolist())

        count  = 0
        for i in res:
            if i[0] > i[1]:
                classIndex = 0
            else:
                classIndex = 1
            pre_y.append(classIndex)
            real_y.append([count, labels[count]])
            count += 1
        print(Test().getScore(real_y, pre_y))



if __name__ == '__main__':
    a = np.array([[1,2], [3, 4]])
    b = np.array([[1, 2], [3, 4]])
    print(np.concatenate((a[:,0:1], b[:,0:1]), axis=1))