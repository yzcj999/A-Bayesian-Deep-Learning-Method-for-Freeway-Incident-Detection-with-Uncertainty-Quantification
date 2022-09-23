import torch
import numpy as np
from torch.nn import functional as F

# CUDA settings
import config_bayesian as cfg
device = cfg.device
mnist_set = None
notmnist_set = None
def get_uncertainty_per_batch(model, batch, T=15, normalized=False):
    batch_predictions = []
    net_outs = []
    batches = batch.unsqueeze(0).repeat(T, 1, 1, 1, 1)

    preds = []
    epistemics = []
    aleatorics = []
    for i in range(T):  # for T batches

        # net_out, _ = model(batches[i].cuda())
        li = np.linspace(0, len(batch), 50).astype(int)
        net_out = []
        for j in range(49):
            y_pred, _ = model(batches[i][li[j]:li[j+1]])
            net_out += y_pred.detach().numpy().tolist()
        net_out = torch.from_numpy(np.array(net_out))

        # net_out, _ = model(batches[i])
        net_outs.append(net_out)
        if normalized:
            prediction = F.softplus(net_out)
            prediction = prediction / torch.sum(prediction, dim=1).unsqueeze(1)
        else:
            prediction = F.softmax(net_out, dim=1)
        batch_predictions.append(prediction)
    
    for sample in range(batch.shape[0]):
        # for each sample in a batch
        pred = torch.cat([a_batch[sample].unsqueeze(0) for a_batch in net_outs], dim=0)
        pred = torch.mean(pred, dim=0)
        preds.append(pred)

        p_hat = torch.cat([a_batch[sample].unsqueeze(0) for a_batch in batch_predictions], dim=0).detach().cpu().numpy()
        p_bar = np.mean(p_hat, axis=0)

        temp = p_hat - np.expand_dims(p_bar, 0)
        epistemic = np.dot(temp.T, temp) / T
        epistemic = np.diag(epistemic)
        epistemics.append(epistemic)

        aleatoric = np.diag(p_bar) - (np.dot(p_hat.T, p_hat) / T)
        aleatoric = np.diag(aleatoric)
        aleatorics.append(aleatoric)

    epistemic = np.vstack(epistemics)  # (batch_size, categories)
    aleatoric = np.vstack(aleatorics)  # (batch_size, categories)
    preds = torch.cat([i.unsqueeze(0) for i in preds]).cpu().detach().numpy()  # (batch_size, categories)

    return aleatoric, epistemic, preds



