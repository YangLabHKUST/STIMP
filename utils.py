import numpy as np
import os
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class StandardScaler:
    def transform(self, x):
        self.mean = np.nanmean(x)
        self.std = np.nanstd(x)
        return (x-self.mean)/(self.std+1e-7)
    def inverse_transform(self, x):
        return x*self.std + self.mean 

class LogScaler:
    def transform(self, x):
        return np.log10(x)
    def inverse_transform(self, x):
        return 10**x

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    return True

def masked_mae(preds, labels, mask):

    residual = torch.abs(labels - preds) * mask 
    num_eval = mask.sum()
    mae= residual.sum() / (num_eval if num_eval > 0 else 1)

    return mae 

def masked_mse(preds, labels, mask):

    residual = (labels - preds) * mask 
    num_eval = mask.sum()
    mse= (residual**2).sum() / (num_eval if num_eval > 0 else 1)

    return mse

def masked_cor(preds, labels, mask):

    mask = ~mask.bool()
    observed_preds = torch.masked.masked_tensor(preds, mask)
    observed_data= torch.masked.masked_tensor(labels, mask)
    vx = observed_preds - observed_preds.mean(1).unsqueeze(1)
    vy = observed_data - observed_data.mean(1).unsqueeze(1)
    corr = torch.sum(vx*vy, dim = 1)/(torch.sqrt(torch.sum(vx**2, dim = 1))*torch.sqrt(torch.sum(vy**2, dim = 1)))

    return corr


def masked_huber_loss(preds, labels, mask):
    delta = 1.
    mask = mask.float()
    huber_mse = 0.5 * (preds - labels)**2
    huber_mae = delta * (torch.abs(preds - labels) - 0.5 * delta)
    loss = torch.where(torch.abs(preds - labels)<=delta, huber_mse, huber_mae)
    loss = loss * mask
    return torch.sum(loss)/torch.sum(mask)

def huber_loss(preds, labels):
    delta = 0.6
    huber_mse = 0.5 * (preds - labels)**2
    huber_mae = delta * (torch.abs(preds - labels) - 0.5 * delta)
    loss = torch.where(torch.abs(preds - labels)<=delta, huber_mse, huber_mae)
    return loss
    
def mse_loss(preds, labels):
    mse = (preds-labels)**2
    return mse
    

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
