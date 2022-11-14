import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
from torch import LongTensor
import pandas as pd 
import typing as ty
import yaml
import numpy as np
from dataset import *
from utils import *
import random
from torchvision.models import  EfficientNet_B6_Weights, EfficientNet_B0_Weights, EfficientNet_B7_Weights
import torchvision.models as models

import wandb
import torch.optim as optim


class TestDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.dir = "/home/ljj0512/private/workspace/CP_urban-datathon_X-ray/test"
    
    def __len__(self):
        return len(os.listdir(self.dir))

    def __getitem__(self, index):
        img_path = sorted(os.listdir(self.dir))[index]
        image = self.transform(Image.open(os.path.join(self.dir,img_path)))
        return image

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def model_infer():
    seed_everything(0) # Seed 고정

    test_set = TestDataset(transform=get_transform("test"))
    dl_test = DataLoader(dataset=test_set,
                            batch_size=1,
                            shuffle=False,
                            num_workers=4)

    submit = pd.read_csv("/home/ljj0512/private/workspace/CP_urban-datathon_X-ray/1000_sample_submission.csv")
    model_path = "/home/ljj0512/private/workspace/CP_urban-datathon_X-ray/log/2022-11-10 23:52:37/checkpoint.pth.tar"

    model = models.efficientnet_b6(weights=EfficientNet_B6_Weights.DEFAULT)
    model.features[0][0] = nn.Conv2d(1, 56, kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=False)
    model.classifier[1] = nn.Linear(in_features=2304, out_features=1, bias=True)
    model.load_state_dict(torch.load(model_path)["state_dict"])       
    model.cuda()
    model = nn.DataParallel(model)

    model.eval()
    model_preds = []
    threshold = 0.5
    with torch.no_grad():
        for img in dl_test:
            print(img.shape)
            img = img.float().cuda()
            model_pred = model(img)
            model_pred = model_pred.squeeze(1).to('cpu')
            model_preds += model_pred.cpu().tolist()
    
    model_preds = np.where(np.array(model_preds) > threshold, 1, 0)
    submit['result'] = model_preds
    print(len(model_preds))
    submit.to_csv('/home/ljj0512/private/workspace/CP_urban-datathon_X-ray/submit13.csv', index=False)


if __name__ == '__main__':
    model_infer()