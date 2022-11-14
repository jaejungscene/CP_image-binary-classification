import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "9"
import torch
import pandas as pd
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataset import get_transform
from PIL import Image
import numpy as np
from train import seed_everything



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


def main():
    seed_everything(0)
    test_set = TestDataset(transform=get_transform("test"))
    test_loader = DataLoader(dataset=test_set,
                            batch_size=16,
                            shuffle=False,
                            num_workers=4)
    path = "/home/ljj0512/private/workspace/CP_urban-datathon_X-ray/log/2022-11-10 23:52:37/checkpoint.pth.tar"
    checkpoint = torch.load(path)

    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
    model.fc = nn.Linear(in_features=512, out_features=1, bias=True)

    model.load_state_dict(checkpoint["state_dict"])
    model = nn.DataParallel(model).cuda()
    inference(model, test_loader)


def inference(model, test_loader):
    model.cuda()
    model.eval()
    preds = []
    threshold = 0.5
    submit = pd.read_csv("./1000_sample_submission.csv")
    with torch.no_grad():
        for img in (test_loader):
            img = img.cuda()
            output = model(img)
            output = output.squeeze(1).to('cpu')
            preds += output.cpu().tolist()
    
    preds = np.where(np.array(preds) > threshold, 1, 0)
    submit['result'] = preds
    submit.to_csv('./1000_submission.csv', index=False)


if __name__ == '__main__':
    main()