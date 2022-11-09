import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import time
import torch.nn as nn
import numpy as np
from sklearn import metrics
from torchvision.models import EfficientNet_B6_Weights, efficientnet_b6
import torchvision.models as models
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        img_path = self.df["image_path"].iloc[index]
        image = self.transform(Image.open(img_path))
        label = self.df["label"].iloc[index]
        return [image, label]


train_df = pd.read_csv("train_df.csv")
train_set = CustomDataset(train_df, transform=transforms.Compose([
                        transforms.Resize((32,32)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]))

train_loader = DataLoader(dataset=train_set,
                                batch_size=16,
                                shuffle=True,
                                num_workers=4)
                            
model = models.efficientnet_b6(weights=EfficientNet_B6_Weights.DEFAULT)
model.classifier[1] = nn.Linear(in_features=2304, out_features=5, bias=True)

model.cuda()
model.eval()
pred_labels = []
true_labels = []
total = 0
correct = 0
threshold = 0.5
criterion = nn.CrossEntropyLoss()

for i, (input, target) in enumerate(train_loader):
    target = target.cuda()
    input = input.cuda()
    true_labels += target.tolist()

    output = model(input)
    loss = criterion(output, target)

    # measure accuracy and record loss
    _, predicted = torch.max(output.data, dim=1)
    total += target.size(0)
    correct += predicted.eq(target.data).cpu().sum()
    output = output.squeeze(1).to('cpu')
    pred_labels += output.tolist()


pred_labels = np.where(np.array(pred_labels) > threshold, 1, 0)
f1_score = metrics.f1_score(y_true=true_labels, y_pred=pred_labels, average='macro')
print(f1_score)