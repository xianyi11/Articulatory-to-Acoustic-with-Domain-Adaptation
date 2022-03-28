import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as io_wav
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm

# Parameters of old / simple continouos vocoder
frameLength = 512 # 23 ms at 22050 Hz sampling
frameShift = 180 # 12 ms at 22050 Hz sampling, correspondong to 81.5 fps (ultrasound)
order = 24
n_mgc = order + 1

mome = 1e-7

# WaveGlow/STFT parameters
samplingFrequency = 22050
n_melspec = 80
hop_length_UTI = 181 # sampling rate of ultrasound files
 
# parameters of ultrasound images
framesPerSec = 22050 / 181
n_lines = 63
n_pixels = 412
n_pixels_reduced = 103

# training parameters
device = torch.device("cuda:0")
model_dir = '..//..//models//' # path to save models
dir_data = '..//..//data//' # path to load dataset
train_params = {'epoches': 7, 'lambda_domain': 0.4, 'batch_size': 512}
lr = 1e-5
mom = 1e-7

class Net(nn.Module):
    def __init__(self, n_melspec, cat_num):
        
        super(Net, self).__init__()
        self.resnet = models.resnet50(pretrained = True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1024)
        
        self.Predict1 = nn.Linear(1024, 256)
        self.Predict2 = nn.Linear(256, n_melspec)
        
        self.Domain1 = nn.Linear(1024, 256)
        self.Domain2 = nn.Linear(256, cat_num)
        
        self.ReLU = nn.ReLU()
        self.Flatten = nn.Flatten()
        
        for para in self.resnet.parameters():
            para.requires_grad = True
    
    def forward(self, X):
        FC = self.ReLU(self.resnet(X))
        
        Pre1 = self.ReLU(self.Predict1(FC))
        mel_pre = self.Predict2(Pre1)
        
        Dom1 = self.ReLU(self.Domain1(FC))
        dom_pre = self.Domain2(Dom1)
        
        return mel_pre, dom_pre
    
    def freezeCNN(self):
        for para in self.resnet.parameters():
            para.requires_grad = False
        
        return 'freezing done'
    
    def unfreezeCNN(self):
        for para in self.resnet.parameters():
            para.requires_grad = True
        
        return 'unfreezing done'

# SO represents Source-Only method
# ST represente ST-Adversarial method
# ID represents ID-Adversarial method

def load_data(dir = dir_data, ds_name = 'SpeakerSep', domain_type = 'ID'):

    if domain_type == 'ST':
        num_dataset = ['train', 'valid']
    elif domain_type == 'ID' or domain_type == 'SO':
        num_dataset = ['train']
    else:
        print('invalid dataset_name')
        return 0
    
    dir_tv = dir + ds_name + '//'
    
    X = []
    y = []
    for train_valid in num_dataset:
        X.append(torch.load(dir_tv + train_valid + '_ult.pt')
        y.append(torch.load(dir_tv + train_valid + '_ult.pt')
    
    if domain_type == 'ST':
        y[0][: , -1] = 1
        y[1][: , -1] = 0
    
    X_data = torch.cat(X, dim = 0).type(torch.FloatTensor).repeat(3,1,1,1)
    X_data = X_data.transpose(0, 1)
    y_data = torch.cat(y, dim = 0).type(torch.FloatTensor)

    domain_cat = set(y_data[: , -1])
    cat_num = len(domain_cat)
    
    dataset = TensorDataset(X_data, y_data)
    
    return dataset, cat_num

def train(model, dataset, domain_type, optimizer, train_params):
    
    epoches = train_params['epoches']
    lambda_domain = train_params['lambda_domain']
    batch_size = train_params['batch_size']

    loss_fun = nn.MSELoss(reduce = False)
    loss_fun_dom = nn.CrossEntropyLoss()
    pre_losses = []
    losses = []
    
    for epoch in tqdm(range(epoches)):
        train_loader = DataLoader(dataset = dataset, batch_size = batch_size,
                                  shuffle = True, drop_last = True)
                                  
        for data in train_loader:
            
            # get ult, melspec and domain
            ult_train, y_train = data
            melspec_train = y_train[: , : -1].to(device)
            domain_train = y_train[: , -1].to(device)
            ult_train = ult_train.reshape((batch_size, 3, n_lines,
                                           n_pixels_reduced)).to(device)
            
            # forward
            mel_pre, dom_pre = model(ult_train)
            
            # calculate domain loss
            if domain_type == 'SO':
                domain_loss = 0
            else:
                domain_train = domain_train.type(torch.LongTensor).to(device)
                domain_loss = loss_fun_dom(dom_pre.to(device), domain_train)
                model.freezeCNN()
                domain_loss.backward(retain_graph=True)
                model.unfreezeCNN()
            
            # calculate prediction loss
            if domain_type == 'SO':
                pred_loss = loss_fun(mel_pre, melspec_train)
                pred_loss[domain_train == 0, : ] = 0
                pre_loss = pred_loss.sum() / torch.sum(domain_train) / n_melspec
            else:
                pre_loss = loss_fun(mel_pre, melspec_train).mean()
            
            # calculate total loss
            loss = pre_loss - lambda_domain * domain_loss
            loss.backward(retain_graph = False)
            
            optimizer.step()
            
            pre_losses.append(pre_loss.item())
            losses.append(loss.item())
            
    plt.plot(range(len(losses)), losses)
    plt.plot(range(len(pre_losses)), pre_losses)
    plt.show()
    
    return model

dataset, cat_num = load_data(ds_name = 'SpeakerSep', domain_type = 'ID')

model = Net(n_melspec, cat_num).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = mom)
print('model initialized')

model = train(model, dataset, domain_type, optimizer, train_params)

model_path = model_dir + ds_name + '_' + domain_type + '_' + str(lambda_domain)
torch.save(model, model_path)
