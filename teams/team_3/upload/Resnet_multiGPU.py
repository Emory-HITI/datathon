#!/usr/bin/env python
# coding: utf-8

## useful link: https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51


import os
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, utils
import random 
import SimpleITK as sitk
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models.resnet import resnet18
from torchvision.models import ResNet18_Weights
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch.multiprocessing as mp



class Embed_data(Dataset):

    def __init__(self, csv_file, transform0=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.            
            transform: transform to be applied on a sample.
        """
        self.info = csv_file
        self.transform0 = transform0        
        
        self.targets = F.one_hot(torch.tensor(list(self.info['cancer'])))
        
    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()            

        img_path0 = self.info['new_path'][idx]        
        image0 = sitk.ReadImage(img_path0, sitk.sitkFloat32)        
        
        im_arr0 = sitk.GetArrayFromImage(image0)
        
        im_arr0 = im_arr0 + abs(im_arr0.min())
        im_arr0 = im_arr0 / im_arr0.max()
        im_tsr0 = torch.from_numpy(np.expand_dims(im_arr0, axis=0))
        lbls = self.targets[idx]      

        if self.transform0 :
            im_tsr0 = self.transform0(im_tsr0)     
                       
        else:
            im_tsr0 = im_tsr0            
            
        return im_tsr0.float(), lbls

def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def setup(rank, world_size,  master_addr, master_port):

    #print('master_port', master_port)
    os.environ['MASTER_ADDR'] = master_addr #'localhost'
    os.environ['MASTER_PORT'] = master_port  
    os.environ['NCCL_P2P_DISABLE'] = '1' # disable p2p
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    

def prepare(dataset, rank, world_size, batch_size=2, pin_memory=False, num_workers=0):
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=True, shuffle=False, sampler=sampler)
    
    return dataloader


def cleanup():
    dist.destroy_process_group()


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=1, bias=False)
        
        self.model.fc = nn.Sequential(nn.Linear(512, 2), nn.Softmax(dim=1))
                
    def forward(self, x):
        x = self.model(x) 
        
        return x    


# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer, device, epoch, epochs):
    net.train()
    total_loss, total_acc_cla, total_num, train_bar = 0.0, 0.0, 0, tqdm(data_loader, disable= (device!=0))
   
    for pos_1, target in train_bar:
        
        pos_1, target = pos_1.to(device), target.to(device)
        
        cla1 = net(pos_1)
       
        # Class loss
        Cla_loss = nn.CrossEntropyLoss()
        loss = Cla_loss(cla1, target.float())
        #if device == 0:
        #    print('loss', loss)
        
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += pos_1.size(0)
        total_loss += loss.clone().detach()  * pos_1.size(0)
        
        # labels predicted by the classifier          
        pred_labels_cla1 = torch.argmax(cla1.clone().detach().round(), dim=1).unsqueeze(dim=-1)
        aa = torch.sum((pred_labels_cla1 == torch.argmax(target, dim=1).unsqueeze(dim=-1)).any(dim=-1).float())    
        total_acc_cla += aa      
        
        avg_loss = total_loss/total_num
        avg_acc = total_acc_cla/total_num*100
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} ACC: {:.2f}%'.format(epoch, epochs, avg_loss, avg_acc ))
        
    return avg_loss, avg_acc 


def validate(net, data_loader, device, epoch, epochs):
    net.eval()
    with torch.no_grad():
        total_loss, total_acc_cla, total_num, val_bar = 0.0, 0.0, 0, tqdm(data_loader, disable= (device!=0))
        y_pred = []
        y_true = []
        i = 0
        for pos_1, target in val_bar:
            pos_1, target = pos_1.to(device), target.to(device)            
            cla1 = net(pos_1)

            # Class loss
            Cla_loss = nn.CrossEntropyLoss()
            loss = Cla_loss(cla1, target.float())

            total_num += pos_1.size(0)
            total_loss += loss.clone().detach() * pos_1.size(0)
            
            # labels predicted by the classifier                       
            pred_labels_cla1 = torch.argmax(cla1.clone().detach().round(), dim=1).unsqueeze(dim=-1)
            target_temp = torch.argmax(target, dim=1).unsqueeze(dim=-1)
            aa = torch.sum((pred_labels_cla1 == target_temp).any(dim=-1).float())   
            total_acc_cla += aa 
                        
            avg_loss = total_loss/total_num
            avg_acc = total_acc_cla/total_num*100
            val_bar.set_description('Validation Epoch: [{}/{}] Loss: {:.4f} ACC: {:.2f}%'.format(epoch, epochs, avg_loss, avg_acc))
                    
        
    return avg_loss, avg_acc


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def main(rank, csv_train, csv_val, batch_size, epochs, lr, world_size, master_addr, master_port, f):
    # setup the process groups
    setup(rank, world_size, master_addr, master_port)
    print('rank{} working'.format(rank))
    
    # prepare the dataloader 
    dataset_train = Embed_data(csv_train, transform0 = transforms.Compose([transforms.Resize((256,256))]))
    dataset_val = Embed_data(csv_val, transform0 = transforms.Compose([transforms.Resize((256,256))]))

    dataloader_train = prepare(dataset_train, rank, world_size, batch_size)
    dataloader_val = prepare(dataset_val, rank, world_size, batch_size)

    # instantiate the model and move it to the right device
    model = Model().to(rank)

    # wrap the model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)     
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)

    loss_list = [torch.tensor(0.0).to(rank) for _ in range(world_size)]
    acc_list = [torch.tensor(0.0).to(rank) for _ in range(world_size)]
    loss_list_val = [torch.tensor(0.0).to(rank) for _ in range(world_size)]
    acc_list_val = [torch.tensor(0.0).to(rank) for _ in range(world_size)]
    
    if rank == 0:
        output_dict = {}
        best_val_loss = 99999.0
    
    for epoch in range(1, epochs + 1):

        # if we are using DistributedSampler, we have to tell it which epoch this is
        dataloader_train.sampler.set_epoch(epoch)  
        dataloader_val.sampler.set_epoch(epoch)  

        avg_loss, avg_acc = train(model, dataloader_train, optimizer, rank, epoch, epochs)
        avg_loss_val, avg_acc_val = validate(model, dataloader_val, rank, epoch, epochs)
        
        if rank == 0:
            
            dist.gather(avg_loss, gather_list=loss_list)
            dist.gather(avg_acc, gather_list=acc_list)
            
            dist.gather(avg_loss_val, gather_list=loss_list_val)
            dist.gather(avg_acc_val, gather_list=acc_list_val)  
            
            val_loss = np.mean([i.item() for i in loss_list_val])
            
            output_dict[epoch] = {'train_loss': np.mean([i.item() for i in loss_list]),
                                  'val_loss': val_loss,
                                  'train_acc': np.mean([i.item() for i in acc_list]),
                                  'val_acc': np.mean([i.item() for i in acc_list_val])}
                               
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'results/epoch{}-{}_fold{}_model.pth'.format(epochs, epoch, f))
            
        else:
            dist.gather(avg_loss, dst=0)
            dist.gather(avg_acc, dst=0)
            
            dist.gather(avg_loss_val, dst=0)
            dist.gather(avg_acc_val, dst=0)            
    if rank == 0:
        #print(output_dict)        
        pd.DataFrame(output_dict).T.to_csv('output_F{}_{}-{}.csv'.format(f, epochs, epoch))
        torch.save(model.state_dict(), 'results/epoch{}-{}_fold{}_model.pth'.format(epochs, epoch, f))
                               
    cleanup()



if __name__ == '__main__':
    
    csv_file_path = './label_train.csv'
    kfold = 5 # for cross-validation
    
    # Parameters
    batch_size = 32 #Number of images in each mini-batch
    epochs = 20 #Number of sweeps over the dataset to train
    lr = 2.5e-4 #learning rate
    world_size = 4 # number of gpus
    
    # suppose we have 3 gpus
    print('Read CSV File...')
    csv_all = pd.read_csv(csv_file_path)
    
    idxs = [i for i in range(len(csv_all))]
    random.shuffle(idxs)
    splits = np.array_split(idxs, kfold)
    
    for f in range(kfold):
        train_idx = []
        for i in range(kfold):
            if i == f:
                val_idx = list(splits[i])
            else:
                train_idx += list(splits[i])

        csv_train = (csv_all.iloc[train_idx]).reset_index()
        csv_val = (csv_all.iloc[val_idx]).reset_index()  
        print('=============Fold {}=============='.format(f))
        print('train data size:', len(csv_train))
        print('validation data size:', len(csv_val))
        
        mp.spawn(main,
            args=[csv_train, csv_val, batch_size, epochs, lr, world_size, 'localhost', find_free_port(), f],
            nprocs=world_size)