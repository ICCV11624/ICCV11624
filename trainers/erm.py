import argparse
import os, pdb
import logging
from PIL import Image

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.distributions import Categorical
    
from torch.utils.tensorboard import SummaryWriter

class ErmTrainer():
    
    def __init__(self, args, model, loaders, optimizer, num_classes):
        print(self)
        self.args = args
        self.model = model 
        self.loaders = loaders
        self.optimizer = optimizer
        
        self.max_epoch = args['max_epoch']
        self.batch_size = args['batch_size']

        self.k = args['k']
        self.num_classes = num_classes
        self.num_groups = np.power(self.num_classes, len(self.args['bias_attrs'])+1)
        self.num_clusters = self.k
        
        self.writer = SummaryWriter(os.path.join(args['checkpoint_dir'],  args['trainer'], 'logs'))
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

        self.best_acc = 0.
        self.save_acc = False
        self.test_best_acc = 0.
    
    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        

    def save_model(self, name, epoch):
        torch.save({
            'epoch': epoch,
            'state_dict': self.model.state_dict(), 
            'optimizer' : self.optimizer.state_dict(),
        }, self.checkpoint_dir / '{}_cfix.pth'.format(name))
        return


    def _extract_features(self):
        self.model.eval()
        envs = ['train_eval', 'valid', 'test']
        features, targets = {}, {}
        ids = {}
        for split in envs:
            data_loader = self.loaders[split]
            features_split = []
            id = []
            t = []
            with torch.no_grad():
                for data, target, _, _, index  in tqdm(data_loader, desc='Feature extraction for clustering..', ncols=5):
                    data = data.cuda()
                    results = self.model(data.detach())
                    features_split.append(results["feature"].cpu().detach())
                    id.append(index.cpu().detach())
                    t.append(target.cpu().detach())

                features[split] = torch.cat(features_split).cpu().detach()
                ids[split] = torch.cat(id).cpu().detach()
                targets[split] = torch.cat(t).cpu().detach()
                
                os.makedirs(os.path.join(self.checkpoint_dir,'features', split), exist_ok=True)
                os.makedirs(os.path.join(self.checkpoint_dir,'ids', split), exist_ok=True)
                os.makedirs(os.path.join(self.checkpoint_dir,'targets', split), exist_ok=True)

                torch.save(features[split], os.path.join(self.checkpoint_dir,'features', split,'tensor.pt'))
                torch.save(ids[split], os.path.join(self.checkpoint_dir,'ids', split,'tensor.pt'))
                torch.save(targets[split], os.path.join(self.checkpoint_dir,'targets', split,'tensor.pt'))
        return 

    def train(self, epoch):
        
        data_loader = self.loaders['train']
        total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader, ncols=100)
    
        
        for data, target, _, _, _, in train_bar:
            B = target.size(0)
                
            data, target = data.cuda(), target.cuda()
            
            results = self.model(data)
            loss = torch.mean(self.criterion(results["out_original"], target.long()))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_num += B
            total_loss += loss.item() * B

        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, self.max_epoch, total_loss / total_num))
        self.writer.add_scalar('Loss/train', total_loss / total_num, epoch)
        
        return 

    def test(self, epoch, train_eval=True):
        self.model.eval()
            
        test_envs = ['valid', 'test']
        for desc in test_envs:
            loader = self.loaders[desc]
            
            total_top1, total_top5, total_num, test_bar = 0.0, 0.0, 0, tqdm(loader, ncols=100)
            num_groups = loader.dataset.num_groups
            
            bias_counts = torch.zeros(num_groups).cuda()
            bias_top1s = torch.zeros(num_groups).cuda()
            best_loss = 0.0
            total_loss = 0.0
            
            with torch.no_grad():
                
                for data, target, biases, group, ids in test_bar:
                    data, target, biases, group = data.cuda(), target.cuda(), biases.cuda(), group.cuda()
                    
                    B = target.size(0)

                    results = self.model(data)
                    pred_labels = results["out_original"].argsort(dim=-1, descending=True)
                    loss = torch.mean(self.criterion(results["out_original"], target.long()))
                    total_loss += loss.item() * B

                    top1s = (pred_labels[:, :1] == target.unsqueeze(dim=-1)).squeeze().unsqueeze(0)
                    group_indices = (group==torch.arange(num_groups).unsqueeze(1).long().cuda())
                    
                    bias_counts += group_indices.sum(1)
                    bias_top1s += (top1s * group_indices).sum(1)
                    
                    
                    total_num += B
                    total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                    total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                    acc1 = total_top1 / total_num * 100
                    bias_accs = bias_top1s / bias_counts * 100
                    
                    avg_acc = np.nanmean(bias_accs.cpu().numpy())
                    worst_acc = np.nanmin(bias_accs.cpu().numpy())
                
                
                self.writer.add_scalar('Loss_original/'+desc, total_loss / total_num, epoch)
                self.writer.add_scalar('Unbiased_avg_acc/'+desc, avg_acc, epoch)
                self.writer.add_scalar('Worst_acc/'+desc, worst_acc, epoch)
                self.writer.add_scalar('Overall_Acc/'+desc, acc1, epoch)

                if desc == 'valid':
                    if (epoch==self.max_epoch):
                        self.save_model('last', epoch=epoch)
                    
                    if acc1 > self.best_acc:
                        self.best_acc = acc1
                        self.save_model('best_acc', epoch)
                        self.save_acc = True
                        
                if desc=='test':
                    if self.save_acc:
                        self.test_best_acc = acc1
                        self.save_acc = False   
                        self.test_avg_acc = avg_acc    
                        self.test_worst_acc = worst_acc

            for i,acc in enumerate(bias_accs):
                self.writer.add_scalar('Unbiased_acc_'+str(i)+'/'+desc, acc, epoch)                    
        self.model.train()
        
        return
