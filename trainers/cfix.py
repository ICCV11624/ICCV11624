import argparse
import os, pdb
from PIL import Image

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.distributions import Categorical
import torchvision

import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib import pyplot as plt
from kmeans_pytorch import kmeans, kmeans_predict

from torch.utils.tensorboard import SummaryWriter
from numpy import loadtxt
from sklearn.metrics import roc_auc_score
from pathlib import Path
import gdown

def entropy(p):
    p[p==0] = 1e-8 # sostituzione dei valori 0 con 1e-8
    return -torch.sum(p * torch.log2(p), dim=1)

class CfixTrainer():
    def __init__(self, args, model, loaders, optimizer, num_classes, t, beta, config):
        self.args = args
        self.model = model 
        self.loaders = loaders
        self.optimizer = optimizer
        
        self.max_epoch = args['max_epoch']
        self.batch_size = args['batch_size']

        self.num_classes = num_classes
        self.num_clusters = args['k']* self.num_classes
        self.num_groups = np.power(self.num_classes, len(self.args['bias_attrs'])+1)


        self.writer = SummaryWriter(os.path.join(args['checkpoint_dir'], args['trainer'], 'logs'))
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.criterion_p = torch.nn.CrossEntropyLoss(reduction='none')

        N = len(loaders['train'].dataset)
        self.pseudo_labels = {}
        self.pseudo_labels['train']= torch.zeros((N)).long().cuda() - 1
        self.corrects = torch.zeros((N)).long().cuda() - 1
        self.losses = torch.zeros((N)).cuda() - 1

        self.corrects_original = torch.zeros((N)).long().cuda() - 1
        self.losses_original = torch.zeros((N)).cuda() - 1
        
        self.weights = torch.ones((N)).cuda()

        N_val = len(loaders['valid'].dataset)
        N_test = len(loaders['test'].dataset)

        self.pseudo_labels['valid']= torch.zeros((N_val)).long().cuda() - 1
        self.pseudo_labels['test']= torch.zeros((N_test)).long().cuda() - 1

        self.losses_val = torch.zeros((N_val)).cuda() - 1
        self.corrects_val = torch.zeros((N_val)).long().cuda() - 1
        
        self.losses_val_original = torch.zeros((N_val)).cuda() - 1
        self.corrects_val_original = torch.zeros((N_val)).long().cuda() - 1

        self.cluster_losses = torch.zeros((self.num_clusters))
        self.cluster_accs = torch.zeros((self.num_clusters))

        self.cluster_losses_val = torch.zeros((self.num_clusters))
        self.cluster_accs_val = torch.zeros((self.num_clusters))

        self.cluster_losses_original = torch.zeros((self.num_clusters))
        self.cluster_losses_val_original = torch.zeros((self.num_clusters))
        self.cluster_accs_original = torch.zeros((self.num_clusters))
        self.cluster_accs_val_original = torch.zeros((self.num_clusters))
        self.momentum = args['momentum']
        self.best_avg_acc= 0.0
        self.best_worst = 0.0
        self.step = 1
        self.save_avg_acc = False
        self.save_worst_acc = False
        self.beta = beta
        self.t = t
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        self.config = config
        
        
    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
    
    def cluster_counts(self):
        return self.pseudo_labels['train'].bincount(minlength=self.num_clusters)
        
    def save_model(self, name, epoch):
        torch.save({
            'epoch': epoch,
            'state_dict': self.model.state_dict(), 
            'optimizer' : self.optimizer.state_dict(),
        }, self.checkpoint_dir / '{}_cfix.pth'.format(name))
        return
        
    
    def _extract_features(self, loaders):
        features, targets = {}, {}
        r = os.path.join(self.config["checkpoint_root"],self.config["dataset"],"self_supervised_"+str(self.config['self_supervised']))
        r_train =  os.path.join(r, "f_train.pth")
        r_val = os.path.join(r, "f_val.pth")
        r_test= os.path.join(r, "f_test.pth")
        if not os.path.exists(r):
            os.makedirs(r,  exist_ok=True)

        if self.args['dataset'] == 'celebA':
            if self.args['self_supervised']:
                url_train = "https://drive.google.com/file/d/1YQLHcBOHZQOHTypbD_RpkbmMxY4A2eq5/view?usp=share_link"
                url_val = "https://drive.google.com/file/d/1rrCnK2ehwDcoi7H_eENJt2ipW5eanNsV/view?usp=share_link"
                url_test = "https://drive.google.com/file/d/1sj3cykK2or-heYeGWN8rOEGNrN9yQFnh/view?usp=share_link"
            else:
                url_train = "https://drive.google.com/file/d/1kjIKg0WVRgQhkXObpR7k9DCEnhA_Nghd/view?usp=share_link"
                url_val = "https://drive.google.com/file/d/1wMPlQN9vXX6Wt8g6oy_HOjVZKvvGXnw-/view?usp=share_link"
                url_test = "https://drive.google.com/file/d/1IA-F3hcjoXjcZYQacFQ_CJGTyAa4q5aH/view?usp=share_link"

            target_index = loaders['train_eval'].dataset.attr_names.index(self.args['target_attr'])
            targets['train_eval'] = loaders['train_eval'].dataset.attr[:, target_index]
            targets['valid'] = loaders['valid'].dataset.attr[:, target_index]
            targets['test'] = loaders['test'].dataset.attr[:, target_index]

        elif self.args['dataset'] == 'waterbirds':
            url_train = "https://drive.google.com/file/d/1DsdNYw26XRQEPJxGMqrIzFGOo6FC0woj/view?usp=share_link"
            url_val = "https://drive.google.com/file/d/1ZEOf-vuPON1g-Npz4BGjVW2UqDbo-sfb/view?usp=share_link"
            url_test = "https://drive.google.com/file/d/1bnHUPkeYxzeS-guCfjonvGaEKDbVd8OA/view?usp=share_link"

            targets['train_eval'] = torch.tensor(loaders['train_eval'].dataset.labels)
            targets['valid'] = torch.tensor(loaders['valid'].dataset.labels)
            targets['test'] = torch.tensor(loaders['test'].dataset.labels)

        if not os.path.exists(r_train):
            gdown.download(url=url_train, output = r_train, quiet=False, fuzzy=True)
            gdown.download(url=url_val, output = r_val, quiet=False, fuzzy=True)
            gdown.download(url=url_test, output = r_test, quiet=False, fuzzy=True)

        features['train_eval'] = torch.load(r_train, map_location=self.device)
        features['train_eval'] = torch.nn.functional.normalize(features['train_eval'])

        features['valid'] = torch.load(r_val, map_location=self.device)
        features['valid'] = torch.nn.functional.normalize(features['valid'])

        features['test'] = torch.load(r_test, map_location=self.device)
        features['test'] = torch.nn.functional.normalize(features['test'])
            
        return features, targets
    
    def _cluster_features(self, features, targets):
        envs = ['train_eval', 'valid', 'test']

        clusters = self.args['k']
        for t in range(self.num_classes):
            cluster_centers = None
            print('CLUSTERING', flush=True)
            for env in envs:
                target_assigns = (targets[env]==t).nonzero().squeeze()
                feautre_assigns = features[env][target_assigns]
                if env == 'train_eval':
                    cluster_ids, cluster_centers = kmeans(X=feautre_assigns, num_clusters=clusters, distance='cosine', device=self.device)
                else:
                    cluster_ids = kmeans_predict(feautre_assigns, cluster_centers, 'cosine', device=self.device)
                
                cluster_ids_ = cluster_ids + t*self.args['k']

                if env == 'train_eval':
                    self.pseudo_labels['train'][target_assigns] = cluster_ids_.cuda()
                else:
                    self.pseudo_labels[env][target_assigns] = cluster_ids_.cuda()

        os.makedirs(os.path.join(self.checkpoint_dir,self.args['trainer'],'pseudo_labels'), exist_ok=True)
        os.makedirs(os.path.join(self.checkpoint_dir,self.args['trainer'],'pseudo_labels_val'), exist_ok=True)
        torch.save(self.pseudo_labels['train'], os.path.join(self.checkpoint_dir, self.args['trainer'], 'pseudo_labels' ,'tensor.pt'))
        torch.save(self.pseudo_labels['valid'], os.path.join(self.checkpoint_dir, self.args['trainer'], 'pseudo_labels_val' ,'tensor.pt'))
        return

    def clustering(self):
        self.model.eval()
        
        with torch.no_grad():
            features, targets = self._extract_features(self.loaders)
            self._cluster_features(features, targets)

        self.model.train()                 
        return 
    
    def get_cluster_weights(self, ids):
        
        cluster_counts = self.cluster_counts()
        cluster_weights = cluster_counts.sum()/(cluster_counts.float())
        assigns_id = self.pseudo_labels['train'][ids]

        if ( self.losses >= 0).nonzero().size(0) > 0:
            cluster_losses_ = self.cluster_losses.view(-1)
            losses_weight = cluster_losses_.float()/cluster_losses_.sum()
            weights_ = losses_weight[assigns_id].cuda() *cluster_weights[assigns_id].cuda()
            weights_ /= weights_.mean()
            weights_ids_ = (1-self.momentum) * self.weights[ids] + self.momentum * weights_
            self.weights[ids] = weights_ids_ 
            weights_ids_ /= weights_ids_.mean()
        else:
            weights_ids_ = self.weights[ids]
            weights_ids_ /= weights_ids_.mean()
        
        return weights_ids_
    
    def compute_weight_cluster(self):
        cluster_counts = self.cluster_counts()
        cluster_weights = cluster_counts.sum()/(cluster_counts.float()).cuda()

        cluster_losses_ = self.cluster_losses.view(-1)
        cluster_losses = (cluster_losses_.float()/cluster_losses_.sum()).cuda()
        weights = cluster_losses*cluster_weights.cuda()

        return weights / weights.mean()

    def compute_stats(self):

        for k in range(self.num_clusters): 
            l = k
            ids = (self.pseudo_labels['train']==l).nonzero()
            
            if ids.size(0) == 0:
                continue

            corrs = self.corrects[ids]
            corrs_nz = (corrs>=0).nonzero()

            corrs_original = self.corrects_original[ids]
            corrs_nz_original = (corrs_original>=0).nonzero()

            if corrs_nz.size(0) > 0:
                self.cluster_accs[k] = corrs[corrs_nz[:, 0]].float().mean(0)
                self.cluster_accs_original[k] = corrs_original[corrs_nz_original[:, 0]].float().mean(0)

            losses = self.losses[ids]
            loss_nz = (losses>=0).nonzero()
            
            losses_original = self.losses_original[ids]
            loss_nz_original = (losses_original>=0).nonzero()
            
            if loss_nz.size(0) > 0:
                if self.args['trainer']=='cfix_split_plus':
                    self.cluster_losses[k] = self.beta*(losses[loss_nz[:, 0]].float().mean(0)) + losses_original[loss_nz_original[:, 0]].float().mean(0)
                else:
                    self.cluster_losses[k] = losses[loss_nz[:, 0]].float().mean(0)
                self.cluster_losses_original[k] = losses_original[loss_nz_original[:, 0]].float().mean(0)
                if self.step % 50 ==0:
                    self.writer.add_histogram("pseudo_loss_in_cluster"+str(k), losses[loss_nz[:, 0]].float(), self.step)
                    self.writer.add_histogram("target_loss_in_cluster"+str(k), losses_original[loss_nz_original[:, 0]].float(), self.step)
                    
        return

    def compute_stats_val(self):
        for k in range(self.num_clusters): 
            l = k
            
            ids = (self.pseudo_labels['valid']==l).nonzero()
            if ids.size(0) == 0:
                continue

            corrs = self.corrects_val[ids]
            corrs_nz = (corrs>=0).nonzero()

            corrs_original = self.corrects_val_original[ids]
            corrs_nz_original = (corrs_original>=0).nonzero()

            if corrs_nz.size(0) > 0:
                self.cluster_accs_val[k] = corrs[corrs_nz[:, 0]].float().mean(0)
                self.cluster_accs_val_original[k] = corrs_original[corrs_nz_original[:, 0]].float().mean(0)

            losses_val = self.losses_val[ids]
            loss_nz = (losses_val>=0).nonzero()

            losses_val_original = self.losses_val_original[ids]
            loss_nz_original = (losses_val_original>=0).nonzero()

            if loss_nz.size(0) > 0:
                self.cluster_losses_val[k] = losses_val[loss_nz[:, 0]].float().mean(0)
                self.cluster_losses_val_original[k] = losses_val_original[loss_nz_original[:, 0]].float().mean(0)
   
        return
    
    def update(self, results, pseudo, target, ids):

        #loss pseudo 
        logits = results['out_pseudo']
        losses = self.criterion_p(logits, pseudo.long())
        self.losses[ids] = losses.detach()
        loss_pseudo = losses.mean()
        corrects = (logits.argmax(1) == pseudo).long()
        self.corrects[ids] = corrects
        
        #loss target
        out = results["out_original"]
        corrects = (out.argmax(1) == target).long()
        self.corrects_original[ids] = corrects

        losses_original = self.criterion(out, target.long()) 
        self.losses_original[ids] = losses_original.detach()
        self.compute_stats()

        with torch.no_grad():
            weight = self.get_cluster_weights(ids)
        
        return torch.mean(losses_original*weight), loss_pseudo

    def update_val(self, results, pseudo, target, ids):

        #loss pseudo classe 0

        logits = results['out_pseudo']
        losses = self.criterion_p(logits, pseudo.long())
        self.losses[ids] = losses.detach()
        loss_pseudo = losses.mean()
        corrects = (logits.argmax(1) == pseudo).long()
        self.corrects_val[ids] = corrects

        #loss target
        out = results["out_original"]
        corrects_val = (out.argmax(1) == target).long()
        self.corrects_val_original[ids] = corrects_val
        losses = self.criterion(out, target.long()).detach()
        self.losses_val_original[ids] = losses

        return torch.mean(losses), loss_pseudo

    
    def print_log(self,total_loss=0., total_loss_pseudo=0., total_num=0., epoch=1, split='train'):
        with torch.no_grad():
            if split=='train':
                self.writer.add_scalar('Loss_original/train', total_loss / total_num, epoch)
                self.writer.add_scalar('Loss_pseudo/train', total_loss_pseudo / total_num, epoch)

                for k in range(self.num_clusters):
                    weights_cluster = self.compute_weight_cluster()
                    self.writer.add_scalar('Weight cluster/'+ str(k), weights_cluster[k], epoch)     
                    self.writer.add_scalar('Loss pseudo cluster train/'+ str(k), self.cluster_losses[k], epoch)
                    self.writer.add_scalar('Acc pseudo cluster train/'+ str(k), self.cluster_accs[k], epoch)
                    self.writer.add_scalar('Loss target cluster train/'+ str(k), self.cluster_losses_original[k], epoch)
                    self.writer.add_scalar('Acc target cluster train/'+ str(k), self.cluster_accs_original[k], epoch)   

            elif split=='valid':
                self.compute_stats_val()
                for k in range(self.num_clusters):
                    self.writer.add_scalar('Loss pseudo cluster val/'+ str(k), self.cluster_losses_val[k], epoch)
                    self.writer.add_scalar('Acc pseudo cluster val/'+ str(k), self.cluster_accs_val[k], epoch)
                    self.writer.add_scalar('Loss target cluster val/'+ str(k), self.cluster_losses_val_original[k], epoch)
                    self.writer.add_scalar('Acc target cluster val/'+ str(k), self.cluster_accs_val_original[k], epoch)


    def train(self, epoch):
        
        data_loader = self.loaders['train']
        total_loss, total_loss_pseudo, total_num, train_bar = 0.0, 0.0, 0, tqdm(data_loader, ncols=100)
        
        for images, target, biases, _, ids in train_bar:
            target =  target.cuda(non_blocking=True)
            biases = biases.cuda(non_blocking=True)
            B = target.size(0)
            images = images.cuda(non_blocking=True)
            pseudo = self.pseudo_labels['train'][ids].cuda(non_blocking=True)
            pseudo[target==1] -= self.args['k']
            results = self.model(images,target)
            loss_original, loss_pseudo = self.update(results, pseudo, target, ids)
            loss = loss_original + self.beta*loss_pseudo
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            total_num += B
            total_loss += loss_original.item() * B
            total_loss_pseudo += loss_pseudo.item() * B
            self.step += 1

        self.print_log(total_loss, total_loss_pseudo, total_num, epoch)     

        return 


    def test(self, epoch):
        self.model.eval()
        test_envs = ['valid', 'test']
            
        for desc in test_envs:
            with torch.no_grad():
                loader = self.loaders[desc]
                total_top1, total_top5, total_num, test_bar = 0.0, 0.0, 0, tqdm(loader, ncols=100)
                total_loss_o , total_loss_p = 0.0, 0.0
                num_groups = loader.dataset.num_groups
                
                bias_counts = torch.zeros(num_groups).cuda(non_blocking=True)
                bias_top1s = torch.zeros(num_groups).cuda(non_blocking=True)
            
                for data, target, biases, group, ids in test_bar:
                    data, target, biases, group = data.cuda(non_blocking=True), target.cuda(non_blocking=True), biases.cuda(non_blocking=True), group.cuda(non_blocking=True)
                    pseudo = self.pseudo_labels[desc][ids].cuda(non_blocking=True)
                    B = target.size(0)
                    pseudo[target==1] -= self.args['k']
                    results = self.model(data, target)

                    if desc =='valid':
                        loss_o, loss_p = self.update_val(results, pseudo, target, ids)

                    logits = results['out_original']
                    pred_labels = logits.argsort(dim=-1, descending=True)
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
                    total_loss_o += loss_o.item() * B
                    total_loss_p += loss_p.item() * B

            self.writer.add_scalar('Loss_original/'+desc, total_loss_o / total_num, epoch)
            self.writer.add_scalar('Loss_pseudo/'+desc, total_loss_p / total_num, epoch) 
            self.writer.add_scalar('Unbiased_avg_acc/'+desc, avg_acc, epoch)
            self.writer.add_scalar('Worst_acc/'+desc, worst_acc, epoch)
            self.writer.add_scalar('Overall_Acc/'+desc, acc1, epoch)

            for i,acc in enumerate(bias_accs):
                self.writer.add_scalar('Unbiased_acc_'+str(i)+'/'+desc, acc, epoch)        

            if desc=='valid':
                if avg_acc > self.best_avg_acc:
                    self.best_avg_acc = avg_acc
                    self.save_model('best_avg_acc', epoch)
                    self.save_avg_acc = True

                if worst_acc > self.best_worst:
                    self.best_worst = worst_acc
                    self.save_model('best_worst_acc', epoch)
                    self.save_worst_acc = True
                if epoch == 100:
                    self.save_model('last', epoch)
                self.print_log(epoch=epoch, split='valid')

            if desc=='test':
                if self.save_avg_acc:
                    self.test_best_avg_acc = avg_acc
                    self.save_avg_acc = False

                if self.save_worst_acc:
                    self.test_best_worst = worst_acc
                    self.save_worst_acc = False
        self.model.train()
        return