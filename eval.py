import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
from utils.config import load_config
from utils.transforms import *
#from model import Model
from pathlib import Path
from factory.factory import ModelFactory, DataLoaderFactory
from tqdm import tqdm
import numpy as np
import gdown

config_debug = Path('./configurations/eval_cfix_celebA.json')

def test(model, loaders):
    model.eval()
    test_envs = ['valid', 'test']
        
    for desc in test_envs:
        with torch.no_grad():
            loader = loaders[desc]
            total_top1, total_top5, total_num, test_bar = 0.0, 0.0, 0, tqdm(loader, ncols=100)
            num_groups = loader.dataset.num_groups
            
            bias_counts = torch.zeros(num_groups).cuda(non_blocking=True)
            bias_top1s = torch.zeros(num_groups).cuda(non_blocking=True)
        
            for data, target, biases, group, ids in test_bar:
                data, target, biases, group = data.cuda(non_blocking=True), target.cuda(non_blocking=True), biases.cuda(non_blocking=True), group.cuda(non_blocking=True)
                B = target.size(0)
                results = model(data, target)

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

    return acc1, avg_acc, worst_acc

def load_model(model, config):
    #Cfix models from Google Drive 
    f_checkpoint = '/mnt/beegfs/homes/gcapitani/ICCV11624/'+config["target_attr"]+".pth"
    if config["dataset"]=='celebA':
        if config["target_attr"]=='Wearing_Necklace':
            #url = "https://drive.google.com/file/d/15WK10DXZlt9xGXkGzl8HUxY2voP-MC34/view?usp=share_link"
            pass
            #f_checkpoint = gdown.download(url=url, output = f_checkpoint, quiet=False, fuzzy=True)
        if config["target_attr"]=='Double_Chin':
            url = ""
            gdown.download(url=url, output=f_checkpoint, quiet=False, fuzzy=True)
        if config["target_attr"]=='Pale_skin':
            url = ""
            gdown.download(url=url, output=f_checkpoint, quiet=False, fuzzy=True)
        if config["target_attr"]=='Wearing_Hat':
            url = ""
            gdown.download(url=url, output=f_checkpoint, quiet=False, fuzzy=True)
        if config["target_attr"]=='Chubby':
            url = ""
            gdown.download(url=url, output=f_checkpoint, quiet=False, fuzzy=True)
        if config["target_attr"]=='Wavy_Hair':
            url = ""
            gdown.download(url=url, output=f_checkpoint, quiet=False, fuzzy=True)
        if config["target_attr"]=='Big_Lips':
            url = "https://drive.google.com/file/d/1JBTDz7hnoppgDS98ubiVSLWyguuniluO/view?usp=share_link"
            gdown.download(url=url, output=f_checkpoint, quiet=False, fuzzy=True)
        if config["target_attr"]=='Bangs':
            url = ""
            gdown.download(url=url, output=f_checkpoint, quiet=False, fuzzy=True)
        if config["target_attr"]=='Receding_Hairline':
            url = ""
            gdown.download(url=url, output=f_checkpoint, quiet=False, fuzzy=True)
        if config["target_attr"]=='Brown_Hair':
            url = ""
            gdown.download(url=url, output=f_checkpoint, quiet=False, fuzzy=True)
    elif config["dataset"]=='waterbirds':
        if config["target_attr"]=='Object':
            url = "https://drive.google.com/file/d/15WK10DXZlt9xGXkGzl8HUxY2voP-MC34/view?usp=share_link"
            f_checkpoint = gdown.download(url=url, output=f_checkpoint, quiet=False, fuzzy=True)

        if config["target_attr"]=='Place':
            url = "https://drive.google.com/file/d/13ooFQ4Vpk3OXxLt9S8ZxDL-ap6louHFt/view?usp=share_link"
            f_checkpoint = gdown.download(url=url, output=f_checkpoint, quiet=False, fuzzy=True)        
    
    checkpoint = torch.load(f_checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    #checkpoint = torch.load(f_checkpoint)
    
    model.eval()

    return model

def main():

    config = load_config(config_debug)
    #SETUP
    torch.backends.cudnn.benchmark = True
    n_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    DATA_ROOT = config['data_root']

    # CFIX CODE
    if config["trainer"] == 'cfix':
        config["arch"] = 'ResNet18Cfix'

        #DATASET BASE
        loaders = DataLoaderFactory.create(config["dataset"], root= DATA_ROOT, batch_size=config["batch_size"],
                                                 num_workers=n_cpus, configs=config)  
        num_classes = 2

        model_args = {
            "name": config['arch'],
            "feature_dim": config['feature_dim'],
            "num_classes": num_classes,
            "pseudo_dim": config['k'], 
            "self_supervised": config['self_supervised'],
            "path_model" : None
        }
        
        #Model setup and optimizer config
        model = ModelFactory.create(**model_args).cuda()
        model = nn.DataParallel(model)

        model = load_model(model, config)
        acc1, avg_acc, worst_acc = test(model, loaders)

        print('Overall: {:.2f} Unbiased Average Acc: {:.2f} Unbiased Worst Acc: {:.2f}'.format(acc1, avg_acc, worst_acc))

if __name__ == "__main__":
    main()

            
            