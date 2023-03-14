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

CHECKPOINT_ROOT = Path('/mnt/beegfs/work/H2020DeciderFicarra/gcapitani/')
config_debug = Path('./configurations/debug_celebA_cfix_eval.json')


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

def main():

    config = load_config(config_debug)
    #SETUP
    torch.backends.cudnn.benchmark = True
    n_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])

    if config["dataset"] == 'celebA':
        DATA_ROOT = Path('/nas/softechict-nas-2/gcapitani/')
    elif config["dataset"] == 'waterbirds':
        DATA_ROOT = Path('/nas/softechict-nas-2/gcapitani/waterbird_complete95_forest2water2/')

    checkpoint_dir = CHECKPOINT_ROOT / config["dataset"] / config["target_attr"] / config["desc"]
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config["checkpoint_dir"] = checkpoint_dir

    # CFIX CODE
    if config["trainer"] == 'cfix':
        config["arch"] = 'ResNet18Cfix'

        #DATASET BASE
        loaders = DataLoaderFactory.create(config["dataset"], root= DATA_ROOT, batch_size=config["batch_size"],
                                                 num_workers=n_cpus, configs=config)  
        num_classes = 2

        if config["dataset"] == 'celebA':
            config['path_model'] = '/mnt/beegfs/work/H2020DeciderFicarra/gcapitani/celebA/self/backbone.pt'
        else:
            raise ValueError('Dataset not supported')
        
        model_args = {
            "name": config['arch'],
            "feature_dim": config['feature_dim'],
            "num_classes": num_classes,
            "pseudo_dim": config['k'], 
            "self_supervised": config['self_supervised'],
            "path_model" : config['path_model']
        }
        
        #Model setup and optimizer config
        model = ModelFactory.create(**model_args).cuda()
        model = nn.DataParallel(model)
        path_model= '/mnt/beegfs/work/H2020DeciderFicarra/gcapitani/celebA/Double_Chin/run6/best_worst_acc_cfix.pth'
        checkpoint = torch.load(path_model)
        model.load_state_dict(checkpoint['state_dict']) # Set CUDA before if error occurs.

        acc1, avg_acc, worst_acc = test(model, loaders)

        print('Overall: {:.2f} Unbiased Average Acc: {:.2f} Unbiased Worst Acc: {:.2f}'.format(acc1, avg_acc, worst_acc))

if __name__ == "__main__":
    main()

            
            