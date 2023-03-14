import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
from utils.config import load_config
from utils.transforms import *
#from model import Model
from pathlib import Path
from factory.factory import ModelFactory, TrainerFactory, DataLoaderFactory, TransformFactory


CHECKPOINT_ROOT = Path('/mnt/beegfs/work/H2020DeciderFicarra/gcapitani/')
config_debug = Path('./configurations/debug_waterbirds_cfix.json')

def main():

    config = load_config(config_debug)
    #SETUP
    torch.backends.cudnn.benchmark = True
    n_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    start_epoch = 1

    if config["dataset"] == 'celebA':
        DATA_ROOT = Path('/nas/softechict-nas-2/gcapitani/')
    elif config["dataset"] == 'waterbirds':
        DATA_ROOT = Path('/nas/softechict-nas-2/gcapitani/waterbird_complete95_forest2water2/')

    checkpoint_dir = CHECKPOINT_ROOT / config["dataset"] / config["target_attr"] / config["desc"]
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config["checkpoint_dir"] = checkpoint_dir

    # BASELINE CODE
    if config["trainer"] == 'erm':
        config["arch"] = 'ResNet18'

        #DATASET BASE
        loaders = DataLoaderFactory.create(config["dataset"], root= DATA_ROOT, batch_size=config["batch_size"],
                                                 num_workers=n_cpus, configs=config)
        num_classes = 2
        #MODEL BASE
        model_args = {
            "name": config["arch"],
            "feature_dim": config["feature_dim"],
            "num_classes": num_classes,
        }

        # Model setup and optimizer config
        model = ModelFactory.create(**model_args).cuda()
        model = nn.DataParallel(model)
        optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["scheduler_T_max"])
        trainer = TrainerFactory.create('erm', config, model, loaders, optimizer, num_classes)
        trainer.set_checkpoint_dir(checkpoint_dir)

        #BASELINE MODEL
        for epoch in range(start_epoch, config["max_epoch"]+1):
            trainer.train(epoch=epoch)
            trainer.test(epoch=epoch)
            scheduler.step()        
        
        with open(os.path.join(checkpoint_dir,'results.txt'), 'w') as f:
            f.write('best_accuracy:'+ str(trainer.test_best_acc)+ "\n")
            f.write('best_avg_accuracy:'+ str(trainer.test_avg_acc)+ "\n")
            f.write('best_worst_accuracy:'+ str(trainer.test_worst_acc)+ "\n")    
    # CFIX CODE
    if config["trainer"] == 'cfix':
        config["arch"] = 'ResNet18Cfix'

        #DATASET BASE
        loaders = DataLoaderFactory.create(config["dataset"], root= DATA_ROOT, batch_size=config["batch_size"],
                                                 num_workers=n_cpus, configs=config)  
        num_classes = 2

        if config["dataset"] == 'celebA':
            config['path_model'] = '/mnt/beegfs/work/H2020DeciderFicarra/gcapitani/celebA/self/backbone.pt'
        elif config["dataset"] == 'waterbirds':
            config['path_model'] = '/nas/softechict-nas-2/gcapitani/Barlow_twins/waterbirds/backbone.pt'
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

        optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["scheduler_T_max"])
            
        trainer = TrainerFactory.create('cfix', config, model, loaders, optimizer, num_classes, config['t'], config['beta'])
        trainer.set_checkpoint_dir(checkpoint_dir)
        trainer.clustering()

        for epoch in range(start_epoch, config['max_epoch']+1):
            trainer.train(epoch=epoch)
            trainer.test(epoch=epoch)
            scheduler.step()

        with open(os.path.join(checkpoint_dir,'results.txt'), 'w') as f:
            f.write('best_avg_accuracy:'+ str(trainer.test_best_avg_acc)+ "\n")
            f.write('best_worst_accuracy:'+ str(trainer.test_best_worst)+ "\n")

if __name__ == "__main__":
    main()

            
            