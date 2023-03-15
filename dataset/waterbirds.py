from abc import abstractmethod
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import pandas as pd
import os
from PIL import Image
import numpy as np
 
class Waterbirds(Dataset):
    """waterbirds dataset"""
    "split values: 0 train, 1 val, 2 test"
    def __init__(self,root, split,transform=None, args=None):
        self.args = args
        self.data_dir = root
        self.transform = transform
        self.metadata_df = pd.read_csv(
            os.path.join(self.data_dir, 'metadata.csv'))

        # Get the datafrmame self.metadata based on the split value
        self.metadata_df = self.metadata_df[self.metadata_df["split"] == split]

        if self.args['target_attr']=='Object':
            self.labels = self.metadata_df["y"].values
            self.bias = self.metadata_df["place"].values
        elif self.args['target_attr']=='Place':
            self.labels = self.metadata_df["place"].values
            self.bias = self.metadata_df["y"].values
        else:
            raise ValueError

        self.n_classes = 2

        # We only support one confounder for CUB for now
        
        # Map to groups
        self.n_groups = pow(2, 2)
        self.group_array = (self.labels * (self.n_groups / 2) +
                            self.bias).astype("int")

        # Extract filenames and splits
        self.filename_array = self.metadata_df["img_filename"].values

    def __len__(self):
        return len(self.metadata_df)
 
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
 
        image_name = os.path.join(self.data_dir,str(self.filename_array[idx]))
        try:
            image = Image.open(str(image_name))
        except:
            raise Exception(f'Cannot read {str(image_name)}.jpg @ idx: {idx}, len: {len(self.metadata_df)}')
 
        if self.transform:
            x = self.transform(image)

        label = self.labels[idx]
        bias  = self.bias[idx]
        group = self.group_array[idx]

        return x, label, bias, group, idx
    
    @property
    def classes(self):
        return ['0', '1']
    
    @property
    def num_classes(self):
        return 2
    
    @property
    def num_groups(self):
        return 4
        
def get_waterbirds_dataloader(root, name, batch_size, num_workers=8, args=None):
    
    from factory.factory import TransformFactory
    waterbirds_train_transform = TransformFactory.create("waterbirds_train")
    waterbirds_test_transform = TransformFactory.create("waterbirds_test")
    
    train = Waterbirds(root, split=0, transform=waterbirds_train_transform, args=args)
    val = Waterbirds(root,split=1, transform=waterbirds_test_transform, args=args)
    test = Waterbirds(root,split=2, transform=waterbirds_test_transform, args=args)
    train_eval = Waterbirds(root, split=0, transform=waterbirds_test_transform, args=args)

    train_loader = torch.utils.data.DataLoader(dataset=train,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=True,
                        #batch_sampler=batch_sampler,
                        drop_last=False
                        )

    valid_loader = torch.utils.data.DataLoader(dataset=val,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=False
                            )
    
    test_loader = torch.utils.data.DataLoader(dataset=test,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=False
                            )
    
    train_eval_loader = torch.utils.data.DataLoader(dataset=train_eval,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=False)

            
    return train_loader, valid_loader, test_loader, train_eval_loader
    
    