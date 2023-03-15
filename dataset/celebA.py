import os
from PIL import Image
import numpy as np
# from models import model_attributes
from pathlib import Path
import torchvision
from torch.utils import data
from utils.uniform_sampler.sampler import SamplerFactory


class CelebA(torchvision.datasets.CelebA):
    
  # Attributes : '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
    
    def __init__(self, root, name = 'celebA', split="train", target_type="attr", transform=None,
                 target_transform=None, download=False,target_attr='', bias_attrs=[], args=None): 
        
        super().__init__(root, split=split, target_type=target_type, transform=transform,
                 target_transform=target_transform, download=download)
        
        self.target_attr = target_attr
        self.bias_attrs = bias_attrs
        self.name = name
        self.target_idx = self.attr_names.index(target_attr)
        self.bias_indices = [self.attr_names.index(bias_att) for bias_att in bias_attrs]
        self.args = args  
    
    @property
    def class_elements(self):
        return self.attr[:, self.target_idx]
    
    @property
    def group_elements(self):
        group_attrs = self.attr[:, [self.target_idx]+self.bias_indices]
        weight = np.power(self.num_classes, np.arange(group_attrs.size(1)))
        group_elems = (group_attrs*weight).sum(1)
        return group_elems
    
    @property
    def group_counts(self):
        group_attrs = self.attr[:, [self.target_idx]+self.bias_indices]
        weight = np.power(self.num_classes, np.arange(group_attrs.size(1)))
        group_elems = (group_attrs*weight).sum(1)
        return group_elems.bincount()
    
    def group_counts_with_attr(self, attr):
        target_idx = self.attr_names.index(attr)
        group_attrs = self.attr[:, [target_idx]+self.bias_indices]
        weight = np.power(self.num_classes, np.arange(group_attrs.size(1)))
        group_elems = (group_attrs*weight).sum(1)
        return group_elems.bincount()
        
    
    def __len__(self):
        len = super().__len__()
        return len
    
    
    def get_sample_index(self, index):
        return index
    
    def __getitem__(self, index_):
        
        index = self.get_sample_index(index_)
        
        img_path = os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index])
        img_ = Image.open(img_path)

        target = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None
    
        target_attr = target[self.target_idx]
        bias_attrs = np.array([target[bias_idx] for bias_idx in self.bias_indices])
        group_attrs = np.insert(bias_attrs, 0, target_attr)  # target first
    
        bit = np.power(self.num_classes, np.arange(len(group_attrs)))
        group = np.sum(bit * group_attrs)
            
        if self.transform is not None:
            transform = self.transform
            img = transform(img_)


        return img, target_attr, bias_attrs, group, index
        
    
    @property
    def classes(self):
        return ['0', '1']
    
    @property
    def num_classes(self):
        return len(self.classes)
    
    @property
    def num_groups(self):
        return np.power(len(self.classes), len(self.bias_attrs)+1)
    
    @property
    def bias_attributes(self):
        return
    
    @property
    def attribute_names(self):
        return ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']


def get_celebA_dataloader(root, batch_size, target_attr, bias_attrs, num_workers=8, args=None):

    from factory.factory import TransformFactory
    

    celebA_transform_train = TransformFactory.create("celebA_train")
    celebA_transform_eval = TransformFactory.create("celebA_test")
    
    ### Dataset split
    celebDataset = CelebA
    
    splits = ['train', 'train_eval', 'valid', 'test']

    for split in splits:
        if split=='train':
            dataset = celebDataset(root, split=split, transform=celebA_transform_train, download=True,
                            target_attr=target_attr, bias_attrs=bias_attrs, args=args)
            class_idxs = []
            sampler_labels = dataset.class_elements
            for c in range(len(sampler_labels.unique())):
                ids = (sampler_labels==c).nonzero().squeeze()
                class_idxs.append(ids.tolist())

            batch_sampler = SamplerFactory().get(
            class_idxs=class_idxs,
            batch_size=batch_size,
            n_batches=len(dataset.class_elements)//batch_size,
            alpha=0.1,
            kind='fixed'
            )
            train_loader = data.DataLoader(dataset=dataset,
                num_workers=num_workers,
                pin_memory=True,
                batch_sampler=batch_sampler,
                prefetch_factor=3
                )
        else:


            if split=='train_eval':
                dataset = celebDataset(root, split='train', transform=celebA_transform_eval, download=True,
                            target_attr=target_attr, bias_attrs=bias_attrs, args=args)
                train_eval_loader = data.DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True,)
            elif split=='valid':
                dataset = celebDataset(root, split='valid', transform=celebA_transform_eval, download=True,
                            target_attr=target_attr, bias_attrs=bias_attrs, args=args)
                valid_loader = data.DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True,)
            elif split=='test':
                dataset = celebDataset(root, split='test', transform=celebA_transform_eval, download=True,
                            target_attr=target_attr, bias_attrs=bias_attrs, args=args)
                test_loader = data.DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True,)

    return train_loader, valid_loader, test_loader, train_eval_loader
    
    