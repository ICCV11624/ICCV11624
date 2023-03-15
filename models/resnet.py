import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import pdb
from pathlib import Path
import gdown
import os

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
    

class ResNet18Erm(nn.Module):
    def __init__(self, feature_dim, num_classes,  arch=''):
        super(ResNet18Erm, self).__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.arch = arch
        resnet = self.get_backbone()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 64
        self.res3 = resnet.layer2 # 1/8, 128
        self.res4 = resnet.layer3 # 1/16, 256
        self.res5 = resnet.layer4 # 1/32, 512
        
        self.f = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                               self.res2, self.res3, self.res4, self.res5)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # classifier
        self.fc = self.get_fc(num_classes)
        self.fc.apply(init_weights)
    
    def get_backbone(self):
        return torchvision.models.resnet18(pretrained=True)
        
    def get_fc(self, num_classes):
        return nn.Linear(512, num_classes, bias=True)
        
    def forward(self, x):
        features = self.f(x)
        features = torch.flatten(self.avgpool(features), start_dim=1)

        logits = self.fc(features)
            
        results = {
            "out_original": logits,
        }
            
        return results


class ResNet18Cfix(nn.Module):
    def __init__(self, feature_dim, num_classes, pseudo_dim, arch='', f_extraction=False, self_supervised=False, config=None, eval=False):
        super(ResNet18Cfix, self).__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.pseudo_dim = pseudo_dim
        self.arch = arch
        self.self_supervised = self_supervised
        self.config = config
        self.evaluation = eval
        resnet = self.get_backbone()
        if self_supervised:
            self.f = resnet
        else:
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.relu = resnet.relu  # 1/2, 64
            self.maxpool = resnet.maxpool

            self.res2 = resnet.layer1 # 1/4, 64
            self.res3 = resnet.layer2 # 1/8, 128
            self.res4 = resnet.layer3 # 1/16, 256
            self.res5 = resnet.layer4 # 1/32, 512
        
            self.f = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                               self.res2, self.res3, self.res4, self.res5)

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            # classifier
        
        self.fc = self.get_fc()
        self.pfc0 = self.get_pfc0()
        self.pfc1 = self.get_pfc1()

        self.fc.apply(init_weights)
        self.pfc0.apply(init_weights)
        self.pfc1.apply(init_weights)
        self.f_extraction= f_extraction

    

    def get_backbone(self):
        if self.self_supervised:
            resnet = torchvision.models.resnet18(zero_init_residual=True)
            backbone = nn.Sequential(*list(resnet.children())[:-1])
            f_root = os.path.join(self.config["checkpoint_root"], self.config["dataset"])
            f_checkpoint = os.path.join(f_root,"backbone.pth")
            if not os.path.exists(f_checkpoint) or self.evaluation==False:
                os.makedirs(f_root,  exist_ok=True)
                url = "https://drive.google.com/file/d/1j6VwqLoS5d-YOxcFxJRchdoHPuvrBWBJ/view?usp=share_link"
                gdown.download(url=url, output = f_checkpoint, quiet=False, fuzzy=True)
                backbone = torch.load(f_checkpoint)
            else:
                backbone = torch.load(f_checkpoint)
            return backbone
        else:
            return torchvision.models.resnet18(pretrained=True)
        
    def get_fc(self):
        return nn.Linear(512, self.num_classes, bias=True)

    def get_pfc0(self):
        return nn.Sequential(
                nn.Linear(512, self.pseudo_dim)
            )
    
    def get_pfc1(self):
        return nn.Sequential(
                nn.Linear(512, self.pseudo_dim)
            )
        
        
    def forward(self, x, target):
        if self.f_extraction:
            features = self.f(x)
            if self.self_supervised:
                features = torch.flatten(features, start_dim=1)
            else:
                features = torch.flatten(self.avgpool(features), start_dim=1)
            return features
        else:
            features = self.f(x)
            if self.self_supervised:
                features = torch.flatten(features, start_dim=1)
            else:
                features = torch.flatten(self.avgpool(features), start_dim=1)

            indexes_0 = (target==0).nonzero().type(torch.LongTensor).squeeze().cuda()
            indexes_1 = (target==1).nonzero().type(torch.LongTensor).squeeze().cuda()
        
            features_p0 = features[indexes_0]
            features_p1 = features[indexes_1]
        
            logits_p0 = self.pfc0(features_p0).squeeze()
            logits_p1 = self.pfc1(features_p1).squeeze()

            logits = self.fc(features)
        
            logits_p = torch.zeros(features.shape[0], self.pseudo_dim).cuda()

            logits_p[indexes_0] = logits_p0
            logits_p[indexes_1] = logits_p1

            results = {
                "out_original": logits,
                "out_pseudo": logits_p
            }
        
        return results

    
    
    