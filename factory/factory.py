from typing import Any, Callable, Dict

from dataset.celebA import  get_celebA_dataloader
from dataset.waterbirds import  get_waterbirds_dataloader

from utils.transforms import *
from models.resnet import ResNet18Erm, ResNet18Cfix

#import trainers (ERM and CFix)
from trainers.erm import ErmTrainer
from trainers.cfix import CfixTrainer


class Factory(object):

    PRODUCTS: Dict[str, Callable] = {}

    def __init__(self):
        raise ValueError(
            f"""Cannot instantiate {self.__class__.__name__} object, use
            `create` classmethod to create a product from this factory.
            """
        )

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> Any:
        r"""Create an object by its name, args and kwargs."""
        if name not in cls.PRODUCTS:
            raise KeyError(f"{cls.__class__.__name__} cannot create {name}.")

        return cls.PRODUCTS[name](*args, **kwargs)
    
    
    
class ModelFactory(Factory):


    MODELS: Dict[str, Callable] = {
        "ResNet18": ResNet18Erm,
        "ResNet18Cfix": ResNet18Cfix,
    }

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> Any:
        
        return cls.MODELS[name](*args, **kwargs)
    
    
class TransformFactory(Factory):
    PRODUCTS: Dict[str, Callable] = {

        "celebA_train": celebA_train_transform,
        "celebA_test": celebA_test_transform,

        "waterbirds_train": waterbirds_train_transform,
        "waterbirds_test": waterbirds_test_transform,

    }
        
    @classmethod
    def create(cls, name: str, *args, **kwargs) -> Any:
        r"""Create an object by its name, args and kwargs."""
        if name not in cls.PRODUCTS:
            raise KeyError(f"{cls.__class__.__name__} cannot create {name}.")

        return cls.PRODUCTS[name]
        
    
    
class DataLoaderFactory(Factory):

    @classmethod
    def create(cls, name: str, root:str, batch_size: int, num_workers: int, configs: Any) -> Any:
        if name == 'celebA':    
            train_loader, valid_loader, test_loader, train_eval_loader = get_celebA_dataloader(
                root= root, batch_size=batch_size, num_workers=num_workers, target_attr=configs['target_attr'], bias_attrs=configs['bias_attrs'], args=configs)
        elif name == 'waterbirds':
            train_loader, valid_loader, test_loader, train_eval_loader = get_waterbirds_dataloader(
                root= root, name = name, batch_size=batch_size, num_workers=num_workers,args=configs)
        else:
            raise ValueError

        data_loaders = {
                'train': train_loader,
                'valid': valid_loader,
                'test': test_loader,
                'train_eval': train_eval_loader,
            }
        return data_loaders
        
class TrainerFactory(Factory):

    TRAINERS: Dict[str, Callable] = {
        "erm": ErmTrainer,
        'cfix': CfixTrainer,
    }

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> Any:
        
        return cls.TRAINERS[name](*args, **kwargs)
    