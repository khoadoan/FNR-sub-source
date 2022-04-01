### common_config.py
import os
import math
import numpy as np
import torch
from torch import nn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.utils as vutils

from termcolor import colored

# Weight initialization
def weights_init(m):
    """weight initialization"""
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2dSameLayer') != -1:
        print(classname)
        for p in m.parameters():
            nn.init.normal_(p.data, 0.0, 0.02)
    elif classname.find('Conv') != -1:
        print(classname)
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        print(classname)
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None and m.bias.data is not None:
            nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.Linear:
        print(classname)
        # nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.normal_(m.weight.data, 0.0, 1)
        if m.bias is not None and m.bias.data is not None:
            nn.init.constant(m.bias.data, 0)
            
def weights_init_v2(m):
    """weight initialization"""
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2dSameLayer') != -1:
        print(classname)
        for p in m.parameters():
            nn.init.normal_(p.data, 0.0, 0.02)
    elif classname.find('Conv') != -1:
        print(classname)
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        print(classname)
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None and m.bias.data is not None:
            nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.Linear:
        print(classname)
        nn.init.normal_(m.weight.data, 0.0, 0.02) #this is different
        # nn.init.normal_(m.weight.data, 0.0, 1)
        if m.bias is not None and m.bias.data is not None:
            nn.init.constant(m.bias.data, 0)   
            
def weights_init_v1(m):
    """weight initialization"""
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2dSameLayer') != -1:
        print(classname)
        for p in m.parameters():
            nn.init.normal_(p.data, 0.0, 0.02)
    elif classname.find('Conv') != -1:
        print(classname)
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        print(classname)
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None and m.bias.data is not None:
            nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.Linear:
        print(classname)
        nn.init.normal_(m.weight.data, 0.0, 0.2) #this is different
        # nn.init.normal_(m.weight.data, 0.0, 1)
        if m.bias is not None and m.bias.data is not None:
            nn.init.constant(m.bias.data, 0)  

def get_optimizer(p, model):
    params = model.parameters()
                
    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])
    
    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer

def adjust_learning_rate(p, optimizer, epoch):
    lr = p['optimizer_kwargs']['lr']
    
    if p['scheduler'] == 'cosine':
        eta_min = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2
         
    elif p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'constant':
        lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def get_model(p, pretrain_path=None, pretrain_states=None):
    # Get backbone
    if p['model'] == 'paircnn':
        from models import TwoTowerHashModel_V0
        model = TwoTowerHashModel_V0(**p['model_kwargs'])
    elif p['model'] == 'two_tower_v1':
        from models import TwoTowerHashModel_V1
        model = TwoTowerHashModel_V1(**p['model_kwargs'])

    elif p['model'] == 'two_tower_v2':
        from models import TwoTowerHashModel_V2
        model = TwoTowerHashModel_V2(**p['model_kwargs'])
        
    elif p['model'] == 'two_tower_v1_triplet':
        from models import TwoTowerHashModel_V1_Triplet
        model = TwoTowerHashModel_V1_Triplet(**p['model_kwargs'])

    elif p['model'] == 'model_DSH':
        from models import TwoTowerHashModel_DSH
        model = TwoTowerHashModel_DSH(**p['model_kwargs'])

    elif p['model'] == 'model_DSH_tanh':
        from models import TwoTowerHashModel_DSH_Tanh
        model = TwoTowerHashModel_DSH_Tanh(**p['model_kwargs'])

    elif p['model'] == 'mlp_concat':
        from models import MLPConcat
        model = MLPConcat(**p['model_kwargs'])

    else:
        raise ValueError('Invalid model {}'.format(p['model']))
    
    if 'weight_init_version' in p:
        print(colored('Intializing the model {}...'.format(p['weight_init_version']), 'red'))
        if p['weight_init_version'] == 'v2':
            model.apply(weights_init_v2)
        elif p['weight_init_version'] == 'v1':
            model.apply(weights_init_v1)
        else:
            print('Use pytorch default initialization')
    else:
        print(colored('Intializing the model...', 'red'))
        model.apply(weights_init)
        
    # Load pretrained weights
    if pretrain_path is not None and os.path.exists(pretrain_path):
        print(colored('Load model from pre-trained path: {}'.format(pretrain_path), 'red'))
        state = torch.load(pretrain_path, map_location='cpu')        
        missing = model.load_state_dict(state['model'], strict=True)
        
    elif pretrain_path is not None and not os.path.exists(pretrain_path):
        raise ValueError('Path with pre-trained weights does not exist {}'.format(pretrain_path))

    else:
        pass
    
    #pretrain_states takes precedence over pretrain_path
    if pretrain_states is not None:
        for k in model.state_dict():
            v = model.state_dict()[k]
            print(k, v.shape)
        print(colored('Load model from pre-trained weights', 'red'))
        missing = model.load_state_dict(pretrain_states, strict=False)

    return model

def get_train_dataset(p):
    data = np.load(p['train_path'], mmap_mode='r')
    print(colored('Load train dataset from {}'.format(p['train_path']), 'red'))
    x1_data = torch.tensor(data['x1'], dtype=torch.float32)
    x2_data = torch.tensor(data['x2'], dtype=torch.float32)
    pos = torch.tensor(data['pos'], dtype=torch.int64) #this is only having topk right now
    neg = torch.tensor(data['neg'], dtype=torch.int64)
    if p['data'] == 'v_random_pair':
        from data import RandomPair_Dataset
        dataset = RandomPair_Dataset(x1_data, x2_data, pos=pos, neg=neg, **p['data_kwargs'])
        
    elif p['data'] == 'v_random_pair_cache':
        from data import CacheRandomPair_Dataset
        dataset = CacheRandomPair_Dataset(x1_data, x2_data, pos=pos, neg=neg, **p['data_kwargs'])

    elif p['data'] == 'v_random_triplet':
        from data import RandomTriplet_Dataset
        dataset = RandomTriplet_Dataset(x1_data, x2_data, pos=pos, neg=neg, **p['data_kwargs'])
        
    elif p['data'] == 'triplet_one':
        from data import NegativeRankedNeuCF_Dataset
        #dataset = NegativeRankedNeuCF_Dataset(x1_data, x2_data, ranking, neg)
        dataset = NegativeRankedNeuCF_Dataset(x1_data, x2_data, ranking, neg)
    
    elif p['data'] == 'triplet':
        from data import RankedNeuCF_Dataset
        dataset = RankedNeuCF_Dataset(x1_data, x2_data, ranking, **p['data_kwargs'])
        
    elif p['data'] == 'random':
        from data import RandomPairNeuCF_Dataset
        if p['data_kwargs']['neural_sim_func'] == 'NeuCFydata_DeepFM':
            dataset = RandomPairNeuCF_Dataset(x1_data, x2_data, 
                                        neural_sim_func=p['data_kwargs']['neural_sim_func'], 
                                        neural_sim_func_N=p['data_kwargs']['neural_sim_func_N'],
                                        neural_sim_func_prefix=p['data_kwargs']['neural_sim_func_prefix']
                                       )
        else:
            dataset = RandomPairNeuCF_Dataset(x1_data, x2_data, neural_sim_func=p['data_kwargs']['neural_sim_func'])

    return dataset

def get_train_non_ranking_dataset(p):
    data = np.load(p['train_path'], mmap_mode='r')
    x1_data = torch.tensor(data['x1'], dtype=torch.float32)
    x2_data = torch.tensor(data['x2'], dtype=torch.float32)
    
    from data import NeuCF_Dataset
    dataset = NeuCF_Dataset(x1_data, x2_data)
    return dataset

def _get_val_dataset(p, data_path):
    data = np.load(data_path, mmap_mode='r')
    x1_data = torch.tensor(data['x1'], dtype=torch.float32)
    x2_data = torch.tensor(data['x2'], dtype=torch.float32)
    pos = torch.tensor(data['pos'], dtype=torch.int64)
    neg = torch.tensor(data['neg'], dtype=torch.int64)    
    
    if p['data'] == 'v_random_pair':
        from data import RandomPair_Dataset
        dataset = RandomPair_Dataset(x1_data, x2_data, pos=pos, neg=neg, **p['data_kwargs'])
        
    elif p['data'] == 'v_random_triplet':
        from data import RandomTriplet_Dataset
        dataset = RandomTriplet_Dataset(x1_data, x2_data, pos=pos, neg=neg, **p['data_kwargs'])
        
    elif p['data'] == 'triplet_one':
        from data import NegativeRankedNeuCF_Dataset
        dataset = NegativeRankedNeuCF_Dataset(x1_data, x2_data, ranking, neg)
    
    elif p['data'] == 'triplet':
        from data import RankedNeuCF_Dataset
        dataset = RankedNeuCF_Dataset(x1_data, x2_data, ranking, **p['data_kwargs'])
        
    elif p['data'] == 'random':
        from data import RandomPairNeuCF_Dataset
        if p['data_kwargs']['neural_sim_func'] == 'NeuCFydata_DeepFM':
            dataset = RandomPairNeuCF_Dataset(x1_data, x2_data, 
                                        neural_sim_func=p['data_kwargs']['neural_sim_func'], 
                                        neural_sim_func_N=p['data_kwargs']['neural_sim_func_N'],
                                        neural_sim_func_prefix=p['data_kwargs']['neural_sim_func_prefix']
                                       )
        else:
            dataset = RandomPairNeuCF_Dataset(x1_data, x2_data, neural_sim_func=p['data_kwargs']['neural_sim_func'])


    return dataset

def get_val_dataset(p):
    return _get_val_dataset(p, p['val_path'])

def get_val_non_ranking_dataset(p):
    data = np.load(p['val_path'], 'mmap_mode')
    x1_data = torch.tensor(data['x1'], dtype=torch.float32)
    x2_data = torch.tensor(data['x2'], dtype=torch.float32)

    from data import NeuCF_Dataset
    dataset = NeuCF_Dataset(x1_data, x2_data)
    return dataset

def get_test_dataset(p):
    data = np.load(p['test_path'], 'r')
    x1_data = torch.tensor(data['x1'], dtype=torch.float32)
    x2_data = torch.tensor(data['x2'], dtype=torch.float32)
    print('Dataset has {} users, {} items'.format(x1_data.shape[0], x2_data.shape[0]))
    pos = torch.tensor(data['pos'], dtype=torch.int64)
    neg = torch.tensor(data['neg'], dtype=torch.int64)    
    
    from data import PairNeuCF_Dataset
    if 'neural_sim_func'  in p['data_kwargs']:
        if p['data_kwargs']['neural_sim_func'] == 'NeuCFydata_DeepFM':
            dataset = PairNeuCF_Dataset(x1_data, x2_data, 
                                        neural_sim_func=p['data_kwargs']['neural_sim_func'], 
                                        neural_sim_func_N=p['data_kwargs']['neural_sim_func_N'],
                                        neural_sim_func_prefix=p['data_kwargs']['neural_sim_func_prefix']
                                       )
        else:
            dataset = PairNeuCF_Dataset(x1_data, x2_data, neural_sim_func=p['data_kwargs']['neural_sim_func'])
    else:
        dataset = PairNeuCF_Dataset(x1_data, x2_data) #support backward comp
    return dataset

def get_train_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'], 
            batch_size=p['batch_size'], pin_memory=True, shuffle=True)

def get_val_dataloader(p, dataset, batch_size=None, num_workers=None):
    return torch.utils.data.DataLoader(dataset, 
            num_workers=p['num_workers'] if num_workers is None else num_workers,
            batch_size=p['batch_size'] if batch_size is None else batch_size, pin_memory=True, shuffle=False)

def get_criterion(p):
    if p['criterion'] == 'v1': #mse regression
        from losses import Loss_V1
        criterion = Loss_V1(**p['criterion_kwargs'])
        
    elif p['criterion'] == 'v1_constrastive': #mse regression
        from losses import Loss_V1_Contrastive
        criterion = Loss_V1_Contrastive(**p['criterion_kwargs'])

    elif p['criterion'] == 'v1_triplet': #mse regression
        from losses import Loss_V1_Triplet
        criterion = Loss_V1_Triplet(**p['criterion_kwargs'])

    elif p['criterion'] == 'v1_triplet_contrastive': #mse regression
        from losses import Loss_V1_Triplet_Contrastive
        criterion = Loss_V1_Triplet_Contrastive(**p['criterion_kwargs'])

    elif p['criterion'] == 'v2': #similar to v1 but with bce regression
        from losses import Loss_V2
        criterion = Loss_V2(**p['criterion_kwargs'])

    elif p['criterion'] == 'v3': #similar to v1 but with 
        from losses import Loss_V1_Neg
        criterion = Loss_V1_Neg(**p['criterion_kwargs'])

    elif p['criterion'] == 'loss_DSH':
        from losses import Loss_DSH
        criterion = Loss_DSH(**p['criterion_kwargs'])

    else:
        raise ValueError('Invalid criterion {}'.format(p['criterion']))

    return criterion