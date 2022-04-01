from os import path
import os
import yaml
from easydict import EasyDict

from utils import mkdir_if_missing

def dict_to_str(h):
    s = []
    for k, v in sorted(h.items()):
        if isinstance(v, list):
            s.append('_'.join([str(e) for e in v]))
        else:
            s.append(f'{v}')
    return '-'.join(s)

def create_bare_config(root_dir, config_file_exp):
    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    
    p = EasyDict()
   
    # Copy
    for k, v in config.items():
        p[k] = v       
                        
    return p

def create_config(root_dir, config_file_exp):
    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    
    p = EasyDict()
   
    # Copy
    for k, v in config.items():
        p[k] = v
    
    basedir = path.join(root_dir, f"{p['criterion']}")
    basedir = path.join(basedir, f"model-{p['model']}-{dict_to_str(p['model_kwargs'])}-data-{p['data']}-{dict_to_str(p['data_kwargs'])}-criterion-{dict_to_str(p['criterion_kwargs'])}-optimizer-{p['optimizer']}-{dict_to_str(p['optimizer_kwargs'])}-scheduler-{p['scheduler']}-{dict_to_str(p['scheduler_kwargs'])}-bsize-{p['batch_size']}-init-{p['weight_init_version']}")
    
    mkdir_if_missing(basedir)    
    p['model_checkpoint'] = os.path.join(basedir, 'checkpoint.pth.tar')
    p['model_file'] = os.path.join(basedir, 'model.pth.tar')
    p['best_model_file'] = os.path.join(basedir, 'best_model.pth.tar')
    p['basedir'] = basedir
                        
    return p 

def create_code_config(root_dir, config_file_exp, **kwargs):
    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    
    p = EasyDict()
   
    # Copy
    for k, v in config.items():
        p[k] = v
    
    # Custom
    for k, v in kwargs.items():
        p[k] = v 

    basedir = path.join(root_dir, f"{p['criterion']}")
    basedir = path.join(basedir, f"model-{p['model']}-{dict_to_str(p['model_kwargs'])}-data-{p['data']}-{dict_to_str(p['data_kwargs'])}-criterion-{dict_to_str(p['criterion_kwargs'])}-optimizer-{p['optimizer']}-{dict_to_str(p['optimizer_kwargs'])}-scheduler-{p['scheduler']}-{dict_to_str(p['scheduler_kwargs'])}-bsize-{p['batch_size']}-init-{p['weight_init_version']}")
    
    mkdir_if_missing(basedir)    
    p['model_checkpoint'] = os.path.join(basedir, 'checkpoint.pth.tar')
    p['model_file'] = os.path.join(basedir, 'model.pth.tar')
    p['best_model_file'] = os.path.join(basedir, 'best_model.pth.tar')
    p['basedir'] = basedir
                        
    return p
                        
                        
def create_legacy_code_config(root_dir, config_file_exp, **kwargs):
    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    
    p = EasyDict()
   
    # Copy
    for k, v in config.items():
        p[k] = v
    
    # Custom
    for k, v in kwargs.items():
        p[k] = v 

    basedir = path.join(root_dir, f"{p['criterion']}")
    basedir = path.join(basedir, f"model-{p['model']}-{dict_to_str(p['model_kwargs'])}-data-{p['data']}-{dict_to_str(p['data_kwargs'])}-k-{p['k']}-criterion-{dict_to_str(p['criterion_kwargs'])}-optimizer-{p['optimizer']}-{dict_to_str(p['optimizer_kwargs'])}-scheduler-{p['scheduler']}-{dict_to_str(p['scheduler_kwargs'])}-bsize-{p['batch_size']}-init-{p['weight_init_version']}")
    
    mkdir_if_missing(basedir)    
    p['model_checkpoint'] = os.path.join(basedir, 'checkpoint.pth.tar')
    p['model_file'] = os.path.join(basedir, 'model.pth.tar')
    p['best_model_file'] = os.path.join(basedir, 'best_model.pth.tar')
    p['basedir'] = basedir
                        
    return p                        