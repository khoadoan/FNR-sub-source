import os
from matplotlib import pyplot as plt
from os import path
import pickle
from tqdm.auto import tqdm
import torch
import numpy as np
from datetime import datetime
from termcolor import colored
import seaborn as sns

from utils import AverageMeter, ProgressMeter
from evaluate_utils import get_test_output, run_external_eval, calculate_ranking_metrics
from config import create_code_config
from common_config import get_train_dataset, get_val_dataset, get_train_dataloader, get_val_dataloader, get_test_dataset
from common_config import get_model, get_optimizer, get_criterion
from common_config import adjust_learning_rate

@torch.no_grad()
def get_output(val_loader, model, device, return_prediction=True, return_target=True):
    all_h1, all_h2, all_prediction, all_targets = [], [], [], []
    for i, (id1, x1, id2, x2, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
        h1, h2, prediction = model(x1.to(device), x2.to(device), return_only_hash=False)
        all_h1.append(h1)
        all_h2.append(h2)
        if return_prediction:
            all_prediction.append(prediction)
        if return_target:
            all_targets.append(target)
    output = {'h1': torch.cat(all_h1, dim=0), 'h2': torch.cat(all_h2, dim=0)}
    if return_prediction:
        output['prediction'] =  torch.cat(all_prediction, dim=0)
    if return_target:
        output['target'] = torch.cat(all_targets, dim=0).to(device)
    return output

@torch.no_grad()
def get_triplet_output(val_loader, model, device, return_prediction=True, return_target=True):
    all_h, all_pos, all_neg, all_prediction_pos, all_prediction_neg, all_target_pos, all_target_neg = [], [], [], [], [], [], []
    for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
        x, pos, neg = data['x'], data['pos'], data['neg']
        target_pos, target_neg = data['target_pos'], data['target_neg']
        
        h, h_pos, h_neg, prediction_pos, prediction_neg = model(x.to(device), pos.to(device), neg.to(device), return_only_hash=False)

        all_h.append(h)
        all_pos.append(h_pos)
        all_neg.append(h_neg)

        if return_prediction:
            all_prediction_pos.append(prediction_pos)
            all_prediction_neg.append(prediction_neg)
        if return_target:
            all_target_pos.append(target_pos)
            all_target_neg.append(target_neg)
    output = {'h': torch.cat(all_h, dim=0), 'h_pos': torch.cat(all_pos, dim=0), 'h_neg': torch.cat(all_neg, dim=0)}
    if return_prediction:
        output['prediction_pos'] =  torch.cat(all_prediction_pos, dim=0)
        output['prediction_neg'] =  torch.cat(all_prediction_neg, dim=0)
    if return_target:
        output['target_pos'] = torch.cat(all_target_pos, dim=0).to(device)
        output['target_neg'] = torch.cat(all_target_neg, dim=0).to(device)
    return output

def print_weight_magnitude(model):
    for n, param in model.named_parameters():
        print('{}: [{:0.4f},{:0.4f}]'.format(n, param.data.min().item(), param.data.max().item()))

@torch.no_grad()
def run_internal_eval(model, criterion, output):
    model.eval()
    h1 = output['h1']
    h2 = output['h2']
    target = output['target']
    prediction = output['prediction']
    print('Prediction [{:.04f}, {:.04f}], Target [{:.04f}, {:.04f}]'.format(
        prediction.min().item(), prediction.max().item(), target.min().item(), target.max().item()))
    total_loss, prediction_loss, uniform_frequency_loss, independent_bit_loss = criterion(h1, h2, prediction, target, model)      
    print_weight_magnitude(model)
    return {'total_loss': total_loss}

@torch.no_grad()
def run_internal_eval_triplet(model, criterion, output):
    model.eval()
    h = output['h']
    h_pos = output['h_pos']
    h_neg = output['h_neg']
    target_pos = output['target_pos']
    target_neg = output['target_neg']
    prediction_pos = output['prediction_pos']
    prediction_neg = output['prediction_neg']
    print((prediction_pos-prediction_neg).abs().max().item())
    print('Prediction [{:.04f}, {:.04f}] [{:.04f}, {:.04f}], Target [{:.04f}, {:.04f}] [{:.04f}, {:.04f}]'.format(
        prediction_pos.min().item(), prediction_pos.max().item(), 
        prediction_neg.min().item(), prediction_neg.max().item(),
        target_pos.min().item(), target_pos.max().item(),
        target_neg.min().item(), target_neg.max().item()
    ))
    total_loss, prediction_loss, uniform_frequency_loss, independent_bit_loss = criterion(h, h_pos, h_neg, prediction_pos, prediction_neg, target_pos, target_neg, model)      
    #print_weight_magnitude(model)
    return {'total_loss': total_loss}

def epoch_train(train_loader, model, criterion, optimizer, epoch, device):
    """ 

    """
    total_losses = AverageMeter('Total', ':.5f')
    prediction_losses = AverageMeter('P', ':.5f')
    uniform_frequency_losses = AverageMeter('U', ':.5f')
    independent_bit_losses = AverageMeter('I', ':.5f')
    
    
    progress = ProgressMeter(len(train_loader),
        [total_losses, prediction_losses, uniform_frequency_losses, independent_bit_losses],
        prefix="Epoch: [{}]".format(epoch+1), sep='  ')

    model.train() # Update BN, Dropout...
    for i, (id1, x1, id2, x2, target) in enumerate(train_loader):
        h1, h2, prediction = model(x1.to(device), x2.to(device), return_only_hash=False)
        
        #print((prediction-target.to(device)).abs())
        
        total_loss, prediction_loss, uniform_frequency_loss, independent_bit_loss = criterion(
            h1, h2, prediction, target.to(device), model=model)

        total_losses.update(total_loss)
        prediction_losses.update(prediction_loss)
        uniform_frequency_losses.update(uniform_frequency_loss)
        independent_bit_losses.update(independent_bit_loss)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)

def epoch_train_triplet(train_loader, model, criterion, optimizer, epoch, device):
    """ 

    """
    total_losses = AverageMeter('Total', ':.5f')
    prediction_losses = AverageMeter('P', ':.5f')
    uniform_frequency_losses = AverageMeter('U', ':.5f')
    independent_bit_losses = AverageMeter('I', ':.5f')
    
    
    progress = ProgressMeter(len(train_loader),
        [total_losses, prediction_losses, uniform_frequency_losses, independent_bit_losses],
        prefix="Epoch: [{}]".format(epoch+1), sep='  ')

    model.train() # Update BN, Dropout...
    for i, data in enumerate(train_loader):
        x, pos, neg = data['x'], data['pos'], data['neg']
        target_pos, target_neg = data['target_pos'], data['target_neg']
        
        h, h_pos, h_neg, prediction_pos, prediction_neg = model(x.to(device), pos.to(device), neg.to(device), return_only_hash=False)
        
        #print((prediction_pos-prediction_neg).abs().max().item())
        
        total_loss, prediction_loss, uniform_frequency_loss, independent_bit_loss = criterion(
            h, h_pos, h_neg, prediction_pos, prediction_neg, target_pos.to(device), target_neg.to(device), model=model)

        total_losses.update(total_loss)
        prediction_losses.update(prediction_loss)
        uniform_frequency_losses.update(uniform_frequency_loss)
        independent_bit_losses.update(independent_bit_loss)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)            

            
def train(p, verbose=0, device=None, pretrain_states=None, epoch_train=epoch_train, get_output=get_output, run_internal_eval=run_internal_eval):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(colored('Using device: {}'.format(device), 'red'))
        
    # Data
    print(colored('Get dataset and dataloaders', 'blue'))
    train_dataset = get_train_dataset(p)
    val_dataset = get_val_dataset(p)
    
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    
    print('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)))

    # Model
    print(colored('Get model', 'blue'))
    model = get_model(p, pretrain_path=None)
    if verbose >= 3:
        print(model)
    
    #TODO: current not supported because of the problem in loading pretrained weights
    #if device.type != 'cpu':
    #    model = torch.nn.DataParallel(model)
    model = model.to(device)

    # Optimizer
    if verbose >= 2:
        print(colored('Get optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    if verbose >= 2:
        print(optimizer)

    # Loss function
    if verbose >= 2:
        print(colored('Get loss', 'blue'))
    criterion = get_criterion(p) 
    criterion.to(device)
    if verbose >= 3:
        print(criterion)

    # Checkpoint
    all_stats = []
    if os.path.exists(p['model_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['model_checkpoint']), 'blue'))
        checkpoint = torch.load(p['model_checkpoint'], map_location='cpu')
        
        #load models
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])        
        
        #load tracking vars
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        best_recall = checkpoint['best_recall']
        all_stats = checkpoint['stats']
        eval_losses = checkpoint['eval_losses'] if 'eval_losses' in checkpoint else {}
        print(best_loss)
    else:
        print(colored('No checkpoint file at {}'.format(p['model_checkpoint']), 'blue'))
        
        # initialize tracking vars
        start_epoch = 0
        best_loss = 1e4
        best_recall = 0.0
        eval_losses = {}
    
    if pretrain_states is not None:
        print(colored('[WARNING] Restart from provided pretrained states instead of checkpoint', 'red'))
        for name, param in model.named_parameters():
            print(name, param.shape)
        model.load_state_dict(pretrain_states)

    # Main loop
    print(colored('Starting main loop', 'blue'))
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
        print(colored('-'*15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train
        print('Train ...')
        start_time = datetime.now()
        epoch_train(train_dataloader, model, criterion, optimizer, epoch, device=device)
        duration = datetime.now() - start_time
        print('Epoch {} finishes in {:.02f} minutes!'.format(epoch+1, duration.total_seconds()/60))
        if (epoch+1) % p['epochs_per_internal_eval'] == 0 or epoch+1 == p['epochs']:                    
            print('Evaluate based internally ...')
            model.eval()
            with torch.no_grad():
                val_output = get_output(val_dataloader, model, device=device)
            internal_result = run_internal_eval(model, criterion, val_output)
            eval_loss = internal_result['total_loss'].item()
            eval_losses[epoch+1] = eval_loss
            
            if best_loss > eval_loss:
                print(colored('New lowest loss: %.6f -> %.6f' %(best_loss, eval_loss), 'green'))
                best_loss = eval_loss
                torch.save({'model': model.state_dict(), 'eval_loss': eval_loss}, p['model_file'])
            else:
                print(colored('No new lowest loss: %.6f -> %.6f' %(best_loss, eval_loss), 'cyan'))

        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
                    'epoch': epoch + 1, 'best_loss': best_loss, 'stats': all_stats, 'eval_losses': eval_losses},
                     p['model_checkpoint'])
        
        if (epoch+1) % p['epochs_per_external_eval'] == 0 or epoch+1 == p['epochs']:
            with torch.no_grad():
                model.eval()
                test_loader = get_val_dataloader(p, get_test_dataset(p), batch_size=1024, num_workers=16)
                test_output = get_test_output(p, test_loader, model, device=device)

            topks = p['k'] if type(p['k']) == list else [p['k']]
            
            for topk in topks:
                raw_metrics, hash_metrics, reranked_metrics, random_metrics, best_metrics = run_external_eval(p, test_output, topk=topk)
                with plt.style.context("seaborn-whitegrid"):
                    plt.rcParams["axes.edgecolor"] = "0.0"
                    plt.rcParams["axes.linewidth"]  = 1.0

                    fig, ax = plt.subplots()
                    N = len(raw_metrics['recall'])
                    computation = np.arange(0, len(raw_metrics['recall'])) / N
                    step=5
                    first_n=1000
                    end_step=500
                    max_n=9999999
                    hash_x = hash_metrics['recall'][:np.where(hash_metrics['recall']==1.0)[0][0]]
                    hash_y = computation[:len(hash_x)]
                    ax.plot(hash_x[:first_n:step].tolist() + hash_x[first_n:max_n:end_step].tolist(),  
                            hash_y[:first_n:step].tolist() + hash_y[first_n:max_n:end_step].tolist(),
                            color='blue', marker='o', fillstyle='none', markersize=7, label='hash')

                    reranked_x = reranked_metrics['recall'][:np.where(reranked_metrics['recall']==1.0)[0][0]]
                    reranked_y = computation[:len(reranked_x)]

                    ax.plot(reranked_x[:first_n:step].tolist() + reranked_x[first_n:max_n:end_step].tolist(),  
                            reranked_y[:first_n:step].tolist() + reranked_y[first_n:max_n:end_step].tolist(),
                            color='red', marker='>', fillstyle='none', markersize=7, label='rerank')
                    ax.set(xlabel='Recall', ylabel='% Distance Computation')       
                    ax.set_xlim(0,1.0)
                    ax.set_ylim(-0.01,0.45)
                    ax.set_xticks(np.linspace(0.0,1.0, 6))
                    ax.legend(frameon=True, framealpha=1, borderpad=0.4, edgecolor='black')
                    plt.savefig(os.path.join(p['basedir'], f'{epoch+1}-top{topk}-recall-vs-computation.png'), bbox_inches='tight')
                    plt.close()

                print(
                    colored('[EPOCH {}] Top-{} Raw:    R@100 {:0.2f}, R@200 {:0.2f}, R@500: {:0.2f}'.format(epoch+1, topk, raw_metrics['recall'][100], raw_metrics['recall'][200], raw_metrics['recall'][500]), 'blue'))
                print(
                    colored('[EPOCH {}] Top-{} Hash:   R@100 {:0.2f}, R@200 {:0.2f}, R@500: {:0.2f}'.format(epoch+1, topk, hash_metrics['recall'][100], hash_metrics['recall'][200], hash_metrics['recall'][500]), 'blue'))
                print(
                    colored('[EPOCH {}] Top-{} Rerank: R@100 {:0.2f}, R@200 {:0.2f}, R@500: {:0.2f}'.format(epoch+1, topk, reranked_metrics['recall'][100], reranked_metrics['recall'][200], reranked_metrics['recall'][500]), 'blue'))
                
            recall500 = hash_metrics['recall'][500]
            if best_recall < recall500:
                print(colored('New best recall: %.2f -> %.2f' %(best_recall, recall500), 'green'))
                best_recall = recall500
                torch.save({'model': model.state_dict(), 'best_recall': recall500}, p['best_model_file'])
            else:
                print(colored('No new best recall: %.2f -> %.2f' %(best_recall, recall500), 'cyan'))




    print('Best loss: {:.04f}'.format(best_loss))
    
# def train(p, verbose=0, device=None, pretrain_states=None, epoch_train=epoch_train, get_output=get_output, run_internal_eval=run_internal_eval):
#     if device is None:
#         device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     print(colored('Using device: {}'.format(device), 'red'))
        
#     # Data
#     print(colored('Get dataset and dataloaders', 'blue'))
#     train_dataset = get_train_dataset(p)
#     val_dataset = get_val_dataset(p)
    
#     train_dataloader = get_train_dataloader(p, train_dataset)
#     val_dataloader = get_val_dataloader(p, val_dataset)
    
#     print('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)))

#     # Model
#     print(colored('Get model', 'blue'))
#     model = get_model(p, pretrain_path=None)
#     if verbose >= 3:
#         print(model)
    
#     #TODO: current not supported because of the problem in loading pretrained weights
#     #if device.type != 'cpu':
#     #    model = torch.nn.DataParallel(model)
#     model = model.to(device)

#     # Optimizer
#     if verbose >= 2:
#         print(colored('Get optimizer', 'blue'))
#     optimizer = get_optimizer(p, model)
#     if verbose >= 2:
#         print(optimizer)

#     # Loss function
#     if verbose >= 2:
#         print(colored('Get loss', 'blue'))
#     criterion = get_criterion(p) 
#     criterion.to(device)
#     if verbose >= 3:
#         print(criterion)

#     # Checkpoint
#     all_stats = []
#     if os.path.exists(p['model_checkpoint']):
#         print(colored('Restart from checkpoint {}'.format(p['model_checkpoint']), 'blue'))
#         checkpoint = torch.load(p['model_checkpoint'], map_location='cpu')
        
#         #load models
#         model.load_state_dict(checkpoint['model'])
#         optimizer.load_state_dict(checkpoint['optimizer'])        
        
#         #load tracking vars
#         start_epoch = checkpoint['epoch']
#         best_loss = checkpoint['best_loss']
#         all_stats = checkpoint['stats']
#         eval_losses = checkpoint['eval_losses'] if 'eval_losses' in checkpoint else {}
#         print(best_loss)
#     else:
#         print(colored('No checkpoint file at {}'.format(p['model_checkpoint']), 'blue'))
        
#         # initialize tracking vars
#         start_epoch = 0
#         best_loss = 1e4
#         eval_losses = {}
    
#     if pretrain_states is not None:
#         print(colored('[WARNING] Restart from provided pretrained states instead of checkpoint', 'red'))
#         for name, param in model.named_parameters():
#             print(name, param.shape)
#         model.load_state_dict(pretrain_states)

#     # Main loop
#     print(colored('Starting main loop', 'blue'))
#     for epoch in range(start_epoch, p['epochs']):
#         print(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
#         print(colored('-'*15, 'yellow'))

#         # Adjust lr
#         lr = adjust_learning_rate(p, optimizer, epoch)
#         print('Adjusted learning rate to {:.5f}'.format(lr))

#         # Train
#         print('Train ...')
#         start_time = datetime.now()
#         epoch_train(train_dataloader, model, criterion, optimizer, epoch, device=device)
#         duration = datetime.now() - start_time
#         print('Epoch {} finishes in {:.02f} minutes!'.format(epoch+1, duration.total_seconds()/60))
#         if (epoch+1) % p['epochs_per_internal_eval'] == 0 or epoch+1 == p['epochs']:                    
#             print('Evaluate based internally ...')
#             model.eval()
#             with torch.no_grad():
#                 val_output = get_output(val_dataloader, model, device=device)
#             internal_result = run_internal_eval(model, criterion, val_output)
#             eval_loss = internal_result['total_loss'].item()
#             eval_losses[epoch+1] = eval_loss
            
#             if best_loss > eval_loss:
#                 print(colored('New lowest loss: %.6f -> %.6f' %(best_loss, eval_loss), 'green'))
#                 best_loss = eval_loss
                
# #                 gt_weights=pretrain_states
# #                 print('BEFORE SAVE')
# #                 learned_params = list(model.named_parameters())
# #                 for name, learned_param in learned_params:
# #                     print((learned_param.detach().cpu() - gt_weights[name]).max().item())
                
#                 torch.save({'model': model.state_dict(), 'eval_loss': eval_loss}, p['model_file'])
                
# #                 print('AFTER SAVE')
# #                 model = get_model(p, pretrain_path=p['model_file']).to(device)
# #                 learned_params = list(model.named_parameters())
# #                 for name, learned_param in learned_params:
# #                     print((learned_param.detach().cpu() - gt_weights[name]).max().item())
#             else:
#                 print(colored('No new lowest loss: %.6f -> %.6f' %(best_loss, eval_loss), 'cyan'))

#         # Checkpoint
#         print('Checkpoint ...')
#         torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
#                     'epoch': epoch + 1, 'best_loss': best_loss, 'stats': all_stats, 'eval_losses': eval_losses},
#                      p['model_checkpoint'])
        
#         if (epoch+1) % p['epochs_per_external_eval'] == 0 or epoch+1 == p['epochs']:
#             with torch.no_grad():
#                 model.eval()
#                 test_loader = get_val_dataloader(p, get_test_dataset(p), batch_size=1024, num_workers=8)
#                 test_output = get_test_output(p, test_loader, model, device=device)

#             topk = 10
#             raw_metrics, hash_metrics, reranked_metrics, random_metrics, best_metrics = run_external_eval(p, test_output, topk=topk)
            
#             fig, ax = plt.subplots(1, 1)
#             step = 1000
#             N = len(raw_metrics['recall'])
#             computation = np.arange(0, len(raw_metrics['recall']), step) / N
#             sns.lineplot(raw_metrics['recall'][::step], computation , color='green', marker='o', label='raw', ax=ax)
#             sns.lineplot(hash_metrics['recall'][::step], computation, color='blue', marker='_', label='hash', ax=ax)
#             sns.lineplot(random_metrics['recall'][::step], computation, color='yellow', marker='*', label='random', ax=ax)
#             sns.lineplot(best_metrics['recall'][::step], computation, color='grey', marker='_', label='GT', ax=ax)

#             ax.set_xlabel('Recall')
#             ax.set_ylabel('% Distance Computation')
#             ax.set_ylim(0, 1)
#             ax.set_xlim(0, 1)
            
#             plt.savefig(os.path.join(p['basedir'], f'{epoch+1}-recall-vs-computation.png'), bbox_inches='tight')

#             print(
#                 colored('[EPOCH {}] Raw:    R@100 {:0.2f}, R@200 {:0.2f}, R@500: {:0.2f}'.format(epoch+1, raw_metrics['recall'][100], raw_metrics['recall'][200], raw_metrics['recall'][500]), 'blue'))
#             print(
#                 colored('[EPOCH {}] Hash:   R@100 {:0.2f}, R@200 {:0.2f}, R@500: {:0.2f}'.format(epoch+1, hash_metrics['recall'][100], hash_metrics['recall'][200], hash_metrics['recall'][500]), 'blue'))
#             print(
#                 colored('[EPOCH {}] Rerank: R@100 {:0.2f}, R@200 {:0.2f}, R@500: {:0.2f}'.format(epoch+1, reranked_metrics['recall'][100], reranked_metrics['recall'][200], reranked_metrics['recall'][500]), 'blue'))
    
#     print('Best loss: {:.04f}'.format(best_loss))