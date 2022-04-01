import os
import numpy as np
import torch
# print(NeuCFYelp_concate.NeuCFYelp_concate(query[0],train[20]))

class RandomPair_Dataset(torch.utils.data.Dataset):
    r"""user/item dataset where pairs are sampled randomly.

    Arguments:
        x1: users
        x2: items
        pos: list of positive indices
        neg: list of negative indices
        topk: select only top top
        negk: select only last negk
        pos_sampling_rate: ratio of sampling pos pairs, for neg, it's (1-pos_sampling_rate)
    """
    def __init__(self, x1, x2, 
                 pos=None, neg=None, topk=None, negk=None, in_out=False, pos_sampling_rate=0.5,
                 neural_sim_func='NeuCFYelp_concate', neg_sampling_type=None, neural_sim_func_prefix=None, neural_sim_func_N=None):
        super().__init__()
        self.x1 = x1
        self.x2 = x2
        self.N1 = self.x1.size(0)
        self.N2 = self.x2.size(0)
        self.pos = pos
        self.neg = neg
        self.topk = topk
        self.negk = negk
        self.in_out = in_out
        self.pos_sampling_rate = pos_sampling_rate
        self.neural_sim_func = neural_sim_func
        
        if neural_sim_func == 'NeuCFydata_DeepFM':
            from neural_functions import create_NeuCFydata_DeepFM
            self.neural_sim_func = create_NeuCFydata_DeepFM(
                os.path.join('saved_models/DeepFM/', '{}-{}.ckpt'.format(neural_sim_func_prefix, neural_sim_func_N)),
                device='cuda'
            )
        else:
            self.neural_sim_func = getattr(__import__(neural_sim_func), neural_sim_func) 
        print('Using neural function: {}'.format(neural_sim_func))
        
        if neg is not None:
            if neg_sampling_type is not None: #only takes effect when neg is not None
                if neg_sampling_type == 'revrank': #reversed ranking, indices in earlier ranked list have higher probs
                    print('Using neg sampling type: {}'.format(neg_sampling_type))
                    if self.negk is not None:
                        probs = 1/np.arange(1, self.negk+1)
                    else:
                        probs = 1/np.arange(1, len(self.neg)+1)
                    print(len(probs))
                    self.probs = probs / np.sum(probs)
                elif 'revrank_' in neg_sampling_type: #only sampling similar to revrank on the top 1k samples
                    topk_neg = int(neg_sampling_type[len('revrank_'):])
                    
                    print('Using neg sampling type: {}, revrank on {} topk neg samples'.format(neg_sampling_type, neg_sampling_type[len('revrank_'):]))
                    if self.negk is not None:
                        if self.negk > topk_neg:
                            probs = [i+1 for i in range(topk_neg)] + [0 for _ in range(topk_neg, self.negk)]
                        else:
                            probs = [i+1 for i in range(self.negk)]
                    else:
                        if len(self.neg) > topk_neg:
                            probs = [i+1 for i in range(topk_neg)] + [0 for _ in range(topk_neg, len(self.neg))]
                        else:
                            probs = [i+1 for i in range(len(self.neg))]
                    probs = np.array(probs)
                    print(len(probs), np.sum(probs), probs[0:10])
                    self.probs = probs / np.sum(probs)
                    
                elif 'uniform_' in neg_sampling_type: #only sampling uniformly on the top 1k samples
                    topk_neg = int(neg_sampling_type[len('uniform_'):])
                    
                    print('Using neg sampling type: {}, uniform on {} topk neg samples'.format(neg_sampling_type, neg_sampling_type[len('uniform_'):]))
                    if self.negk is not None:
                        if self.negk > topk_neg:
                            probs = [1 for _ in range(topk_neg)] + [0 for _ in range(topk_neg, self.negk)]
                        else:
                            probs = [1 for _ in range(self.negk)]
                    else:
                        if len(self.neg) > topk_neg:
                            probs = [1 for _ in range(topk_neg)] + [0 for _ in range(topk_neg, len(self.neg))]
                        else:
                            probs = [1 for _ in range(len(self.neg))]
                    probs = np.array(probs)
                    print(len(probs), np.sum(probs), probs[0:10])
                    self.probs = probs / np.sum(probs)
            else:
                self.probs = None
        
    def _sample_id2(self, id1):
        toss = np.random.rand()
        is_pos = None
        if toss > self.pos_sampling_rate:
            if self.neg is not None:
                if self.negk is not None:
                    #print('Sampling negative')
                    is_pos = False
                    id2 = np.random.choice(self.neg[id1, -self.negk:], 1, p=self.probs)[0]
                else:
                    id2 = np.random.choice(self.neg[id1], 1, p=self.probs)[0]
            else:
                id2 = np.random.choice(self.N2, 1)[0]
        else:
            if self.pos is not None and self.topk is not None:
                #print('Sampling positive')
                id2 = np.random.choice(self.pos[id1, :self.topk], 1)[0]
                is_pos = True
            else:
                id2 = np.random.choice(self.pos[id1], 1)[0]
        return id2, is_pos #this flag is for 1-0 flag in specfic training case
        
    def __getitem__(self, index):
        x1 = self.x1[index]
        id2, is_pos = self._sample_id2(index)
        x2 = self.x2[id2]
        
        if self.in_out and is_pos is not None:
            sim = 1.0 if is_pos else 0.0
        else:
            sim = self.neural_sim_func(x1.numpy(), x2.numpy())

        return (index, x1, id2, x2, torch.tensor(sim))

    def __len__(self):
        return self.N1 

class CacheRandomPair_Dataset(RandomPair_Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, x1, x2, pos=None, neg=None, topk=None, negk=None, in_out=False, pos_sampling_rate=0.5):
        super().__init__()
        self.x1 = x1
        self.x2 = x2
        self.N1 = self.x1.size(0)
        self.N2 = self.x2.size(0)
        self.pos = pos
        self.neg = neg
        self.topk = topk
        self.negk = negk
        self.in_out = in_out
        self.pos_sampling_rate = pos_sampling_rate
        
        self.cache = {}
        
    def __getitem__(self, index):
        x1 = self.x1[index]
        id2, is_pos = self._sample_id2(index)
        x2 = self.x2[id2]
        
        k = (index, id2)
        if k in self.cache:
            sim = self.cache[k]
        else:
            if self.in_out and is_pos is not None:
                sim = 1.0 if is_pos else 0.0
            else:
                sim = NeuCFYelp_concate.NeuCFYelp_concate(x1.numpy(), x2.numpy())

            self.cache[k] = sim
        return (index, x1, id2, x2, torch.tensor(sim))

    def __len__(self):
        return self.N1 
    
class RandomTriplet_Dataset(RandomPair_Dataset):
    def __init__(self, x1, x2, pos=None, neg=None, topk=None, negk=None):
        super().__init__(x1, x2, pos=pos, neg=neg, topk=topk, negk=negk)
        if self.topk is not None:
            self.topk = pos.shape[1]
        if self.negk is None:
            self.negk = neg.shape[1]
        
    def __getitem__(self, index):
        x = self.x1[index]
        
        pos_id = np.random.choice(self.pos[index, :self.topk], 1)[0]
        pos = self.x2[pos_id]
        pos_sim = torch.tensor(NeuCFYelp_concate.NeuCFYelp_concate(x.numpy(), pos.numpy()))
        
        neg_id = np.random.choice(self.neg[index, -self.negk:], 1)[0]
        neg = self.x2[neg_id] 
        neg_sim = torch.tensor(NeuCFYelp_concate.NeuCFYelp_concate(x.numpy(), neg.numpy()))
        
        return {'id': index, 'x': x, 
                'pos_id': pos_id, 'pos': pos, 'target_pos': pos_sim,
                'neg_id': neg_id, 'neg': neg, 'target_neg': neg_sim}

    def __len__(self):
        return self.N1
    
class RankedNeuCF_Dataset(torch.utils.data.Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, x1, x2, ranking, k=100):
        super().__init__()
        self.x1 = x1
        self.x2 = x2
        self.N1 = self.x1.size(0)
        self.N2 = self.x2.size(0)
        self.ranking = ranking
        self.topk = ranking[:, :k]
        
    def _sample_id2(self, id1):
        return np.random.choice(self.topk[id1], 1)[0]
    
    def __getitem__(self, index):
        id1 = index
        id2 = self._sample_id2(id1)
        x1 = self.x1[id1]
        x2 = self.x2[id2]
        sim = NeuCFYelp_concate.NeuCFYelp_concate(x1.numpy(), x2.numpy())
        return (id1, x1, id2, x2, torch.tensor(sim))

    def __len__(self):
        return self.N1 

class NegativeRankedNeuCF_Dataset(torch.utils.data.Dataset):
    def __init__(self, x1, x2, topk, neg):
        super().__init__()
        self.x1 = x1
        self.x2 = x2
        self.N1 = self.x1.size(0)
        self.N2 = self.x2.size(0)
        self.topk = topk
        self.neg = neg
        
    def __getitem__(self, index):
        x = self.x1[index]
        
        pos_id = np.random.choice(self.topk[index], 1)[0]
        pos = self.x2[pos_id]
        pos_sim = torch.tensor(NeuCFYelp_concate.NeuCFYelp_concate(x.numpy(), pos.numpy()))
                   
        neg_id = np.random.choice(self.neg[index], 1)[0]
        neg = self.x2[neg_id] 
        neg_sim = torch.tensor(NeuCFYelp_concate.NeuCFYelp_concate(x.numpy(), neg.numpy()))
        
        return {'id': index, 'x': x, 
                'pos_id': pos_id, 'pos': pos, 'pos_sim': pos_sim,
                'neg_id': neg_id, 'neg': neg, 'neg_sim': neg_sim}

    def __len__(self):
        return self.N1     
    
# class NegativeRankedNeuCF_Dataset(torch.utils.data.Dataset):
#     def __init__(self, x1, x2, topk, neg):
#         super().__init__()
#         self.x1 = x1
#         self.x2 = x2
#         self.N1 = self.x1.size(0)
#         self.N2 = self.x2.size(0)
#         self.ranking = ranking
#         self.topk = topk
#         self.neg = neg
        
#     def __getitem__(self, index):
#         x = self.x1[index]
        
#         pos_id = np.random.choice(self.topk[index], 1)[0]
#         pos = self.x2[pos_id]
#         pos_sim = torch.tensor(NeuCFYelp_concate.NeuCFYelp_concate(x.numpy(), pos.numpy()))
                   
#         neg_id = np.random.choice(self.neg[index], 1)[0]
#         neg = self.x2[neg_id] 
#         neg_sim = torch.tensor(NeuCFYelp_concate.NeuCFYelp_concate(x.numpy(), neg.numpy()))
        
#         return {'id': index, 'x': x, 
#                 'pos_id': pos_id, 'pos': pos, 'pos_sim': pos_sim,
#                 'neg_id': neg_id, 'neg': neg, 'neg_sim': neg_sim}

#     def __len__(self):
#         return self.N1 
    
class TripletRankedNeuCF_Dataset(torch.utils.data.Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, x1, x2, ranking, k=100, neg_k=1000, neg_m=10):
        super().__init__()
        self.x1 = x1
        self.x2 = x2
        self.N1 = self.x1.size(0)
        self.N2 = self.x2.size(0)
        self.ranking = ranking
        self.topk = ranking[:, :k]
        self.neg = ranking[:, -neg_k:]
        
    def _sample_neg_id2(self, id1, m):
        return np.random.choice(self.neg[id1], m)
    
    def __getitem__(self, index):
        x = self.x1[index]
        
        pos_id = np.random.choice(self.topk[index], 1)[0]
        pos = self.x2[pos_id]
        pos_sim = torch.tensor(NeuCFYelp_concate.NeuCFYelp_concate(x.numpy(), pos.numpy()))
                   
        neg_ids = self._sample_neg_id2(index) #multiple
        neg = self.x2[id2s_neg] 
        neg_sims = torch.tensor([NeuCFYelp_concate.NeuCFYelp_concate(x.numpy(), x2.numpy()) for x2 in neg])
        
        return {'id': index, 'x': x, 
                'pos_id': pos_id, 'pos': pos, 'pos_sim': pos_sim,
                'neg_ids': neg_ids, 'neg': neg, 'neg_sims': neg_sims}

    def __len__(self):
        return self.N1 
           
class PairNeuCF_Dataset(torch.utils.data.Dataset):
    r"""Iterate over pair of tensor for pairwise output

    Arguments:
        x1 (Tensor): 
        x2 (Tensor)
    """

    def __init__(self, x1, x2, neural_sim_func='NeuCFYelp_concate', neural_sim_func_prefix=None, neural_sim_func_N=None):
        super().__init__()
        self.x1 = x1
        self.N1 = self.x1.size(0)
        self.x2 = x2
        self.N2 = self.x2.size(0)
        
        #self.neural_sim_func = getattr(__import__(neural_sim_func), neural_sim_func)
        if neural_sim_func == 'NeuCFydata_DeepFM':
            from neural_functions import create_NeuCFydata_DeepFM
            self.neural_sim_func = create_NeuCFydata_DeepFM(
                os.path.join('saved_models/DeepFM/', '{}-{}.ckpt'.format(neural_sim_func_prefix, neural_sim_func_N)),
                device='cuda'
            )
        else:
            self.neural_sim_func = getattr(__import__(neural_sim_func), neural_sim_func) 
        
    def __getitem__(self, index):
        try:
            id1 = int(index / self.N2)
            id2 = index % self.N2
            x1 = self.x1[id1]
            x2 = self.x2[id2]
        except Exception as e:
            print(index, id1, id2, self.N1, self.N2)
            raise e
        sim = self.neural_sim_func(x1.numpy(), x2.numpy())
        return (id1, x1, id2, x2, torch.tensor(sim))

    def __len__(self):
        return self.N1 * self.N2
    
    
class RandomPairNeuCF_Dataset(torch.utils.data.Dataset):
    r"""Iterate over pair of tensor for pairwise output

    Arguments:
        x1 (Tensor): 
        x2 (Tensor)
    """

    def __init__(self, x1, x2, neural_sim_func='NeuCFYelp_concate', neural_sim_func_prefix=None, neural_sim_func_N=None):
        super().__init__()
        self.x1 = x1
        self.N1 = self.x1.size(0)
        self.x2 = x2
        self.N2 = self.x2.size(0)
        
        #self.neural_sim_func = getattr(__import__(neural_sim_func), neural_sim_func)
        if neural_sim_func == 'NeuCFydata_DeepFM':
            from neural_functions import create_NeuCFydata_DeepFM
            self.neural_sim_func = create_NeuCFydata_DeepFM(
                os.path.join('saved_models/DeepFM/', '{}-{}.ckpt'.format(neural_sim_func_prefix, neural_sim_func_N)),
                device='cuda'
            )
        else:
            self.neural_sim_func = getattr(__import__(neural_sim_func), neural_sim_func) 
        
    def __getitem__(self, index):
        try:
            id1 = index
            id2 = np.random.randint(0, self.N2)
            x1 = self.x1[id1]
            x2 = self.x2[id2]
        except Exception as e:
            print(index, id1, id2, self.N1, self.N2)
            raise e
        sim = self.neural_sim_func(x1.numpy(), x2.numpy())
        return (id1, x1, id2, x2, torch.tensor(sim))

    def __len__(self):
        return self.N1
        

# class NeuCF_Dataset(torch.utils.data.Dataset):
#     r"""Dataset wrapping tensors.

#     Each sample will be retrieved by indexing tensors along the first dimension.

#     Arguments:
#         *tensors (Tensor): tensors that have the same size of the first dimension.
#     """

#     def __init__(self, x, y, train=False):
#         super().__init__()
#         self.x = data
#         self.train = train
#         self.N = self.data.size(0)
        
#     def __getitem__(self, index):
#         x1 = self.data[index]
#         if self.train:
#             x2 = self.data[np.random.choice(self.N, 1)[0]]
#             sim = NeuCFYelp_concate.NeuCFYelp_concate(x1.numpy(), x2.numpy())
#             return (x1, x2, torch.tensor(sim))
#         else:
#             return (x1,)

#     def __len__(self):
#         return self.N