import torch
from torch import nn

def print_tensor_info(*tensors):
    print('  '.join(['[{:0.5f}, {:0.5f}]'.format(tensor.min().item(), tensor.max().item()) for tensor in tensors]))

def get_activation_fn(n):
    if n == 'lrelu':
        return nn.LeakyReLU(inplace=True)
    elif n == 'relu':
        return nn.ReLU(inplace=True)
    else:
        return None

class HashModel(nn.Module):
    def __init__(self):
        super().__init__()

def forward_module_list(modules, x):
    for m in modules:
        x = m(x)
    return x

def add_mlp_layers(m, current_dim, dims, use_bn=False, dropout=0.0, activation=nn.ReLU(inplace=True)):
    for idx, dim in enumerate(dims):
        m.append(nn.Linear(current_dim, dim))
        if idx >= 1:
            if use_bn:
                m.append(nn.BatchNorm1d(dim))
        if activation is not None:
            m.append(activation)
        if dropout > 0:
            m.append(nn.Dropout(dropout))
        current_dim = dim
    return current_dim

class PairHashModel(HashModel):
    """https://github.com/popfido/PairCNN-Ranking/blob/master/models/PairCNN.py"""
    def __init__(self, x_dim, tower_dims, common_dims, h_dim, use_bn=False, dropout=0.0):
        super().__init__()
        
        self.tower1 = nn.ModuleList([])
        self.tower2 = nn.ModuleList([])
        self.common = nn.ModuleList([])
        current_dim = x_dim
        
        # Towers
        _ = add_mlp_layers(self.tower1, current_dim, tower_dims, use_bn=use_bn, dropout=dropout)
        current_dim = add_mlp_layers(self.tower2, current_dim, tower_dims, use_bn=use_bn, dropout=dropout)        
        current_dim = add_mlp_layers(self.common, current_dim, common_dims,  use_bn=use_bn, dropout=dropout)
        self.hash_output = nn.Linear(current_dim, h_dim)

class MLPConcat(HashModel):
    """ This is a non-hashing model for testing, hash code is the same as input
    """
    def __init__(self, x_dim, hidden_dims, h_dim, use_bn=False, dropout=0.0, activation=None):
        super().__init__()
        
        assert x_dim == h_dim
        
        self.hidden = nn.ModuleList([])

        current_dim = add_mlp_layers(self.hidden, x_dim * 2, hidden_dims, 
                                     use_bn=use_bn, dropout=dropout, activation=get_activation_fn(activation))
        self.output = nn.Linear(current_dim, 1)
    
    def forward(self, x1, x2, return_only_hash=False):
        out1 = x1
        out2 = x2
        if return_only_hash:
            return out1, out2 
        else:
            x_join = torch.cat([x1, x2], dim=1)
            similarity = torch.sigmoid(self.output(forward_module_list(self.hidden, x_join))).squeeze()
            return out1, out2, similarity
        
class TwoTowerHashModel_V0(HashModel):
    """ Two tower model where similarity~<TANH(h1), TANH(h2)>
    """
    def __init__(self, x_dim, tower_dims, h_dim, mlp_dims, use_bn=False, dropout=0.0):
        super().__init__()
        
        self.tower1 = nn.ModuleList([])
        self.tower2 = nn.ModuleList([])
        self.common = nn.ModuleList([])

        current_dim = add_mlp_layers(self.tower1, x_dim, tower_dims, use_bn=use_bn, dropout=dropout)
        self.hash1 = torch.nn.Linear(current_dim, h_dim)
        current_dim = add_mlp_layers(self.tower2, x_dim, tower_dims, use_bn=use_bn, dropout=dropout)
        self.hash2 = torch.nn.Linear(current_dim, h_dim)
        
        # self.M = nn.Parameter(torch.ones(h_dim, h_dim))
        self.similarity_layer = nn.Bilinear(h_dim, h_dim, 1)
        
        self.linear = nn.ModuleList([])
        current_dim = add_mlp_layers(self.linear, h_dim * 2 + 1, mlp_dims, use_bn=False, dropout=0.0)
        self.output = nn.Linear(current_dim, 1)

    def _forward_module_list(self, modules, x):
        for m in modules:
            x = m(x)
        return x
    
    def forward(self, x1, x2, return_only_hash=False):
        out1 = self.hash1(self._forward_module_list(self.common, self._forward_module_list(self.tower1, x1)))
        out2 = self.hash2(self._forward_module_list(self.common, self._forward_module_list(self.tower2, x2)))
        if return_only_hash:
            return out1, out2 
        else:
            b, n = out1.size()
            sim = self.similarity_layer(out1, out2).view(b, 1)
            x_join = torch.cat([out1, sim, out2], dim=1)
            similarity = torch.sigmoid(self.output(self._forward_module_list(self.linear, x_join))).squeeze()
            return out1, out2, similarity 
        
# class TwoTowerHashModel_V1(HashModel):
#     """ Two tower model where similarity~<TANH(h1), TANH(h2)>
#     """
#     def __init__(self, x_dim, tower_dims, common_dims, h_dim, use_bn=False, dropout=0.0):
#         super().__init__()
        
#         self.tower1 = nn.ModuleList([])
#         self.tower2 = nn.ModuleList([])
#         self.common = nn.ModuleList([])
#         current_dim = x_dim
        
#         _ = add_mlp_layers(self.tower1, current_dim, tower_dims, use_bn=use_bn, dropout=dropout)
#         current_dim = add_mlp_layers(self.tower2, current_dim, tower_dims, use_bn=use_bn, dropout=dropout)
#         current_dim = add_mlp_layers(self.common, current_dim, common_dims,  use_bn=use_bn, dropout=dropout)
#         self.hash_output = nn.Linear(current_dim, h_dim)

#     def _forward_module_list(self, modules, x):
#         for m in modules:
#             x = m(x)
#         return x
    
#     def forward(self, x1, x2, return_only_hash=False):
#         out1 = self.hash_output(self._forward_module_list(self.common, self._forward_module_list(self.tower1, x1)))
#         out2 = self.hash_output(self._forward_module_list(self.common, self._forward_module_list(self.tower2, x2)))
#         if return_only_hash:
#             return out1, out2 
#         else:
#             h1 = torch.tanh(out1)
#             h2 = torch.tanh(out2)
#             b, n = h1.size()
#             similarity = torch.bmm(h1.view(b, 1, n), h2.view(b, n, 1)).squeeze() / n
#             # print_tensor_info(h1, h2, similarity)
#             similarity = (similarity + 1) / 2
#             return out1, out2, similarity 
        
class TwoTowerHashModel_V1(HashModel):
    """ Two tower model where similarity~<TANH(h1), TANH(h2)>
    """
    def __init__(self, x_dim=None, tower_dims=None, common_dims=None, h_dim=None, use_bn=False, dropout=0.0,
                x1_dim=None, x2_dim=None, tower1_dims=None, tower2_dims=None):
        super().__init__()
        
        if x_dim is not None: #x_dim takes priority over individual dims
            x1_dim = x_dim
            x2_dim = x_dim
            
        if tower_dims is not None:
            tower1_dims = tower_dims
            tower2_dims = tower_dims
        else:
            assert tower1_dims[-1] == tower2_dims[-1]
        
        self.tower1 = nn.ModuleList([])
        self.tower2 = nn.ModuleList([])
        self.common = nn.ModuleList([])

        _ = add_mlp_layers(self.tower1, x1_dim, tower1_dims, use_bn=use_bn, dropout=dropout)
        current_dim = add_mlp_layers(self.tower2, x2_dim, tower2_dims, use_bn=use_bn, dropout=dropout)
        current_dim = add_mlp_layers(self.common, current_dim, common_dims,  use_bn=use_bn, dropout=dropout)
        self.hash_output = nn.Linear(current_dim, h_dim)

    def _forward_module_list(self, modules, x):
        for m in modules:
            x = m(x)
        return x
    
    def forward(self, x1, x2, return_only_hash=False):
        out1 = self.hash_output(self._forward_module_list(self.common, self._forward_module_list(self.tower1, x1)))
        out2 = self.hash_output(self._forward_module_list(self.common, self._forward_module_list(self.tower2, x2)))
        if return_only_hash:
            return out1, out2 
        else:
            h1 = torch.tanh(out1)
            h2 = torch.tanh(out2)
            b, n = h1.size()
            similarity = torch.bmm(h1.view(b, 1, n), h2.view(b, n, 1)).squeeze() / n
            # print_tensor_info(h1, h2, similarity)
            similarity = (similarity + 1) / 2
            return out1, out2, similarity 
        
        
class MultipleHeadsTwoTowerHashModel_V1(HashModel):
    """ Two tower model where similarity~<TANH(h1), TANH(h2)>
    """
    def __init__(self, x_dim, tower_dims, common_dims, h_dim, num_h_tables, use_bn=False, dropout=0.0):
        super().__init__()
        
        self.tower1 = nn.ModuleList([])
        self.tower2 = nn.ModuleList([])
        self.common = nn.ModuleList([])
        current_dim = x_dim
        
        _ = add_mlp_layers(self.tower1, current_dim, tower_dims, use_bn=use_bn, dropout=dropout)
        current_dim = add_mlp_layers(self.tower2, current_dim, tower_dims, use_bn=use_bn, dropout=dropout)
        current_dim = add_mlp_layers(self.common, current_dim, common_dims,  use_bn=use_bn, dropout=dropout)
        
        self.hash_output = nn.ModuleList([nn.Linear(current_dim, h_dim) for i in range(num_h_tables)])
        self.num_h_tables = num_h_tables

    def _forward_module_list(self, modules, x):
        for m in modules:
            x = m(x)
        return x
    
    def forward(self, x1, x2, return_only_hash=False):
        out1 = self._forward_module_list(self.common, self._forward_module_list(self.tower1, x1))
        out2 = self._forward_module_list(self.common, self._forward_module_list(self.tower2, x2))
        
        out1 = [self.hash_output[i](out1) for i in range(num_h_tables)]
        out2 = [self.hash_output[i](out2) for i in range(num_h_tables)]
        
        if return_only_hash:
            return out1, out2 
        else:
            h1 = [torch.tanh(o) for o in out1]
            h2 = [torch.tanh(o) for o in out2]
            b, n = h1[0].size()
            similarity = [torch.bmm(e1.view(b, 1, n), e2.view(b, n, 1)).squeeze() / n for (e1, e2) in zip(h1, h2)]
            # print_tensor_info(h1, h2, similarity)
            similarity = [(s + 1) / 2 for s in similarity]
            return out1, out2, similarity
        
class TwoTowerHashModel_V2(TwoTowerHashModel_V1):
    """ Two tower model where similarity=SIGMOID(<out1, out2>)
    """
    def forward(self, x1, x2, return_only_hash=False):
        out1 = self.hash_output(self._forward_module_list(self.common, self._forward_module_list(self.tower1, x1)))
        out2 = self.hash_output(self._forward_module_list(self.common, self._forward_module_list(self.tower2, x2)))
        if return_only_hash:
            return out1, out2 
        else:
            b, n = out1.size()
            similarity = torch.bmm(out1.view(b, 1, n), out2.view(b, n, 1)).squeeze()
            similarity = torch.sigmoid(similarity)
            return out1, out2, similarity  
        
class TwoTowerHashModel_DSH(HashModel):
    def __init__(self, x_dim, tower_dims, common_dims, h_dim, use_bn=False, dropout=0.0):
        super().__init__()
        
        self.tower1 = nn.ModuleList([])
        self.tower2 = nn.ModuleList([])
        self.common = nn.ModuleList([])
        current_dim = x_dim
        
        _ = add_mlp_layers(self.tower1, current_dim, tower_dims, use_bn=use_bn, dropout=dropout) #tower 1
        current_dim = add_mlp_layers(self.tower2, current_dim, tower_dims, use_bn=use_bn, dropout=dropout) #tower 2
        current_dim = add_mlp_layers(self.common, current_dim, common_dims,  use_bn=use_bn, dropout=dropout) #common lane
        self.hash_output = nn.Linear(current_dim, h_dim) #output hash layer
        
    def forward(self, x1, x2, return_only_hash=False):
        #x = self._forward_module_list(self.tower1, x1)
        #print(x.min().item(), x.max().item())
        out1 = self.hash_output(forward_module_list(self.common, forward_module_list(self.tower1, x1)))
        out2 = self.hash_output(forward_module_list(self.common, forward_module_list(self.tower2, x2)))

        if return_only_hash:
            return out1, out2 
        else:
            b, n = out1.size()
            similarity = torch.bmm(out1.view(b, 1, n), out2.view(b, n, 1)).squeeze()
            similarity = 0.5 * (1 + similarity / n) # range [0, 1]
            return out1, out2, similarity
        
class TwoTowerHashModel_DSH_Tanh(TwoTowerHashModel_DSH):
    def forward(self, x1, x2, return_only_hash=False):
        #x = self._forward_module_list(self.tower1, x1)
        #print(x.min().item(), x.max().item())
        out1 = self.hash_output(forward_module_list(self.common, forward_module_list(self.tower1, x1)))
        out2 = self.hash_output(forward_module_list(self.common, forward_module_list(self.tower2, x2)))

        if return_only_hash:
            return out1, out2 
        else:
            b, n = out1.size()
            h1 = torch.tanh(out1)
            h2 = torch.tanh(out2)
            similarity = torch.bmm(h1.view(b, 1, n), h2.view(b, n, 1)).squeeze() #range [-n, n]
            similarity = 0.5 * (1 + similarity / n) # range [0, 1]
            return out1, out2, similarity
        
class TwoTowerHashModel_V1_Triplet(TwoTowerHashModel_V1):
    def forward(self, x, pos, neg, return_only_hash=False):
        out = self.forward_tower_1(x)
        out_pos = self.forward_tower_2(pos)
        out_neg = self.forward_tower_2(neg)

        if return_only_hash:
            return out, out_pos, out_neg
        else:
            b, n = out.size()

            h = torch.tanh(out)
            h_pos = torch.tanh(out_pos)
            h_neg = torch.tanh(out_neg)
            
            similarity_pos = torch.bmm(h.view(b, 1, n), h_pos.view(b, n, 1)).squeeze() / n
            similarity_neg = torch.bmm(h.view(b, 1, n), h_neg.view(b, n, 1)).squeeze() / n
            
            similarity_pos = 0.5 * (1 + similarity_pos)
            similarity_neg = 0.5 * (1 + similarity_neg)

            return out, out_pos, out_neg, similarity_pos, similarity_neg
        
    def forward_similarity(self, x1, x2, return_only_hash=False):
            out1 = self.forward_tower_1(x1)
            out2 = self.forward_tower_2(x2)
            h1 = torch.tanh(out1)
            h2 = torch.tanh(out2)

            if return_only_hash:
                return out1, out2
            else:
                b, n = h1.size()
                similarity = torch.bmm(h1.view(b, 1, n), h2.view(b, n, 1)).squeeze() / n
                return out1, out2, similarity
            
    def forward_tower_1(self, x):
        return self.hash_output(forward_module_list(self.common, forward_module_list(self.tower1, x)))
    def forward_tower_2(self, x):
        return self.hash_output(forward_module_list(self.common, forward_module_list(self.tower2, x)))
        
class TwoTowerHashModel_V2_Neg(TwoTowerHashModel_V1):
    def forward(self, x, pos, neg=None, return_only_hash=False):
        h = self.hash_output(self._forward_module_list(self.common, self._forward_module_list(self.tower1, x)))
        h_pos = self.hash_output(self._forward_module_list(self.common, self._forward_module_list(self.tower2, pos)))
        if neg is not None:
            h_neg = self.hash_output(self._forward_module_list(self.common, self._forward_module_list(self.tower2, neg)))
        else:
            h_neg = None
        if return_only_hash:
            return h, h_pos, h_neg
        else:
            b, n = h.size()
            similarity_pos = torch.bmm(h.view(b, 1, n), h_pos.view(b, n, 1)).squeeze()
            similarity_pos = torch.sigmoid(similarity_pos)
            
            if neg is not None:
                similarity_neg = torch.bmm(h.view(b, 1, n), h_neg.view(b, n, 1)).squeeze()
                similarity_neg = torch.sigmoid(similarity_neg)
            else:
                similarity_neg = None
            return h, h_pos, h_neg, similarity_pos, similarity_neg
        
class TwoTowerHashModel_V2_Triplet(TwoTowerHashModel_V1):
    def forward(self, data):
        x = data['x']
        x_pos = data['pos']
        x_neg = data['neg']
        b, num_neg, n = x_neg.size()
        x_neg = n_neg.view(b*num_neg, n) #view for forward, reshape later
          
        h = self.forward_tower_1(x)
        h_pos = self.forward_tower_2(x_pos)
        h_neg = self.forward_tower_2(x_neg).view(b, num_neg, n)
        
        pos_similarity = torch.bmm(h.view(b, 1, n), h_pos.view(b, n, 1)).squeeze()
        
        neg_similarity = []
        for i in range(num_neg):
            neg_similarity.append(torch.bmm(h.view(b, 1, n), h_neg[:,i,:].view(b, n, 1)))
        neg_similarity = torch.cat(neg_similarity)
              
        return h, h_pos, h_neg, torch.sigmoid(pos_similarity), torch.sigmoid(neg_similarity)
        
    def forward_tower_1(self, x):
        return self.hash_output(self._forward_module_list(self.common, self._forward_module_list(self.tower1, x)))
    def forward_tower_2(self, x):
        return self.hash_output(self._forward_module_list(self.common, self._forward_module_list(self.tower2, x)))