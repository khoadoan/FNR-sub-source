from torch import nn
import torch

EPS = 1e-6

class Loss_DSH(nn.Module):
    def __init__(self, contrastive_weight, quantization_weight, m=None):
        super().__init__()
        self.mse = nn.MSELoss()
        
        self.contrastive_weight = contrastive_weight
        self.quantization_weight = quantization_weight
        self.m = m

    def _to_hash_code(self, anchors):
        anchors_prob = torch.sign(anchors)            
        return anchors_prob    
    
    def _calculate_prediction_loss(self, prediction, target):
        consistency_loss = self.mse(prediction, target)
        return consistency_loss
    
    def _calculate_contrastive_loss(self, target, out1, out2):
        dim = out1.shape[1]
        if self.m is None:
            m = dim * 2
        else:
            m = self.m
        d = (out1-out2).square().sum(dim=1)
        contrastive_loss = 0.5 * target * d  + 0.5 * (1-target) * (m-d).clamp(min=0) 
        #LABELS ARE REVERSED OF SDH PAPER: 0 FOR NOT DISSIM (SIM IN DSH), 1 FOR SIM (DISSIM IN DSH)
        #print((0.5 * (1-target) * d).max().item(), (0.5 * target * (self.m-d)).max().item())
        contrastive_loss = contrastive_loss.mean()
        
        return contrastive_loss
    def _calculate_quantization_loss(self, out1, out2):
        reg1_loss = (out1.abs()-1).abs().sum(dim=1).mean()
        reg2_loss = (out2.abs()-1).abs().sum(dim=1).mean()
        regularization_loss = 0.5 * (reg1_loss + reg2_loss)
        return regularization_loss
    
    def forward(self, h1, h2, prediction, target, model=None):
        """
        input:
            - h1: ex1 output (before binarization)
            - h2: ex2 output (before binarization)
            - prediction: predicted similarity score [0, 1]
            - target: target similarity score [0, 1]

        output:
            - Loss
        """
        # Instead of the original Softmax, we use sigmoid, so each bit is assume inpdenently distributed
        b, n = h1.size()

        # Similarity in output space
        prediction_loss = self._calculate_prediction_loss(prediction, target)
        
        # Uniform frequency
        if self.contrastive_weight > 0:
            contrastive_loss = self._calculate_contrastive_loss(target, h1, h2)
        else:
            contrastive_loss = 0.0
        
        # Independent bit loss
        if self.quantization_weight > 0:
            quantization_loss = self._calculate_quantization_loss(h1, h2)
        else:
            quantization_loss = 0.0
        
        total_loss = prediction_loss + self.contrastive_weight * contrastive_loss \
                    + self.quantization_weight * quantization_loss
        
        return total_loss, prediction_loss, contrastive_loss, quantization_loss
    
class Loss_V1(nn.Module):
    """Compute loss
    1) mse prediction loss, 2) uniform freq on tanh, 3) independent on W
    """
    def __init__(self, uniform_frequency_weight, independent_bit_weight):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        
        self.uniform_frequency_weight = uniform_frequency_weight # Default = 2.0
        self.independent_bit_weight = independent_bit_weight

    def _to_hash_code(self, anchors):
        anchors_prob = torch.tanh(anchors)            
        return anchors_prob    
    
    def _calculate_prediction_loss(self, prediction, target, h1, h2, input_as_hash_codes=False):
        consistency_loss = self.mse(prediction, target)
        return consistency_loss

    def _calculate_uniform_frequency_loss(self, h, input_as_hash_codes=False):
        bsize = h.size(0)
        
        if not input_as_hash_codes:
            h = self._to_hash_code(h)
            
        # This is equivalent to the original paper, uniform frequency, but on sigmoid activation
        f = torch.mean(h, dim = 0)
        uniform_frequency_bits =  f.abs().mean()  # maximum entropy

        return uniform_frequency_bits
    
    def _calculate_independent_bit_loss(self, model):
        """ Orthogonal loss
        https://github.com/kevinzakka/pytorch-goodies
        """
        param = list(model.parameters())[-2]
        param_flat = param.view(param.shape[0], -1)

        sym = torch.mm(param_flat, torch.t(param_flat))
        sym -= torch.eye(param_flat.shape[0]).to(param.get_device())
        orth_loss = sym.abs().sum() / param.shape[0]
        
        return orth_loss   
    
    def forward(self, h1, h2, prediction, target, model=None):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        # Instead of the original Softmax, we use sigmoid, so each bit is assume inpdenently distributed
        b, n = h1.size()
        h1 = self._to_hash_code(h1)
        h2 = self._to_hash_code(h2)

        # Similarity in output space
        prediction_loss = self._calculate_prediction_loss(prediction, target, h1, h2, input_as_hash_codes=True)
        
        # Uniform frequency
        if self.uniform_frequency_weight > 0:
            uniform_frequency_loss = 0.5 * ( 
                self._calculate_uniform_frequency_loss(h1, input_as_hash_codes=True) 
                + self._calculate_uniform_frequency_loss(h2, input_as_hash_codes=True))
        else:
            uniform_frequency_loss = 0.0
        
        # Independent bit loss
        if self.independent_bit_weight > 0:
            independent_bit_loss = self._calculate_independent_bit_loss(model)
        else:
            independent_bit_loss = 0.0
        
        total_loss = prediction_loss + self.uniform_frequency_weight * uniform_frequency_loss \
                    + self.independent_bit_weight * independent_bit_loss
        
        return total_loss, prediction_loss, uniform_frequency_loss, independent_bit_loss
    
class Loss_V1_Contrastive(Loss_V1):
    """Compute loss
    1) mse prediction loss, 2) uniform freq on tanh, 3) independent on W
    """
    def __init__(self, uniform_frequency_weight, independent_bit_weight, consistency_weight, m=None):
        super().__init__(uniform_frequency_weight, independent_bit_weight)
        self.consistency_weight = consistency_weight
        self.m = m
        
    def _calculate_prediction_loss(self, prediction, target, h1, h2, input_as_hash_codes=False):
        consistency_loss = self.mse(prediction, target)
        
        dim = h1.shape[1]
        if self.m is None:
            m = dim * 2
        d = (h1-h2).square().sum(dim=1)
        #print(h1.min().item(), h1.max().item(), d.min().item(), d.max().item())
        contrastive_loss = 0.5 * target * d  + 0.5 * (1-target) * (m-d).clamp(min=0) 
        
        return consistency_loss * self.consistency_weight + contrastive_loss.mean()
    
class Loss_V1_Triplet(Loss_V1):
    """Compute loss
    1) mse prediction loss, 2) uniform freq on tanh, 3) independent on W
    """
    
    def __init__(self, uniform_frequency_weight, independent_bit_weight, margin):
        super().__init__(uniform_frequency_weight, independent_bit_weight)
        self.margin = margin
        
    def _calculate_prediction_loss(self, prediction_pos, prediction_neg, target_pos, target_neg, input_as_hash_codes=False):
        consistency_pos = (prediction_pos-target_pos).square()
        consistency_neg = (prediction_neg-target_neg).square()
        
        consistency_loss = consistency_pos.mean() + consistency_neg.mean()#torch.clamp(self.margin-consistency_neg, min=0).mean()
        
        return consistency_loss
    
    def forward(self, h, h_pos, h_neg, prediction_pos, prediction_neg, target_pos, target_neg, model=None):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        # Instead of the original Softmax, we use sigmoid, so each bit is assume inpdenently distributed
        b, n = h.size()
        h_pos = self._to_hash_code(h_pos)
        h_neg = self._to_hash_code(h_neg)
        
        # Similarity in output space
        prediction_loss = self._calculate_prediction_loss(prediction_pos, prediction_neg, target_pos, target_neg, input_as_hash_codes=True)
        
        # Uniform frequency
        if self.uniform_frequency_weight > 0:
            uniform_frequency_loss = ( 
                self._calculate_uniform_frequency_loss(h, input_as_hash_codes=True) 
                + self._calculate_uniform_frequency_loss(h_pos, input_as_hash_codes=True)
                + self._calculate_uniform_frequency_loss(h_neg, input_as_hash_codes=True)) / 3
        else:
            uniform_frequency_loss = 0.0
        
        # Independent bit loss
        if self.independent_bit_weight > 0:
            independent_bit_loss = self._calculate_independent_bit_loss(model)
        else:
            independent_bit_loss = 0.0
        
        total_loss = prediction_loss + self.uniform_frequency_weight * uniform_frequency_loss \
                    + self.independent_bit_weight * independent_bit_loss
        
        return total_loss, prediction_loss, uniform_frequency_loss, independent_bit_loss

class Loss_V1_Triplet_Contrastive(Loss_V1):
    """Compute loss
    1) mse prediction loss, 2) uniform freq on tanh, 3) independent on W
    """
    
    def __init__(self, uniform_frequency_weight, independent_bit_weight, margin, contrastive_weight):
        super().__init__(uniform_frequency_weight, independent_bit_weight)
        self.margin = margin
        self.contrastive_weight = contrastive_weight
        
    def _calculate_prediction_loss(self, h, h_pos, h_neg, prediction_pos, prediction_neg, target_pos, target_neg, input_as_hash_codes=False):
        consistency_pos = (prediction_pos-target_pos).square()
        consistency_neg = (prediction_neg-target_neg).square()
        
        triplet_loss = (h-h_pos).square().sum(dim=1).mean() + torch.clamp(self.margin - (h-h_neg).square().sum(dim=1), min=0).mean()
        
#         print('HASH pos {:0.2f}, neg {:0.2f}'.format((h-h_pos).square().sum(dim=1).max().item(), (h-h_neg).square().sum(dim=1).max().item()))
#         print('RAW pos {:0.2f}, neg {:0.2f}'.format(consistency_pos.max(), consistency_neg.max()))
        
        consistency_loss = consistency_pos.mean() + consistency_neg.mean() + self.contrastive_weight*triplet_loss#torch.clamp(self.margin-consistency_neg, min=0).mean()
        
        return consistency_loss
    
    def forward(self, h, h_pos, h_neg, prediction_pos, prediction_neg, target_pos, target_neg, model=None):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        # Instead of the original Softmax, we use sigmoid, so each bit is assume inpdenently distributed
        b, n = h.size()
        h = self._to_hash_code(h)
        h_pos = self._to_hash_code(h_pos)
        h_neg = self._to_hash_code(h_neg)
        
        # Similarity in output space
        prediction_loss = self._calculate_prediction_loss(h, h_pos, h_neg, prediction_pos, prediction_neg, target_pos, target_neg, input_as_hash_codes=True)
        
        # Uniform frequency
        if self.uniform_frequency_weight > 0:
            uniform_frequency_loss = ( 
                self._calculate_uniform_frequency_loss(h, input_as_hash_codes=True) 
                + self._calculate_uniform_frequency_loss(h_pos, input_as_hash_codes=True)
                + self._calculate_uniform_frequency_loss(h_neg, input_as_hash_codes=True)) / 3
        else:
            uniform_frequency_loss = 0.0
        
        # Independent bit loss
        if self.independent_bit_weight > 0:
            independent_bit_loss = self._calculate_independent_bit_loss(model)
        else:
            independent_bit_loss = 0.0
        
        total_loss = prediction_loss + self.uniform_frequency_weight * uniform_frequency_loss \
                    + self.independent_bit_weight * independent_bit_loss
        
        return total_loss, prediction_loss, uniform_frequency_loss, independent_bit_loss
               
class Loss_V1_Sigmoid(nn.Module):
    """Compute loss
    1) mse prediction loss, 2) uniform freq on sigmoid, 3) independent on W
    """
    def _to_hash_code(self, anchors):
        anchors_prob = torch.sigmoid(anchors)            
        return anchors_prob   
    
    def _calculate_uniform_frequency_loss(self, h, input_as_hash_codes=False):
        bsize = h.size(0)
        
        if not input_as_hash_codes:
            h = self._to_hash_code(h)
            
        # This is equivalent to the original paper, uniform frequency, but on sigmoid activation
        f = torch.mean(h, dim = 0)
        f =  torch.clamp(f, min = EPS) #numerically stable log
        frequency_entropy = - f * torch.log(f)
        uniform_frequency_bits =  - frequency_entropy.sum()  # maximum entropy

        return uniform_frequency_bits
    
class Loss_V2(Loss_V1):
    """Compute loss
    1) bce prediction loss, 2) uniform freq on tanh, 3) independent on W
    """
    def _calculate_prediction_loss(self, prediction, target, input_as_hash_codes=False):
        consistency_loss = self.bce(prediction, target)
        return consistency_loss
    
class Loss_V2_Triplet(Loss_V1):
    """Compute loss
    1) bce prediction loss, 2) uniform freq on tanh, 3) independent on W
    """
    def _calculate_prediction_loss(self, pos_prediction, neg_prediction, pos_target, neg_target, input_as_hash_codes=False):
        #todo
        pass
    
    def forward(self, h, h_pos, h_neg, prediction_pos, prediction_neg, target_pos, target_neg, model=None):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        # Instead of the original Softmax, we use sigmoid, so each bit is assume inpdenently distributed
        b, n = h.size()
        h_pos = self._to_hash_code(h_pos)
        h_neg = self._to_hash_code(h_neg)
        
        # Similarity in output space
        prediction_loss = self._calculate_prediction_loss(pos_prediction, neg_prediction, pos_target, neg_target, input_as_hash_codes=True)
        
        # Uniform frequency
        if self.uniform_frequency_weight > 0:
            uniform_frequency_loss = ( 
                self._calculate_uniform_frequency_loss(h, input_as_hash_codes=True) 
                + self._calculate_uniform_frequency_loss(h_pos, input_as_hash_codes=True)
                + self._calculate_uniform_frequency_loss(h_neg, input_as_hash_codes=True)) / 3
        else:
            uniform_frequency_loss = 0.0
        
        # Independent bit loss
        if self.independent_bit_weight > 0:
            independent_bit_loss = self._calculate_independent_bit_loss(model)
        else:
            independent_bit_loss = 0.0
        
        total_loss = prediction_loss + self.uniform_frequency_weight * uniform_frequency_loss \
                    + self.independent_bit_weight * independent_bit_loss
        
        return total_loss, prediction_loss, uniform_frequency_loss, independent_bit_loss
    
class Loss_V2_Sigmoid(Loss_V2):
    """Compute loss
    1) bce prediction loss, 2) uniform freq on sigmoid, 3) independent on W
    """
    def _calculate_uniform_frequency_loss(self, h, input_as_hash_codes=False):
        bsize = h.size(0)
        
        if not input_as_hash_codes:
            h = self._to_hash_code(h)
            
        # This is equivalent to the original paper, uniform frequency, but on sigmoid activation
        f = torch.mean(h, dim = 0)
        f =  torch.clamp(f, min = EPS) #numerically stable log
        frequency_entropy = - f * torch.log(f)
        uniform_frequency_bits =  - frequency_entropy.sum()  # maximum entropy

        return uniform_frequency_bits
        
class Loss_V3(Loss_V1):

    def _calculate_prediction_loss(self, prediction_pos, prediction_neg, target_pos, target_neg, input_as_hash_codes=False):
        consistency_loss = 0.3 + (prediction_pos-target_pos).square() - (prediction_neg-target_neg).square()
        consistency_loss = torch.clamp(consistency_loss, min=0).mean()
        return consistency_loss        
    
    def forward(self, h, h_pos, h_neg, prediction_pos, prediction_neg, target_pos, target_neg, model=None):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        # Instead of the original Softmax, we use sigmoid, so each bit is assume inpdenently distributed
        b, n = h.size()
        h = self._to_hash_code(h)
        h_pos = self._to_hash_code(h_pos)
        h_neg = self._to_hash_code(h_neg)

        # Similarity in output space
        prediction_loss = self._calculate_prediction_loss(prediction_pos, prediction_neg, target_pos, target_neg, input_as_hash_codes=True)
        
        # Uniform frequency
        if self.uniform_frequency_weight > 0:
            uniform_frequency_loss = (
                self._calculate_uniform_frequency_loss(h, input_as_hash_codes=True) 
                + self._calculate_uniform_frequency_loss(h_pos, input_as_hash_codes=True)
                + self._calculate_uniform_frequency_loss(h_neg, input_as_hash_codes=True)
            ) / 3
        else:
            uniform_frequency_loss = 0.0
        
        # Independent bit loss
        if self.independent_bit_weight > 0:
            independent_bit_loss = self._calculate_independent_bit_loss(model)
        else:
            independent_bit_loss = 0.0
        
        total_loss = prediction_loss + self.uniform_frequency_weight * uniform_frequency_loss \
                    + self.independent_bit_weight * independent_bit_loss
        
        return total_loss, prediction_loss, uniform_frequency_loss, independent_bit_loss

