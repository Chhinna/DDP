from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm

from .NCEAverage import NCEAverage, NCESoftmax

from .NCECriterion import NCECriterion


class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T):
        """
        Initialize DistillKL class
        Args:
            T: Temperature for distillation
        Returns: 
            None: Does not return anything
        - Store temperature T as attribute
        - Call parent class initializer"""
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        """
        Calculate KL divergence between predicted and target distributions
        Args:
            y_s: Predicted distribution
            y_t: Target distribution
        Returns:
            loss: KL divergence loss between distributions
        - Calculate softmax normalized predicted distribution p_s
        - Calculate softmax normalized target distribution p_t  
        - Return KL divergence between p_s and p_t averaged over batch, multiplied by temperature T^2"""
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        return F.kl_div(p_s, p_t, size_average=False) * self.T ** 2 / y_s.shape[0]


class NCELoss(nn.Module):
    """NCE contrastive loss"""
    def __init__(self, opt, n_data):
        """
        Initialize NCE loss module
        Args:
            opt: Options class
            n_data: Number of samples in the dataset 
        Returns: 
            loss: Combined loss value from temporal and spatial criteria
        - Initialize NCEAverage module to sample negative pairs
        - Initialize NCECriterion modules for temporal and spatial criteria
        - Combine losses from both criteria"""
        super(NCELoss, self).__init__()
        self.contrast = NCEAverage(opt.feat_dim, n_data, opt.nce_k, opt.nce_t, opt.nce_m)
        self.criterion_t = NCECriterion(n_data)
        self.criterion_s = NCECriterion(n_data)

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        """
        Calculates loss for source and target domains.
        Args:
            f_s: {Source feature vector} 
            f_t: {Target feature vector}
            idx: {Sample index}
        Returns: 
            loss: {Combined source and target loss}
        - Computes contrastive output for source and target features 
        - Calculates source loss using source criterion
        - Calculates target loss using target criterion  
        - Returns sum of source and target losses
        """
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
        s_loss = self.criterion_s(out_s)
        t_loss = self.criterion_t(out_t)
        return s_loss + t_loss


class NCESoftmaxLoss(nn.Module):
    """info NCE style loss, softmax"""
    def __init__(self, opt, n_data):
        """
        Initialize NCESoftmaxLoss module
        Args:
            opt: Options class
            n_data: Number of samples
        Returns: 
            loss: Combined loss from softmax and NCE
        - Initialize NCESoftmax module for computing NCE loss
        - Initialize CrossEntropyLoss for computing softmax loss
        - Loss is computed as a weighted sum of softmax and NCE losses during training"""
        super(NCESoftmaxLoss, self).__init__()
        self.contrast = NCESoftmax(opt.feat_dim, n_data, opt.nce_k, opt.nce_t, opt.nce_m)
        self.criterion_t = nn.CrossEntropyLoss()
        self.criterion_s = nn.CrossEntropyLoss()

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        """
        Calculates the loss for a batch of source and target features.
        Args:
            f_s: {Source feature tensor}: Source feature tensor 
            f_t: {Target feature tensor}: Target feature tensor
            idx: {Index tensor}: Index tensor specifying positive/negative pairs
            contrast_idx: {Contrast index tensor (optional)}: Contrast index tensor for negative sampling
        Returns: 
            loss: {Loss tensor}: Combined source and target loss
        - Computes contrastive loss for source and target features
        - Calculates source and target loss using criterion 
        - Returns sum of source and target loss
        """
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
        bsz = f_s.shape[0]
        label = torch.zeros([bsz, 1]).cuda().long()
        s_loss = self.criterion_s(out_s, label)
        t_loss = self.criterion_t(out_t, label)
        return s_loss + t_loss


class Attention(nn.Module):
    """attention transfer loss"""
    def __init__(self, p=2):
        """
        Initialize Attention layer
        Args:
            p: Attention power
        Returns:
            None: Does not return anything
        - Save attention power p as a layer parameter
        - Initialize base class (super) with no arguments
        - Attention layer is initialized and ready for use"""
        super(Attention, self).__init__()
        self.p = p

    def forward(self, g_s, g_t):
        """
        Forward propagates loss through the network.
        Args:
            g_s: Source graph embeddings
            g_t: Target graph embeddings  
        Returns:
            loss: Loss between source and target graph embeddings
        - Zip source and target graph embeddings to pair corresponding embeddings
        - Calculate loss between each pair of embeddings using at_loss function
        - Return list of losses for each embedding pair"""
        return [self.at_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def at_loss(self, f_s, f_t):
        """
        Calculates the loss between source and target features.
        Args:
            f_s: {Source feature map} 
            f_t: {Target feature map}
        Returns:
            loss: {Mean squared error between source and target features}
        Processing Logic:
            - Compare source and target feature map heights and widths
            - Adaptively average pool smaller feature map to match larger one
            - Subtract corresponding elements between source and target feature maps
            - Return mean of squared differences between source and target features
        """
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        """Computes the attention weights of the text with respect to the query using a softmax.
        Args:
            f: {text embeddings}: Matrix of text embeddings to attend over 
            self.p: {query vector}: Vector containing the query
        Returns: 
            {attention weights}: Matrix of attention weights for text with respect to query
        - Raise the query vector to the power of the text embeddings
        - Take the mean of the powered query vector for each text embedding
        - Normalize using softmax to get attention weights
        - Return attention weights matrix"""
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))


class HintLoss(nn.Module):
    """regression loss from hints"""
    def __init__(self):
        """Initializes HintLoss class
        Args:
            self: HintLoss class instance
        Returns: 
            None: Does not return anything
        - Initializes HintLoss class by calling parent class' __init__ method
        - Initializes MSELoss criterion for calculating loss"""
        super(HintLoss, self).__init__()
        self.crit = nn.MSELoss()

    def forward(self, f_s, f_t):
        """Forward propagation of the critic. 
        Args:
            f_s: Current state features 
            f_t: Next state features
        Returns: 
            value: Estimated value for state s
        - Calculate the value of the current state by applying the critic to the current state features
        - Forward pass through the critic model
        - Return the estimated value"""
        return self.crit(f_s, f_t)


if __name__ == '__main__':
    pass
