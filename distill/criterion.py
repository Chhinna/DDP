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
        - Call parent class __init__ method"""
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        """
        Calculate KL divergence between softmax outputs of source and target networks
        Args:
            y_s: Output of source network
            y_t: Output of target network
        Returns:
            loss: KL divergence loss between source and target networks
        Processing Logic:
            - Take log softmax of normalized source network output
            - Take softmax of normalized target network output 
            - Calculate KL divergence between the two distributions
            - Average over batch size
            - Multiply by temperature squared to scale the loss appropriately
        """
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        return F.kl_div(p_s, p_t, size_average=False) * self.T ** 2 / y_s.shape[0]


class NCELoss(nn.Module):
    """NCE contrastive loss"""
    def __init__(self, opt, n_data):
        """
        Initialize NCELoss module
        Args:
            opt: Options class
            n_data: Number of samples in the dataset 
        Returns: 
            loss: Combined loss value from temporal and spatial criteria
        - Initialize NCEAverage module to sample negative pairs
        - Initialize NCECriterion modules for temporal and spatial losses
        - Combine temporal and spatial losses during forward pass"""
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
            opt: options
            n_data: number of samples
        Returns: 
            loss: computed loss
        - Initialize NCESoftmax module for contrastive learning
        - Initialize CrossEntropyLoss for target loss
        - Initialize CrossEntropyLoss for similarity loss
        """
        super(NCESoftmaxLoss, self).__init__()
        self.contrast = NCESoftmax(opt.feat_dim, n_data, opt.nce_k, opt.nce_t, opt.nce_m)
        self.criterion_t = nn.CrossEntropyLoss()
        self.criterion_s = nn.CrossEntropyLoss()

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        """
        Calculates the loss for a source-target pair and their contrastive examples.
        Args:
            f_s: {Source feature vector} 
            f_t: {Target feature vector}
            idx: {Index of the current sample}
            contrast_idx: {Index of contrastive examples}
        Returns: 
            loss: {Total loss value}
        - Calculates contrastive outputs for source and target using self.contrast
        - Initializes label tensor with all zeros 
        - Applies source and target criterion to contrastive outputs  
        - Returns sum of source and target losses
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
        - Initialize base Attention layer using super().__init__()"""
        super(Attention, self).__init__()
        self.p = p

    def forward(self, g_s, g_t):
        """
        Forward pass through the attention layer
        Args:
            g_s: Source graph embeddings
            g_t: Target graph embeddings  
        Returns:
            loss: Attention loss between source and target graphs
        - Zip source and target graph embeddings to pair corresponding nodes
        - Calculate attention loss between each pair of source and target node embeddings
        - Return a list of attention losses for each node pair
        """
        return [self.at_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def at_loss(self, f_s, f_t):
        """
        Calculates the average squared distance between embeddings.
        Args:
            f_s: {Feature maps of source domain} 
            f_t: {Feature maps of target domain}
        Returns:
            loss: {Average squared distance between embeddings}
        Processing Logic:
            - Compare shape of feature maps and use adaptive average pooling to make them same size
            - Apply transformation to get embeddings of source and target feature maps
            - Return mean of squared distance between embeddings
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
        """Computes the attention weights of the given features based on the query.
        Args: 
            f: {Features to attend over}
        Returns: 
            {Attention weights}: {Normalized weights for each feature}
        - Raise the query (self.p) to the power of the attention weight (default 0.5)
        - Take the mean of each row to get a single attention weight per query 
        - Normalize the weights to sum to 1"""
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))


class HintLoss(nn.Module):
    """regression loss from hints"""
    def __init__(self):
        """Initialize HintLoss class
        Args:
            self: HintLoss class instance
        Returns: 
            None: Does not return anything
        - Initialize HintLoss class with MSELoss criterion
        - MSELoss will be used to calculate loss between predicted and target hints"""
        super(HintLoss, self).__init__()
        self.crit = nn.MSELoss()

    def forward(self, f_s, f_t):
        """Forward propagation of the critic. 
        Args:
            f_s: Current state features 
            f_t: Next state features
        Returns: 
            value: Estimated value for state s
        - Calculate the value of the current state by forwarding f_s through the critic network
        - Also calculate the value of the next state by forwarding f_t through the critic network  
        - Return the value for the current state"""
        return self.crit(f_s, f_t)


if __name__ == '__main__':
    pass
