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
        Initialize DistillKL model
        Args:
            T: Temperature for distillation in one line
        Returns: 
            None: Does not return anything
        - Call super constructor to initialize base class
        - Set temperature T for distillation"""
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
        - Calculate softmax of predicted distribution with temperature
        - Calculate softmax of target distribution with temperature 
        - Return KL divergence between distributions averaged over batch size"""
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
        Contrasts feature representations and returns the loss.
        Args:
            f_s: Source feature representation
            f_t: Target feature representation 
            idx: Index of anchor sample
            contrast_idx: Index of contrastive sample (optional)
        Returns: 
            Loss: Total loss from source and target representations
        - Contrasts source and target representations using specified index
        - Computes loss for source representation using criterion_s 
        - Computes loss for target representation using criterion_t
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
        - Applies source and target criterion to outputs and label
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
        - Save attention power p as layer parameter
        - Initialize base class"""
        super(Attention, self).__init__()
        self.p = p

    def forward(self, g_s, g_t):
        """
        Forward propagates the loss through the network.
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
            - Return mean of squared differences
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
        - Raise the query (self.p) to the power of the scaling factor (default 0.5)
        - Take the mean of each scaled query row over the features dimension
        - Normalize the mean weighted features to get the attention weights
        """
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
            f_s: Current state features. 
            f_t: Next state features.
        Returns:
            value: Estimated value for state pair (f_s, f_t)
        - Calculate the value of the state pair (f_s, f_t) using the critic network
        - Pass the state features into the critic network 
        - Critic network returns the estimated value"""
        return self.crit(f_s, f_t)


if __name__ == '__main__':
    pass
