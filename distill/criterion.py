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
            T: Temperature parameter for distillation
        Returns: 
            None: Does not return anything
        - Store temperature parameter T as attribute
        - Call parent class' __init__ method to initialize base model"""
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
            - Take log softmax of source network output divided by temperature
            - Take softmax of target network output divided by temperature 
            - Calculate KL divergence between the two distributions
            - Multiply KL divergence by temperature squared and divide by number of samples to get loss
        """
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
        - Contrastive learning loss will be computed between positive pairs and sampled negative pairs from these modules"""
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
            Loss: Combined source and target loss
        - Contrasts source and target features using specified index
        - Computes source loss on contrasted source features 
        - Computes target loss on contrasted target features
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
            idx: {Index of the current example}
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
            p: Attention power value
        Returns:
            None: Does not return anything
        - Initialize base class with super call
        - Set internal power value from input p parameter"""
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
        Calculates the loss between two feature maps.
        Args:
            f_s: {Feature map from source domain in one line}
            f_t: {Feature map from target domain in one line}
        Returns:
            loss: {Loss between the two feature maps in one line}
        Processing Logic:
            - Checks the shape of the feature maps and uses adaptive average pooling to make them the same size
            - Subtracts the feature maps and takes the mean of the squared difference
            - Returns the loss between the two feature maps
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
            {Attention weights}: {Normalized attention weights over the features}
        - Raise the query (self.p) to the power of the features (f)
        - Take the mean over the last (second) dimension 
        - View the result as a 2D tensor of size (batch_size, features)
        - Normalize the weights using F.normalize"""
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))


class HintLoss(nn.Module):
    """regression loss from hints"""
    def __init__(self):
        """Initialize HintLoss class
        Args:
            self: HintLoss class instance
        Returns: 
            None: Does not return anything
        - Initialize MSELoss criterion for loss computation
        - Call parent class __init__ method"""
        super(HintLoss, self).__init__()
        self.crit = nn.MSELoss()

    def forward(self, f_s, f_t):
        """Forward propagation of the critic. 
        Args:
            f_s: Current state features. 
            f_t: Next state features.
        Returns:
            value: Estimated value for state transition.
        - Calculate estimated value (Q-value) of transitioning from state f_s to f_t using critic network.
        - Return estimated value (Q-value) as output."""
        return self.crit(f_s, f_t)


if __name__ == '__main__':
    pass
