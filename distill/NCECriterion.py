import torch
from torch import nn

eps = 1e-7


class NCECriterion(nn.Module):

    def __init__(self, nLem):
        """
        Initialize NCECriterion class
        Args:
            nLem: Number of lemmas
        Returns: 
            None: Does not return anything
        - Call super class __init__ method to initialize base class
        - Set the number of lemmas attribute nLem to the passed in value"""
        super(NCECriterion, self).__init__()
        self.nLem = nLem

    def forward(self, x):
        """
        Calculates the loss for a forward pass through the model.
        Args: 
            x: Input data tensor of shape (batchSize, K+1): The first column corresponds to targets and remaining K columns correspond to noise samples  
        Returns:
            loss: Loss value of type float
        Calculates loss by:
            1. Computing P(origin=model) and P(origin=noise) probabilities for each sample using equations 5.1 and 5.2
            2. Taking log of the probabilities
            3. Summing log probabilities across samples
            4. Returning negative of average log probability as loss
        """
        batchSize = x.size(0)
        K = x.size(1) - 1
        Pnt = 1 / float(self.nLem)
        Pns = 1 / float(self.nLem)

        # eq 5.1 : P(origin=model) = Pmt / (Pmt + k*Pnt)
        Pmt = x.select(1, 0)
        Pmt_div = Pmt.add(K * Pnt + eps)
        lnPmt = torch.div(Pmt, Pmt_div)

        # eq 5.2 : P(origin=noise) = k*Pns / (Pms + k*Pns)
        Pon_div = x.narrow(1, 1, K).add(K * Pns + eps)
        Pon = Pon_div.clone().fill_(K * Pns)
        lnPon = torch.div(Pon, Pon_div)

        # equation 6 in ref. A
        lnPmt.log_()
        lnPon.log_()

        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.view(-1, 1).sum(0)

        loss = - (lnPmtsum + lnPonsum) / batchSize

        return loss