from __future__ import print_function

import torch.nn as nn


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        """Initializes embedding layer
        Args:
            dim_in: Dimension of input embeddings
            dim_out: Dimension of output embeddings 
        Returns: 
            self: Initialized Embed layer object
        Initializes a linear layer that maps inputs to outputs and l2 normalizes outputs.
        - Creates a linear layer that maps from dim_in to dim_out
        - Adds an l2 normalization layer to normalize the outputs"""
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        """Forward pass of the model. 
        Args:
            x: Input tensor of shape (batch_size, *).
        Returns: 
            x: Output tensor of shape (batch_size, hidden_size).
        - Flatten input tensor x to shape (batch_size, -1) 
        - Apply linear transformation
        - Apply L2 normalization
        - Return transformed and normalized tensor"""
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class LinearEmbed(nn.Module):
    """Linear Embedding"""
    def __init__(self, dim_in=1024, dim_out=128):
        """
        Initialize a linear embedding layer
        Args:
            dim_in: Dimension of the input features
            dim_out: Dimension of the output features 
        Returns: 
            None: Does not return anything
        - Creates a linear layer that maps from dim_in to dim_out
        - Initializes the weights and biases of the linear layer
        - Stores the linear layer as an attribute for future use"""
        super(LinearEmbed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        """Forward pass of the network
        Args: 
            x: Input tensor 
        Returns: 
            x: Transformed input tensor
        Processes input tensor:
        - Reshapes input to flatten dimensions except batch
        - Passes input through linear layer 
        - Returns transformed input tensor"""
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x


class MLPEmbed(nn.Module):
    """non-linear embed by MLP"""
    def __init__(self, dim_in=1024, dim_out=128):
        """
        Initializes an MLP embedding module
        Args:
            dim_in: Dimension of input features
            dim_out: Dimension of output features 
        Returns:
            self: Initialized MLP embedding module
        Processing Logic:
            - Applies a linear transformation to project inputs to a higher dimensional space
            - Applies ReLU activation
            - Applies another linear transformation to project to output dimension 
            - Applies L2 normalization"""
        super(MLPEmbed, self).__init__()
        self.linear1 = nn.Linear(dim_in, 2 * dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(2 * dim_out, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        """Forward pass of the network.
        Args:
            x: Input tensor of shape [batch_size, *].
        Returns: 
            x: Output tensor of shape [batch_size, num_features].
        - Flatten input tensor 
        - Apply first linear layer and ReLU activation
        - Apply second linear layer and l2 normalization
        - Return output tensor"""
        x = x.view(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.l2norm(self.linear2(x))
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        """Initializes normalization parameters
        Args:
            power: The power to raise values to before normalizing
        Returns:
            self: The initialized Normalize object
        - Stores the power parameter for future reference
        - Calls the parent class' initialization"""
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        """Normalizes the input x to have unit norm along the specified dimension.
        Args:
            x: Input tensor of shape (N, *).
            self.power: The power of the norm.
        Returns: 
            Output tensor of same shape as x with unit norm along dim.
        - Computes the norm of each sample along dim 1 by taking the power of self.power and summing. 
        - Takes the power of 1/self.power to invert the operation.
        - Divides the input x by the computed norm to normalize each sample to have unit norm."""
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        return x.div(norm)


if __name__ == '__main__':
    pass
