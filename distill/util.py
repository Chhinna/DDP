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
            x: Transformed tensor of shape (batch_size, hidden_size).
        - Flatten input tensor x to shape (batch_size, -1) 
        - Apply linear transformation
        - Apply L2 normalization
        - Return transformed tensor x"""
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class LinearEmbed(nn.Module):
    """Linear Embedding"""
    def __init__(self, dim_in=1024, dim_out=128):
        """Initializes a linear embedding layer
        Args:
            dim_in: Dimension of the input features
            dim_out: Dimension of the output features 
        Returns: 
            self: An initialized LinearEmbed object
        Initializes a linear layer that maps inputs of shape (-1, dim_in) to outputs of shape (-1, dim_out)
        - Creates a linear layer with input dimension dim_in and output dimension dim_out
        - Stores the linear layer in self.linear"""
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
            - Applies a linear transformation to project inputs to 2 * dim_out dimensions
            - Applies ReLU activation
            - Applies another linear transformation to project to dim_out dimensions 
            - Applies L2 normalization"""
        super(MLPEmbed, self).__init__()
        self.linear1 = nn.Linear(dim_in, 2 * dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(2 * dim_out, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        """Forward pass through the network.
        Args:
            x: Input tensor of shape [batch_size, input_dim]
        Returns: 
            x: Output tensor of shape [batch_size, output_dim] after applying linear transformations and activations.
        - Apply first linear transformation and ReLU activation
        - Apply second linear transformation 
        - Apply L2 normalization
        - Return transformed and normalized output"""
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
        - Calls the parent class' initializer"""
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        """Normalizes the input x to have unit norm along the dimension 1.
        Args:
            x: Input tensor to normalize
        Returns: 
            x: Normalized tensor with unit norm along dimension 1
        Processes input tensor x as follows:
            - Raises each element of x to the power of self.power
            - Sums along dimension 1 and keeps dimension to get norm
            - Takes the power of 1/self.power to the norm to invert the operation
            - Divides x by the norm to normalize along dimension 1"""
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        return x.div(norm)


if __name__ == '__main__':
    pass
