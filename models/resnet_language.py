# This script is largely based on https://github.com/WangYueFt/rfs

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli
import os
import pickle
from models.util import get_embeds


class LinearMap(nn.Module):
    def __init__(self, indim, outdim):
        """
        Initialize a linear map layer
        Args:
            indim: Input dimension
            outdim: Output dimension 
        Returns: 
            None: Does not return anything
        - Creates a linear layer using PyTorch's nn.Linear module to map from input to output dimensions
        - Initializes the parent class 
        - Stores the linear layer in an attribute called 'map'"""
        super(LinearMap, self).__init__()
        self.map = nn.Linear(indim, outdim)
        
    def forward(self, x):
        """Forward pass of the layer. 
        Args: 
            x: Input tensor 
        Returns: 
            Tensor: Transformed input tensor
        - Apply linear transformation: y = Wx + b
        - Apply activation: output = activation(y)"""
        return self.map(x)
        
class LangPuller(nn.Module):
    def __init__(self,opt, vocab_base, vocab_novel):
        """Initializes the LangPuller model
        
        Args: 
            self: The LangPuller object
            opt: Training options
            vocab_base: Base vocabulary 
            vocab_novel: Novel vocabulary
        
        Returns: 
            None
        
        Processing Logic:
            - Initializes base and novel word embeddings 
            - Retrieves base and novel embeddings from files
            - Puts embeddings on GPU
            - Initializes softmax for label attractors
            - Truncates embeddings to first 300 dimensions if using GloVe
        """
        super(LangPuller, self).__init__()
        self.mapping_model = None
        self.opt = opt
        self.vocab_base = vocab_base
        self.vocab_novel = vocab_novel
        self.temp = opt.temperature
        dim = opt.word_embed_size # TODO

        # Retrieve novel embeds
        embed_pth = os.path.join(opt.word_embed_path, "{0}_dim{1}.pickle".format(opt.dataset, dim)) # TODO
        self.novel_embeds = get_embeds(embed_pth, vocab_novel).float().cuda()

        # Retrieve base embeds
        if opt.use_synonyms:
            embed_pth = os.path.join(opt.word_embed_path,
                                     "{0}_dim{1}_base_synonyms.pickle".format(opt.dataset, dim)) # TOdo
            with open(embed_pth, "rb") as openfile:
                label_syn_embeds = pickle.load(openfile)
            base_embeds = []
            for base_label in vocab_base:
                base_embeds.append(label_syn_embeds[base_label])
        else:
            embed_pth = os.path.join(opt.word_embed_path,
                                     "{0}_dim{1}.pickle".format(opt.dataset, dim)) # TODO
            base_embeds = get_embeds(embed_pth, vocab_base)

        self.base_embeds = base_embeds.float().cuda()
        # This will be used to compute label attractors.
        self.softmax = nn.Softmax(dim=1)
        # If Glove, use the first 300 TODO
        if opt.glove:
            self.base_embeds = self.base_embeds[:,:300]
            self.novel_embeds = self.novel_embeds[:,:300]
            
    def update_novel_embeds(self, vocab_novel):
        """
        Update novel embeddings
        Args: 
            vocab_novel: Vocabulary of novel words
        Returns: 
            self: Updated model object
        Processing Logic:
            - Retrieve novel embeddings from file
            - Set novel embeddings on model to new embeddings
            - If using Glove, truncate novel embeddings to first 300 dimensions
        """
        # Retrieve novel embeds
        opt = self.opt
        dim = opt.word_embed_size
        embed_pth = os.path.join(opt.word_embed_path, "{0}_dim{1}.pickle".format(opt.dataset, dim))
        new_novel_embeds = get_embeds(embed_pth, vocab_novel).float().cuda()
        self.novel_embeds = new_novel_embeds
        if opt.glove: #todo
            self.novel_embeds = self.novel_embeds[:,:300] # First 300 of the saved embeddings are Glove.
#         self.novel_embeds = torch.cat((self.novel_embeds, new_novel_embeds), 0)

    def create_pulling_mapping(self, state_dict, base_weight_size=640):
        """
        Maps novel embeddings to base weight size.
        Args: 
            state_dict: State dictionary of pretrained linear mapping model
            base_weight_size: Size of base weights (default 640)
        Returns: 
            self: Object with embedded mapping model
        - Loads state dictionary into a LinearMap module
        - Sets input and output dimensions of LinearMap
        - Moves LinearMap to GPU
        - Attaches LinearMap to self as mapping_model
        """
        indim = self.novel_embeds.size(1)
        outdim = base_weight_size
        self.mapping_model = LinearMap(indim, outdim)
        self.mapping_model.load_state_dict(state_dict)
        self.mapping_model = self.mapping_model.cuda()
        

    def forward(self, base_weight, mask=False):
        """
        Forward pass through the model.
        Args:
            base_weight: Base weight matrix to pull from.
            mask: Whether to mask diagonal in similarity computation.
        Returns: 
            scores: Weighted base embeddings pulled for each novel example.
        Processing Logic:
            - Compute similarity between novel and base embeddings.
            - Apply softmax with temperature. 
            - Mask diagonal in similarity matrix if specified.
            - Weight base embeddings by similarity scores.
            - Alternatively, apply linear mapping model to novel embeddings.
        """
        if self.mapping_model is None:
            # Default way of computing pullers is thru sem. sub. reg.:
            scores = self.novel_embeds @ torch.transpose(self.base_embeds, 0, 1)
            if mask:
                scores.fill_diagonal_(-9999)
            scores = self.softmax(scores / self.temp)
            return scores @ base_weight # 5 x 640 for fine-tuning.
        else:
            # Linear Mapping:
            with torch.no_grad():
                inspired = self.mapping_model(self.novel_embeds)
            return inspired

    @staticmethod
    def loss1(pull, inspired, weights):
        """Calculates loss between pull and distance between inspired and weights
        Args:
            pull: Pull value
            inspired: Inspired tensor 
            weights: Weights tensor
        Returns: 
            Loss: Loss value between pull and distance
        - Calculates L2 norm of difference between inspired and weights tensors
        - Multiplies pull value with squared L2 norm
        - Returns product as loss value"""
        return pull * torch.norm(inspired - weights)**2

    @staticmethod
    def get_projected_weight(base_weight, weights):
        """Project base weight onto subspace spanned by weights.
        Args:
            base_weight: Base weight vector to project. 
            weights: Weight vectors spanning the subspace.
        Returns: 
            projected_weight: Projected base weight vector.
        - Transpose base_weight for matrix multiplication.
        - Perform QR decomposition on transposed base_weight to get orthonormal basis Q.
        - Project weights onto Q to get coefficients mut.  
        - Normalize mut by column norms of Q to ignore scaling."""
        tr = torch.transpose(base_weight, 0, 1)
        Q, R = torch.qr(tr, some=True) # Q is 640x60
        mut = weights @ Q # mut is 5 x 60
        mutnorm = mut / torch.norm(Q.T, dim=1).unsqueeze(0)
        return mutnorm @ Q.T
        


class ResNet(nn.Module):

    def __init__(self, block, n_blocks, keep_prob=1.0, avg_pool=False, drop_rate=0.0,
                 dropblock_size=5, num_classes=-1, use_se=False, vocab=None, opt=None):
        """
        Initializes a ResNet model
        Args:
            block: {Block type to use (BasicBlock, Bottleneck, etc)}
            n_blocks: {Number of blocks for each layer of the ResNet} 
            keep_prob: {Dropout keep probability}
        Returns: 
            self: {Initialized ResNet model}
        Processing Logic:
            1. Defines the initial parameters like input channels, use of SE, etc.
            2. Calls _make_layer to create each layer of the ResNet 
            3. Adds average pooling and dropout
            4. Initializes weights
            5. Adds a linear classifier if num_classes is specified
        """
        if vocab is not None:
            assert opt is not None

        super(ResNet, self).__init__()

        self.inplanes = 3
        self.use_se = use_se
        self.layer1 = self._make_layer(block, n_blocks[0], 64,
                                       stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, n_blocks[1], 160,
                                       stride=2, drop_rate=drop_rate)
        if opt.no_dropblock:
            drop_block = False
            dropblock_size = 1
        self.layer3 = self._make_layer(block, n_blocks[2], 320,
                                       stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, n_blocks[3], 640,
                                       stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        if avg_pool:
            # self.avgpool = nn.AvgPool2d(5, stride=1)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        self.vocab = vocab
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.num_classes = num_classes
        if self.num_classes > 0:
            self.classifier = nn.Linear(640, self.num_classes, bias=opt.linear_bias)

    def _make_layer(self, block, n_block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        """
        Creates a layer by stacking blocks
        Args:
            block: {Block class to use} 
            n_block: {Number of blocks to return}
            planes: {Number of output channels (feature maps)} 
            stride: {Stride of the first block. All others will have stride=1}
            drop_rate: {Dropout rate}
            drop_block: {Use DropBlock instead of Dropout}
            block_size: {Size of each DropBlock if using DropBlock}
        Returns:
            Sequential: {Sequentially stacked blocks}
        Processing Logic:
            - Creates downsample if needed
            - Adds first block with specified stride
            - Adds remaining blocks with stride=1
            - Returns stacked blocks in Sequential
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if n_block == 1:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size, self.use_se)
        else:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, self.use_se)
        layers.append(layer)
        self.inplanes = planes * block.expansion

        for i in range(1, n_block):
            if i == n_block - 1:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, drop_block=drop_block,
                              block_size=block_size, use_se=self.use_se)
            else:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, use_se=self.use_se)
            layers.append(layer)

        return nn.Sequential(*layers)


    def forward(self, x, is_feat=False, get_alphas=False):
        """
        Forward pass through the network.
        Args:
            x: Input image or feature map
            is_feat: If True, returns intermediate feature maps
            get_alphas: If True, returns attention weights
        Returns: 
            Output of the network or intermediate feature maps:
                - If is_feat is True, returns list of intermediate feature maps
                - If is_feat is False, returns output of the network
        Processing Logic:
            - Passes input through 4 convolutional layers
            - Optionally applies average pooling
            - Flattens output and passes through classifier 
            - Optionally returns attention weights from classifier
        """
        x = self.layer1(x)
        f0 = x
        x = self.layer2(x)
        f1 = x
        x = self.layer3(x)
        f2 = x
        x = self.layer4(x)
        f3 = x
        if self.keep_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        feat = x
        if self.num_classes > 0:
            if self.vocab is not None:
                x = self.classifier(x, get_alphas=get_alphas)
            else: # linear classifier has no attribute get_alphas
                x = self.classifier(x)

        if is_feat:
            return [f0, f1, f2, f3, feat], x
        else:
            return x

    def _get_base_weights(self):
        """
        Gets the base weights of the classifier.
        Args:
            self: The classifier object.
        Returns: 
            base_weight: The base weight of the classifier.
            base_bias: The base bias of the classifier or None.
        Processing Logic:
        - Detach and clone the weight and bias of the classifier to get the base weights
        - Set requires_grad to False to prevent tracking history
        - Return base weight and bias. If bias is None, return base weight and None
        """
        base_weight = self.classifier.weight.detach().clone().requires_grad_(False)
        if self.classifier.bias is not None:
            base_bias = self.classifier.bias.detach().clone().requires_grad_(False)
            return base_weight, base_bias
        else:
            return base_weight, None

    def augment_base_classifier_(self,
                                 n,
                                 novel_weight=None,
                                 novel_bias=None):
        """
        Augments the base classifier with novel classes
        Args:
            n: {Number of novel classes to add}
            novel_weight: {Weight matrix for novel classes} 
            novel_bias: {Bias vector for novel classes}
        Returns: 
            None: {Does not return anything, augments classifier in-place}
        Processing Logic:
            - Creates classifier weights for novel classes if not provided
            - Concatenates base and novel weights/biases
            - Sets augmented weights/biases as classifier parameters
        """

        # Create classifier weights for novel classes.
        base_device = self.classifier.weight.device
        base_weight = self.classifier.weight.detach()
        if self.classifier.bias is not None:
            base_bias = self.classifier.bias.detach()
        else:
            base_bias = None

        if novel_weight is None:
            novel_classifier = nn.Linear(base_weight.size(1), n, bias=(base_bias is not None)) # TODO!!
            novel_weight     = novel_classifier.weight.detach()
            if base_bias is not None and novel_bias is None:
                novel_bias = novel_classifier.bias.detach()

        augmented_weight = torch.cat([base_weight, novel_weight.to(base_device)], 0)
        self.classifier.weight = nn.Parameter(augmented_weight, requires_grad=True)

        if base_bias is not None:
            augmented_bias = torch.cat([base_bias, novel_bias.to(base_device)])
            self.classifier.bias = nn.Parameter(augmented_bias, requires_grad=True)


    def regloss(self, lmbd, base_weight, base_bias=None):
        """Calculates the regularization loss between the current model and a base model
        Args:
            lmbd: Regularization hyperparameter
            base_weight: Weight matrix of base model 
            base_bias: Bias vector of base model (optional)
        Returns: 
            reg: Regularization loss value
        - Calculates L2 norm between current and base weights
        - If bias is provided, adds L2 norm between current and base biases
        - Returns total regularization loss"""
        reg = lmbd * torch.norm(self.classifier.weight[:base_weight.size(0),:] - base_weight)
        if base_bias is not None:
            reg += lmbd * torch.norm(self.classifier.bias[:base_weight.size(0)] - base_bias)**2
        return reg
    
    def reglossnovel(self, lmbd, novel_weight, novel_bias=None):
        """
        Calculates the regularization loss for novel classes
        Args:
            lmbd: Regularization coefficient
            novel_weight: Novel class weights 
            novel_bias: Novel class biases (optional)
        Returns: 
            reg: Regularization loss
        - Calculates regularization loss between novel class weights/biases and classifier weights/biases
        - Adds L2 norm of weight and bias differences  
        - Returns total regularization loss
        """
        rng1, rng2 = self.num_classes, self.num_classes + novel_weight.size(0)
        reg = lmbd * torch.norm(self.classifier.weight[rng1:rng2, :] - novel_weight) #**2??
        if novel_bias is not None:
            reg += lmbd * torch.norm(self.classifier.bias[rng1:rng2, :] - novel_bias)**2
        return reg
    

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False,
                 block_size=1, use_se=False):
        """
        Initializes a BasicBlock module
        Args:
            inplanes: {Number of input channels} 
            planes: {Number of output channels}
            stride: {Stride of the convolution}
            downsample: {Downsampling layer}
        Returns: 
            self: {BasicBlock module}
        Processing Logic:
            - Apply 3x3 convolution with batch normalization and ReLU on input
            - Apply 3x3 convolution with batch normalization 
            - Apply 3x3 convolution with batch normalization
            - Apply max pooling if stride > 1
            - Apply downsampling if provided
            - Initialize attributes like stride, drop rate etc.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)
        self.use_se = use_se
        if self.use_se:
            self.se = SELayer(planes, 4)

    def forward(self, x):
        """
        Performs forward pass of ResNet block.
        Args:
            x: {Input tensor}: Input tensor of shape (batch_size, channels, height, width)  
        Returns:
            out: {Output tensor}: Output tensor of same shape as input
        Processing Logic:
            - Computes residual as input tensor
            - Passes input through convolution, batch normalization and ReLU layers
            - Adds residual to output
            - Applies ReLU and max pooling
            - Optionally applies dropout
        """
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out

class DropBlock(nn.Module):
    def __init__(self, block_size):
        """
        Initializes DropBlock regularization layer
        Args:
            block_size: Size of block to drop in pixels
        Returns: 
            None: Does not return anything
        - Sets block size attribute from input
        - Calls parent class' init method"""
        super(DropBlock, self).__init__()

        self.block_size = block_size
        #self.gamma = gamma
        #self.bernouli = Bernoulli(gamma)

    def forward(self, x, gamma):
        """
        Forward pass of the network
        Args:
            x: Input tensor of shape (bsize, channels, height, width) 
            gamma: Dropout probability
        Returns: 
            Output: Output tensor after applying dropout
        Processing Logic:
            - Sample Bernoulli random variable with probability gamma
            - Generate mask of same shape as input by thresholding samples
            - Compute block mask by applying max pooling on mask
            - Elementwise multiply input with block mask 
            - Scale output to preserve expected value
        """
        # shape: (bsize, channels, height, width)

        if self.training:
            batch_size, channels, height, width = x.shape

            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample((batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1))).cuda()
            block_mask = self._compute_block_mask(mask)
            countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        """
        Computes block mask from input mask
        Args: 
            mask: Mask tensor to compute block mask from
        Returns: 
            block_mask: Block masked version of input mask
        Processing Logic:
            - Calculate left and right padding for blocks
            - Get non-zero indices from mask
            - Generate offsets tensor for blocks
            - Repeat non-zero indices and offsets for number of blocks
            - Add offsets to non-zero indices to get block indices
            - Pad input mask and set block region values to 1
            - Return 1 - padded mask as the block mask
        """
        left_padding = int((self.block_size-1) / 2)
        right_padding = int(self.block_size / 2)

        batch_size, channels, height, width = mask.shape
        #print ("mask", mask[0][0])
        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1), # - left_padding,
                torch.arange(self.block_size).repeat(self.block_size), #- left_padding
            ]
        ).t().cuda()
        offsets = torch.cat((torch.zeros(self.block_size**2, 2).cuda().long(), offsets.long()), 1)

        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            #block_idxs += left_padding
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))

        block_mask = 1 - padded_mask#[:height, :width]
        return block_mask

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y



    def forward(self, x, get_alphas=False):
        if self.attention is not None:
            if self.transform_query_size is not None:
                q = x @ self.transform_W_query
                logits = q @ torch.transpose((self.embed @ self.transform_W_key),0,1) # Bxnum_classes key values
                c = self.softmax(logits) @ (self.embed @ self.transform_W_value)  # B x cdim context vector (or transform_query_size if provided)
            else:
                logits = x @ torch.transpose((self.embed @ self.transform_W_key),0,1) # Bxnum_classes key values
                c = self.softmax(logits) @ (self.embed @ self.transform_W_value)  # B x cdim context vector (or transform_query_size if provided)

            if self.attention == "sum":
                x = self.dropout(x) + c
            elif self.attention == "concat":
                x = torch.cat((self.dropout(x),c),1)
            else: # context only
                x = c
            if get_alphas:
                return F.linear(x, self.weight, self.bias), logits

        else:
            raise NotImplementedError()

        return F.linear(x, self.weight, self.bias)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def resnet12(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model


def resnet18(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 2, 2], keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model


def resnet24(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-24 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model


def resnet50(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-50 model.
    indeed, only (3 + 4 + 6 + 3) * 3 + 1 = 49 layers
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model


def resnet101(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-101 model.
    indeed, only (3 + 4 + 23 + 3) * 3 + 1 = 100 layers
    """
    model = ResNet(BasicBlock, [3, 4, 23, 3], keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model


def seresnet12(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], keep_prob=keep_prob, avg_pool=avg_pool, use_se=True, **kwargs)
    return model


def seresnet18(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 2, 2], keep_prob=keep_prob, avg_pool=avg_pool, use_se=True, **kwargs)
    return model


def seresnet24(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-24 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], keep_prob=keep_prob, avg_pool=avg_pool, use_se=True, **kwargs)
    return model


def seresnet50(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-50 model.
    indeed, only (3 + 4 + 6 + 3) * 3 + 1 = 49 layers
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], keep_prob=keep_prob, avg_pool=avg_pool, use_se=True, **kwargs)
    return model


def seresnet101(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-101 model.
    indeed, only (3 + 4 + 23 + 3) * 3 + 1 = 100 layers
    """
    model = ResNet(BasicBlock, [3, 4, 23, 3], keep_prob=keep_prob, avg_pool=avg_pool, use_se=True, **kwargs)
    return model


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--model', type=str, choices=['resnet12', 'resnet18', 'resnet24', 'resnet50', 'resnet101',
                                                      'seresnet12', 'seresnet18', 'seresnet24', 'seresnet50',
                                                      'seresnet101'])
    args = parser.parse_args()

    model_dict = {
        'resnet12': resnet12,
        'resnet18': resnet18,
        'resnet24': resnet24,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'seresnet12': seresnet12,
        'seresnet18': seresnet18,
        'seresnet24': seresnet24,
        'seresnet50': seresnet50,
        'seresnet101': seresnet101,
    }

    model = model_dict[args.model](avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=64)
    data = torch.randn(2, 3, 84, 84)
    model = model.cuda()
    data = data.cuda()
    feat, logit = model(data, is_feat=True)
    print(feat[-1].shape)
    print(logit.shape)
