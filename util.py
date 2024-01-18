import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import argparse



class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        """
        Calculate loss for model prediction against target.
        Args:
            x: Input tensor: Model predictions 
            target: Target tensor: Correct labels
        Returns: 
            loss: Loss tensor: Average loss over batch
        - Calculate log probabilities of predictions using softmax
        - Calculate negative log likelihood loss against one-hot encoded targets
        - Calculate smoothing loss as negative average log probability  
        - Combine NLL and smoothing losses with confidence and smoothing weights
        - Return average loss over batch"""
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class BCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None, num_classes=64):
        """
        Initialize BCEWithLogitsLoss loss function
        Args:
            weight: Weight of each class for loss computation
            size_average: Normalize loss by the batch size
            reduce: Reduce loss over samples/batches or not 
            reduction: Type of loss reduction ('mean' or 'sum')
            pos_weight: Weight of positive examples for unbalanced classes
            num_classes: Number of classes
        Returns: 
            loss: Computed loss value
        - Computes BCE loss between input and target 
        - Applies weights to loss if provided
        - Averages loss over samples/batches based on arguments
        - Returns scalar loss value
        """
        super(BCEWithLogitsLoss, self).__init__()
        self.num_classes = num_classes
        self.criterion = nn.BCEWithLogitsLoss(weight=weight,
                                              size_average=size_average,
                                              reduce=reduce,
                                              reduction=reduction,
                                              pos_weight=pos_weight)
    def forward(self, input, target):
        """Computes the loss between input and target
        Args:
            input: Input tensor 
            target: Target tensor 
        Returns: 
            loss: Loss value computed by criterion
        Computes one-hot encoding of target
            Applies criterion to compute loss between input and one-hot encoded target
            Returns loss value"""
        target_onehot = F.one_hot(target, num_classes=self.num_classes)
        return self.criterion(input, target_onehot)

def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

def create_and_save_embeds(opt, vocab):
    """
    Creates and saves word embeddings
    Args: 
        opt: Options class containing parameters
        vocab: Vocabulary containing words
    Returns: 
        None: Does not return anything, saves embeddings to file
    Processing Logic:
        - Gets word embedding path and dimension from options
        - Checks if embedding path exists, creates if not
        - Gets all words from vocabulary
        - Loads GloVe embeddings 
        - Loops through words and extracts embeddings
        - Saves embeddings in dictionary to file
    """

    word_embeds = opt.word_embed_path
    dim = opt.word_embed_size
    embed_pth = "{0}_dim{1}.pkl".format(opt.dataset, dim)

    if not os.path.isdir(word_embeds):
        os.makedirs(word_embeds)

    words = []
    for token in vocab:
        words = words + token.split(' ')

    embed_pth = os.path.join(word_embeds, embed_pth)
    print(embed_pth)
    if os.path.exists(embed_pth):
        print("Found {}.".format(embed_pth))
        return
    else:
        print("Loading dictionary...")
        print(words)
        """
        from torchnlp.word_to_vector import Vico
        pretrained_embedding = Vico(name='linear',
                                    dim=dim,
                                    is_include=lambda w: w in set(words))

        embeds = []
        keys = pretrained_embedding.token_to_index.keys()
        for w in keys:
            embeds.append(pretrained_embedding[w].numpy())
        d = dict(zip(keys, embeds))

        # Pickle the dictionary for later load
        print("Pickling word embeddings...")
        with open(embed_pth, 'wb') as f:
            pickle.dump(d, f)
        print("Pickled.")
        """
        import torchtext
        glove = torchtext.vocab.GloVe(name='6B', dim=500)
        embeds = []
        words = list(set(words))
        for w in words:
            embeds.append(glove.vectors[glove.stoi[w]])
        d = dict(zip(words, embeds))
        print("Pickling word embeddings...")
        with open(embed_pth, 'wb') as f:
            pickle.dump(d, f)
        print("Pickled.")
        print(embed_pth)



def create_and_save_descriptions(opt, vocab):
    """
    Create and save description embeddings
    Args: 
        opt: Options object containing parameters
        vocab: List of vocabulary words
    Returns: 
        None: Description embeddings are pickled to file
    Processing Logic:
        - Check if output directory exists, create if not
        - Construct pickle path from parameters
        - Check if pickle exists, if not continue
        - Initialize tokenizer and transformer model
        - Get definitions from WordNet for each vocab word  
        - Generate embeddings for each definition using transformer hidden states
        - Zip vocab words and embeddings into dictionary
        - Pickle dictionary to file
    """

    if not os.path.isdir(opt.description_embed_path):
        os.makedirs(opt.description_embed_path)

    embed_pth = os.path.join(opt.description_embed_path,
                             "{0}_{1}_layer{2}_prefix_{3}.pickle".format(opt.dataset,
                                                             opt.desc_embed_model,
                                                             opt.transformer_layer,
                                                             opt.prefix_label))

    if os.path.exists(embed_pth):
        return
    else:
        print("Path {} not found.".format(embed_pth))
        with torch.no_grad():
            print("Creating tokenizer...")
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            tokenizer = AutoTokenizer.from_pretrained(opt.desc_embed_model)
            print("Initializing {}...".format(opt.desc_embed_model))
            model = AutoModelForMaskedLM.from_pretrained(opt.desc_embed_model, output_hidden_states=True)

            # Create wordnet
            from nltk.corpus import wordnet
            defs = [wordnet.synsets(v.replace(" ", "_"))[0].definition() for v in vocab]
    #         defs = torch.cat(defs, 0)
            embeds = []
            for i,d in enumerate(defs):
                inp = vocab[i]+" "+d if opt.prefix_label else d
                inp = tokenizer(inp, return_tensors="pt")
                outputs = model(**inp)
                hidden_states = outputs[1]
                embed = torch.mean(hidden_states[opt.transformer_layer], dim=(0,1))
                embeds.append(embed)

            d = dict(zip(vocab, embeds))
            # Pickle the dictionary for later load
            print("Pickling description embeddings from {}...".format(opt.desc_embed_model))
            with open(embed_pth, 'wb') as f:
                pickle.dump(d, f)
            print("Pickled.")

def restricted_float(x):
    """Restricts a float to the range [0.0, 1.0]
    Args: 
        x: The value to convert to a float and check
    Returns:
        x: The float restricted to the range [0.0, 1.0]
    Processing Logic:
    - Converts the input to a float
    - Checks if the float is in the range [0.0, 1.0]
    - Raises an error if not in range
    - Returns the float if it is in range"""
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x
