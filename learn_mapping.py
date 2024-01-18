import torch
import torch.nn as nn
import pickle
from models.util import get_embeds
import pdb
import os
from models.resnet_language import LinearMap

WORD_EMBED_PATH = "word_embeds/CIFAR_Glove_300d.pkl"


GLOVE = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1.0
WD = 5e-4
EPOCHS = 1000



def load_pickle(pth):
    """
    Loads pickled data from a file path
    Args:
        pth: Path to pickle file
    Returns: 
        d: Loaded pickled data
    Loads pickled data from a file:
    - Opens file at given path for reading in binary mode
    - Loads pickled data from file handle using pickle.load()
    - Returns loaded data
    """
    with open(pth, 'rb') as f:
        d = pickle.load(f)
    return d

def get_base_labels(pth):
    """
    Load labels from pickle file
    Args:
        pth: Path to pickle file 
    Returns:
        ls: List of labels
    - Load dictionary from pickle file at given path
    - Initialize empty list with same length as dictionary
    - Iterate through dictionary and add values to list using keys as indices  
    - Return the list of labels
    """
    d = load_pickle(pth)
    ls = [""]*len(d)
    for k,v in d.items():
        ls[k] = v
    return ls

def get_classifier_weights(pth, device):
    """Get weights of classifier from PyTorch checkpoint
    Args:
        pth: Path to PyTorch checkpoint file
        device: Device to load checkpoint weights
    Returns: 
        weights: Weights of classifier layer from checkpoint
    Loads PyTorch checkpoint, returns checkpoint object and classifier weights:
    - Load checkpoint from pth using device for mapping
    - Return checkpoint object and classifier weights from model.classifier layer"""
    ckpt = torch.load(pth, map_location=device)
    return ckpt, ckpt['model']['classifier.weight']

def save_model(ckpt, model, nickname, save_path):
    """
    Saves a model state dict to a checkpoint file
    Args:
        ckpt: Checkpoint dictionary to save model state dict
        model: Model to extract state dict from 
        nickname: Name to assign the model state dict in the checkpoint
        save_path: Path to save the checkpoint file
    Returns: 
        None: Does not return anything
    - Extracts the state dict of the model
    - Saves it to the checkpoint dictionary under the given nickname 
    - Saves the entire checkpoint dictionary to the given save path location"""
    ckpt[nickname] = model.state_dict()
    torch.save(ckpt, save_path)

def main(MODEL_HOME, MODEL_PATH, SAVE_PATH):
    """
    Trains a linear mapping between label embeddings and image embeddings.
    
    Args:
        MODEL_HOME: Path to model home directory
        MODEL_PATH: Path to pretrained model weights
        SAVE_PATH: Path to save trained linear mapping model
    
    Returns: 
        None
    
    Processing Logic:
        - Load pretrained classifier weights and extract label and image embeddings
        - Load label embeddings from GloVe/Word2Vec
        - Initialize linear mapping model 
        - Train model to minimize MSE loss between predicted and actual image embeddings
        - Save trained linear mapping model
    """
    
    ckpt, base_embeds = get_classifier_weights(MODEL_PATH, DEVICE) #Tensor
    base_labels = [name for name in ckpt['label2human'] if name != ''] 
    label_embeds = get_embeds(WORD_EMBED_PATH, vocab=base_labels).float().to(DEVICE) #Tensor
    label_embed_size = 300 if GLOVE else 500
    label_embeds = label_embeds[:, :label_embed_size].to(DEVICE)
    model = LinearMap(label_embed_size, base_embeds.size(1)).to(DEVICE) # e.g. for glove 300x640
    
    optimizer = torch.optim.SGD(model.parameters(),
                          lr=LR,
                          weight_decay=WD)
    criterion = nn.MSELoss()
    
    for ep in range(EPOCHS):
        
        output = model(label_embeds)
        loss = criterion(output, base_embeds)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (ep+1) % 10 == 0:
            print(f"Epoch [{ep+1}/{EPOCHS}] Loss: {loss}")
    
    save_model(ckpt, model, "mapping_linear_label2image", SAVE_PATH)
        
if __name__ == "__main__":
    for i in range(3,11):
        
        MODEL_HOME = f"/home/gridsan/akyurek/git/rfs-incremental/dumped/backbones/continual/resnet18/{i}/"
        MODEL_PATH = os.path.join(MODEL_HOME, "resnet18_last.pth")
        SAVE_PATH = os.path.join(MODEL_HOME, "resnet18_last_with_mapping.pth")
#         BASE_LABELS = os.path.join(MODEL_HOME, "label2human.pickle")
        main(MODEL_HOME, MODEL_PATH, SAVE_PATH)
