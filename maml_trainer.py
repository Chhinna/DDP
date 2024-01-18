from __future__ import print_function

import os
import socket
import time
import sys
import subprocess
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models import model_pool
from models.util import create_model
from models.resnet_language import LangPuller

from dataset.mini_imagenet import ImageNet, MetaImageNet
from dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet
from dataset.transform_cfg import transforms_options, transforms_list

from util import adjust_learning_rate, create_and_save_embeds, create_and_save_descriptions
from eval.util import accuracy, AverageMeter, validate

import ipdb
from configs import parse_option_supervised
import pandas as pd
import datetime
import learn2learn as l2l

def main():
    """Calculates accuracy between predictions and targets
    Args:
        predictions: Predictions from the model
        targets: True target values  
    Returns: 
        accuracy: Accuracy between predictions and targets
    Processing Logic:
        1. Get predictions in one-hot format to match targets
        2. Calculate number of correct predictions
        3. Divide correct predictions by total number of targets to get accuracy
    """
    def accuracy(predictions, targets):
        predictions = predictions.argmax(dim=1).view(targets.shape)
        return (predictions == targets).sum().float() / targets.size(0)
        
    def training_step(
        data,
        target_here,
        learner,
        epoch: int = 0,
    ):
        device = torch.device('cuda')
        #print(data)
        #print(len(data))
        #s_inputs, s_labels = batch.support
        #q_inputs, q_labels = batch.query
        s_inputs = data[:32]
        s_labels = target[:32]
        q_inputs = data[32:]
        q_labels = target[32:]
        query_loss = torch.tensor(.0, device=device)

        s_inputs = s_inputs.float().cuda(device=device)
        s_labels = s_labels.cuda(device=device)
        q_inputs = q_inputs.float().cuda(device=device)
        q_labels = q_labels.cuda(device=device)
        
        inner_criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        # Adapt the model on the support set
        for step in range(3):
            # forward + backward + optimize
            pred = learner(s_inputs)
            #pred = learner(s_inputs, step)
            new_labels = []
            #print(s_labels)
            for i in s_labels:
                #print(i)
                new_labels.append(int(i))
            torch_new_labels = torch.LongTensor(new_labels)
            torch_new_labels = torch_new_labels.to(device)
            support_loss = inner_criterion(pred, torch_new_labels)
            
            learner.adapt(support_loss)
            

        # Evaluate the adapted model on the query set
        q_pred = learner(q_inputs)
        #q_pred = learner(q_inputs, step)
        q_new_labels = []
        for i in q_labels:
            q_new_labels.append(int(i))
        torch_q_new_labels = torch.LongTensor(q_new_labels)
        torch_q_new_labels = torch_q_new_labels.to(device)
        query_loss = inner_criterion(q_pred, torch_q_new_labels)
        acc = accuracy(q_pred, torch_q_new_labels).detach()
        
        return query_loss, acc
        
    opt = parse_option_supervised()
    torch.manual_seed(opt.set_seed)
    np.random.seed(opt.set_seed)
    if opt.dataset == 'miniImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        train_loader = DataLoader(ImageNet(args=opt, split="train", phase="train", transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_loader = DataLoader(ImageNet(args=opt, split="train", phase="val", transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)
        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 64
            if opt.continual:
                n_cls = 60
        
        fast_lr = 0.01
        meta_lr = 0.001
        vocab = None
        model = create_model(opt.model, n_cls, opt, vocab=vocab)
        model = model.cuda()
        maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=True)
        
        optim = torch.optim.AdamW(maml.parameters(), meta_lr, betas=(0, 0.999))
        meta_bsz = 2
        # train_samples = not sure
        k_shots = 60
        n_queries = 60
        epochs = 1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim,
            T_max=epochs,
            eta_min=0.0001,
        )
        
        for epoch in range(epochs):
            epoch_meta_train_loss, epoch_meta_train_acc = 0.0, 0.0
            for idx, (input, target,  _) in enumerate(train_loader):
                optim.zero_grad()
                meta_train_losses, meta_train_accs = [], []
                
                meta_loss, meta_acc = training_step(
                    input,
                    target,
                    maml.clone(),
                    epoch=epoch,
                )
                
                meta_loss.backward()
                #final_meta_loss = final_meta_loss + meta_loss
                meta_train_losses.append(meta_loss.detach())
                meta_train_accs.append(meta_acc)
            
            epoch_meta_train_loss += torch.Tensor(meta_train_losses).mean().item()
            epoch_meta_train_acc += torch.Tensor(meta_train_accs).mean().item()
            
            print(epoch_meta_train_loss)
            print(epoch_meta_train_acc)
            # Average the accumulated gradients and optimize
            
            with torch.no_grad():
                for p in maml.parameters():
                    # Remember the MetaBatchNorm layer has parameters that don't require grad!
                    if p.requires_grad:
                        p.grad.data.mul_(1.0 / meta_bsz)
            
            optim.step()
            scheduler.step()
        
        torch.save(maml.model.state_dict(), 'model_weights.pth')
        
        return
if __name__ == "__main__":
    print("inside main")
    main() 
