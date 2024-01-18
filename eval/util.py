import torch
import time
import numpy as np
import io
import base64
from PIL import Image
import scipy

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        """
        Initialize the object
        Args:
            self: The object
        Returns:
            None: Nothing is returned
        - Reset the internal state of the object by calling self.reset()
        """
        self.reset()

    def reset(self):
        """Resets the statistics tracker object.
        Args:
            self: The statistics tracker object. 
        Returns: 
            None: Resets the internal state of the tracker.
        - Sets the value (val) to 0
        - Sets the average (avg) to 0 
        - Sets the running sum (sum) to 0
        - Sets the sample count (count) to 0"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Updates the running statistics
        Args:
            val: New value to include in the statistics
            n: Number of samples (default is 1) 
        Returns: 
            None: Does not return anything
        - Adds the new value to the running sum
        - Increases the sample count by n 
        - Recomputes the running average from the new sum and count"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def mean_confidence_interval(data, confidence=0.95):
    """
    Calculate a mean confidence interval from data
    Args:
        data: Sample data to calculate statistics on
        confidence: Confidence level as a decimal percentage (default 0.95)
    Returns: 
        m, h: Mean and half-width of confidence interval
    - Calculate mean (m) and standard error (se) of data
    - Use student's t-distribution to calculate half-width (h) based on confidence level
    - Return mean and half-width to define confidence interval as m ± h"""
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h

def normalize(x):
    """Normalizes a tensor
    Args: 
        x: Tensor to normalize
    Returns: 
        Tensor: Normalized tensor
    - Squares each element of the input tensor x
    - Sums the squared elements along dimension 1, keeping other dimensions the same size 
    - Takes the square root of the sum to calculate the norm
    - Divides each element of the input tensor by the calculated norm to normalize the tensor"""
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    return x.div(norm)

def image_formatter(im):
    """
    Formats image for display in HTML
    Args:
        im: Image to format in one line
    Returns: 
        String: Formatted image string in one line
    Processing Logic:
        - Normalize image pixels to range 0-255
        - Convert to uint8 and transpose dimensions
        - Save image to BytesIO stream
        - Encode BytesIO to base64 string
        - Return HTML image tag with base64 data URI
    """
    im = ((im / np.max(im, axis=(1,2), keepdims=True)) * 255).astype('uint8').transpose((1,2,0))
    im = Image.fromarray(im)
    rawBytes = io.BytesIO()
    im.save(rawBytes, "PNG") # TODO: why this is required here ?
    rawBytes.seek(0)  # return to the start of the file
    decoded = base64.b64encode(rawBytes.read()).decode()
    return f'<img src="data:image/jpeg;base64,{decoded}">'

def freeze_backbone_weights(backbone, opt, epoch, exclude=['classifier.transform']):
    """
    Freeze backbone weights at a specified epoch
    Args:
        backbone: Backbone model
        opt: Training options
        epoch: Current epoch number
    Returns: 
        None: Function does not return anything
    Freeze all parameters in backbone model except those in exclude list:
        - Check if current epoch matches opt.freeze_backbone_at
        - Set requires_grad=False for all backbone parameters
        - Check parameter name against exclude list and set requires_grad=True if matches
    """
    if opt.freeze_backbone_at == epoch:
        print("Freezing the backbone.")
        for name, param in backbone.named_parameters():
            param.requires_grad = False
            if any(map(lambda s: name.startswith(s), exclude)): # why not; name in exclude:
                print("Not frozen: ", name)
                param.requires_grad = True

def NN(support, support_ys, query):
    """nearest classifier"""
    support = np.expand_dims(support.transpose(), 0)
    query = np.expand_dims(query, 2)
    diff = np.multiply(query - support, query - support)
    distance = diff.sum(1)
    min_idx = np.argmin(distance, axis=1)
    pred = [support_ys[idx] for idx in min_idx]
    return pred

def Cosine(support, support_ys, query):
    """Cosine classifier"""
    support_norm = np.linalg.norm(support, axis=1, keepdims=True)
    support = support / support_norm
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    query = query / query_norm
    cosine_distance = query @ support.transpose()
    max_idx = np.argmax(cosine_distance, axis=1)
    pred = [support_ys[idx] for idx in max_idx]
    return pred

def get_optim(net, opt):
    """Initializes an optimizer for training a network
    Args:
        net: Network to optimize
        opt: Optimizer options
    Returns: 
        optimizer: Initialized optimizer 
    Initializes either an Adam or SGD optimizer based on opt.adam flag:
        - If opt.adam is True, initializes Adam optimizer
        - Else initializes SGD optimizer using learning rate, momentum, and weight decay from opt"""
    if opt.adam:
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=opt.learning_rate,
                                     weight_decay=0.0005)
    else:
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=opt.learning_rate,
                                    momentum=opt.momentum,
                                    weight_decay=opt.weight_decay)
    return optimizer

def get_vocab(loaders):
    """Get vocabularies from multiple loaders
    Args:
        loaders: Loaders to extract vocabularies from
    Returns: 
        vocabs: List of vocabularies extracted from each loader
    - Iterate through each loader
    - Extract label to human mapping from dataset
    - Filter out empty names
    - Append filtered names to vocab list
    - Return list of vocabularies"""
    vocabs = []
    for loader in loaders:
        label2human = loader.dataset.label2human
        vocab = [name for name in label2human if name != '']
        vocabs.append(vocab)
    return vocabs

def get_vocabs(base_loader=None, novel_loader=None, query_ys=None):
    """
    Gets vocabularies from base and novel loaders.
    Args:
        base_loader: Base loader to get base vocab from
        novel_loader: Novel loader to get novel vocab from 
        query_ys: Query labels to get novel vocab from
    Returns: 
        vocab_base: Base vocabulary
        vocab_all: Concatenated base and novel vocabulary
        vocab_novel: Novel vocabulary
        orig2id: Mapping from original novel labels to ids
    Processing Logic:
        - Extract base vocabulary from base loader if provided
        - Extract novel vocabulary from novel loader/query_ys if provided 
        - Concatenate base and novel vocabularies
        - Return vocabularies and original to id mapping
    """
    vocab_all = []
    vocab_base = None
    if base_loader is not None:
        label2human_base = base_loader.dataset.label2human
        vocab_base  = [name for name in label2human_base if name != '']
        vocab_all  += vocab_base

    vocab_novel, orig2id = None, None

    if novel_loader is not None:
        print(query_ys)
        novel_ids = np.sort(np.unique(query_ys))
        label2human_novel = novel_loader.dataset.label2human
        print("Some important shit")
        print(novel_loader.dataset.label2human)
        print(novel_ids)
        vocab_novel = [label2human_novel[i] for i in novel_ids]
        orig2id = dict(zip(novel_ids, len(vocab_base) + np.arange(len(novel_ids))))
        vocab_all += vocab_novel

    return vocab_base, vocab_all, vocab_novel, orig2id

def drop_a_dim(data): #TODO why do we need this in the first place?
    """
    Drops a dimension from the input data.
    Args:
        data: Tuple of tensors containing support and query data.
    Returns: 
        Tuple: Restructured support and query data with dropped dimension.
    Processing Logic:
        - Extract support and query data from input tuple
        - View support and query data to drop batch dimension
        - View labels to drop batch dimension
        - Return restructured data tuple
    """
    support_xs, support_ys, query_xs, query_ys = data
    batch_size, _, height, width, channel = support_xs.size()
    print(support_xs.shape)
    print(type(support_xs))
    support_xs = support_xs.view(-1, height, width, channel)
    print(query_xs.shape)
    print(type(query_xs))
    
    query_xs = query_xs.view(-1, height, width, channel)
    support_ys = support_ys.view(-1).detach().numpy() # TODO
    query_ys = query_ys.view(-1).detach().numpy()
    return (support_xs, support_ys, query_xs, query_ys)

def get_batch_cycle(meta_trainloader_it, meta_trainloader):
    """
    Gets next batch from dataloader iterator or resets iterator if exhausted
    Args:
        meta_trainloader_it: Iterator over dataloader
        meta_trainloader: Dataloader
    Returns: 
        data: Next batch of data
    - Tries to get next batch from iterator
    - If iterator is exhausted, resets it to start of dataloader 
    - Returns next batch
    """
    try:
        data = next(meta_trainloader_it)
    except StopIteration:
        meta_trainloader_it = iter(meta_trainloader)
        data = next(meta_trainloader_it)
    return data

def log_episode(novel_labels,
                vocab_novel,
                epoch,
                novel_acc,
                base_acc,
                running_base,
                running_novel):
    """
    Logs episode results
    Args:
        novel_labels: Novel class labels in one line
        vocab_novel: Novel vocabulary in one line 
        epoch: Current epoch number
        novel_acc: Novel accuracy score
        base_acc: Base accuracy score
        running_base: Running average of base accuracy
        running_novel: Running average of novel accuracy
    Returns: 
        None: Does not return anything, just prints the results
    Processing Logic:
        - Calculates average of novel and base accuracy
        - Calculates running average of novel and base accuracies
        - Prints the results in a formatted string with class labels, vocab, accuracies and averages
    """
    avg_score = (novel_acc + base_acc) / 2
    running_avg = (running_base + running_novel) / 2
    print('\n{:25} {:}\n'
          '{:25} {:}\n'
          '{:25} {:}\n'
          '{:25} {:.4f}\n'
          '{:25} {:.4f}\n'
          '{:25} {:.4f}\n'
          '{:25} {:.4f}\n'
          '{:25} {:.4f}\n'
          '{:25} {:.4f}\n'.format("Classes:",
                                  novel_labels,
                                  "Labels:",
                                  vocab_novel,
                                  "Fine-tuning epochs:",
                                  epoch-1,
                                  "Novel acc:",
                                  novel_acc,
                                  "Base acc:",
                                  base_acc,
                                  "Average:",
                                  avg_score,
                                  "Runnning Base Avg:",
                                  running_base,
                                  "Running Novel Avg:",
                                  running_novel,
                                  "Running Average:",
                                  running_avg,
                                  ), flush=True)

def validate(val_loader, model, criterion, opt):
    """One epoch validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target, _) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda().long()

            # compute output
            output = model(input)
            if opt.dataset == "tieredImageNet" and opt.augment_pretrain_wtrainb:
                output = output[:,:200]
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg
