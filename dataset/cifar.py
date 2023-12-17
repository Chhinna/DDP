from __future__ import print_function

import os
import pickle
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class CIFAR100(Dataset):
    """support FC100 and CIFAR-FS"""
    def __init__(self, args, partition='train', pretrain=True, is_sample=False, k=4096,
                     transform=None):
            """
            Initializes a dataset
            Args: 
                self: The object
                args: Arguments
                partition: Dataset partition ('train' or 'test')
                pretrain: Whether to pretrain
                is_sample: Whether to sample
                k: Number of negative samples
                transform: Image transformations
            Returns:
                None
            Processes data:
                - Loads data from file
                - Normalizes images
                - Preprocesses labels
                - Samples positive and negative pairs if is_sample is True
            """
            super(Dataset, self).__init__()
            self.data_root = args.data_root
            self.partition = partition
            self.data_aug = args.data_aug
            self.mean = [0.5071, 0.4867, 0.4408]
            self.std = [0.2675, 0.2565, 0.2761]
            self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
            self.pretrain = pretrain
    
            setup_transformations(transform, self)
    
    
            self.file_pattern = '%s.pickle'
            if not self.pretrain:
                pass
    
            self.data = {}
    
            load_and_preprocess_data(self, partition)
    
            # pre-process for contrastive sampling
            self.k = k
            self.is_sample = is_sample
            if self.is_sample:
                self.labels = np.asarray(self.labels)
                self.labels = self.labels - np.min(self.labels)
                num_classes = np.max(self.labels) + 1
    
                self.cls_positive = [[] for _ in range(num_classes)]
                for i in range(len(self.imgs)):
                    self.cls_positive[self.labels[i]].append(i)
    
                self.cls_negative = [[] for _ in range(num_classes)]
                for i in range(num_classes):
                    for j in range(num_classes):
                        if j == i:
                            continue
                        self.cls_negative[i].extend(self.cls_positive[j])
    
                self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
                self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]
                self.cls_positive = np.asarray(self.cls_positive)
                self.cls_negative = np.asarray(self.cls_negative)
    
    def setup_transformations(transform, self):
        if transform is None:
            if self.partition == 'train' and self.data_aug:
                self.transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
            else:
                self.transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
        else:
            self.transform = transform
    def load_and_preprocess_data(self, partition):
        with open(os.path.join(self.data_root, self.file_pattern % partition), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            self.imgs = data['data']
            labels = data['labels']
            # adjust sparse labels to labels from 0 to n.
            cur_class = 0
            label2label = {}
            for idx, label in enumerate(labels):
                if label not in label2label:
                    label2label[label] = cur_class
                    cur_class += 1
            new_labels = [label2label[label] for (idx, label) in enumerate(labels)]
            self.labels = new_labels
    


class MetaCIFAR100(CIFAR100):

    def __init__(self, args, partition='train', train_transform=None, test_transform=None, fix_seed=True):
            """
            Initialize MetaCIFAR100 dataset
            Args:
                args: Arguments - Contains dataset configuration
                partition: Dataset partition ('train' or 'test') 
                train_transform: Transformations for support set
                test_transform: Transformations for query set
                fix_seed: Fix random seed
            Returns: 
                self: MetaCIFAR100 object
            Processing Logic:
                - Initialize base class
                - Set dataset hyperparameters
                - Define transformations
                - Split data into classes
                - Shuffle class data
            """
            super(MetaCIFAR100, self).__init__(args, partition, False)
            self.fix_seed = fix_seed
            self.n_ways = args.n_ways
            self.n_shots = args.n_shots
            self.n_queries = args.n_queries
            self.classes = list(self.data.keys())
            self.n_test_runs = args.n_test_runs
            self.n_aug_support_samples = args.n_aug_support_samples
            define_train_transformations(train_transform, self)
    
            define_test_transformations(test_transform, self)
    
            load_images_and_labels(self)
    
    def define_train_transformations(train_transform, self):
        if train_transform is None:
            self.train_transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.train_transform = train_transform
    
    def define_test_transformations(test_transform, self):
        if test_transform is None:
            self.test_transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.test_transform = test_transform
    def load_images_and_labels(self):
        self.data = {}
        for idx in range(self.imgs.shape[0]):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.imgs[idx])
        self.classes = list(self.data.keys())
    


if __name__ == '__main__':
    args = lambda x: None
    args.n_ways = 5
    args.n_shots = 1
    args.n_queries = 12
    # args.data_root = 'data'
    args.data_root = '/home/yonglong/Downloads/FC100'
    args.data_aug = True
    args.n_test_runs = 5
    args.n_aug_support_samples = 1
    imagenet = CIFAR100(args, 'train')
    print(len(imagenet))
    print(imagenet.__getitem__(500)[0].shape)

    metaimagenet = MetaCIFAR100(args, 'train')
    print(len(metaimagenet))
    print(metaimagenet.__getitem__(500)[0].size())
    print(metaimagenet.__getitem__(500)[1].shape)
    print(metaimagenet.__getitem__(500)[2].size())
    print(metaimagenet.__getitem__(500)[3].shape)