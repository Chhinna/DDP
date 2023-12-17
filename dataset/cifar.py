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
    
            set_image_transformations(transform, self)
    
    
            self.file_pattern = '%s.pickle'
            if not self.pretrain:
                pass
    
            self.data = {}
    
            load_and_process_data(self, partition)
    
            # pre-process for contrastive sampling
            self.k = k
            self.is_sample = is_sample
            prepare_for_sampling(self)
    
    def set_image_transformations(transform, self):
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
    
    def load_and_process_data(self, partition):
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
    def prepare_for_sampling(self):
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
    

    def __getitem__(self, item):
        """
        Get item from dataset at index.
        Args:
            item: Index of item to retrieve
        Returns: 
            img: Image at index as tensor
            target: Label of image 
            item: Index
        Processing Logic:
            - Get image and target from lists using index
            - Normalize target by subtracting min label value 
            - Optionally return negative samples if sampling
            - Return image, target, index and negative samples if sampling
        """
        img = np.asarray(self.imgs[item]).astype('uint8')
        img = self.transform(img)
        target = self.labels[item] - min(self.labels)

        if not self.is_sample:
            return img, target, item
        pos_idx = item
        replace = True if self.k > len(self.cls_negative[target]) else False
        neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
        sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
        return img, target, item, sample_idx

    def __len__(self):
        """
        Returns the length of the labels attribute
        Args:
            self: The object whose labels attribute length is returned
        Returns:
            An integer representing the length of the labels attribute
        Calculates the length of the labels attribute of the object:
        - Gets the labels attribute of the passed in object 
        - Calls the built-in len() function on the labels attribute to get its length
        - Returns the length"""
        return len(self.labels)


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
            set_train_transform(train_transform, self)
    
            set_test_transform(test_transform, self)
    
            self.data = {}
            arrange_classes(self)
    
    def set_train_transform(train_transform, self):
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
    
    def set_test_transform(test_transform, self):
        if test_transform is None:
            self.test_transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.test_transform = test_transform
    def arrange_classes(self):
        for idx in range(self.imgs.shape[0]):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.imgs[idx])
        self.classes = list(self.data.keys())
    

    def __getitem__(self, item):
            """
            Samples data for few-shot classification tasks
            Args: 
                self: The dataset object
                item: The index of the task
            Returns:
                support_xs: Support set images
                support_ys: Support set labels 
                query_xs: Query set images
                query_ys: Query set labels
            Processing Logic:
                1. Samples classes and splits data into support and query sets
                2. Applies data augmentation and transformations
                3. Reshapes data and splits into batches
                4. Returns support and query examples and labels
            """
            query_xs, query_ys, support_xs, support_ys = sample_classes_and_prepare_data(self, item)
            support_xs, query_xs, support_ys, query_ys = reshape_data(query_xs, query_ys, support_xs, self, support_ys)
    
            support_xs, query_xs = apply_transformations(self, support_xs, query_xs)
    
            return support_xs, support_ys, query_xs, query_ys
    
    def sample_classes_and_prepare_data(self, item):
        if self.fix_seed:
            np.random.seed(item)
        cls_sampled = np.random.choice(self.classes, self.n_ways, False)
        support_xs = []
        support_ys = []
        query_xs = []
        query_ys = []
        for idx, cls in enumerate(cls_sampled):
            imgs = np.asarray(self.data[cls]).astype('uint8')
            support_xs_ids_sampled = np.random.choice(range(imgs.shape[0]), self.n_shots, False)
            support_xs.append(imgs[support_xs_ids_sampled])
            support_ys.append([idx] * self.n_shots)
            query_xs_ids = np.setxor1d(np.arange(imgs.shape[0]), support_xs_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)
            query_xs.append(imgs[query_xs_ids])
            query_ys.append([idx] * query_xs_ids.shape[0])
        support_xs, support_ys, query_xs, query_ys = np.array(support_xs), np.array(support_ys), np.array(
            query_xs), np.array(query_ys)
        return query_xs, query_ys, support_xs, support_ys
    
    def reshape_data(query_xs, query_ys, support_xs, self, support_ys):
        num_ways, n_queries_per_way, height, width, channel = query_xs.shape
        query_xs = query_xs.reshape((num_ways * n_queries_per_way, height, width, channel))
        query_ys = query_ys.reshape((num_ways * n_queries_per_way,))
    
        support_xs = support_xs.reshape((-1, height, width, channel))
        if self.n_aug_support_samples > 1:
            support_xs = np.tile(support_xs, (self.n_aug_support_samples, 1, 1, 1))
            support_ys = np.tile(support_ys.reshape((-1,)), (self.n_aug_support_samples))
        support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
        query_xs = query_xs.reshape((-1, height, width, channel))
        query_xs = np.split(query_xs, query_xs.shape[0], axis=0)
        return support_xs, query_xs, support_ys, query_ys
    def apply_transformations(self, support_xs, query_xs):
        support_xs = torch.stack([self.train_transform(x.squeeze()) for x in support_xs])
        query_xs = torch.stack([self.test_transform(x.squeeze()) for x in query_xs])
        return support_xs, query_xs
    

    def __len__(self):
        """
        Returns the number of test runs in the object
        Args:
            self: The object whose number of test runs is returned
        Returns: 
            n_test_runs: The number of test runs in the object
        - The function returns the attribute n_test_runs of the object
        - n_test_runs stores the number of test runs performed by the object
        """
        return self.n_test_runs


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