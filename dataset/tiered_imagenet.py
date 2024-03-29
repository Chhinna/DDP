import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class TieredImageNet(Dataset):
    def __init__(self, args, partition='train', pretrain=True, is_sample=False, k=4096,
                 transform=None):
        """Initializes a Dataset object
        
        Args:
            args: Namespace of arguments - {Argument description in one line}
            partition: str - {Argument description in one line}
            pretrain: bool - {Argument description in one line}
            is_sample: bool - {Argument description in one line}
            k: int - {Argument description in one line}
            transform: transform - {Argument description in one line}
        
        Returns:
            self: Dataset - Initialized Dataset object
        
        Processing Logic:
            - Loads image and label data from files
            - Normalizes images
            - Applies data augmentation if partition is 'train'
            - Pools train data from two files if pretrain and augment_pretrain_wtrainb
            - Samples positive and negative pairs for contrastive learning if is_sample
        """
        super(Dataset, self).__init__()
        self.data_root = args.data_root
        self.partition = partition
        self.data_aug = args.data_aug
        self.mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        self.std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]

        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.pretrain = pretrain

        if transform is None:
            if self.partition == 'train' and self.data_aug:
                self.transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.RandomCrop(84, padding=8),
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

        if self.pretrain:
            self.image_file_pattern = 'train_a_train_a_phase_%s_images.npz'
            self.label_file_pattern = 'train_a_train_a_phase_%s_labels.pkl'
        else:
            self.image_file_pattern = '%s_images.npz'
            self.label_file_pattern = '%s_labels.pkl'

        self.data = {}

        # modified code to load tieredImageNet 
        image_file = os.path.join(self.data_root, self.image_file_pattern % partition)
        self.imgs = np.load(image_file)['images']
        label_file = os.path.join(self.data_root, self.label_file_pattern % partition)
        self.labels = self._load_labels(label_file)['label_specific']

        
        # if partition is train, we'll pool two files together to use all 351 classes in backbone training.
        if self.pretrain and partition == "train" and args.augment_pretrain_wtrainb:
            second_image_file = os.path.join(self.data_root, 'train_b_images.npz')
            second_label_file = os.path.join(self.data_root, 'train_b_labels.pkl')
            num_classes = np.max(self.labels) + 1
            second_imgs = np.load(second_image_file)['images']
            second_labels = self._load_labels(second_label_file)['label_specific']
            second_labels = num_classes + second_labels
            self.imgs = np.concatenate((self.imgs, second_imgs), axis=0)
            self.labels = np.concatenate((self.labels, second_labels), axis=0)
        

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
            - Return image and target
            - Optionally also return positive and negative samples
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
            len: The length of the labels attribute
        Calculates the length of the labels attribute by calling the built-in len() function on it."""
        return len(self.labels)

    @staticmethod
    def _load_labels(file):
        """Load labels from a pickle file
        Args:
            file: File path to load labels from
        Returns: 
            data: Loaded labels data
        - Try to load data from file using pickle.load
        - If that fails, try loading with latin1 encoding
        - Return loaded data"""
        try:
            with open(file, 'rb') as fo:
                data = pickle.load(fo)
            return data
        except:
            with open(file, 'rb') as f:
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                data = u.load()
            return data


class MetaTieredImageNet(TieredImageNet):

    def __init__(self, args, partition='train', train_transform=None, test_transform=None, fix_seed=True):
        """
        Initialize MetaTieredImageNet dataset
        Args: 
            args: Arguments - Contains dataset configuration
            partition: Dataset partition ('train' or 'test')
            train_transform: Transformations for training data
            test_transform: Transformations for test data 
            fix_seed: Fix random seed
        Returns:
            self: MetaTieredImageNet object
        Processing Logic:
            - Initialize superclass with args and partition
            - Set attributes like n_ways, n_shots, n_queries from args
            - Define train and test transforms if not provided
            - Organize images into classes based on labels
            - Set class list attribute
        """
        super(MetaTieredImageNet, self).__init__(args, partition, False)
        self.fix_seed = fix_seed
        self.n_ways = args.n_ways
        self.n_shots = args.n_shots
        self.n_queries = args.n_queries
        self.classes = list(self.data.keys())
        self.n_test_runs = args.n_test_runs
        self.n_aug_support_samples = args.n_aug_support_samples
        if train_transform is None:
            self.train_transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.RandomCrop(84, padding=8),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.train_transform = train_transform

        if test_transform is None:
            self.test_transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.test_transform = test_transform

        self.data = {}
        for idx in range(self.imgs.shape[0]):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.imgs[idx])
        self.classes = list(self.data.keys())

    def __getitem__(self, item):
        """
        Samples data for few-shot classification tasks
        Args: 
            self: The data sampler object
            item: The random seed
        Returns:
            support_xs, support_ys, query_xs, query_ys: The sampled support and query sets
        Processing Logic:
            1. Samples N-way K-shot classification tasks from dataset
            2. Samples support and query examples for each class
            3. Applies data augmentation and transformations
            4. Reshapes and splits data into mini-batches
        """
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

        support_xs = torch.stack([self.train_transform(x.squeeze()) for x in support_xs])
        query_xs = torch.stack([self.test_transform(x.squeeze()) for x in query_xs])

        return support_xs, support_ys, query_xs, query_ys

    def __len__(self):
        """
        Returns the number of test runs in the object
        Args:
            self: The object
        Returns: 
            n_test_runs: The number of test runs in the object
        Calculates the number of test runs by accessing the n_test_runs attribute of the object.
        """
        return self.n_test_runs


if __name__ == '__main__':
    args = lambda x: None
    args.n_ways = 5
    args.n_shots = 1
    args.n_queries = 12
    # args.data_root = 'data'
    args.data_root = '/home/yonglong/Data/tiered-imagenet-kwon'
    args.data_aug = True
    args.n_test_runs = 5
    args.n_aug_support_samples = 1
    imagenet = TieredImageNet(args, 'train')
    print(len(imagenet))
    print(imagenet.__getitem__(500)[0].shape)

    metaimagenet = MetaTieredImageNet(args)
    print(len(metaimagenet))
    print(metaimagenet.__getitem__(500)[0].size())
    print(metaimagenet.__getitem__(500)[1].shape)
    print(metaimagenet.__getitem__(500)[2].size())
    print(metaimagenet.__getitem__(500)[3].shape)
