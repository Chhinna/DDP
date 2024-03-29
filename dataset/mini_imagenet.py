# This script is largely based on https://github.com/WangYueFt/rfs

import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import re

# torch.multiprocessing.set_sharing_strategy('file_system')
class ImageNet(Dataset):
    def __init__(self, 
                 args, 
                 split='train',
                 phase=None,
                 is_sample=False, 
                 k=4096,
                 transform=None):
        """Initializes a Dataset object
        
        Args: 
            args: Dataset configuration arguments
            split: Dataset split ('train', 'val', 'test')
            phase: Dataset phase ('train', 'val', 'test') for continual learning
            is_sample: Whether to sample positive/negative pairs
            k: Number of negative pairs per positive sample
            transform: Image transforms
        
        Returns:
            self: Dataset object
        
        Processing Logic:
            1. Sets attributes like normalization params
            2. Applies transforms to images 
            3. Loads images and labels from file
            4. Maps labels to integers
            5. Samples positive/negative pairs if is_sample is True
        """
        super(Dataset, self).__init__()
        self.split = split
        self.phase = phase
        self.data_aug = args.data_aug
        self.mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        self.std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.unnormalize = transforms.Normalize(mean=-np.array(self.mean)/self.std, std=1/np.array(self.std))
        
        np.random.seed(args.set_seed)

        if transform is None:
            if self.split == 'train' and self.data_aug:
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


        if args.continual:
            file_pattern = "all.pickle" # data root should be data/continual
        else:
            if self.split == "train":
                file_pattern = 'miniImageNet_category_split_train_phase_{}.pickle'.format(phase)
            else:
                file_pattern = 'miniImageNet_category_split_{}.pickle'.format(split)
            
        self.data = {}
        with open(os.path.join(args.data_root, file_pattern), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            self.imgs = data['data']
            self.labels = data['labels']
            self.cat2label = data['catname2label']

        # If continual, we read the whole data.
        # Based on the seed and split read the classes. 
        # Filter accordingly.
        if args.continual: # Multi-session.
            all_classes = np.arange(100)
            np.random.shuffle(all_classes) # Shuffles in place.
            basec = np.sort(all_classes[:60])
            
            # Create mapping for base classes as they are not consecutive anymore.
            self.basec_map = dict(zip(basec, np.arange(len(basec))))
            
            valc = all_classes[60:]
            
            if split == "train":
                base_samples = [i for i, e in enumerate(data['labels']) if e in basec]
                
                np.random.shuffle(base_samples) # Shuffle the images that belong in base classes.
                nbc = len(basec)
                ttrain, tval, ttest = base_samples[:500*nbc], base_samples[500*nbc:500*nbc+50*nbc], base_samples[500*nbc+50*nbc:]
                ttrain, tval, ttest = np.array(ttrain), np.array(tval), np.array(ttest)
                if phase == "train":
                    self.labels = [self.labels[i] for i in ttrain] 
                    self.imgs = self.imgs[ttrain,:] 
                elif phase == "val":
                    self.labels = [self.labels[i] for i in tval] 
                    self.imgs = self.imgs[tval,:] 
                elif phase == "test":
                    self.labels = [self.labels[i] for i in ttest] 
                    self.imgs = self.imgs[ttest,:] 
                else:
                    raise ValueError(f"Phase {phase} is unrecognized for split train.")
                    
                # Map labels to be consecutive
                self.labels = [self.basec_map[e] for e in self.labels]
                
                # Set the specific cat2label dict for base classes.
                new_cat2label = {k: self.basec_map[v] for (k, v) in self.cat2label.items() if v in basec}
                
                
            
            elif split == "val":
                val_samples = [i for i, e in enumerate(data['labels']) if e in valc]
                val_samples = np.array(val_samples)
                self.labels = [self.labels[i] for i in val_samples] 
                self.imgs = self.imgs[val_samples,:] 
                
                # Set the specific cat2label dict for val classes.
                new_cat2label = {k: v for (k, v) in self.cat2label.items() if v in valc}
                

                
            else:
                raise ValueError(f"No such split as {split}.")


            self.cat2label = new_cat2label
        self.label2human = [""]*100 # Total of 100 classes in mini.

        # Labels are available by codes by default. Converting them into human readable labels.
        with open(os.path.join(args.data_root, 'class_labels.txt'), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                catname, humanname = line.strip().lower().split(' ')
                humanname = " ".join(humanname.split('_'))
                if catname in self.cat2label:
                    label = self.cat2label[catname]
                    self.label2human[label]= humanname
        
        
        self.global_labels = self.labels

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
            output: Image, target class, and other data
        Processing Logic:
            - Retrieve image and target from lists using index
            - Normalize target class
            - Optionally return negative samples
            - Return image, target, and other requested data
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


class MetaImageNet(ImageNet):

    def __init__(self, 
                 args, 
                 split, 
                 phase=None,
                 train_transform=None, 
                 test_transform=None, 
                 fix_seed=True,
                 use_episodes=False,
                 disjoint_classes=False):
        
        """
        Initialize a MetaImageNet dataset
        Args: 
            args: Namespace of arguments
            split: Dataset split (train/val/test) 
            phase: Dataset phase (train/val/test)
            train_transform: Transform for training samples
            test_transform: Transform for test samples
            fix_seed: Fix random seed
            use_episodes: Use pre-defined episodes
            disjoint_classes: Use disjoint classes for base/novel
        Returns: 
            N/A
        Processing Logic:
            - Initialize base attributes from args
            - Load episode support/query IDs if using episodes
            - Define train/test transforms 
            - Organize data into classes
            - Shuffle classes if fixing seed
        """
        super(MetaImageNet, self).__init__(args, split, phase)
        self.fix_seed = fix_seed
        self.n_ways = args.n_ways
        self.n_shots = args.n_shots
        self.n_queries = args.n_queries
        self.classes = list(self.data.keys())
        self.n_test_runs = args.n_test_runs
        self.eval_mode = args.eval_mode
        self.n_aug_support_samples = args.n_aug_support_samples
        self.n_base_aug_support_samples = args.n_base_aug_support_samples
        self.n_base_support_samples = args.n_base_support_samples
        self.use_episodes = use_episodes
        self.phase = phase
        self.split = split
        self.disjoint_classes = disjoint_classes
        
        if self.split != "train":
            assert self.phase is None

        if self.use_episodes: # this is to match the exact examples used in XtarNet.
            
            self.episode_support_ids = []
            self.episode_query_ids = []

            with open(os.path.join(args.data_root, f'episodes_{self.n_ways}_{self.n_shots}.txt'), 'r', encoding='utf-8') as f: # FIX: episodes is only for 5 shot x 5 way experiments
                is_val = True
                for line in f.readlines():
                    if line.startswith("TEST"):
                        is_val = False

                    if split == 'train' and (phase == "test" or is_val) and (phase == 'val' or not is_val):

                        if line.startswith("Base Query"):
                            arr = re.split(': ', line)[1].rstrip()
                            arr = list(map(int,filter(None,
                                  arr.lstrip('[').rstrip(']').split(" "))))
                            self.episode_query_ids.append(arr)

                    if (split == "val" and is_val) or (split == "test" and not is_val):

                        if line.startswith("Novel"):
                            arr = re.split(': ', line)[1].rstrip()
                            arr = list(map(int,filter(None,
                                  arr.lstrip('[').rstrip(']').split(","))))
                            if line.startswith("Novel Support"):
                                self.episode_support_ids.append(arr)
                            else:
                                self.episode_query_ids.append(arr)


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
        
        if self.fix_seed:
            np.random.seed(args.set_seed)
            np.random.shuffle(self.classes)
            

    def __getitem__(self, item):
        """
        Generates episodes for few-shot learning
        Args: 
            item: Episode index
        Returns:
            support_xs: Support images
            support_ys: Support labels 
            query_xs: Query images
            query_ys: Query labels
        Processing Logic:
        1. Samples support and query examples from dataset 
        2. Applies data augmentation and transformations
        3. Returns support and query examples in episode format
        4. Handles training, validation and test phases differently
        """
        if not self.use_episodes:
            
            if self.split == "train" and self.phase == "train" and self.n_base_support_samples > 0:
                    assert self.n_base_support_samples > 0
                    # These samples will be stored in memory for every episode.
                    support_xs = []
                    support_ys = []
                    if self.fix_seed:
                        np.random.seed(item)
                    cls_sampled = np.random.choice(self.classes, len(self.classes), False)
                    
                    for idx, cls in enumerate(np.sort(cls_sampled)):
                        imgs = np.asarray(self.data[cls]).astype('uint8')
                        support_xs_ids_sampled = np.random.choice(range(imgs.shape[0]),
                                                                  self.n_base_support_samples,
                                                                  False)
                        support_xs.append(imgs[support_xs_ids_sampled])
                        support_ys.append([cls] * self.n_base_support_samples)    
                    support_xs, support_ys = np.array(support_xs), np.array(support_ys)
                    num_ways, n_queries_per_way, height, width, channel = support_xs.shape
                    support_xs = support_xs.reshape((-1, height, width, channel))
                    if self.n_base_aug_support_samples > 1:
                        support_xs = np.tile(support_xs, (self.n_base_aug_support_samples, 1, 1, 1))
                        support_ys = np.tile(support_ys.reshape((-1, )), (self.n_base_aug_support_samples))
                    support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
                    support_xs = torch.stack([self.train_transform(x.squeeze()) for x in support_xs])

                    # Dummy query.
                    query_xs = support_xs
                    query_ys = support_ys
            else:
            
                if self.fix_seed:
                    np.random.seed(item)

                if self.disjoint_classes:
                    cls_sampled = self.classes[:self.n_ways] # 
                    self.classes = self.classes[self.n_ways:]
                else:
                    cls_sampled = np.random.choice(self.classes, self.n_ways, False)
                support_xs = []
                support_ys = []
                query_xs = []
                query_ys = []
                for idx, cls in enumerate(np.sort(cls_sampled)):
                    imgs = np.asarray(self.data[cls]).astype('uint8')
                    support_xs_ids_sampled = np.random.choice(range(imgs.shape[0]), self.n_shots, False)
                    support_xs.append(imgs[support_xs_ids_sampled])
                    lbl = idx
                    if self.eval_mode in {"few-shot-incremental-fine-tune"}:
                        lbl = cls
                    support_ys.append([lbl] * self.n_shots) #
                    query_xs_ids = np.setxor1d(np.arange(imgs.shape[0]), support_xs_ids_sampled)
                    query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)
                    query_xs.append(imgs[query_xs_ids])
                    query_ys.append([lbl] * query_xs_ids.shape[0]) #
                support_xs, support_ys, query_xs, query_ys = np.array(support_xs), np.array(support_ys), np.array(query_xs), np.array(query_ys)
                num_ways, n_queries_per_way, height, width, channel = query_xs.shape

                query_xs = query_xs.reshape((num_ways * n_queries_per_way, height, width, channel))
                query_ys = query_ys.reshape((num_ways * n_queries_per_way, ))

                support_xs = support_xs.reshape((-1, height, width, channel))
                if self.n_aug_support_samples > 1:
                    support_xs = np.tile(support_xs, (self.n_aug_support_samples, 1, 1, 1))
                    support_ys = np.tile(support_ys.reshape((-1, )), (self.n_aug_support_samples))
                support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
                query_xs = query_xs.reshape((-1, height, width, channel))
                query_xs = np.split(query_xs, query_xs.shape[0], axis=0)

                support_xs = torch.stack([self.train_transform(x.squeeze()) for x in support_xs])
                query_xs = torch.stack([self.test_transform(x.squeeze()) for x in query_xs])
            
        else: # to match XtarNet  
            if self.split == "train" and self.phase == "train":
                    assert self.n_base_support_samples > 0
                    # These samples will be stored in memory for every episode.
                    support_xs = []
                    support_ys = []
                    if self.fix_seed:
                        np.random.seed(item)
                    cls_sampled = np.random.choice(self.classes, len(self.classes), False)
                    
                    for idx, cls in enumerate(np.sort(cls_sampled)):
                        imgs = np.asarray(self.data[cls]).astype('uint8')
                        support_xs_ids_sampled = np.random.choice(range(imgs.shape[0]),
                                                                  self.n_base_support_samples,
                                                                  False)
                        support_xs.append(imgs[support_xs_ids_sampled])
                        support_ys.append([cls] * self.n_base_support_samples)    
                    support_xs, support_ys = np.array(support_xs), np.array(support_ys)
                    num_ways, n_queries_per_way, height, width, channel = support_xs.shape
                    support_xs = support_xs.reshape((-1, height, width, channel))
                    if self.n_base_aug_support_samples > 1:
                        support_xs = np.tile(support_xs, (self.n_base_aug_support_samples, 1, 1, 1))
                        support_ys = np.tile(support_ys.reshape((-1, )), (self.n_base_aug_support_samples))
                    support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
                    support_xs = torch.stack([self.train_transform(x.squeeze()) for x in support_xs])

                    # Dummy query.
                    query_xs = support_xs
                    query_ys = support_ys
                    
            else:
     
                # Actual query.
                query_xs_ids = self.episode_query_ids[item]
                query_xs = np.array(self.imgs[query_xs_ids])
                query_ys = np.array([self.labels[i] for i in query_xs_ids])
                _, height, width, channel = query_xs.shape
                num_ways, n_queries_per_way = (self.n_ways, len(query_xs_ids) // self.n_ways)

                query_xs = query_xs.reshape((num_ways * n_queries_per_way, height, width, channel))
                query_ys = query_ys.reshape((num_ways * n_queries_per_way, ))
                query_xs = query_xs.reshape((-1, height, width, channel))
                query_xs = np.split(query_xs, query_xs.shape[0], axis=0)
                query_xs = torch.stack([self.test_transform(x.squeeze()) for x in query_xs])
            
            
                if self.split == "train" and self.phase in {'val', "test"} :
                    # Dummy support if phase is val or test.
                    support_xs = query_xs.squeeze(0)
                    support_ys = query_ys
                
                else:
                    # Actual support.
                    support_xs_ids_sampled = self.episode_support_ids[item]
                    support_xs = np.array(self.imgs[support_xs_ids_sampled])

                    support_ys = np.array([self.labels[i] for i in support_xs_ids_sampled])
                    assert len(np.unique(support_ys)) == self.n_ways

                    support_xs = support_xs.reshape((-1, height, width, channel))
                    if self.n_aug_support_samples > 1:
                        support_xs = np.tile(support_xs, (self.n_aug_support_samples, 1, 1, 1))
                        support_ys = np.tile(support_ys.reshape((-1, )), (self.n_aug_support_samples))
                    support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
                    support_xs = torch.stack([self.train_transform(x.squeeze()) for x in support_xs])
                

        return support_xs.float(), support_ys, query_xs.float(), query_ys

    def __len__(self):
        """
        Returns the length of the dataset
        Args:
            self: The dataset object
        Returns:
            length: The length of the dataset
        - Checks if dataset is in train split and train phase, returns number of test runs if disjoint classes else returns n_test_runs
        - If using episodes, returns length of episode query ids 
        - Else returns n_test_runs"""
        if (self.split == "train" and self.phase == "train"):
            if self.disjoint_classes:
                return 8
            return self.n_test_runs
        if self.use_episodes:
            return len(self.episode_query_ids)
        return self.n_test_runs
            


if __name__ == '__main__':
    args = lambda x: None
    args.n_ways = 5
    args.n_shots = 1
    args.n_queries = 12
    args.data_root = 'data'
    args.data_aug = True
    args.n_test_runs = 5
    args.n_aug_support_samples = 1
    imagenet = ImageNet(args, 'val')
    print(len(imagenet))
    print(imagenet.__getitem__(500)[0].shape)

    metaimagenet = MetaImageNet(args)
    print(len(metaimagenet))
    print(metaimagenet.__getitem__(500)[0].size())
    print(metaimagenet.__getitem__(500)[1].shape)
    print(metaimagenet.__getitem__(500)[2].size())
    print(metaimagenet.__getitem__(500)[3].shape)