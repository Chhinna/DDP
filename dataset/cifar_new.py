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
    def __init__(self, args, split='train', phase=None, is_sample=False, k=4096,
                 transform=None):
        """Initializes the dataset for training, validation or testing
        Args: 
            args: Namespace of arguments
            split: String indicating if dataset is for 'train', 'val' or 'test'
            phase: String indicating 'train', 'val' or 'test' phase
            is_sample: Boolean indicating if samples need to be generated
            k: Integer number of negative samples per positive sample
            transform: Transform to apply on samples
        Returns:
            None
        Processing Logic:
            - Loads CIFAR-FS dataset from pickle files
            - Applies transforms like normalization, random crop etc
            - Splits data into train, val and test based on args.continual, phase and split
            - Maps labels to consecutive numbers
            - Generates positive and negative sample indices if is_sample is True
        """
        super(Dataset, self).__init__()
        self.data_root = args.data_root
        self.split = split
        self.data_aug = args.data_aug
        self.mean = [0.5071, 0.4867, 0.4408]
        self.std = [0.2675, 0.2565, 0.2761]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.unnormalize = transforms.Normalize(mean=-np.array(self.mean)/self.std, std=1/np.array(self.std))

        np.random.seed(args.set_seed)

        if transform is None:
            if self.split == 'train' and self.data_aug:
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

        with open(os.path.join(args.data_root, 'CIFAR_FS_train.pickle'), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            imgs1 = data['data']
            labels1 = data['labels']


        with open(os.path.join(args.data_root, 'CIFAR_FS_test.pickle'), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            imgs2 = data['data']
            labels2 = data['labels']

        with open(os.path.join(args.data_root, 'CIFAR_FS_val.pickle'), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            imgs3 = data['data']
            labels3 =  data['labels']

        self.imgs = np.concatenate((imgs1, imgs2, imgs3), axis=0)
        self.labels = np.concatenate((labels1, labels2, labels3), axis = 0)
        
        if args.continual: # Multi-session.
            all_classes = np.arange(100)
            np.random.shuffle(all_classes) # Shuffles in place.
            basec = np.sort(all_classes[:60])
            
            # Create mapping for base classes as they are not consecutive anymore.
            self.basec_map = dict(zip(basec, np.arange(len(basec))))
            
            valc = all_classes[60:]
            
            if split == "train":
                base_samples = [i for i, e in enumerate(self.labels) if e in basec]
                
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
                self.main_classes = [    "apple",    "aquarium_fish",    "baby",    "bear",    "beaver",    "bed",    "bee",    "beetle",    "bicycle",    "bottle",    "bowl",    "boy",    "bridge",    "bus",    "butterfly",    "camel",    "can",    "castle",    "caterpillar",    "cattle",    "chair",    "chimpanzee",    "clock",    "cloud",    "cockroach",    "couch",    "crab",    "crocodile",    "cup",    "dinosaur",    "dolphin",    "elephant",    "flatfish",    "forest",    "fox",    "girl",    "hamster",    "house",    "kangaroo",    "keyboard",    "lamp",    "lawn_mower",    "leopard",    "lion",    "lizard",    "lobster",    "man",    "maple_tree",    "motorcycle",    "mountain",    "mouse",    "mushroom",    "oak_tree",    "orange",    "orchid",    "otter",    "palm_tree",    "pear",    "pickup_truck",    "pine_tree",    "plain",    "plate",    "poppy",    "porcupine",    "possum",    "rabbit",    "raccoon",    "ray",    "road",    "rocket",    "rose",    "sea",    "seal",    "shark",    "shrew",    "skunk",    "skyscraper",    "snail",    "snake",    "spider",    "squirrel",    "streetcar",    "sunflower",    "sweet_pepper",    "table",    "tank",    "telephone",    "television",    "tiger",    "tractor",    "train",    "trout",    "tulip",    "turtle",    "wardrobe",    "whale",    "willow_tree",    "wolf",    "woman",    "worm"]

                self.label2human = [""]*100
                print("Printing basec map")
                print(self.basec_map)
                for i in self.basec_map:
                    self.label2human[self.basec_map[i]] = " ".join(self.main_classes[i].split('_'))
                print(self.label2human)   

            elif split == "val":
                print("=============going into split val==============")
                self.main_classes = [    "apple",    "aquarium_fish",    "baby",    "bear",    "beaver",    "bed",    "bee",    "beetle",    "bicycle",    "bottle",    "bowl",    "boy",    "bridge",    "bus",    "butterfly",    "camel",    "can",    "castle",    "caterpillar",    "cattle",    "chair",    "chimpanzee",    "clock",    "cloud",    "cockroach",    "couch",    "crab",    "crocodile",    "cup",    "dinosaur",    "dolphin",    "elephant",    "flatfish",    "forest",    "fox",    "girl",    "hamster",    "house",    "kangaroo",    "keyboard",    "lamp",    "lawn_mower",    "leopard",    "lion",    "lizard",    "lobster",    "man",    "maple_tree",    "motorcycle",    "mountain",    "mouse",    "mushroom",    "oak_tree",    "orange",    "orchid",    "otter",    "palm_tree",    "pear",    "pickup_truck",    "pine_tree",    "plain",    "plate",    "poppy",    "porcupine",    "possum",    "rabbit",    "raccoon",    "ray",    "road",    "rocket",    "rose",    "sea",    "seal",    "shark",    "shrew",    "skunk",    "skyscraper",    "snail",    "snake",    "spider",    "squirrel",    "streetcar",    "sunflower",    "sweet_pepper",    "table",    "tank",    "telephone",    "television",    "tiger",    "tractor",    "train",    "trout",    "tulip",    "turtle",    "wardrobe",    "whale",    "willow_tree",    "wolf",    "woman",    "worm"]
                val_samples = [i for i, e in enumerate(self.labels) if e in valc]
                
                val_samples = np.array(val_samples)
                self.labels = [self.labels[i] for i in val_samples] 
                self.imgs = self.imgs[val_samples,:] 
                print("++++++++++++++Printing valc++++++++++++")
                print(valc)
                print(len(self.labels))
                print(len(self.imgs))
                # Set the specific cat2label dict for val classes.
                print(valc)
                self.label2human = [""]*100
                for i in valc:
                    self.label2human[i] = " ".join(self.main_classes[i].split('_'))
                print(self.label2human)
        
        
        # This has to have all the 100 classes
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
            item: Index of item to retrieve. 
        Returns:
            img: Image at index as tensor.
            target: Label of image as integer. 
            item: Index of retrieved item.
        Processing Logic:
            - Retrieve image and label at given index from dataset
            - Transform image to tensor
            - Normalize label
            - Return image, label and index
            - Optionally also return negative samples if sampling
        """
        img = np.asarray(self.imgs[item]).astype('uint8')
        img = self.transform(img)
        target = self.labels[item] - min(self.labels)
        if not self.is_sample:
            return img, target, item
        else:
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
        Calculates the length of the labels attribute by calling len() on it."""
        return len(self.labels)

class MetaCIFAR100(CIFAR100):

    def __init__(self, args, split, phase=None, train_transform=None, test_transform=None, fix_seed=True, use_episodes = False, disjoint_classes=False):
        """
        Initialize the MetaCIFAR100 dataset
        Args: 
            args: Arguments - Contains hyperparameters
            split: Dataset split (train/val/test)
            phase: Dataset phase (train/test) 
            train_transform: Transformations for training data
            test_transform: Transformations for test data
            fix_seed: Fix random seed
            use_episodes: Use episode sampling 
            disjoint_classes: Use disjoint classes
        Returns:
            None
        Processing Logic:
            1. Initialize hyperparameters like n_ways, n_shots etc from args
            2. Apply train and test transformations 
            3. Split data into classes and shuffle classes if fix_seed is True
        """
        super(MetaCIFAR100, self).__init__(args, split, phase)
        self.fix_seed = fix_seed
        self.n_ways = args.n_ways
        self.n_shots = args.n_shots
        self.n_queries = args.n_queries
        self.classes = []
        self.n_test_runs = args.n_test_runs
        self.eval_mode = args.eval_mode
        self.n_aug_support_samples = args.n_aug_support_samples
        self.n_base_aug_support_samples = args.n_base_aug_support_samples
        self.n_base_support_samples = args.n_base_support_samples
        self.use_episodes = use_episodes
        self.phase = phase
        self.split = split
        self.disjoint_classes = disjoint_classes
        print("eval_mode")
        print(self.eval_mode)
        print(self.n_aug_support_samples)
        print(self.n_base_aug_support_samples)
        print(self.n_base_support_samples)

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

        if test_transform is None:
            self.test_transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.test_transform = test_transform

        self.data = {}
        print(len(self.imgs))
        for idx in range(self.imgs.shape[0]):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.imgs[idx])
        self.classes = list(self.data.keys())
        print("====[[[printing classes]]]=======")
        print(self.classes)
        if self.fix_seed:
            np.random.seed(args.set_seed)
            np.random.shuffle(self.classes)

    def __getitem__(self, item):
        """
        Generate episodes for few-shot learning
        Args: 
            self: The class instance
            item: The index of the episode
        Returns:
            support_xs: Support set images
            support_ys: Support set labels 
            query_xs: Query set images
            query_ys: Query set labels
        Processing Logic:
            1. Sample support and query sets from the dataset for each class
            2. Apply data augmentation and normalization to the images
            3. Return support and query sets for the episode
        """
        print("using episodes")
        print(self.use_episodes)

        if not self.use_episodes:
            
            if self.split == "train" and self.phase == "train" and self.n_base_support_samples > 0:
                    assert self.n_base_support_samples > 0
                    # These samples will be stored in memory for every episode.
                    support_xs = []
                    support_ys = []
                    print("==================")
                    print(self.fix_seed)
                    print(item)
                    if self.fix_seed:
                        np.random.seed(item)
                    print(self.classes)
                    print(len(self.classes))
                    cls_sampled = np.random.choice(self.classes, len(self.classes), False)
                    print(cls_sampled)
                    print(cls_sampled.shape)
                    for idx, cls in enumerate(np.sort(cls_sampled)):
                        imgs = np.asarray(self.data[cls]).astype('uint8')
                        support_xs_ids_sampled = np.random.choice(range(imgs.shape[0]),
                                                                  self.n_base_support_samples,
                                                                  False)
                        support_xs.append(imgs[support_xs_ids_sampled])
                        support_ys.append([cls] * self.n_base_support_samples)    
                    support_xs, support_ys = np.array(support_xs), np.array(support_ys)
                    num_ways, n_queries_per_way, height, width, channel = support_xs.shape
                    print("here")
                    print(num_ways)
                    print(n_queries_per_way)
                    print(height)
                    print(width)
                    print(channel)
                    support_xs = support_xs.reshape((-1, height, width, channel))
                    if self.n_base_aug_support_samples > 1:
                        support_xs = np.tile(support_xs, (self.n_base_aug_support_samples, 1, 1, 1))
                        support_ys = np.tile(support_ys.reshape((-1, )), (self.n_base_aug_support_samples))
                    support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
                    print(support_xs[0].shape)
                    support_xs = torch.stack(list(map(lambda x: self.train_transform(x.squeeze()), support_xs)))

                    # Dummy query.
                    query_xs = support_xs
                    query_ys = support_ys
            else:
                print("I was hereeeeeeeee")
                if self.fix_seed:
                    np.random.seed(item)
                print(self.n_ways)
                print(self.classes)
                print(self.classes[:self.n_ways])
                # Only 5 novel classes should be sampled
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
                    if self.eval_mode in ["few-shot-incremental-fine-tune"]:
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

                support_xs = torch.stack(list(map(lambda x: self.train_transform(x.squeeze()), support_xs)))
                query_xs = torch.stack(list(map(lambda x: self.test_transform(x.squeeze()), query_xs)))
            
        
        return support_xs.float(), support_ys, query_xs.float(), query_ys
        
    def __len__(self):
        """
        Returns the length of the dataset
        Args:
            self: The dataset object
        Returns:
            length: The length of the dataset
        - Checks if dataset is in train split and train phase, returns number of test runs if disjoint classes else returns test runs
        - If using episodes, returns length of episode query ids 
        - Else returns number of test runs"""
        if (self.split == "train" and self.phase == "train"):
            if self.disjoint_classes:
                return 8
            return self.n_test_runs
        elif self.use_episodes:
            return len(self.episode_query_ids)
        else:
            return self.n_test_runs

