from __future__ import print_function

import os
import pickle
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class Cub200(Dataset):
    def __init__(self, args, split='train', phase=None, is_sample=False, k=4096,
        transform=None):
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

        # with open(os.path.join(args.data_root, 'CIFAR_FS_train.pickle'), 'rb') as f:
        #     data = pickle.load(f, encoding='latin1')
        #     imgs1 = data['data']
        #     labels1 = data['labels']


        # with open(os.path.join(args.data_root, 'CIFAR_FS_test.pickle'), 'rb') as f:
        #     data = pickle.load(f, encoding='latin1')
        #     imgs2 = data['data']
        #     labels2 = data['labels']

        # with open(os.path.join(args.data_root, 'CIFAR_FS_val.pickle'), 'rb') as f:
        #     data = pickle.load(f, encoding='latin1')
        #     imgs3 = data['data']
        #     labels3 =  data['labels']

        # self.imgs = np.concatenate((imgs1, imgs2, imgs3), axis=0)
        # self.labels = np.concatenate((labels1, labels2, labels3), axis = 0)

        self.imgs = []
        self.labels = []
        iter_here = 0
        lowest_width = 1000
        lowest_height = 1000
        lowest_depth = 1000
        faulty = 0
        # Iterate over all folders inside the main folder
        for folder_name in os.listdir(r'/nfs4/anurag/subspace-reg/data/CUB_200_2011/CUB_200_2011/images'):
            folder_dir = os.path.join(r'/nfs4/anurag/subspace-reg/data/CUB_200_2011/CUB_200_2011/images', folder_name)
            if not os.path.isdir(folder_dir):
                continue

            # Iterate over all image files inside each folder
            for file_name in os.listdir(folder_dir):
                file_path = os.path.join(folder_dir, file_name)
                if not os.path.isfile(file_path):
                    continue

                # Open the image and convert it to a numpy array
                image = Image.open(file_path)
                image = image.resize((120, 120))
                image_array = np.array(image)
                
                lowest_depth = min(image_array.shape[0], lowest_depth)
                lowest_height = min(image_array.shape[1], lowest_height)
                try:
                    lowest_width = min(image_array.shape[2], lowest_width)
                    self.imgs.append(image_array)
                    self.labels.append(iter_here)
                except Exception as e:
                    pass
                # Add the image array to the list
                
            
            
            iter_here += 1
        
        print(lowest_depth)
        print(lowest_width)
        print(lowest_height)
        print(faulty)
        self.imgs = np.stack(self.imgs)
        # print(len(self.imgs))
        # print(len(self.labels))
        # print(self.imgs[0].shape)
        # print(self.imgs[0:3])
        # print(self.labels[0:3])
        if args.continual: # Multi-session.
            all_classes = np.arange(200)
            np.random.shuffle(all_classes) # Shuffles in place.
            basec = np.sort(all_classes[:100])
            
            # Create mapping for base classes as they are not consecutive anymore.
            self.basec_map = dict(zip(basec, np.arange(len(basec))))
            
            valc = all_classes[100:]
            
            if split == "train":
                base_samples = [i for i, e in enumerate(self.labels) if e in basec]
                
                np.random.shuffle(base_samples) # Shuffle the images that belong in base classes.
                nbc = len(basec)
                ttrain, tval, ttest = base_samples[:50*nbc], base_samples[50*nbc:50*nbc+5*nbc], base_samples[50*nbc+5*nbc:]
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
                self.main_classes = ['Black_footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross', 'Groove_billed_Ani', 'Crested_Auklet', 'Least_Auklet', 'Parakeet_Auklet', 'Rhinoceros_Auklet', 'Brewer_Blackbird', 'Red_winged_Blackbird', 'Rusty_Blackbird', 'Yellow_headed_Blackbird', 'Bobolink', 'Indigo_Bunting', 'Lazuli_Bunting', 'Painted_Bunting', 'Cardinal', 'Spotted_Catbird', 'Gray_Catbird', 'Yellow_breasted_Chat', 'Eastern_Towhee', 'Chuck_will_Widow', 'Brandt_Cormorant', 'Red_faced_Cormorant', 'Pelagic_Cormorant', 'Bronzed_Cowbird', 'Shiny_Cowbird', 'Brown_Creeper', 'American_Crow', 'Fish_Crow', 'Black_billed_Cuckoo', 'Mangrove_Cuckoo', 'Yellow_billed_Cuckoo', 'Gray_crowned_Rosy_Finch', 'Purple_Finch', 'Northern_Flicker', 'Acadian_Flycatcher', 'Great_Crested_Flycatcher', 'Least_Flycatcher', 'Olive_sided_Flycatcher', 'Scissor_tailed_Flycatcher', 'Vermilion_Flycatcher', 'Yellow_bellied_Flycatcher', 'Frigatebird', 'Northern_Fulmar', 'Gadwall', 'American_Goldfinch', 'European_Goldfinch', 'Boat_tailed_Grackle', 'Eared_Grebe', 'Horned_Grebe', 'Pied_billed_Grebe', 'Western_Grebe', 'Blue_Grosbeak', 'Evening_Grosbeak', 'Pine_Grosbeak', 'Rose_breasted_Grosbeak', 'Pigeon_Guillemot', 'California_Gull', 'Glaucous_winged_Gull', 'Heermann_Gull', 'Herring_Gull', 'Ivory_Gull', 'Ring_billed_Gull', 'Slaty_backed_Gull', 'Western_Gull', 'Anna_Hummingbird', 'Ruby_throated_Hummingbird', 'Rufous_Hummingbird', 'Green_Violetear', 'Long_tailed_Jaeger', 'Pomarine_Jaeger', 'Blue_Jay', 'Florida_Jay', 'Green_Jay', 'Dark_eyed_Junco', 'Tropical_Kingbird', 'Gray_Kingbird', 'Belted_Kingfisher', 'Green_Kingfisher', 'Pied_Kingfisher', 'Ringed_Kingfisher', 'White_breasted_Kingfisher', 'Red_legged_Kittiwake', 'Horned_Lark', 'Pacific_Loon', 'Mallard', 'Western_Meadowlark', 'Hooded_Merganser', 'Red_breasted_Merganser', 'Mockingbird', 'Nighthawk', 'Clark_Nutcracker', 'White_breasted_Nuthatch', 'Baltimore_Oriole', 'Hooded_Oriole', 'Orchard_Oriole', 'Scott_Oriole', 'Ovenbird', 'Brown_Pelican', 'White_Pelican', 'Western_Wood_Pewee', 'Sayornis', 'American_Pipit', 'Whip_poor_Will', 'Horned_Puffin', 'Common_Raven', 'White_necked_Raven', 'American_Redstart', 'Geococcyx', 'Loggerhead_Shrike', 'Great_Grey_Shrike', 'Baird_Sparrow', 'Black_throated_Sparrow', 'Brewer_Sparrow', 'Chipping_Sparrow', 'Clay_colored_Sparrow', 'House_Sparrow', 'Field_Sparrow', 'Fox_Sparrow', 'Grasshopper_Sparrow', 'Harris_Sparrow', 'Henslow_Sparrow', 'Le_Conte_Sparrow', 'Lincoln_Sparrow', 'Nelson_Sharp_tailed_Sparrow', 'Savannah_Sparrow', 'Seaside_Sparrow', 'Song_Sparrow', 'Tree_Sparrow', 'Vesper_Sparrow', 'White_crowned_Sparrow', 'White_throated_Sparrow', 'Cape_Glossy_Starling', 'Bank_Swallow', 'Barn_Swallow', 'Cliff_Swallow', 'Tree_Swallow', 'Scarlet_Tanager', 'Summer_Tanager', 'Artic_Tern', 'Black_Tern', 'Caspian_Tern', 'Common_Tern', 'Elegant_Tern', 'Forsters_Tern', 'Least_Tern', 'Green_tailed_Towhee', 'Brown_Thrasher', 'Sage_Thrasher', 'Black_capped_Vireo', 'Blue_headed_Vireo', 'Philadelphia_Vireo', 'Red_eyed_Vireo', 'Warbling_Vireo', 'White_eyed_Vireo', 'Yellow_throated_Vireo', 'Bay_breasted_Warbler', 'Black_and_white_Warbler', 'Black_throated_Blue_Warbler', 'Blue_winged_Warbler', 'Canada_Warbler', 'Cape_May_Warbler', 'Cerulean_Warbler', 'Chestnut_sided_Warbler', 'Golden_winged_Warbler', 'Hooded_Warbler', 'Kentucky_Warbler', 'Magnolia_Warbler', 'Mourning_Warbler', 'Myrtle_Warbler', 'Nashville_Warbler', 'Orange_crowned_Warbler', 'Palm_Warbler', 'Pine_Warbler', 'Prairie_Warbler', 'Prothonotary_Warbler', 'Swainson_Warbler', 'Tennessee_Warbler', 'Wilson_Warbler', 'Worm_eating_Warbler', 'Yellow_Warbler', 'Northern_Waterthrush', 'Louisiana_Waterthrush', 'Bohemian_Waxwing', 'Cedar_Waxwing', 'American_Three_toed_Woodpecker', 'Pileated_Woodpecker', 'Red_bellied_Woodpecker', 'Red_cockaded_Woodpecker', 'Red_headed_Woodpecker', 'Downy_Woodpecker', 'Bewick_Wren', 'Cactus_Wren', 'Carolina_Wren', 'House_Wren', 'Marsh_Wren', 'Rock_Wren', 'Winter_Wren', 'Common_Yellowthroat']

                self.label2human = [""]*200
                print("Printing basec map")
                print(self.basec_map)
                for i in self.basec_map:
                    self.label2human[self.basec_map[i]] = " ".join(self.main_classes[i].split('_'))
                print(self.label2human)   

            elif split == "val":
                print("=============going into split val==============")
                self.main_classes = ['Black_footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross', 'Groove_billed_Ani', 'Crested_Auklet', 'Least_Auklet', 'Parakeet_Auklet', 'Rhinoceros_Auklet', 'Brewer_Blackbird', 'Red_winged_Blackbird', 'Rusty_Blackbird', 'Yellow_headed_Blackbird', 'Bobolink', 'Indigo_Bunting', 'Lazuli_Bunting', 'Painted_Bunting', 'Cardinal', 'Spotted_Catbird', 'Gray_Catbird', 'Yellow_breasted_Chat', 'Eastern_Towhee', 'Chuck_will_Widow', 'Brandt_Cormorant', 'Red_faced_Cormorant', 'Pelagic_Cormorant', 'Bronzed_Cowbird', 'Shiny_Cowbird', 'Brown_Creeper', 'American_Crow', 'Fish_Crow', 'Black_billed_Cuckoo', 'Mangrove_Cuckoo', 'Yellow_billed_Cuckoo', 'Gray_crowned_Rosy_Finch', 'Purple_Finch', 'Northern_Flicker', 'Acadian_Flycatcher', 'Great_Crested_Flycatcher', 'Least_Flycatcher', 'Olive_sided_Flycatcher', 'Scissor_tailed_Flycatcher', 'Vermilion_Flycatcher', 'Yellow_bellied_Flycatcher', 'Frigatebird', 'Northern_Fulmar', 'Gadwall', 'American_Goldfinch', 'European_Goldfinch', 'Boat_tailed_Grackle', 'Eared_Grebe', 'Horned_Grebe', 'Pied_billed_Grebe', 'Western_Grebe', 'Blue_Grosbeak', 'Evening_Grosbeak', 'Pine_Grosbeak', 'Rose_breasted_Grosbeak', 'Pigeon_Guillemot', 'California_Gull', 'Glaucous_winged_Gull', 'Heermann_Gull', 'Herring_Gull', 'Ivory_Gull', 'Ring_billed_Gull', 'Slaty_backed_Gull', 'Western_Gull', 'Anna_Hummingbird', 'Ruby_throated_Hummingbird', 'Rufous_Hummingbird', 'Green_Violetear', 'Long_tailed_Jaeger', 'Pomarine_Jaeger', 'Blue_Jay', 'Florida_Jay', 'Green_Jay', 'Dark_eyed_Junco', 'Tropical_Kingbird', 'Gray_Kingbird', 'Belted_Kingfisher', 'Green_Kingfisher', 'Pied_Kingfisher', 'Ringed_Kingfisher', 'White_breasted_Kingfisher', 'Red_legged_Kittiwake', 'Horned_Lark', 'Pacific_Loon', 'Mallard', 'Western_Meadowlark', 'Hooded_Merganser', 'Red_breasted_Merganser', 'Mockingbird', 'Nighthawk', 'Clark_Nutcracker', 'White_breasted_Nuthatch', 'Baltimore_Oriole', 'Hooded_Oriole', 'Orchard_Oriole', 'Scott_Oriole', 'Ovenbird', 'Brown_Pelican', 'White_Pelican', 'Western_Wood_Pewee', 'Sayornis', 'American_Pipit', 'Whip_poor_Will', 'Horned_Puffin', 'Common_Raven', 'White_necked_Raven', 'American_Redstart', 'Geococcyx', 'Loggerhead_Shrike', 'Great_Grey_Shrike', 'Baird_Sparrow', 'Black_throated_Sparrow', 'Brewer_Sparrow', 'Chipping_Sparrow', 'Clay_colored_Sparrow', 'House_Sparrow', 'Field_Sparrow', 'Fox_Sparrow', 'Grasshopper_Sparrow', 'Harris_Sparrow', 'Henslow_Sparrow', 'Le_Conte_Sparrow', 'Lincoln_Sparrow', 'Nelson_Sharp_tailed_Sparrow', 'Savannah_Sparrow', 'Seaside_Sparrow', 'Song_Sparrow', 'Tree_Sparrow', 'Vesper_Sparrow', 'White_crowned_Sparrow', 'White_throated_Sparrow', 'Cape_Glossy_Starling', 'Bank_Swallow', 'Barn_Swallow', 'Cliff_Swallow', 'Tree_Swallow', 'Scarlet_Tanager', 'Summer_Tanager', 'Artic_Tern', 'Black_Tern', 'Caspian_Tern', 'Common_Tern', 'Elegant_Tern', 'Forsters_Tern', 'Least_Tern', 'Green_tailed_Towhee', 'Brown_Thrasher', 'Sage_Thrasher', 'Black_capped_Vireo', 'Blue_headed_Vireo', 'Philadelphia_Vireo', 'Red_eyed_Vireo', 'Warbling_Vireo', 'White_eyed_Vireo', 'Yellow_throated_Vireo', 'Bay_breasted_Warbler', 'Black_and_white_Warbler', 'Black_throated_Blue_Warbler', 'Blue_winged_Warbler', 'Canada_Warbler', 'Cape_May_Warbler', 'Cerulean_Warbler', 'Chestnut_sided_Warbler', 'Golden_winged_Warbler', 'Hooded_Warbler', 'Kentucky_Warbler', 'Magnolia_Warbler', 'Mourning_Warbler', 'Myrtle_Warbler', 'Nashville_Warbler', 'Orange_crowned_Warbler', 'Palm_Warbler', 'Pine_Warbler', 'Prairie_Warbler', 'Prothonotary_Warbler', 'Swainson_Warbler', 'Tennessee_Warbler', 'Wilson_Warbler', 'Worm_eating_Warbler', 'Yellow_Warbler', 'Northern_Waterthrush', 'Louisiana_Waterthrush', 'Bohemian_Waxwing', 'Cedar_Waxwing', 'American_Three_toed_Woodpecker', 'Pileated_Woodpecker', 'Red_bellied_Woodpecker', 'Red_cockaded_Woodpecker', 'Red_headed_Woodpecker', 'Downy_Woodpecker', 'Bewick_Wren', 'Cactus_Wren', 'Carolina_Wren', 'House_Wren', 'Marsh_Wren', 'Rock_Wren', 'Winter_Wren', 'Common_Yellowthroat']
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
                self.label2human = [""]*200
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
        return len(self.labels)

class MetaCub200(Cub200):

    def __init__(self, args, split, phase=None, train_transform=None, test_transform=None, fix_seed=True, use_episodes = False, disjoint_classes=False):
        super(MetaCub200, self).__init__(args, split, phase)
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

        print(train_transform)
        print(test_transform)

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
                transforms.RandomCrop(84, padding=8),
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
        if (self.split == "train" and self.phase == "train"):
            if self.disjoint_classes:
                return 8
            return self.n_test_runs
        elif self.use_episodes:
            return len(self.episode_query_ids)
        else:
            return self.n_test_runs