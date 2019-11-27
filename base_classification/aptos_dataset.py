import os
import cv2
import torch
import pandas as pd
import numpy as np

from preprocessing import preprocessing
from torch.utils.data import SubsetRandomSampler

class aptos_dataset(object):
    """
        input : data folder path

        output : train, test, validation set
        [numpy array with mini batch]

        method : None

    """

    def __init__(self, d_path = './', da_root_path = './', label_file = './', preprocess = True, da_method = None, da = True):

        super(aptos_dataset, self).__init__()

        self.data_path = d_path

        self.image_list = os.listdir(d_path)
        self.da = da
        if da:
            self.da_image_list, self.da_label_list = self.make_da_image_list()

        self.da_root_path = da_root_path

        self.train_csv = pd.read_csv(label_file)
        self.preprocess = preprocess
        self.da_method = da_method

    def __len__(self):
        if not self.da:
            return len(self.image_list)
        else:
            return len(self.da_image_list)

    def make_da_image_list(self):

        labels = os.listdir(self.da_root_path)
        da_image_list = []
        da_label_list = []

        original_image_list = []
        original_label_list = []

        for label in labels:

            label_list = [os.path.join(self.da_root_path, label, a) for a in os.listdir(os.path.join(self.da_root_path, label))]
            
            da_image_list.extend(label_list)
            da_label_list.extend([int(label)] * len(label_list))

        for image_name in self.image_list:

            label = self.read_label(image_name.rstrip('.png'))
            original_label_list.append(label)
            original_image_list.append(os.path.join(self.data_path, image_name))

        da_image_list.extend(original_image_list)
        da_label_list.extend(original_label_list)

        return da_image_list, da_label_list

    def __getitem__(self, index):
        
        if not self.da:
            # read image
            image_name = self.image_list[index]
            img_id = image_name.rstrip('.png')

            image = self.read_image(os.path.join(self.data_path, image_name))

            # get labels
            label = self.read_label(img_id)

        else:
            image_name = self.da_image_list[index]
            image = self.read_image(image_name)
            label = self.da_label_list[index]

        if self.preprocess:
            image = preprocessing(image)
        else:
            image = self.resize(image)
            if len(image.shape) != 3:
                image = self.expand_channel(image)

        # To Tensor
        image = self.ToTensor(image)

        sample = {'image' : image, 'label' : label}

        return sample

    #TODO : Images with half sizes should be labeled individually

    def resize(self, image, target_shape=(256, 256)):
        # 299 is fixed size of inception v3 network
        resized_image = cv2.resize(image, target_shape, interpolation = cv2.INTER_CUBIC)
        return resized_image

    def expand_channel(self, image):
        # merges single channel image to 3 channel image ( all same images )
        expanded_image = cv2.merge((image, image, image))
        return expanded_image

    def read_label(self, img_id):
        return int(self.train_csv[self.train_csv['id_code'] == img_id]['diagnosis'])

    def read_image(self, image_name):
        return cv2.imread(image_name)

    def ToTensor(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)

    # TODO : split data into train, validation set

    def split_train_val():

        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        return train_sampler, valid_sampler

def dataloaders(dataset, validation_split = 0.2, shuffle_dataset = True, random_seed = 102, batch_size = 4):

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                            sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)

    dataloaders = dict(train = train_loader, val = validation_loader)
    dataset_sizes = dict(train = len(train_sampler), val = len(valid_sampler))

    return dataloaders, dataset_sizes