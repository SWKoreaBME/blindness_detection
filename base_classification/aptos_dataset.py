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

    def __init__(self, d_path = './', label_file = './', preprocess = True, da = True, da_method = None):
        super(aptos_dataset, self).__init__()

        self.data_path = d_path
        self.image_list = os.listdir(d_path)
        self.train_csv = pd.read_csv(label_file)
        self.preprocess = preprocess
        self.da = da
        self.da_method = da_method

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        # read image
        image_name = self.image_list[index]
        img_id = image_name.rstrip('.png')

        image = self.read_image(os.path.join(self.data_path, image_name))
        if self.preprocess:
            image = preprocessing(image)
        # image = self.resize(image)
        # if len(image.shape) != 3:
        #     image = self.expand_channel(image)

        # get labels
        label = self.read_label(img_id)

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