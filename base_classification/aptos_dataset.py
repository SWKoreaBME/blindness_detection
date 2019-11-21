import os
import cv2
import torch
import pandas as pd

from preprocessing import preprocessing

class aptos_dataset(object):
    """
        input : data folder path

        output : train, test, validation set
        [numpy array with mini batch]

        method : None

    """

    def __init__(self, d_path = './', label_file = './', preprocess = True):
        super(aptos_dataset, self).__init__()

        self.data_path = d_path
        self.image_list = os.listdir(d_path)
        self.train_csv = pd.read_csv(label_file)
        self.preprocess = preprocess

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

    def resize(self, image, target_shape=(299, 299)):
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